# K8V4 register-staged prompt attention

Validated on RTX 2060 SUPER (sm75), CUDA 12.4, IQ3_XXS, 2026-10-22.
`STRATA_PA_K8V4_STAGE=1` enables the kernel only on an actual sm75 device.
The global default remains off. Other devices and formats retain their existing
paths. Local adoption is limited to the measured IQ3_XXS configuration.

The kernel keeps query fragments in registers and prefetches INT8 K codes and
packed Q4 V blocks. Consumed K storage becomes score scratch. Each Q4 product
rounds to FP16 before p.v. Each warp computes an unscaled q.k group, and the
softmax reader applies the original sequential scale FMAs. A warp barrier
protects V and scale reuse between tiles. These choices preserve mode 3's
rounding order while reducing shared storage from 46,096 to 29,968 bytes.

## Full-model results

Two paired comparisons in ABBA order, one isolated binary with only the feature
switch changed. Both arms use 1,403 cache slots, 6,400-token prompt chunks,
39.60 GiB pinned expert RAM and a 575 MiB VRAM reserve. Requests generate up to
256 tokens. Reported values are medians; both long-context pairs improve.

| Prompt | Baseline prefill tok/s | Candidate prefill tok/s | Change | Complete request, baseline → candidate |
| --- | ---: | ---: | ---: | ---: |
| 1K | 285.09 | 287.83 | +0.96% | 12.431 → 12.363 s |
| 16K | 664.58 | 707.48 | +6.46% | 32.279 → 30.366 s |
| 32K | 688.96 | 736.11 | +6.84% | 55.457 → 52.215 s |

The 1K paired changes cross zero. The original 4K measurements are confounded
by first-loan pinned allocations: candidate 1 spent 1,379.3 ms in experts_lend,
and baseline 2 spent 28,299.6 ms, versus approximately 107 ms for ordinary loans.
The preceding 1K request has 1,023 tokens, below the 1,024-token lending threshold.
Do not attribute the apparent 60.5% 4K prefill improvement to this kernel.
A separate candidate/base confirmation pair measured 291.59 → 291.61 tok/s at
1K and 492.38 → 498.67 tok/s at 4K (+1.28%). Short-request latency is variable;
no strong short-context or decode improvement is established.

The isolated engine SHA-256 is
`240c316f8d326f5557c3ece3bef6356979566ac6bf83ca03935b34f9300890be`.
All campaign artifacts live under
`Strata-data/bench/2026-10-21-volta-turing/` in the surrounding workspace:

- `k8v4-stage-fma-isolated-abba/`: 16 timing rows, source/binary manifest,
  configuration snapshots, telemetry, and `audit.json` with paired TTFT/latency.
- `k8v4-stage-short-confirm/`: the additional 1K/4K pair and audit. Telemetry
  started partway through this confirmation; it does not cover the whole run.
- `k8v4-stage/isolated-build.json`: attention-only source and binary identity.

## Correctness and launch evidence

`k8v4-stage/screen-fma-final/` passed all 12 output SHA-256 comparisons against
both the disabled path and separately linked original v19 kernels. Cases cover
33/1500/4096/16384/32768-token contexts, full 4096-query prefill, shuffled pages,
ragged/empty selections, and graph replay with changed queries and poisoned
outputs. `screen-isolated/` repeats the full 4K-query check on the isolated build.
Median kernel speed ratios range from 1.292x to 1.532x; full 4K-query ratio is
1.462x. Kernel timing is separate from model timing above.

The FP64 oracle uses rounded V operands. Legacy decode retains unrounded Q4
products, so that comparison permits the existing FP16-boundary difference
(1e-3 of output scale); candidate-versus-baseline prompt comparisons require
byte equality, including signed zeros. Nonfinite outputs fail.

`memcheck-fma.log`, `racecheck-fma.log` and `synccheck-fma.log` report zero
errors/hazards at context 4096, 33 queries, graph replay and empty selection.
The all-format CUDA suite passes INT8/FP16/K8V4; pure Q4 correctly skips sm75.
The final translation unit compiles for sm70/sm75/sm80. Only sm75 runtime was
tested. V100, Ampere and HIP/SYCL runtime remain unqualified.

`k8v4-stage-fma-quality/` passed all 16 retrieval requests at 1K/4K/16K/32K,
with natural termination and byte-identical first-token logits (248,320 values
per request) across all four arms/repeats; KL is zero. That build contained
disabled wide experiments. `k8v4-stage-isolated-quality/` checks the isolated
candidate on the same four requests: exact answers and logits identical to the
previous baseline. These checks establish equivalence on these requests, not
broad model quality or full-continuation teacher-forced KL. The isolated kv1188
probe also passes its counting and colors checks.

Compiled candidate resources: 168 registers, no stack/spills, 29,968 shared
bytes per block. The current stage kernel already keeps its p.v output accumulator in register array `acc[8][4]` (source `qsa_prompt_k8v4_stage.cuh`). Future C08 work must first profile the remaining p.v loads/scales or launch-bound constraints; adding another register accumulator is not a new operation and may reduce the measured two-block occupancy. `k8v4-stage/occupancy-runtime.log` reports CUDA occupancy API
capacity increasing from one to two blocks per SM (12.5% → 25% theoretical
occupancy). Nsight Compute could not access counters (`ERR_NVGPUCTRPERM`).
Nsight Systems exited without a usable report; its raw stream is retained.
Neither attempt establishes achieved occupancy.

A lightweight CUDA launch interposer was checked on baseline/candidate parity
with equal outputs. `k8v4-stage/launch-trace-model/` then completed a real
4096-token prompt plus 256-token generation and recorded 12 successful direct
K8V4 stage launches (grid 4083×2×1, shared memory 29,968 bytes). This confirms
model dispatch; it does not measure GPU time or achieved occupancy.

## Reproduction

Use the surrounding workspace's CUDA build and existing model configurations:

```sh
CCACHE_DIR=/tmp/strata-v20-ccache cmake --build Strata-v20/build -j6 \
  --target qsa_prompt_attn_parity strata
python Strata-v20/bench/volta-turing/k8v4-screen.py \
  Strata-v20/build/qsa_prompt_attn_parity /tmp/k8v4-new-screen
STRATA_PA_K8V4_STAGE=1 QA_GRAPH=1 QA_EMPTY=1 compute-sanitizer \
  --tool racecheck --error-exitcode 1 \
  Strata-v20/build/qsa_prompt_attn_parity 4096 33 1 3
```

`k8v4-screen.py --reference-binary ORIGINAL` adds original-kernel comparisons;
`--cases 4096:4096` selects the full-query case. `prefill-quality.py BASE
CANDIDATE OUT` runs model retrieval/logit ABBA; `--arms candidate --reference-leg
PATH` compares an isolated build with saved baseline requests and logits.

`compare.sh LABEL BASE CANDIDATE [TARGETS] [PAIRS]` uses the local
`Strata-data/bench/{cfg_ab,kv1188}.sh` harnesses, which are workspace dependencies,
and records telemetry and source identity. `analyze-attention.py DIRECTORY`
requires complete results, matched prompts/configurations, memory budgets,
unchanged binary hashes and cold prefill before writing the latency audit.
Run these with the GPU and benchmark port 18080 free.

Compile the optional launch logger with:

```sh
g++ -O2 -std=c++20 -fPIC -shared \
  Strata-v20/bench/volta-turing/trace-attention-launches.cpp \
  -o /tmp/strata-attention-launches.so -ldl -pthread
python Strata-v20/bench/volta-turing/trace-attention.py \
  CANDIDATE_CONFIG SAVED_REQUEST /tmp/strata-attention-launches.so NEW_OUT
```

## Rejected prototype

The first prototype changed scale-FMA ordering. Its `k8v4-stage-abba/` timing
showed gains, but `k8v4-stage-quality/` reached first-token KL 0.214 at 32K;
it was rejected. Earlier `screen/` also contains a reproduced warp reuse race.
Those historical results are retained for diagnosis and are excluded from the
adoption evidence above. The corrected source preserves the original scale-FMA
order and includes the warp barrier.
