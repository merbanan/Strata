# QSA halving scorer and FP16 pooled keys

Status, 2026-10-22: source/data-layout mismatch confirmed; corrected standalone
CUDA builds and a separate engine link pass. GPU reproduction, sanitizer,
model-quality and performance checks remain pending. Nothing installed.

The local configuration enables `STRATA_IDX_F16=1`. The indexer allocates and
writes two bytes per pooled-key element. Turing enables `scores_halve_on()` by
default; for up to eight decode queries with no host active-block count,
`qsa_block_scores()` selects `block_scores_halve_kernel`. That kernel loads
`float4` directly from the pooled pointer, without checking the storage type.
The ordinary warp and multi-query scorers already use `ld_key4<true>` for FP16.
The dead/tail key remains FP32 in both storage modes.

This is a local integration interaction between the FP16 storage work
(`a76ed0c91c3749b498c31ac61feb284effd688cf`) and the Turing halving scorer
(`4a6d11a66a7e0e18708f405c42a1100296dfeaa1`, upstream
[PR #2072](https://github.com/Niko1221/Strata/pull/2072)). The source hash in the
current clean engine build matches the audited source. Current workload logs
also confirm FP16 storage. The original upstream scorer was tested with FP32
keys; this finding does not establish an upstream default-storage defect.

A CPU byte-layout witness packs two FP16 values of 1.0 as `003c003c`.
Interpreting those bytes as FP32 gives 0.007826805114746094. With constant-one
queries and keys, four heads of dimension 128 should score 512; the incorrect
load predicts 4.00732421875 for the first block. This is a source/data-layout
demonstration, not a GPU measurement. At larger active contexts the doubled
stride can also exceed the logical pooled-key allocation; whether a specific
engine run crosses an allocation boundary needs runtime evidence.

The isolated correction adds a storage-type template parameter and reuses
`ld_key4<F16>` in the halving scorer. Dispatch selects the storage variant;
the FP32 load, reduction order, ReLU, head sum and tail-key handling are
retained. No new numerical algorithm is introduced. Both CUDA targets build:

| Variant | sm75 registers (1/2/4 query slots) | sm70 registers | Spills |
| --- | --- | --- | --- |
| FP32 | 42 / 64 / 113 | 40 / 64 / 113 | 0 |
| FP16 | 45 / 64 / 112 | 45 / 64 / 112 | 0 |

Four-slot sm70 variants report a 16-byte stack frame; sm75 reports no stack
frame. These compiler resources do not establish performance. HIP was not
built, and V100 runtime testing remains outstanding.

The queued regression suite contains:

- An original-FP16 negative control with padded key backing, requiring the
  predicted score mismatch rather than accepting an arbitrary CUDA failure.
- Eight reference/candidate arms covering FP32, FP16, default halving,
  forced halving and the existing fallback. Each checks 60 graph captures and
  180 replays across 1–16 queries and contexts from 32 to 32768 tokens.
- Exact score bytes and selected IDs, causal/ragged tails, poisoned unused
  output, allocation guards, changing keys/queries/step records, and three
  sanitizers on the corrected FP16 path.

Nine host parser checks pass, including rejected missing/duplicate replays,
wrong storage metadata, absent completion markers, sanitizer warnings and
nonzero exits. They are synthetic protocol checks, not GPU qualification.

`STRATA_SCORES_HALVE=0` selects the existing storage-aware fallback and is a
candidate correctness control, not an adopted recommendation yet. Once the
GPU checks pass, compare the corrected engine with this fallback, then repeat
task-quality tests. The previous failed tasks cannot yet be attributed to this
issue, and existing throughput records remain measurements of their original
configuration rather than evidence for the corrected engine.

Artifacts under `Strata-data/bench/2026-10-21-volta-turing/`:

- `qsa-halve-f16-v1/`: source audit/witness, original and corrected source,
  patch, regression probe, build logs, host checks, pinned GPU queue and
  upstream PR snapshots.
- `qsa-halve-f16-engine-v1/`: clean engine link with only the corrected QSA
  object; no IQ2 fusion, router fusion or graph-cache prototype mixed in.

Public issue/PR searches for `STRATA_SCORES_HALVE` and `STRATA_IDX_F16` found
no matching storage-fix report in that search. PR #3233 concerns gfx12 scoring;
its saved diff retains the existing halving dispatch. This is a limited search,
not a claim that every upstream discussion was inspected for this interaction.


## Full-model qualification prepared

`qsa-halve-f16-fixed-v1` links the corrected score object and the previously
validated diagnostic draft-cost policy into the clean engine. A 1000-round
host negative control varies only measured timing: the original policy makes
987 different window choices, while the fixed-cost policy has zero differing
choices, chains or costs. This demonstrates why learned timing must be
controlled for numerical isolation; it does not validate model inference.

`qsa-halve-f16-workload-v1` queues the same 19 original/4K/16K/32K tasks and
full generation caps, in four ABBA legs. Prepared case bytes match the prior
suite exactly. Both arms use one corrected binary with FP16 keys; the base
selects the existing fallback (`STRATA_SCORES_HALVE=0`), the candidate selects
the corrected halving scorer (`=1`). Both disable adaptive swaps and deferred
CPU work, use blocking adaptation, fix the PCIe fraction to 0.11, and use
fixed draft costs. No IQ2 fusion is enabled.

The queue requires a fresh independent review of the complete kernel,
negative-control and sanitizer logs before starting GPU inference. It retains
all task failures, output tokens, prefix reuse and memory budgets. The workload
reviewer now recognizes this explicitly controlled comparison and rejects
missing fixed-policy evidence or drift in these controls. Nine host checks
pass. Replaying the prior 76 fused-prefill records preserves every task
classification. GPU task results, per-request allocation review and production
performance remain pending; this diagnostic alone cannot justify adoption.


## Timing comparison prepared

`qsa-halve-f16-timing-v1` builds a separate host harness against the corrected
sm75 object with `-Wall -Wextra -Werror`. Seven parser controls pass. It queues
eight arms: fallback/halving/halving/fallback separately for FP32 and FP16,
after the full-workload process releases the GPU and independent kernel review
passes. Each arm includes the existing 60-graph/180-replay parity matrix plus
120 timing graphs, with 64 repeated calls per graph. Score-only and complete
score-plus-selection timings produce 19,200 samples across the eight arms.
Outputs and guards are checked again after timing. This is stationary synthetic
CUDA-event timing, not inference throughput; model promotion remains separate.


The production CUDA source now contains the same storage-aware correction as
the isolated candidate, with formatting-only differences. A separate
production-source compile succeeds for sm70 and sm75; the corrected halving
instances have no spills. The only reported spill is the pre-existing
33-wide register top-k instance, not the changed scorer. The production edit
remains uncommitted and uninstalled until GPU regression/model review finishes.
Evidence: `qsa-halve-f16-v1/production-source-validation/build.json`.
