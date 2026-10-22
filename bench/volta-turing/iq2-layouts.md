# IQ2 storage and compact-codebook screening

Status, 2026-10-22: CPU codec/storage audit complete. A compact IQ2_XS
codebook prototype builds for sm70 and sm75; GPU qualification is queued
behind the existing inference campaign. No model weights or installed
configuration have changed.

## Upstream arithmetic and storage

1Cat's [IQ2 nibble work](https://github.com/1CatAI/1Cat-vLLM/pull/1011)
expands signed magnitudes at loading. Its corrected reader retains original
scales, FP16 reconstructed weights and activation boundaries, FP32 accumulation,
and the native reduction order. The
[arithmetic fix](https://github.com/1CatAI/1Cat-vLLM/commit/7e08fa26ff9bfb40b2f972ccd63ae602b6d993d1)
also restores contiguous gate/up reduction partitions and the intermediate
FP16 SiLU result. The earlier faster operator revision changed acceptance and
did not improve emitted-token latency. Those V100 projection results do not
qualify a Strata port.

Strata's current grouped expert path instead uses packed signed-byte DP4A,
integer local-scale arithmetic with truncating division, then floating-point
scaling/reduction and its existing SwiGLU/Q8 boundaries. Replacing that path
with the upstream FP16 projection reader would change arithmetic. A compatible
storage experiment must preserve Strata's own operations.

The upstream representation uses 128 code bytes and 16 metadata bytes per
256 weights. Comparing with the current GGUF blocks:

| Format | Original block bytes | Expanded bytes | Increase |
| --- | ---: | ---: | ---: |
| IQ2_XXS | 66 | 144 | 118.2% |
| IQ2_XS | 74 | 144 | 94.6% |
| IQ2_S | 82 | 144 | 75.6% |

The local pack has 29 IQ2 gate/up layers, 512 experts per layer, and gate/up
dimensions 2560/640. Expanding all of those weights would add 13,251,379,200
bytes (12.34 GiB) across the full expert set. This is a storage calculation,
not measured GPU or pinned-memory allocation. A selective conversion would
have a smaller cost that must be measured against actual cache residency.
No conversion has been performed; C04's GPU and acceptance checks remain open.

## Compact codebook prototype

The related C25 experiment keeps every packed model byte unchanged. Each
eight-magnitude IQ2_XS codebook entry contains only 8, 25 or 43. Encoding those
values as two-bit codes reduces the 512-entry table from 4096 to 1024 bytes.
The decoder spreads four codes into byte lanes and computes
`8 + 17 * code + (code >> 1)` per lane. Original sign handling and all later
integer/scaling operations remain unchanged. The generated table is derived
from the existing MIT-licensed ggml table; its source/license attribution is
retained in the prototype.

CPU checks cover all 6561 eight-value ternary vectors, all 65536 combinations
of the actual grid and sign code, and 131072 integer sub-dot/local-scale cases.
The compact-only patch reproduces the candidate source exactly from the frozen
IQ2_XS v2 source. CUDA 12.4 builds pass for both targets:

| Kernel resource | Existing v2 | Compact prototype |
| --- | ---: | ---: |
| Registers/thread | 62 | 63 |
| Static shared bytes/block | 5120 | 2048 |
| Dynamic shared bytes/block | 11520 | 11520 |
| Stack/spill bytes | 0 | 0 |

Fewer shared bytes do not establish a speedup: reconstruction adds integer
instructions, and actual occupancy has not been measured. The candidate is
isolated under the existing fusion opt-in and measured geometry; other formats
and disabled paths retain their original source. It is not linked into the
engine currently running continuation tests.

The queued GPU suite checks original/unset/zero/one plus v2-enabled controls,
300 route cases per arm, 24 graphs/144 changed-input replays per arm, and all
three sanitizers. Timing uses **v2/compact/compact/v2 with fusion enabled in
every leg**, retaining automatic/0/4/8 and separate phases. This isolates the
compact table from the already measured fusion gain. No V100 runtime result,
full-model gain or adoption is claimed.

An independent reviewer is prepared to recompute timing medians and compare
raw parity, graph and sanitizer logs. Five host checks pass using archived v2
logs as synthetic fixtures, including rejection of a disabled timing arm,
wrong binary hash, missing sample and pending run. These checks validate the
reviewer only; compact GPU results remain pending.

Evidence under `Strata-data/bench/2026-10-21-volta-turing/`:

- `iq2-layout-audit-v1/audit.py`, `audit.json`, `compact-grid.json`, and archived
  upstream patches: exact CPU checks, storage inventory and source provenance.
- `expert-conversion-audit-v1/iq2xs-gu-compact-v1/`: candidate source,
  compact-only patch, patch replay, compiler logs/resources, binaries and
  pinned GPU-test queue.

Compilation ran at lower CPU priority during continuation qualification.
Those workload timings are not used to claim a gain; the dedicated timing
queue remains separate.


## Compact-table GPU result (RTX 2060 SUPER, 2026-10-22)

`iq2xs-gu-compact-v1/results-review.json` independently reviews 6000 exact
parity comparisons, 120 changed-input graphs/720 replays, three clean
sanitizers and 115,200 ABBA timing samples. The populated VRAM whole-call
shape loses at T=1, T=4 and T=8 in both format pairs and both ABBA pairs;
T=16 is mixed. For IQ2_XS gate/up with type-20 down, T=1 v2/compact speed
ratios are 0.882/0.916 in the two pairs, and T=4 ratios are 0.888/0.927.
A ratio below 1 means the compact table is slower. The unchanged mode-0 and
down-path controls are retained in the report. This prototype is not selected
for engine integration despite its lower static shared-memory use. These are
synthetic-weight timings, so they do not prove a model speed result.

`iq2xs-realweights-v1/results-review.json` checks all 47,769,600 copied bytes
against 90 GGUF/index slices from 30 experts in ten IQ2_XS layers, then reviews
five arms and 6000 exact active Q8/output/guard comparisons in four modes.
Activations are synthetic; there is no real activation replay or engine timing.
The real-weight result strengthens correctness coverage of the v2 fusion but
does not override failed continuation/task-quality gates.
