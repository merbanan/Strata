# IQ2_XS grouped gate/up fusion experiment

Status, 2026-10-22: an isolated CUDA proposal builds for sm_70 and sm_75.
GPU correctness, sanitizer and timing tests are queued after the existing
model-quality and tile runs. It is not installed or adopted.

## Measured reason to investigate

The installed IQ3_XXS pack uses ten gate/up/down format pairs across 48 layers.
Ten layers have IQ2_XS gate/up weights (type 17). This format lacks the
gate/up-to-Q8 epilogue used by mode 8 for the other IQ formats. The existing
default already combines standalone SwiGLU and Q8 conversion; this proposal
does not add another version of that standalone operation.

The baseline route experiment completed 120 cases, 480 exact-output/Q8
comparisons and 57,600 timing samples on the RTX 2060 SUPER. An independent
parser checked every sample, recomputed the summary medians and verified
the input hashes. For IQ2_XS/IQ4_NL VRAM calls:

| Tokens | Automatic complete call, us | Mode 0, us | Mode 8, us |
| --- | ---: | ---: | ---: |
| 1 | 128.59 | 118.07 | 115.44 |
| 4 | 490.74 | 443.93 | 430.29 |
| 8 | 838.79 | 571.57 | 580.27 |
| 16 | 1536.22 | 826.01 | 881.25 |

These are medians over 40 samples, with automatic/0/4/8/8/4/0/automatic
order in each of 20 measured rounds after five warm-up rounds. Each graph
contains 48 calls using nonoverlapping weight regions. Packed weights are
synthetic; formats and dimensions match the pack. T=1/4 use groups of one
entry, T=8 groups of two, and T=16 groups of four. Whole-call, GU plus epilogue,
and down timings were measured separately and are not additive.

Mode 8 currently falls back for IQ2_XS gate/up. These results motivate testing
its missing fusion; they do not show that the new kernel is faster. Mode 0 is
also a necessary control, especially at larger token counts. There are no
measured production route frequencies, CPU/PCIe transfer costs or full-model
speed gains in this experiment. The earlier full mode-8 inference comparison
did not establish a consistent decode win.

## Isolated proposal

`STRATA_IQ2XS_GU_FUSED=1` enables the proposed format only when the effective
expert mode is 8, the device is sm_70/sm_75, and dimensions are 2560/640.
Unset, zero and unsupported geometry retain the existing dispatch. The CUDA
strided-PCIe rule still forces mode 0. Other formats and modes retain their
old dispatch. New code is excluded under `__HIPCC__`; no HIP or SYCL runtime
qualification is claimed.

`GridOf<17>` supplies the 512-entry, 64-bit IQ2_XS table in shared memory.
`DotG<17>` copies the original dot-product body, changing only the table
pointer. It retains the byte-SIMD sign arithmetic, integer division and
floating-point scale order. The existing fused kernel supplies the reduction,
SwiGLU and Q8 rounding. Patch replay exactly reproduces the archived source.

CUDA 12.4 builds pass for both architectures. The new specialization uses
62 registers, 5,120 static shared bytes plus 11,520 dynamic shared bytes for
this geometry, and zero stack/spill bytes on both targets. These compiler
results do not measure runtime occupancy or V100 performance.

## Queued validation

- Original binary and candidate unset/zero/one controls: 300 cases per arm
  across all ten format pairs, automatic/0/4/8, empty/strided/VRAM routes,
  ragged groups and repeated four-entry stages. Check complete output bytes,
  active Q8 bytes, poisoned unused entries and allocation guards.
- Six captured graphs and 36 changed-input/group-plan replays per arm.
  Count kernel nodes to prove fusion selection: two for enabled IQ2_XS VRAM,
  three for the fallback. Check strided PCIe and unsupported geometry too.
- Memcheck, racecheck and synccheck on the graph/replay suite. Require clean
  exit, complete coverage and zero errors/warnings; missing/tracking reports
  cannot pass.
- Four timing legs, disabled/enabled/enabled/disabled, over 60 IQ2_XS cases.
  Keep automatic/0/4/8 phase timings and all raw samples. Review kernel results
  before preparing any full-engine dispatch or adoption change.

Host parser checks accept the real baseline timing log and reject missing
records, nonfinite timings, wrong graph dispatch, incomplete replays,
tracking warnings and nonzero sanitizer exits. Synthetic parser fixtures
are labelled and are not GPU test results.

Artifacts are under
`Strata-data/bench/2026-10-21-volta-turing/expert-conversion-audit-v1/`:

- `route-bench/timing-review.json`: reviewed baseline measurements.
- `iq2xs-gu-v1/proposal.patch`, `iq_kernels.base.cu`, `iq_kernels.cu`: exact proposal.
- `prepared.json`, `build-result.json`, `checks-build-result.json`,
  `compiler-resources.json`, `proposal-replay.json`: source/build provenance.
- `tests-prepared.json`, `queue-prepared.json`, `queue.json`: pinned tests and queue.
- `run-tests.py` checks inputs without GPU work by default; `--run` requires
  an idle GPU and creates a new `run/` directory without overwriting results.

The live source tree, frozen fused-prefill library and installed engine were
not replaced by these builds.
