# IQ2_XS grouped gate/up fusion experiment

Status, 2026-10-22: both the isolated mode-8 proposal and its extension to
Turing's normal mode 4 pass GPU correctness, sanitizers and reviewed repeated
kernel timings. The clean engine build is complete; disabled logits match the
installed reference. Enabled/disabled model checks are running. Neither
proposal is installed or adopted.

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

## First isolated proposal

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

## Completed first-proposal validation

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

Independent review of the v1 raw logs and pinned inputs is in
`iq2xs-gu-v1/results-review.json`: 4,800 exact comparisons, 24 captured graphs,
144 changed-input replays, all three sanitizers clean, and 115,200 timing
samples. Enabled IQ2_XS VRAM graphs have two kernels; disabled/fallback graphs
have three. Original/unset/zero/one output hashes match across all 300 cases.

For balanced VRAM shapes, complete-call disabled/enabled ratios in the two
ABBA pairs are:

| Down format | Tokens | First pair | Second pair |
| --- | ---: | ---: | ---: |
| IQ4_NL | 1 | 1.126 | 1.181 |
| IQ4_NL | 4 | 1.154 | 1.216 |
| IQ4_NL | 8 | 1.255 | 1.309 |
| IQ4_NL | 16 | 1.327 | 1.400 |
| Q2_0 | 1 | 1.171 | 1.205 |
| Q2_0 | 4 | 1.205 | 1.222 |
| Q2_0 | 8 | 1.312 | 1.347 |
| Q2_0 | 16 | 1.426 | 1.424 |

Both arms use explicit mode 8. Unchanged automatic/down controls show clock
drift between legs; the report retains raw costs and control-normalized ratios.
These are synthetic kernel results, not an engine speedup.

## Selective Turing dispatch and engine qualification

`iq2xs-gu-v2` reuses the exact v1 device-kernel body and additionally admits
effective mode 4 when `STRATA_IQ2XS_GU_FUSED=1`. On Turing this includes normal
automatic dispatch, so no global `STRATA_EXP_MODE` override is needed. Mode 0,
strided PCIe, unsupported geometry and other formats keep their old paths.
sm70/sm75 builds and complete patch replay pass.

V2 expands graph coverage to automatic/0/4/8: 24 graphs and 144 changed-input
replays per arm. All four 300-case controls and all three sanitizers passed.
`iq2xs-gu-v2/review-results.py` rechecks frozen inputs, all raw parity/graph/
sanitizer logs and 115,200 timing samples. It independently reconstructs
sample counts and medians, compares them with saved summaries, and checks
all timing-leg output hashes against the original control. Results are in
`iq2xs-gu-v2/results-review.json`.

For automatic dispatch, complete-call disabled/enabled ratios for balanced
VRAM shapes are:

| Down format | Tokens | First pair | Second pair |
| --- | ---: | ---: | ---: |
| IQ4_NL | 1 | 1.241 | 1.298 |
| IQ4_NL | 4 | 1.234 | 1.287 |
| IQ4_NL | 8 | 1.778 | 1.867 |
| IQ4_NL | 16 | 2.516 | 2.588 |
| Q2_0 | 1 | 1.301 | 1.337 |
| Q2_0 | 4 | 1.298 | 1.316 |
| Q2_0 | 8 | 1.857 | 1.889 |
| Q2_0 | 16 | 2.472 | 2.513 |

Unchanged mode-0 and down controls are retained with each comparison to
expose clock drift. Automatic dispatch is now an affected path, so it cannot
serve as the unchanged control used for v1. Two CPU compilation jobs overlapped
part of this run. These synthetic kernel ratios justify full-model testing;
they do not establish an inference speedup or V100 performance.

`iq2xs-engine-v1` builds a fresh production-source baseline and links only the
IQ2_XS CUDA object into a candidate. It does not reuse experimental fused-IQ
prefill or group-tail objects. Its queued model gate waits for both the exact
build and kernel-test processes, then runs four disabled controls against the
installed-reference logits, followed by 16 disabled/enabled ABBA requests.
Every first-logit byte must match the installed reference and observed memory
budgets must match. The local installed configuration remains unchanged;
performance/continuation qualification and adoption are separate later steps.

The clean build completed and all four disabled first-logit vectors match the
installed reference byte for byte. The retrieval ABBA is running.
`iq2xs-workload-v1` queues continuation checks behind that exact process and
requires all 16 reference matches plus a strict allocation audit. Its prepared
19 cases are byte-identical to the earlier workload (SHA256
`7262da4ecbd575c22755d08f91faabe91a77ff564d7b21da650e5fb3c1adc36a`):
seven original cases and twelve separately named 4K/16K/32K variants, 76
requests across ABBA. Only `STRATA_IQ2XS_GU_FUSED=0/1` differs. Memory budgets
must match; no diagnostic relaxation is enabled. Existing task failures remain
failures even if continuations are identical. Full-model performance and
adoption remain pending.
