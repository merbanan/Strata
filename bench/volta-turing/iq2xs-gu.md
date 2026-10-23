# IQ2_XS grouped gate/up fusion experiment

Status, 2026-10-22: both the isolated mode-8 proposal and its extension to
Turing's normal mode 4 pass GPU correctness, sanitizers and reviewed repeated
kernel timings. The clean engine build is complete; disabled logits match the
installed reference. All 16 enabled/disabled retrieval requests also pass
with exact logits. A corrected traced run now also passes independent
retrieval/allocation review. All 76 continuation requests are now independently
reviewed: four warmups qualify and all 72 substantive task responses fail.
Exact-continuation timing qualification declined. Neither proposal is adopted.

## Corrected-baseline full-workload decision (2026-10-23)

The fixed-policy one-switch ABBA on RTX 2060 SUPER completed all 76 original
requests with the same binary in both arms. Independent review qualifies 11/19
tasks per leg; all ten successful edits have identical tokens across four legs.
Startup and per-request allocations match, including verified window-only reads
for the two short requests in each leg. No task has identical non-timing logged
work in both pairs. On the successful equal-output edits, candidate/base decode
geometric means are 1.001x and 0.976x; prompt-processing means are 1.001x and
0.997x. There is no repeatable full-model speedup, so fusion is **not adopted**.

The saved binary, source provenance, complete workload, independent task and
allocation reviews, timing screen and promotion decision are under
`Strata-data/bench/2026-10-21-volta-turing/iq2xs-requalified-fixed-v1/`.
The isolated kernel gains below remain valid only at that scope. No V100 runtime
result is claimed.

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
installed reference byte for byte. The initial retrieval ABBA completed:
16/16 marker responses pass, with all first-logit bytes identical to the
installed reference. `iq2xs-engine-v1/quality-review.json` independently checks
raw responses, all 20 finite vectors (248320 values each), configuration
isolation and the original audit. Allocation qualification remains incomplete:
the configurations set `STRATA_PF_TRACE` but omitted `STRATA_TRACE`, so all four
logs lack per-request loan records. Matching startup budgets alone cannot pass.
The failed audit and stopped dependent queues are preserved.

`iq2xs-engine-trace-v1` repeats the same retrieval ABBA with `STRATA_TRACE=1`
in both arms. It completed with 16/16 exact responses and byte-identical first
logits. `iq2xs-engine-trace-v1/quality-review.json` independently rechecks raw
responses, finite vectors, configuration isolation and the allocation audit.
All request/loan records are complete and equal across ABBA. Startup budgets
also match: 1403 expert slots, chunk 6400, ring 96, maximum 1185 borrowed slots,
39.60 GiB pinned complement, and 675 MiB/303 extra MTP-room slots. It rechecks
the preserved disabled controls and does not rebuild or change the candidate.
`iq2xs-workload-v1/run-queue-resume1.py` queues continuation checks behind that exact process and
requires all 16 reference matches plus a strict allocation audit. Its prepared
19 cases are byte-identical to the earlier workload (SHA256
`7262da4ecbd575c22755d08f91faabe91a77ff564d7b21da650e5fb3c1adc36a`):
seven original cases and twelve separately named 4K/16K/32K variants, 76
requests across ABBA. Only `STRATA_IQ2XS_GU_FUSED=0/1` differs. Memory budgets
must match; no diagnostic relaxation is enabled. Existing task failures remain
failures even if continuations are identical. Full-model performance and
adoption remain pending.

The resumed continuation queue has passed those gates and is now running.

`iq2xs-performance-v1/run-queue-resume1.py` then waits for the continuation
job. It requires all 18 non-warmup tasks to retain identical emitted tokens
and prefix reuse across ABBA, plus matching memory budgets. The timing runner
uses the unchanged historical 256-token benchmark at 1K/4K/16K/32K, records
host/GPU telemetry, and requires matching requests, response text, generation
counts and finish reasons. Eleven host checks cover changed outputs, prefix
reuse, missing records, invalid timings and configuration drift. These are
host checks, not new performance results. Timing budgets still require final
independent review; capped output throughput does not imply task success.

The workload reviewer now checks the declared experiment and configuration
isolation. It retains fused-prefill coverage requirements for that experiment,
without imposing those unrelated kernels on IQ2_XS. Replaying all 76 earlier
fused-prefill records retains the same four warmup passes and 72 task failures.
IQ2_XS launch coverage is a separate gate, not inferred from prefill traces.

## Expert launch diagnostic

`trace-expert-launches.cpp` is a separate Linux/CUDA interposer for grouped
gate/up, down and standalone SwiGLU/Q8 kernels. It records launch shapes and
capture state, inspects graph nodes (including child graphs), and records
actual graph replays. `analyze-expert-launches.py` counts direct launches plus
replayed nodes, excluding capture records from submitted-kernel totals.
Unknown replay handles, incomplete metadata, failed CUDA calls and conditional
graphs fail the trace qualification. Current engine sources do not update
graphs or disable nodes; such mutations are outside this tracer's scope.

The interposer builds with `-Wall -Wextra -Werror`; eleven host parser checks
pass. GPU qualification is queued after performance, in
`expert-launch-trace-v1`. It first runs the existing 24-graph/144-replay tests
with the tracer on, requiring unchanged output and exact fused node counts.
Then it runs four retrieval contexts with each feature setting, requiring
matching output tokens and installed-reference logits while checking that
IQ2_XS fused kernels are submitted only when enabled. These remain pending
GPU checks. Full-process kernel counts include startup; launch capacities do
not reveal active routed groups. No timing or occupancy is inferred.


## Partial continuation divergence (2026-10-22)

The first 28 of 76 requests have completed and passed saved-text/token/protocol
review. All 19 baseline tasks are reviewed: one warmup pass and 18 substantive
task failures. The first nine candidate requests contain one warmup pass and
eight substantive failures. No independent response reviews remain pending
within this snapshot (`iq2xs-workload-v1/reviews/progress-5.json`).

Seven of the eight completed substantive baseline/candidate pairs have
different emitted token sequences; all have zero prefix reuse in both arms.
First differences occur at zero-based output indices 30 through 2344, depending
on the task. The instruction-first rename output matches exactly. Evidence:
`iq2xs-workload-v1/reviews/divergence-progress-1.json`, which pins the raw
record prefix and lists lengths, first differing tokens and finish reasons.

Both repeated arms are still pending. These observations fail the exact-output
requirement but do not yet distinguish feature effects from run-to-run
variation. Keep the ABBA run intact, compare repeated arms when complete, and
then isolate any reproducible difference. Retrieval parity and synthetic
kernel tests do not override continuation failures. The performance queue
must decline this run if the exact-continuation requirement remains failed;
no new inference throughput or adoption is qualified.


## Repeatability confound and controlled diagnostics

The original continuation configuration enables `--adapt-async 1`. Both logs
confirm the asynchronous tier is active. `src/program/generate.cpp` and
`docs/DETAILS.md` explicitly describe its run-to-run numerical variation:
copy completion determines which window first computes an expert on GPU,
whose rounding differs from CPU execution. In addition, the default PCIe
fraction is computed continuously from a startup bandwidth probe; the logs
print only two decimals. Identical printed fractions do not establish identical
routing decisions. The original exact-output gate therefore cannot, by itself,
isolate this kernel change. This confound should have been accounted for when
preparing the comparison.

The second candidate pass already differs from the first on `edit-rename`,
after 2343 common tokens. This establishes variation with the same feature
setting, without excluding a simultaneous feature effect. Preserve all four
original legs to assess repeats. Evidence: `iq2xs-workload-v1/reviews/divergence-progress-2.json`
and `iq2xs-blocking-diagnostic-v1/configuration-audit.json`.

Two additional diagnostics are queued, with exact process identities and
input hashes, after the existing GPU jobs:

- `iq2xs-realweights-v1`: unmodified gate/up/down slices for experts 0, 127 and
  511 in all ten IQ2_XS layers (47,769,600 bytes total). The host probe builds
  against original, v2 and compact kernels with warnings as errors. Five arms
  check 300 cases each, four modes, active Q8 bytes, full output and guards.
  Cases use 1/3/4/5/16 tokens, uneven groups, two seeds and synthetic activation
  standard deviations 0.03125/1/8. Weight slices retain offsets and hashes.
  These are real weights with synthetic activations, not actual activation
  replay or model inference. GPU results remain pending.
- `iq2xs-blocking-diagnostic-v1`: ABBA with `--adapt-async 0` and explicit
  `--pcie-frac 0.11` in both arms, holding the fusion switch as the arm
  difference. The original warmup plus fresh-code, prose and instruction-first
  logging prompts retain exact token IDs; the three continuations are capped
  at 256 tokens to cover early observed differences. First-logit vectors,
  emitted tokens, same-arm repeatability and observed memory budgets are
  retained. This selected diagnostic does not replace full-task qualification
  or production performance testing. Host preparation checks pass; GPU pending.

The first baseline and first candidate task reviews are complete. At the
40-request snapshot, all saved text matches emitted tokens, there are no
pending independent reviews, three warmups pass and 37 substantive responses
fail the requested tasks (`reviews/progress-7.json`). These failures do not
establish a candidate-specific regression. No feature has been adopted.


## Completed continuation review (2026-10-22)

`iq2xs-workload-v1/reviews/qualification.json` verifies all 76 original requests,
raw emitted tokens, saved responses and applicable protocol/prefill traces.
There are no pending manual reviews. Each arm has two passing warmups and
36 failed substantive responses. Invalid/incomplete code and background-only
answers remain failures; the original cache modules also fail the independent
expiration-boundary check.

`reviews/divergence-complete.json` compares the complete saved token vectors:
no substantive task is identical across all four legs. Baseline repeats match
on 4/18 tasks, candidate repeats on 1/18, and the two cross-arm pairs match on
2/18 and 3/18. Prefix reuse matches across all four legs for all 18 tasks.
This does not isolate a fusion defect: asynchronous placement, measured draft
costs and deferred CPU decisions remain uncontrolled in this original run.
The baseline QSA FP16 storage issue is separately under investigation.

The historical timing runner declined the exact-continuation/prefix gate at
`run-performance-resume1.py:95`; its trace is preserved. There is no new
qualified full-model timing or adoption result. The launch diagnostic is now
running; real-weight, compact-table and controlled execution tests remain queued.


## Model dispatch and full-workload allocations reviewed

The independent launch review reconciles all raw traces, graph controls, saved
model tokens and finite 248,320-value logits. Disabled/enabled runs submit
18,432/17,792 expert kernels in total, including startup. The enabled run
captures 50 IQ2_XS fused nodes and submits 640 through replay; disabled submits
none. All eight retrieval responses and first logits match their references.
These are submission counts, not per-request active groups or GPU timing.
Evidence: `expert-launch-trace-v1/results-review.json`.

The complete 76-request continuation allocation audit now matches startup
budgets and all observed per-request borrowed mappings across the four legs.
Requests 4/5 in each leg read entirely through verifier windows, so no prompt
buffer loan is expected: complete read counts, prefix position and refill
completion prove this. The original strict retrieval auditor is unchanged.
Eight host positive/negative controls pass. Evidence:
`iq2xs-workload-v1/reviews/allocation.json`. Matching allocation counts do not
prove equal expert identities and do not resolve the failed task/token gates.

The real-weight probe's 90 recorded slices were independently compared against
the GGUF and native-expert index: all 47,769,600 copied bytes match across 30
experts in ten layers. `iq2xs-realweights-v1/provenance-review.json` covers only
those slices; actual GPU parity is still pending and activations are synthetic.
