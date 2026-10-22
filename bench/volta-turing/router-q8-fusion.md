# Router and expert-input Q8 fusion

Initial experiments, RTX 2060 SUPER, 2026-10-22. None of the three candidates
is selected for adoption. Original-selection fusion passes kernel and retrieval
checks but has no established end-to-end gain. Sorted lists are slower than
that fusion; winner-only exponentials change expert IDs for a small kernel-time
saving. The production source edits have been archived and restored to the
adopted baseline. The installed release was never changed.

The archived `STRATA_ROUTE_Q8_FUSE=1` proposal combines independent routing and expert-input
quantization blocks in one grid. It writes `nat_xq_`; it does not change the
shared expert's activation buffer. The existing full softmax and top-10
selection arithmetic remain intact, including tie handling and normalization
by winning-expert lane. Sorted lists and winner-only exponentials remain
separate C02 experiments.

The router translation unit uses fast math. Q8 reductions and division use
explicit PTX operations without flush-to-zero so the producer can match
`iq_kernels.cu`'s different arithmetic contract. Four warps per block distribute
the work better than the first eight-warp prototype. The existing router
kernels remain unchanged for the default path. The new wrapper returns false
without launching when disabled or unsupported, and checks its output spans.
HIP takes this fallback; it has not been built or runtime-tested here.

Integration covers both `moe_route`'s single-row path and the verifier's direct
multi-row path. The later expert quantizer is skipped only after successful
fusion. An already-produced QFUSE image prevents redundant fusion; QDEDUP can
reuse the image. These interactions still require engine coverage.

## Kernel evidence

`build-router-q8-probe.py BUILD OUT --warps 4` builds the independent probe
against the engine's real reference library. It extracts the routing row body
without changing its arithmetic and saves sources, commands and hashes. With
the integrated library, `probe --production` instead exercises the production
wrapper; repeat with the environment switch unset, `0` and `1`.

The initial sanitizer matrix has 468 cases: rows 1/2/3/4/5/6/7/8/9/10/12/16/31,
columns 32/2560/5120, and 12 input patterns. It covers random values, ties,
large logits, subnormals, broad exponents, quantization thresholds from both
sides, half saturation, infinities and NaNs. IDs, weights, Q8 bytes and trailing
guards match exactly, including captured graphs replayed with changed inputs.
The actual production wrapper passes memcheck unset/0/1 and racecheck and
synccheck enabled, with zero errors or hazards. This establishes those kernel
contracts, not model quality or serving performance.

Timing uses ordinary finite activations, 128 operations per captured graph,
four warm-up replays and 16 timed replays. Each row count compares separate
route+Q8, fusion and route-only in order A/B/C/C/B/A. Route-only estimates the
extra wait a following CPU doorbell would incur; it does not measure actual
CPU overlap. Mean microseconds for the four-warp standalone prototype:

| Rows | Separate route+Q8 | Fused | Route only | Saved | Extra routing-completion wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.198 | 3.908 | 3.819 | 1.290 | 0.089 |
| 4 | 5.379 | 4.087 | 3.983 | 1.293 | 0.103 |
| 8 | 6.151 | 4.706 | 4.562 | 1.445 | 0.144 |
| 16 | 6.679 | 4.853 | 4.583 | 1.825 | 0.271 |

The eight-warp version regressed at 16 rows (about 6.99 versus 6.68 us).
Keep that result; the four-warp variant addresses it without omitting the
larger rows. The standalone compiler reports 78 registers, no spill loads or
stores, and a 64-byte stack frame. These are compiler resource counts, not
measured occupancy. The short microbenchmarks have limited thermal sampling.

## Validation scope and missing gates

The integrated sm75 engine builds; the modified CUDA translation unit also
compiles for sm70 and sm80 (compile-only, no runtime claims). Full-model ABBA checks 1K/4K/16K/32K
retrieval, first-token distributions against the adopted snapshot release,
unchanged source/binary identity, and observed fusion graph captures. These
first-token checks largely cover prefill, so longer decode and lookup-window
output checks remain necessary for this decode optimization. First-token
agreement alone cannot qualify it.

Before any future adoption, cover QFUSE/QDEDUP and n=1/n>1 engine paths, and measure
successful longer tasks and complete verification windows with matching cache,
chunks and accepted outputs. Remove the launch interposer and logit dumps from
timing configurations. Keep the local release unchanged until these gates
pass. All prototype versions, including regressions, are preserved under
`Strata-data/bench/2026-10-21-volta-turing/router-q8-fusion/`.

The longer comparison started after successful quality completion using
`decode-compare.py`. It retains the original two edit, fresh-code and prose
tasks, and separately names instruction-first versions of the same edit tasks.
It records every generated token ID, natural-stop/cap outcomes, prefix reuse,
cache budgets and both ABBA timing pairs. A qualified 1K warmup precedes timing.
Original failures remain evidence; they do not count as successful-task speedups.
Edit qualification uses the requested AST change, compilation without execution
and logger placement. Fresh code and prose still need independent review.
The direct-protocol runner passed a CPU smoke covering two requests, DONE
fields, trailing logs and clean QUIT. Instrumentation is absent from timing
configs. Both full-workload ABBAs finished before source restoration. The exact
experimental executable is `engine/strata-experimental`; the complete proposal
patch recreates all six archived source hashes from base `a189efb5`. The
temporary build still contains that experimental binary and must be rebuilt
before further engine work. The standalone probe works without the proposal;
its `--production` and fourth timing arm require applying it and rebuilding.

## Current model evidence

The completed retrieval ABBA recorded 480 successful fused-kernel captures in
candidate legs and none in baseline legs. Captured grids correspond to rows
1, 2 and 8 at 2560 columns. Capture records establish construction, not replay
counts. A second diagnostic interposer was compiled for graph replay tracking,
but its original queue declined the failed decode parity gate. A later isolated
compatibility check (`report-tool-compat/summary.json`) validates it on GPU:
78 graph lifetimes, 624 successful submissions and 312 fused-node submissions.
The expanded 624-case exact probe passes with the archived production API
unset/0/1. The current standalone helper also builds against baseline headers
without a reference to the proposed API, using the preserved reference library.
This is not a fresh baseline engine build. QFUSE/QDEDUP model pairs remain
untested. Neither interposer was enabled in the timing runs.

The first longer baseline failed all four edit variants despite natural stops.
The original rename did not rename the class and removed a method; the original
logging edit omitted logging and duplicated a dictionary entry. Instruction-first
rename did rename the class but removed a method; instruction-first logging added
the requested lines but duplicated several methods. These failures are retained.

Independent sandboxed checks of the generated LRU module passed ordinary LRU,
get/put/delete, independent TTL/update and a short concurrent-access smoke.
The exact TTL-deadline check failed: the entry remains accessible at its expiry.
An expired entry also displaced a live entry, recorded as a limitation because
the prompt did not specify expired-capacity policy. The prose has the requested
length and topics but incorrectly generalizes loss recovery and describes growth
during idle periods; compare [RFC 5681 sections 3.1 and 4.1](https://www.rfc-editor.org/rfc/rfc5681.html).
These baseline outputs do not establish successful-task throughput. Review
artifacts and hashes live alongside the run; the original task records remain
unchanged. Full-token equality across arms is a separate fusion-correctness gate.

The first baseline also reached 26–30 MiB free VRAM. Preserve that observation
with the graph-memory TODO and compare all arms' budgets and telemetry before
interpreting timing. Do not change reserve or chunk settings mid-comparison.

The first candidate matches the original edits and fresh-code token sequences,
but its prose first differs at token index 90 and its instruction-first logging
output also differs. Repeated candidate runs vary too. Source inspection found
that the selected `--adapt-async 1` configuration explicitly permits run-to-run
differences: the window where an expert moves to GPU execution depends on copy
completion, and GPU/CPU arithmetic differs (`src/program/generate.cpp`, adaptive
tier implementation). The strict whole-output parity gate therefore used a
configuration that cannot promise it. Preserve its failure; do not attribute
the differences to fusion without a controlled comparison. The audit now
separately reports within-arm repeats, acceptance, prefix reuse and pair timings.
The final baseline's logging edit also differs from its first baseline repeat,
confirming that variation is present with fusion disabled. A synchronous-
adaptation ABBA retained all original tasks and also completed 28 requests.
The original timings remain workload observations, not a speedup. The original
interaction queue's parity-gate refusal is retained.

The async ABBA completed all 28 requests with clean engine exits and matching
1403 cache slots, 6400-token chunks and 39.60 GiB pinned expert budgets. All
requests stopped naturally. The instruction-first rename is the only longer
task with identical outputs across all four legs, but it fails the requested
edit and has different acceptance counts. Its wall-time pairs change by +6.82%
and -4.64%; there is no consistent end-to-end win to adopt. Other mismatched
outputs and acceptance counts prevent treating their pair timings as controlled
speedups. The original interaction queue did not run because its parity gate
failed; this is recorded, not waived.

The first synchronous-adaptation pair also differs on fresh code, prose and
instruction-first logging. The original logging edit has identical output but
different suffix-window counts (141 baseline, 145 candidate). Async copy
completion is therefore not a complete explanation. `DraftPolicy` chooses
windows from measured round costs; blocking adaptation still follows windows,
and deferred CPU computation is enabled. Those interactions are hypotheses to
isolate, not proof that the fused kernel is responsible or exonerated. The full
control again varies within both arms. Its original rename is identical across
all four outputs, but fails the edit and has different acceptance in the second
pair; wall-time changes are -2.51% and +0.65%. Neither ABBA qualifies adoption.

## Sorted-selection result

`router-sorted-selection.cuh` is a standalone prototype, not engine code. It
sorts each lane's 16 post-softmax probabilities with a fixed comparator network,
then merges the lane-local lists for ten winners. Ties use the original expert
index; selected probabilities still accumulate in their winning lanes. Full
softmax, Q8 arithmetic and normalization remain intact. Build it with
`build-router-q8-probe.py BUILD OUT --warps 4 --selection sorted`.

A host mirror passed 6,400 sort/tie cases. The sm75 standalone build then passed
all 468 exact-byte cases, guards and changing-input graph replay. Memcheck and
synccheck report zero errors; racecheck reports zero errors and warnings. The
compiler reports 75 registers, zero stack bytes and no spills.

The same captured-graph A/B/C/C/B/A microbenchmark gives these mean microseconds
at 2560 columns. The route-only baseline uses the original unsorted router.

| Rows | Separate route + Q8 | Sorted fused | Original route only |
| --- | ---: | ---: | ---: |
| 1 | 5.361 | 5.347 | 3.941 |
| 2 | 5.325 | 5.389 | 3.984 |
| 4 | 5.545 | 5.464 | 4.112 |
| 8 | 6.340 | 6.461 | 4.707 |
| 16 | 6.885 | 6.581 | 4.732 |
| 31 | 7.666 | 6.825 | 4.735 |

The 1/2/8-row shapes observed in this engine gain essentially nothing or regress,
while routing completion is about 1.4–1.8 us later than the original route-only
kernel. Larger rows do save combined kernel time, but do not establish a win
for this engine's split windows. This first sorted-list variant is not being
integrated or adopted. The original fusion microbenchmarks were a separate run;
do not report their difference as a same-run comparison. All sources, hashes,
logs and the executable are preserved under `router-q8-fusion/sorted-selection-v1/`.

## Winner-only exponential experiment

`router-winner-exp.cuh` is a separate approximate standalone candidate. It
selects raw finite logits, exponentiates only the ten winners, redistributes
those probabilities to their winning lanes and applies the same reduction
shape. Nonfinite inputs fall back to the original full-softmax router. Even
with that fallback, raw-logit order and rounded-softmax order are not equivalent:
adjacent float32 logits near 0.01 can produce equal float32 exponentials. The
CUDA audit confirms changes in expert IDs for both this case and underflow ties.

The probe now adds underflow ties, adjacent finite logits, all-negative-infinity
and mixed-negative-infinity patterns: 624 cases. It also checks guard sentinels
against their initialized value independently in each arm. The optional
`--audit-route-differences` mode completes the matrix and records ID, weight and
nonfinite differences, but still exits with failure if any route bytes differ.
Q8 output and guard checks remain strict. The default mode still stops on the
first mismatch. Sanitizer success cannot turn numerical failure into parity.

Original and sorted controls pass all 624 expanded cases. The winner-only
candidate changes expert IDs in 78 cases and weight bytes in 308; the largest
finite difference between weight positions is 2.38418579e-7. That small weight
number does not bound the effect of choosing a different expert. Nonfinite
fallback output matches. Q8 bytes and independent guard checks remain exact.
Its strict run exits on a weight mismatch in the first random case; the full
audit also returns a parity failure. Memcheck/synccheck report zero errors and
racecheck reports zero errors or warnings, independently of numerical failure.
The compiler reports 81 registers, 128 stack bytes and no spills.

Same-run microtiming adds the integrated original-selection fusion as arm D,
using A/B/C/D/D/C/B/A and the same replay counts. Winner-only mean microseconds:

| Rows | Original fused | Winner-only fused | Change |
| --- | ---: | ---: | ---: |
| 1 | 4.038 | 3.886 | -3.76% |
| 2 | 4.085 | 3.963 | -2.98% |
| 8 | 4.849 | 4.848 | -0.01% |
| 16 | 5.003 | 4.917 | -1.72% |
| 31 | 5.324 | 5.172 | -2.87% |

The same-run sorted comparison is 31–37% slower than original fusion at rows
1/2/8. The original standalone implementation calibrates within about 1% of
the integrated implementation at those rows. These short microbenchmarks do
not establish a serving gain. The unguarded winner-only variant is not selected:
it changes routing for at most a few tenths of a microsecond saved here. No
model quality claim or integration follows from it. All builds, strict failures,
complete audits, sanitizer logs and timing pairs remain under
`router-q8-fusion/winner-exponentials-v1/`. The broader C02 lead and missing
engine qualification gates remain open if a substantially better variant is proposed.
