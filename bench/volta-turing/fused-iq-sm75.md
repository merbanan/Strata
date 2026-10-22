# Native IQ prefill on Turing

Experimental port, 2026-10-22. Kernel validation passed; model qualification in progress; not adopted. Target: RTX 2060 SUPER, CUDA 12.4, IQ3_XXS.

The v14 change `540953d7ac292af48f5749473099b54bf55eada3` applies to the current
`src/prefill/moe_fused_iq.cu` without offsets or fuzz. A detached source copy and
patch are saved under
`Strata-data/bench/2026-10-21-volta-turing/fused-iq-sm75-v1/`. Production source
and the installed engine have not changed for this experiment.

The opt-in `STRATA_PF_FUSED_SM75_IQ=1`, alongside the existing fused/SM75
switches, admits the IQ gate/up and IQ4_NL down kernels on Turing. It sets up
two warps per row for every covered format and four where shared memory fits.
IQ2_S needs 67,840 bytes at four warps, beyond this card's 64 KiB limit, so a
layer containing it uses two. This does not enable a Volta implementation.

The old report (`2026-10-18-fusediq/TODO.md`) measured 1.2–1.67x isolated layer
speedups but no engine gain: 545.3 versus 549.1 prefill tokens/s. That comparison
also changed prompt chunks from 6656 to 6400 tokens. Its first-token KL of
0.35 against v13 is retained as an unresolved numerical result, not a quality
pass. Fused and MMQ activation quantization/accumulation differ; their outputs
are not expected to be byte-identical.

## Validation

The graph-cache full-model timing run finished before compilation/GPU work
began. v1 builds and passes all 11 independent format references. Q2 output
bytes match the original library with the new switch unset, zero and one.
Memcheck then reports two CUDA API errors from probing IQ2_S's oversized
four-warp shared-memory request; the reference calculation itself passes. The
engine build correctly declined that failed gate.

v2 adds a device-capacity check, including static shared memory, before calling
`cudaFuncSetAttribute`. It keeps the kernel arithmetic and two-warp fallback.
The corrected source and rerun are under `fused-iq-sm75-v2/`; all v1 evidence
is retained. All 11 references and exact Q2 controls pass. Memcheck and
synccheck report zero errors; racecheck reports zero hazards, errors or warnings
on IQ3_XXS/IQ4_NL at 256 tokens. Repeated full-layer timings (grouping,
quantization and both products included) are 1.634/1.635x at 2048 tokens,
1.311/1.311x at 6400, and 1.228/1.222x at 8192. These are isolated layer gains;
IQ output is not byte-identical to MMQ. See `micro-validation.json`.

- Compile the detached native IQ object and the existing independent reference
  test, with an optional canonical pair-order output dump.
- Compare the existing Q2 path's output bytes against the original library with
  the new switch unset, zero and one. Grouping storage order is normalized before
  comparison.
- Require all 11 supported format pairs to run against the double-precision
  reference and MMQ. Skips are failures for this experiment.
- Run memcheck, racecheck and synccheck on IQ3_XXS/IQ4_NL.
- Repeat full-layer timing at 2048, 6400 and 8192 tokens, including grouping,
  quantization and both expert products.

The builder reuses hashed, unmodified libraries and places the candidate object
before the archive; only its native IQ member is replaced. It strips CMake's
link dependency-output option so artifact links cannot overwrite build metadata.
Commands, source and library hashes are in `build-prepared.json`.

A separate checkout at `/tmp/strata-v20-iq-prefill`, based on `5e4d56b7`,
contains the admission patch plus resource-capacity fix. Its complete sm75
engine build passed after the microchecks. Binary SHA-256:
`e153886094b55c9172d8723a39dcc65f3ab1ef4b332e272fcff88287aeee1cf0`.
Pinned source/build evidence is in `engine-prepared.json` and
`engine-queue.json`; the graph-cache and router prototypes are absent.

## Original-planner model diagnostic

All 16 retrieval requests in the 1K/4K/16K/32K ABBA pass. Both baseline legs'
first logits match the adopted-release reference exactly. Each arm repeats
exactly. Baseline-to-fused first-token KL is 0, 0.0040404, 0.0172764 and
0.1086496, with unchanged top-1 on these four retrieval prompts. The 1K case
keeps MMQ; candidate logs confirm fused execution at longer prompts. This is
not a broad model-quality pass. See `model-diagnostic-audit.json`.

The allocation audit fails equality: startup chunks are 6400/6144 tokens,
maximum borrowed slots 1185/1192, pinned expert RAM 39.60/39.57 GiB. Cache slots
(1403), startup ring (96) and MTP room (675 MiB, 303 extra slots) agree. Actual
request loans are baseline/fused 317/317, 859/956, 1076/1132 and 1185/1192.
The 16K request uses 5632-token chunks in both arms, but still borrows different
numbers of slots. All traces are complete with no chunk-reduction warning.
No throughput claim uses this instrumented comparison.

`prefill-teacher.py` uses the existing `pin=P` protocol cut and short-read
windows to score a fixed 128-token tail after batched prefill. It verifies the
exact prefix/tail read paths, target IDs and score count, keeps within-arm
repeats, and reports full-vocabulary target NLL and top-1 agreement. Its top-64
lists do not establish full-distribution KL. Corpus repetitions are explicit;
these numerical probes do not replace full editing/code/prose task review.
The 4K code smoke test passed: exactly 4096 batched prefix tokens and 128
scored window positions, with fused execution confirmed. The complete sweep
finished successfully: five corpus segments, four context lengths, ABBA
(80 requests and 10,240 scored positions). `audit-teacher.py` independently
checks the raw score rows against the fixed target IDs, exact read-path traces
and saved per-case results. Both arms repeat all logged TSV bytes exactly;
these are rounded scores, not a proof of identical full logits or states.

Each context below pools five segments and 640 positions per arm. Both ABBA
pairs give the same results:

| Prefix | Top-1 agreement | NLL change (fused minus MMQ) | Perplexity ratio |
| --- | ---: | ---: | ---: |
| 1K | 100% | 0 | 1.0000 |
| 4K | 98.4375% | -0.07713 | 0.9258 |
| 16K | 92.1875% | +0.02298 | 1.0233 |
| 32K | 88.9063% | -0.00218 | 0.9978 |

Do not hide individual-case variation behind the averages: 16K code has
NLL +0.21054 and perplexity 1.2343x; 32K docs has only 82.8125% top-1 agreement
although its target NLL improves slightly. The original planner's allocation
changes are included, so this does not isolate kernel error. Raw repeated-corpus
scores do not qualify task quality. No adoption or throughput claim follows.
Evidence: `teacher-audit.json`, `teacher/summary.json`, and raw TSV/trace files.

## Model gates still open

Compare the isolated engine with identical
chunk size, expert residency, ring and MTP-loan settings. Current baseline logs
show 6400-token chunks, a 96-slot ring and 303 additional loaned slots. The ring
can be fixed with `STRATA_PREFILL_RING`; verify actual allocations in both arms.
Do not infer equality from identical configuration alone: the fused layout
changes scratch allocation and the automatic planner can choose other budgets.

`audit-prefill-allocation.py` compares observed startup slots/chunks/rings,
maximum borrowed slots, pinned RAM and MTP room, plus each request's loan
counts. Use `STRATA_TRACE=1` in separate diagnostic runs; do not use their
timings as throughput measurements. Missing trace coverage, incomplete prefill,
multi-device lending or chunk-reduction warnings cannot pass. Example:

```sh
python3 bench/volta-turing/audit-prefill-allocation.py base/engine.log candidate/engine.log \
  --expected-requests 4 --output allocation-comparison.json
```

Nine parser/gate checks pass, including mismatched rings, maximum/request loans,
prompt shapes, missing/incomplete traces and reduced chunks. A real untraced
baseline compared with itself correctly fails completeness. These checks use
synthetic request traces and are recorded in `allocation-audit-validation.json`;
they are not new GPU measurements. Equal counts do not prove equal expert
identities or owned bytes. Source audit: the existing loan trace prints the
requested chunk (`want`), so a logged reduction to `want_p` requires review.

Retain short/4K/16K/32K prompts and the full editing, code and prose workload.
Measure first-token and teacher-forced numerics plus task quality before treating
any timing as an adoption result. A kernel win alone is insufficient.

## Grouped scratch for fused layouts' short tails

A detached prototype is built under
`Strata-data/bench/2026-10-21-volta-turing/fused-iq-group-tail-v1/`.
`STRATA_PF_FUSED_GROUP_TAIL=1` requires existing group scratch support. It
reserves grouped GU/H/Hq fallback rows, keeps the full quantized input rows,
and sets the per-chunk group cap only when a fused layout runs a short MMQ
tail. Every later fused chunk resets that cap to zero. Unsupported-layer and
buffer bounds guards remain. Production source and the installed engine are
unchanged. The complete patch replays to the archived source byte-for-byte.

The source investigation found why an accounting-only change would be unsafe:
`run_impl` reuses the fused layout for its final small chunk; it does not
recarve it. The original layout retains `m.group_cap=0` there. Both reservation
and execution therefore need the new switch.

The prototype compiles without warnings (CUDA build's host translation unit).
Binary SHA-256:
`d7f49dc1a1c94456fd592efab464f03f882a50c94110860f838a82e285d981e7`.
Host probes use the extracted original/proposed allocation functions and
actual linked MMQ/fused byte calculators. Each of six runs checks 1,179,648
layout combinations (T=1..32768; thresholds 1024/3072/4096; 64/128/512 experts;
grouped and fused modes). Unset/zero controls preserve the original sizes;
enabled runs check 24,567 short-tail capacities each. Tests also cover cap
reset across full/short/full chunks and a forced group cap of 5120.

At T=6400, threshold=3072, 512 experts and the normal group-cap rule, the four
MoE buffers calculate to 359,712,384 -> 177,012,608 bytes (343.0 -> 168.8 MiB).
This is buffer arithmetic, not observed VRAM savings: the enclosing arena is
the maximum of attention, MoE and other scratch requirements.

The teacher-forced sweep finished and the queued handoff checked its process
identity and successful completion. All GPU boundary controls now pass, with
pinned inputs. It compares the original engine and
the new binary with unset/zero/one flags, fixed 4096-token chunks, CPU share
off and an explicit 3072 streaming threshold. Fifteen tail cases include
1/63/64/65, 511, 1023/1024/1025, 2047/2048/2049, 3071/3072/3073, then a full
chunk to test reset. Exact chunk and group-cap traces are required, and unset
and zero first logits must match the original engine byte-for-byte. A separate
full-engine memcheck covers the 3071-token tail. Model numerics, actual budget
changes, original task quality and speed remain pending; the prototype is not
adopted. See `queue.json` and `allocation-probe-results.json`.

The completed boundary test has 60 requests: 15 each for the original binary,
new binary with the switch unset, zero, and one. All three new-binary arms
match all original first-logit bytes. The expected full/short chunk sequence
and enabled group-cap selection/reset traces pass, with clean exits. An
independent raw-vector/config/trace audit confirms finite 248,320-entry logits
and all 60 request shapes. Observed loans are 956 slots for original/unset/zero
and 844 with the switch on, at the same fixed 4096-token chunk: 112 fewer
resident mappings displaced. This is not a global VRAM-byte or speed result.
Full-engine boundary memcheck now passes under the instrumentation settings
below. Full model/automatic-planner task qualification remains pending.
Evidence: `boundary-audit.json` and `boundaries/summary.json`.

## Compiler resource audit

`fused-iq-sm75-v2/compiler-resource-audit.json` maps all 19 ptxas entries to
format, projection and warp layout. IQ3_XXS gate/up uses 124 registers and
IQ4_NL down uses 125, with zero stack or spills in either layout. The six
spilling entries are IQ2_S and IQ2_XS gate/up (16-byte stack, 16-byte spill
stores/24-byte loads each) and Q2_0 gate/up (8 bytes each), two layouts each.
These are compiler counts, not runtime traffic or per-format occupancy.

Retain two tuning leads: compare the existing `STRATA_PF_FUSED_TILE=64|128`
controls by routed rows, and investigate a smaller block/tile mapping for the
512-thread, register-heavy IQ3 kernels. The latter requires consistent tile
coverage/group-list changes and independent numerical checks; a block-size
constant change alone is insufficient. Spills in other formats do not explain
this model's IQ3_XXS/IQ4_NL layer timing.

## Full task coverage beyond the fusion threshold

The existing seven-case workload has 45/46-token prose/code prompts,
2442/2452-token edits and a 1024-token warmup. With this configuration's
3072-token streaming threshold, those prompts do not exercise fused prefill.
They remain useful default/decode controls and are retained exactly.

`fused-iq-workload-v1/` prepares 12 additional cases: the original rename,
logging, fresh-code and prose requests at each of 4K/16K/32K. Explicitly labelled
synthetic background records precede the unchanged task. This is extra
long-context coverage, not a replacement corpus or a claim about real agent
traffic. `case-audit.json` verifies all seven original bodies/IDs/hashes are
identical and all 12 additions retain their original request and token cap.
The four-leg suite has 76 requests, with independent review still required.

A detached harness adds diagnostic budget handling: missing budget evidence
still fails; the default still requires equal budgets. Diagnostic mode retains
actual allocations and task failures while suppressing timing comparisons.
Host gate tests and configuration preflight pass. The original repository
harness remains frozen for its active users; the proposal patch is archived.

The queue waits for the grouped-tail prototype's default controls and memcheck.
It then runs a 16-request retrieval/allocation ABBA with the new binary, requires
both baseline legs to match the adopted-release logits and all retrieval tasks
to pass, and only then runs the 76-request task suite. Neither stage adopts the
feature automatically. See `fused-iq-workload-v1/queue-prepared.json`.

## Existing tile-control experiment and smaller-block constraints

`fused-iq-tiles-v1/` prepares the existing auto/64/128-row controls with the
unchanged v2 kernel-test binary. It is queued after the full task suite, with
no GPU overlap. All 11 format references run under each setting; canonical
IQ3_XXS/IQ4_NL output bytes are retained at the reference shape. IQ2_S's
requested 64-row layout still falls back to 128 because four warps do not fit.

The balanced auto/64/128/128/64/auto timing order covers 1024, 2048, 2867,
2868, 4096, 5632, 5734, 5735, 6144, 6400 and 8192 tokens. The adjacent shapes
straddle the existing average-routed-row cutoffs of 56 and 112. Small shapes
also include cases below this engine configuration's fusion threshold; their
kernel timings cannot establish a current engine gain. The parser was checked
against real logs and rejects wrong shapes, incomplete runs and zero timings.
GPU results are pending. Larger-shape output hashes are checksums, while the
reference-shape comparison retains the full output bytes.

The source-level smaller-block audit rules out changing only THREADS from
512 to 256: at WW=4, half the weight rows and half the routed rows would be
uncovered; at WW=2, half the routed rows would be uncovered. A consistent
256-thread WW=2/TR=64 mapping covers the intended rows but still needs 36,096
shared bytes for IQ3_XXS with two activation stages. Turing has 64 KiB shared
memory per SM, so shared memory alone limits that proposed layout to one
block. This is an inference from the source's size formula and NVIDIA's
[occupancy limits](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#occupancy),
not a runtime occupancy measurement.

One activation stage would calculate to 30,976 bytes, but the existing loop
loads that stage after its barrier and would then let other warps read it
without another barrier. A one-stage variant needs explicit synchronization
and updated host/device accounting before any numerical or sanitizer tests.
Neither smaller-block variant is implemented. See `smaller-block-audit.json`;
benchmark the existing tile controls before changing the kernel mapping.

## Full-engine sanitizer setup and queue recovery

The 60 normal-runtime boundary controls above passed with their original
context and memory settings. The seventh full-engine memcheck attempt passed;
the failed attempts are retained to distinguish incomplete checking:

- At 575 MiB reserve, cuBLAS library loading failed with CUDA out of memory
  before READY. At 1000 MiB, cuBLAS loaded, but only 53 MiB remained for the
  128 MiB verifier arena; that attempt also failed before READY.
- At context 16384 and reserve 1536 MiB, initialization succeeded, but the
  788-slot cache could not fund the requested chunk. The engine reduced 4096
  to 2048, below the 3072 fusion threshold. After confirming two MMQ chunks,
  that process was explicitly stopped. It is not a boundary or sanitizer pass.
- The fourth attempt used context 8192, resident-KV argument 8192, reserve
  1408 MiB, and prefill keep/KV-grow floor 16, solely for instrumentation.
  KV is fully resident at this context (INFO reports kv_resident=0); the
  implementation clamps streamed residency to at least 20480 cells. It
  initialized with 925 expert slots and borrowed 786 for a 4096-token chunk.
  The trace confirms `fused group tail T 4096 cap 0`. However, the sanitizer
  then exhausted tracking memory and reported missed launches (1459 messages
  at the explicit stop). This is incomplete checking, not a pass. The settings
  do not change the installed configuration.
- The fifth attempt keeps that engine configuration and request, adding only
  `--force-synchronization-limit 16` to Compute Sanitizer. This bounds launches
  between forced stream synchronizations, a documented way to reduce tracking
  memory. It changes instrumented scheduling, so it cannot establish a race
  or throughput result. See NVIDIA's
  [Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
  and the archived installed-tool `compute-sanitizer-help.txt`.
  It completed exact chunks 4096+3071 with caps 0/5952 and returned the lent
  buffers, but then failed in the verifier graph: exit 97, 553 internal tracking
  errors and watchdog release of the GPU waits. It is not a sanitizer pass.
- The sixth attempt adds the existing `STRATA_VERIFY_ASYNC_LAUNCH=1` option.
  The earlier runner omitted this documented requirement for instrumented
  blocking graph launches. It launches the same graph from a host thread so
  the caller can service CPU doorbells; see `docs/DETAILS.md` and upstream
  commit `d8684360`. The option is present in the frozen prototype binary.
  A separate one-token graph smoke process must pass before repeating the
  full boundary process. No graph, request or kernel coverage is removed from
  that boundary test. Both stages retain the synchronization limit and strict
  report checks. The one-token smoke has passed: clean exit, zero errors and
  diagnostic lines, with 248,320 finite first logits. Independent inspection
  is recorded in `graph-smoke-review.json`. The full boundary process also
  completed its request and generated one token, but still reported 553
  tracking errors. It had 165 MiB free before graph capture versus 257 MiB
  in the clean smoke. The CPU-service failure is resolved for that request;
  complete memory checking is not. See `memcheck-async-failure-audit.json`.
- The seventh attempt raises only the sixth's reserve from 1408 to 1536 MiB.
  It initialized with 383 MiB free and 849 cache slots; the exact 4096-token
  chunk borrowed 784, followed by the 3071-token tail. Async launch,
  synchronization limit 16, context, floor, input and binary are unchanged.
  No repeated short smoke substitutes for the full request. The complete
  request and one-token decode passed: clean exit, zero sanitizer errors and
  diagnostics, all 248,320 first logits finite. Independent inspection checks
  config/source hashes, exact chunks, caps 0/5952 and the real fused route.
  Evidence: `full-engine-memcheck-review.json`; inputs and commands are in
  `memcheck-reserve1536-async-prepared.json`. The graph had 293 MiB free at
  capture. This does not qualify production long-context scheduling or speed.

The fourth runner now rejects startup chunk reduction before issuing a
request; it retains the post-request exact chunk/cap and fused-route checks.
All attempts use the same prototype binary and 7168-token request, with MTP,
K8V4, a 96-slot base ring and streaming threshold 3072. The smaller context
does not establish long-context sanitizer coverage. No API error is suppressed.
Artifacts: `fused-iq-group-tail-v1/memcheck-*-state.json`, corresponding logs,
`memcheck-context16k-stop.json`, `memcheck-fit8k-stop.json`, and
`memcheck-sync16-prepared.json`. The fifth failure is independently classified
in `memcheck-sync16-failure-audit.json`; the sixth inputs/commands are pinned
by `memcheck-async-prepared.json` and its runner.

`audit-memcheck.py` now requires a clean process exit, error summaries with
zero counts, and no error/warning/incomplete-tracking diagnostics. It accepts
the archived clean microbenchmark report and rejects the actual internal-error
report, that report with an appended zero-error summary, a missing summary,
a nonzero exit and two nonzero-summary cases. The seventh check ensures a
zero summary cannot hide another summary with errors. These are report checks;
exact chunks, real fused/cap traces and finite outputs remain separate gates.
Evidence: `memcheck-report-gate-validation.json`.

The original, `queue-resume2` and `queue-resume3` workload/tile queues declined their failed
prerequisites without running their GPU suites. Workload resumption requires
a successful sanitizer result plus the independent boundary audit. Tile tests
follow the retrieval and full-task suite. All stages retain independent
review/adoption gates; no failed queue or log has been overwritten.
The subsequent `queue-resume4` also declined the sixth attempt's failed full
result. `queue-resume5` now waits for the seventh attempt and the independent
baseline expert-route timing job to release the GPU. Those unchanged-kernel
timings may run regardless of this feature's result, but model-quality work
still requires a clean full boundary sanitizer report. No adoption gate is
bypassed; tile work still follows the full task suite.
