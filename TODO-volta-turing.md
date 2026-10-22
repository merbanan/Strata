# Volta and Turing inference work

Original comparison baseline: v19 `e3a32c9c`. Adopted local attention release:
`703eab3d` with `STRATA_PA_K8V4_STAGE=1`; INT8 race fix committed `fd6fa044`.
Worktree: Strata-v20, branch v20. Start on the
RTX 2060 SUPER (sm_75, 8 GiB); test T4 and V100 separately later. Keep new
paths opt-in until correctness and same-day end-to-end measurements pass.

This is the implementation backlog. The matching [benchmark TODO](../Strata-data/bench/2026-10-21-volta-turing/TODO.md)
tracks execution and evidence; the [optimization ledger](../Strata-data/bench/optimizations_tests.md)
indexes both. Unchecked items include conditional investigations, not promises
that every proposed port will help. Deferred ideas require new evidence first.

## Execution status

- [x] Investigate local code/history and upstream open/closed PRs and issues.
- [x] Create an isolated v20 worktree and this complete work list.
- [x] Resolve GPU access: host execution works after the user changed the setting.
- [x] Establish a same-day RTX 2060 SUPER inference baseline (4K/16K ABBA).
- [ ] Compare default expert dispatch with explicit modes 0, 4 and 8 on v19.
  Explicit mode 4 disables the default large-down fallback: measure it separately.
- [x] Record hardware, clocks, memory, build revision and exact configuration.

First implementation validation, 2026-10-21:

- [x] Build the sm_75 engine and mmvq_il_parity with CUDA 12.4.
- [x] Compile the changed native_mmvq translation unit for sm_70.
- [x] Pass all three dispatch CTests and reject invalid TMAX values 0, 9 and text.
- [x] Save configs, build logs, resource reports and validation.json under
  Strata-data/bench/2026-10-21-volta-turing.
- [x] Execute GPU parity, compute-sanitizer and repeated kernel timing on the
  RTX 2060 SUPER: 33 shapes, T=1..8 pass; memcheck reports zero errors. Logs:
  Strata-data/bench/2026-10-21-volta-turing/20261021-215005-micro.
  Initial device-access failures are resolved. HIP/SYCL builds remain pending.

To reproduce the archived wide experiment, apply its saved patch in an isolated
worktree first, then use `run-2060.sh micro`, `modes` and `wide` as described in
bench/volta-turing/README.md. Current attention validation commands and evidence
are in bench/volta-turing/k8v4-stage.md.

## 1. Interleaved dense matvecs at five through eight columns (first implementation)

- [x] Port upstream 75fabbd94152 wide kernels to v19, preserving default dispatch.
- [x] Extend fused quantization, interleave validation and verifier scratch sizing.
- [x] Allow opt-in wide row overrides for sm_70/sm_75; do not copy Blackwell tuning.
- [x] Add device-free tests for default dispatch, wide overrides and their disable switch.
- [x] Prepare GPU parity, pipeline timing, compute-sanitizer and engine A/B runners.
- [x] Check bitwise parity for T=1..8, all supported formats, odd output rows,
  and both fused quantization and standalone interleave.
- [x] Benchmark real dense/head shapes, including interleave cost, and inspect
  register usage/spills. Generate sm_75 tables from repeated measurements.
  Two runs select 52/100 wide cells at a conservative 3% threshold against
  group16; the Q5_K head stays off because it is slower. Selection is opt-in in
  configs/wide-tuned.json, generated with bench/volta-turing/select-wide.py.
  Compile inspection: 40 wide sm_75 instances have zero local/stack bytes;
  one-row variants use 80–168 registers, two-row variants 116–248. This is not
  a runtime measurement; high register use still needs occupancy/timing checks.
- [x] Execute lookup windows 8/16 and normal speculative decode end to end:
  16 requests in ABBA order, saved under wide-lookup-abba-2208. No repeatable
  overall throughput win; leave the measured table opt-in.
  The five modified wide source files and their patch are preserved under
  `Strata-data/bench/2026-10-21-volta-turing/wide-experiment-preserved/`; the patch
  applies cleanly to v19. They have been removed from the active v20 source so
  attention can be built, benchmarked and committed independently.
- [ ] Qualify model numerics and task quality before promoting wide kernels.
  All eight edit responses across both arms omit the requested change. Candidate
  fresh-code tests pass, but baseline/candidate outputs differ and this small
  sample is not a quality qualification.
  The baseline uses STRATA_MMVQ_GROUP=16 (different summation order); exact-layout
  kernel parity alone cannot establish baseline engine equivalence. Microbench
  timing now selects group16 separately from its exact-layout parity reference.
- [ ] Measure V100 separately before adding sm_70 defaults.

## 2. K8V4 prompt attention with register staging

- [x] Extend the Turing register-staged INT8 attention path to K8V4.
- [x] Preserve existing Q4-to-FP16 conversion/rounding for the first candidate.
- [x] Make selection opt-in; check parity and run attention microbenchmarks.
  `STRATA_PA_K8V4_STAGE=1` selects the new path on actual sm75 only. The
  synthetic screen covers 33/1500/4096/16384/32768-token contexts, shuffled
  pages, ragged/empty selections and changed-input graphs. A missing warp
  barrier found by racecheck was fixed; corrected memcheck and racecheck pass
  with zero errors/warnings. The first arithmetic prototype improved engine
  prefill 4.0/6.8/8.1% at 4K/16K/32K but failed numerical qualification (32K
  first-token KL 0.214). It is not adopted. A correction preserves the original
  sequential scale FMAs: all ten initial synthetic comparisons have zero
  numerical difference. The rebuilt correction passes all 12 byte-for-byte
  comparisons against v19 and the disabled path, including full 4K-query prefill.
  Memcheck/racecheck/synccheck pass at 4096/33 with graph replay and an empty
  selection. Corrected model-quality ABBA passes all 16 retrieval tasks with
  byte-identical first-token logits (248,320 values) across every arm and repeat
  at 1K/4K/16K/32K; all KL values are zero. The isolated attention-only build passes the same logits and the completed
  performance ABBA improves 16K/32K prefill by 6.46%/6.84%;
  see [K8V4 notes](bench/volta-turing/k8v4-stage.md).
- [x] Test short/4k/16k/32k prompts, occupancy and full-engine prefill/TTFT.
  Short-context gains are inconclusive; first-loan stalls confound the initial
  4K timing (L01). CUDA reports theoretical occupancy 12.5% → 25%; hardware
  counters were unavailable. A completed real model request records 12 direct
  candidate launches. Committed as `703eab3d`; locally adopted for IQ3_XXS
  through `Strata/start-current.sh` → `start-v20.sh`, with the opt-in enabled.
  The installed release passed a 4K retrieval smoke test. Global default is off.
- [ ] Investigate direct code/scale arithmetic only as a separate numerical change.
- [x] Run racecheck on the existing `STRATA_PA_V2T` INT8 pipeline. Reproduced
  25 hazards at 65/33; a warp barrier fixes same-stage V/scales reuse. At
  4096/33, racecheck/memcheck/synccheck now pass, including changed-input
  graphs and empty selections. Default and V2T outputs each match their old
  path byte-for-byte across three context/query cases. All-format regression
  and sm70/sm75/sm80 compiles pass. V2T remains opt-in, without a model
  speedup claim. See [race report](bench/volta-turing/int8-v2t-race.md).

## 3. Expert dispatch by format, phase and shape

- [x] Run the mode 0/4/8 kernel screen first; record routed entries and GU/down timings.
  90 cases across 10 real GGUF format pairs and G16/T1, G16/T4, G40/T1:
  all bitwise equal to default. Mode 8 reduces median kernel time by
  17.2%, 58.0%, 18.9% respectively. Full mode-8 inference ABBA completed:
  no consistent decode win; keep default dispatch. Both paired decode delta
  ranges cross zero. The second 4K candidate prefill is an unexplained outlier.
- [ ] Add opt-in format/shape dispatch only for measured winning cells.
- [x] Recheck IQ formats with current mode-8 fused GU/SiLU/quantization kernels.
- [ ] Validate numerical parity/teacher-forced output and accepted tokens/second.

## 4. Revisit fused IQ prefill on the current stack

- [ ] Port the v14 sm_75 IQ candidate (540953d7) behind its opt-in switch.
  Detached port plus IQ2_S shared-memory admission fix built successfully.
  All 11 format references, exact Q2 controls and three sanitizers pass.
  Isolated layer speedups repeat at 1.63x/1.31x/1.22x for 2K/6.4K/8K.
  Production source/release remain unchanged pending model qualification.
  See [IQ prefill notes](bench/volta-turing/fused-iq-sm75.md).
  All 16 original-planner retrieval requests pass, within-arm first logits
  repeat exactly, and baseline matches the installed reference. Fused first-token
  KL is 0/0.00404/0.01728/0.10865 at 1K/4K/16K/32K; broad quality remains open.
- [ ] Compare against MMQ with identical chunk size, ring and expert residency.
  Observed-allocation audit detects real differences: chunks 6400/6144,
  max loans 1185/1192, pinned RAM 39.60/39.57 GiB; 4K and 16K loans also differ.
  Equal configuration or startup rings do not qualify allocation parity.
- [ ] Score fixed continuations after batched prefill, then retain the full
  original editing/code/prose workload. Added `prefill-teacher.py`; the 4K smoke
  test passes exact prefix/tail coverage. Five segments at 1K/4K/16K/32K
  are now running in ABBA (80 requests, 10,240 score positions). Target NLL/top-1
  diagnostics do not substitute for full-distribution KL or task qualification.
  Coverage audit: all original task prompts are below the 3072-token fusion
  threshold. A detached 76-request task suite retains all seven original cases
  exactly and adds four task types at 4K/16K/32K with labelled background records.
  Default budget equality remains strict; diagnostic mode records mismatches
  and suppresses timing comparisons. Host gates/configuration preflight pass.
  It is queued behind group-tail GPU gates and a fresh retrieval/allocation ABBA.
- [ ] Reduce fused fallback scratch under `STRATA_PF_GROUP_SCRATCH`.
  Detached opt-in `STRATA_PF_FUSED_GROUP_TAIL` prototype builds; it changes
  both buffer reservations and per-chunk group-cap selection/reset. Host size,
  capacity and disabled-path probes pass. Calculated four-buffer size at
  T=6400/threshold3072 is 343.0 -> 168.8 MiB; actual arena/VRAM savings unmeasured.
  Original/unset/zero/one GPU boundary controls plus full-engine memcheck are
  queued behind the current teacher sweep. Validate original tasks and auto
  planner behavior before adoption; do not just lower baseline chunks.
- [ ] Profile per-format/routed-M performance and use selective dispatch.
  Static audit maps all 19 compiler entries: IQ3_XXS/IQ4_NL use 124/125
  registers without spills. Spill entries belong to IQ2_S/IQ2_XS/Q2_0 GU.
  Compare existing TILE=64/128 first. Investigate smaller blocks only with
  consistent tile/group coverage; do not assume other formats' spills explain
  current-model performance.
- [ ] Check scratch/grouping interactions; PF_DM_GROUP changes combine order.
- [ ] Require a full-engine win: the old isolated-kernel win did not improve prefill.

## 5. Reduce long-lookup CUDA graph memory

- [ ] Measure live bytes and use counts for each solo/batch graph and the arena.
  Source audit: `instantiate_evicting()` evicts only `exec_bm_`/`commit_bm_`
  batch layouts on OOM; solo `exec_`/`exec_nr_` graphs remain until destruction.
  The current ABBA baseline captures T=8/3/1/2 and reports 67 MiB free before
  its 4K request. Instrument those retained graphs and prompt-read size choice
  before assuming a new solo cache policy can save 75–100 MiB.
  The C02 longer-decode baseline subsequently reached 26–30 MiB free VRAM
  at the same 575 MiB reserve. Keep its telemetry and graph captures as another
  workload for this investigation; no mid-run budget changes.
  Completed diagnostic on the installed snapshot release: 21 executable
  lifetimes, peak 21 retained, 8,544 successful host submissions, all destroyed.
  Five verifier graphs (T=3/1/16/2/8) coincide with 144 MiB less free VRAM
  during instantiation; every destruction shows zero immediate free increase.
  These are observed deltas, not owned bytes. See
  [graph memory notes](bench/volta-turing/graph-memory.md). Batch/arena coverage
  remains open. Standalone recreation matrix passes 960 changed-input cases:
  cap 1/2/3/5 uses 28/60/90/136 MiB; metadata is reused without global free
  recovery. Cap 2 saves 76 MiB synthetically, at more instantiations; test an
  opt-in engine cap before adoption.
- [x] Prototype a budgeted solo-graph LRU beyond the existing OOM-only eviction.
  `STRATA_VERIFY_SOLO_GRAPHS=N` prototype builds; default 0 retains all graphs.
  Actual Verifier policy probe passes plain/memcheck/synccheck: resident versus
  doorbell recency, pipeline guards, in-flight refusal, 37 graph creations and
  35 changed-input checks. Both original seven-request diagnostic legs completed:
  peak solo graphs 5 -> 2, minimum observed free VRAM 25.125 -> 105.125 MiB,
  unchanged 1403 slots/6400-token chunks/39.60 GiB pinned RAM/128.8 MiB arena.
  Candidate has 174 graph lifetimes and 172 evictions. Every graph is destroyed,
  no recorded API errors, but all six long continuations differ. Retrieval/logit
  ABBA passes all 16 tasks and raw adopted-reference logits. Full controlled
  decode (swaps/deferred CPU off) finishes 28 requests but still varies within
  baseline repeats. Matched prose wall time regresses 5.07%/5.14% with cap 2;
  cap 2 is not selected for adoption. Cap 3 diagnostic completes: 53 lifetimes,
  50 evictions, 75.125 MiB minimum free (+50 MiB), at the same budgets.
  Fixed draft-cost full-workload diagnostic completed: all 28 requests stop
  naturally with identical tokens and work counts across all four legs (11,593
  tokens per leg); each candidate has 149 captures and 147 evictions across all
  five window sizes. This is controlled continuation parity, not broad task
  quality: independent review retains all six long-task failures. See
  `fixed-policy/qualification.json` and its hash-pinned reviews.
  The original-config cap-0/3 ABBA also finished all 28 requests (27 natural
  stops; one candidate edit hit 4096 tokens). Both arms vary between repeats;
  no long pair matches both output and acceptance. Minimum logged free VRAM
  is 29/75 MiB; cap 3 has 37/25 evictions. No qualified speedup established.
  Prototype not selected for adoption; source restored. The complete patch and
  probes are archived and replay-verified under `engine/proposal-complete.patch`.
  Future promotion requires a demonstrated benefit and remaining quality gates.
- [ ] Explore graph-exec parameter updates where topology permits safe reuse.
  Measured solo T=1/2/3/8/16 has 2300/4300/4307/4346/4440 nodes. All 20
  cross-size pairs violate CUDA 12.4's equal-node-count update requirement;
  direct whole-graph replacement cannot avoid these recaptures. See the
  graph-memory report and `graph-update-topology-audit.json`. Common padded
  topology, same-size recaptures and batch layouts remain separate leads.
- [ ] Account for recurring lookup preparation cost when evaluating a bounded
  graph cache: `graph_fits()` prepares before `round0`, so current learned
  lookup round cost excludes recapture; MTP misses inside `run()` are timed.
  Source-audited lead only; measure its effect before changing policy/defaults.
- [ ] Preserve CPU/GDN/KV state for padded rows; test graph eviction/recreation.
- [ ] Target lookup-16 without shrinking prompt chunks or increasing reserve;
  75–100 MiB savings is a hypothesis, not a measured result.

## 6. Volta register-fragment IQ kernels (after RTX 2060 SUPER)

- [ ] Start from merbanan/v100 e24e2b2e and compare upstream PF_WMMA2.
- [ ] Extend the existing Q2 register-fragment approach to IQ4_NL down and IQ2_S
  gate/up, then IQ3 if useful; keep DP4A for small routed batches.
- [ ] Explore lossless compact prepacking/LUT layout without a global FP16 cache.
- [ ] Measure register pressure, spills, bitwise/numerical parity and full inference.
- [ ] Do not repeat the losing IQ3 pair-reader without a new measured reason.

## Instrumentation, correctness and measurement

- [ ] Evaluate upstream PR #3631 per-window tracing for acceptance and dispatch cost.
- [ ] Review PR #3483 pinned-stage per-stream fences before testing affected paths.
- [ ] Profile PLE latency separately; existing local PLE measurements do not show
  the HDD bottleneck reported in issue #2239.
- [ ] Use alternating A/B/B/A runs with same-day baseline, matching memory/chunks,
  editing, fresh-code and prose prompts, plus short/4k/16k/32k context lengths.
  Completed: mode-8 4K/16K, wide edit/code/prose and attention 1K/4K/16K/32K
  ABBA. Attention has separate exact retrieval/logit qualification; the wide
  edit workload still needs successful-task qualification. Repeat matched
  comparisons for each new feature.
- [ ] Record accepted tokens/s, TTFT, prefill, graph memory, cache slots/chunks and
  variance. Reject speedups caused by changed acceptance or incorrect output.
- [ ] Run bitwise tests where expected; otherwise teacher-forced KL and quality
  checks. Run kv1188 and compute-sanitizer for changed CUDA memory layouts.
- [ ] Resolve the existing lookup workload's quality failures before using it as
  successful editing throughput: the first baseline omitted both requested edits
  and capped the fresh-code answer. Static AST/completeness checks are implemented
  in bench/volta-turing/analyze-lookup.py; qualify baseline and candidate separately.
- [ ] Build touched CUDA/HIP/SYCL backends where available; record untested targets.

## Lower-priority ideas retained from the investigation

- [ ] Revisit layer-major prefill copies only if new profiles justify it (current
  copy time about 8.2% on the 2060 SUPER and 4.1% on T4 at long prompts).
- [ ] Revisit chunked GDN only with the new shared-buffer lifetime design in
  1Cat PR #1228; prior Strata kernels were 4–7x slower and exceeded Turing's
  practical shared-memory budget. See the new experiments below.
- [ ] Keep generic BF16 tensor-core caching deferred because of memory cost.
- [ ] Keep lossy routing/tail skipping deferred; check the prior near-full-cache
  regression (#1884) before any reconsideration.
- [ ] Avoid duplicating existing adaptive draft policy; use window measurements
  to find specific missing decisions before revisiting closed generic MTP work.

Sources: upstream commit 75fabbd94152; issues #2197, #1884, #2239;
PRs #2092, #2146, #2356, #2357, #3631, #3483, #1098, #2529;
local Strata-data/bench/optimizations_tests.md and v14/v19/V100 history.

## 1Cat-vLLM follow-up, researched 2026-10-21

- [x] Inventory code/history and all open/closed PR and issue records; inspect
  architecture-specific kernels, relevant discussions and eight open PR diffs.
  See [the investigation](bench/volta-turing/1cat-investigation.md) for source
  commits, measured scope, existing Strata overlap and rejected experiments.
- [ ] Extend weight reuse to T=9..16 only if lookup profiling justifies it;
  preserve packed weights and compare the full pipeline against group16 (#1142).
- [ ] Measure whether router selection can share a required Q8 quantizer launch
  after accounting for QFUSE/QDEDUP; preserve FP32 logits and exact routes (#1147).
- [ ] Prototype the SM75 chunked GDN shared-buffer schedule from #1228 in native
  CUDA: no transposed K copy, A/P lifetime reuse, loop-end synchronization.
  Compare FP32 recurrent state, preparation/output costs and cache-slot loss.
- [ ] Audit launch bounds and spills where shared memory already limits occupancy
  (#1226, issue #1132); benchmark selected variants instead of globally raising caps.
- [ ] Trace SM7x QSA verifier scoring; test shared-key reuse for 2..8 queries only
  where existing batching does not already provide it (#1127). Check exact top-k.
- [ ] Apply architecture-specific accumulator mapping and shared-memory admission
  lessons from #1236 to the K8V4 attention work; keep sm70/sm75 tuning separate.
- [ ] Retain #797's separate SM75 FA2 build/worker-device selection as a reference
  for future general FP16 attention or mixed-GPU stages; it is not a replacement
  for Strata's current quantized sparse attention.
- [ ] Measure mapped-host expert reads alongside concurrent compute; compare a
  bounded DMA staging threshold for small decode misses, including fences and
  lost expert-cache capacity (#1203). Existing prefill DMA/chunk balancing stays.
- [ ] Compare KV probe-LRU policy and bounded staging under shared-prefix and
  disjoint-context churn; retain K8V4 encoded storage (#1204/#1242).
- [ ] Use QPN register-fragment scheduling as a V100 design reference, preserving
  original GGUF bytes; do not treat NVFP4/MXFP4 support as an IQ3 drop-in (#742/#959).
- [ ] Add shuffled-page/alignment, poisoned-scratch and changing-input graph checks
  to relevant future kernels; hold chunk/capacity settings fixed (issues #98/#1003).
- [ ] Keep resident expert task pools (#1078), 43%-larger IQ3 packets (#1068/#1121),
  and acceptance-changing IQ2 expansion (#1011 first revision) out of default
  implementation. Require a new measured argument before retrying them.

### Remaining leads from the report and saved PR discussions

IDs below supplement the six workstreams and follow-ups above. Sources refer
to the saved 2026-10-21 snapshot in the benchmark directory. Upstream results
on four V100s are hypotheses for our single RTX 2060 SUPER, not local gains.

#### Dense, router and expert kernels

- [ ] C01 — For T=9..16 reuse, keep independent FP32 accumulators and each row's
  reduction order while decoding each weight once; include quantization/interleave
  costs and retain the large Q5_K head fallback. Treat M17..32 FP8/QPN8 reuse as
  a later format-specific extension, not a reason to expand current verifier
  capacity ([#1142](https://github.com/1CatAI/1Cat-vLLM/pull/1142),
  [#1157](https://github.com/1CatAI/1Cat-vLLM/pull/1157)).
- [ ] C02 — Test warp-local sorted router lists and winner-only exponentials
  inside a still-needed Q8 launch. Preserve FP32 route keys, ties, nonfinites,
  normalization and Q8 bytes; avoid global completion counters. Measure existing
  QFUSE/QFUSE_HC/QDEDUP coverage first; standalone selection may lose
  ([#1147](https://github.com/1CatAI/1Cat-vLLM/pull/1147)).
  Source audit on v20: CUDA's `g_qfuse_hc()` is always false; the default shared
  expert stream prevents QDEDUP, and the current config leaves QFUSE off. A
  separate expert-input quantizer remains after routing. The first fusion must
  write `nat_xq_`, using `iq_kernels.cu::quantize_q8_1_rows` as its byte reference,
  rather than the dense/shared expert's `xq_` producer. `native_router.cu` is
  compiled with `--use_fast_math`; `iq_kernels.cu` is not. A combined translation
  unit must preserve both arithmetic contracts, including rounding thresholds,
  subnormal and large finite activation blocks, before engine integration.
  Preserve the current full-softmax normalization before top-k: winner-only
  exponentials can change rounded ties/weights despite algebraic cancellation.
  Measure launch cost and exact route/quantizer bytes before changing selection.
  Further integration audit: `--native` enables the native router. In
  `VerifyWindow::record` the n>1 path calls the multi-token router directly,
  whereas n=1 goes through `moe_route`; cover both if integrating a fused launch.
  The CPU doorbell is published after routing and before expert quantization.
  Moving quantization into routing may delay CPU-expert work even when one GPU
  launch is removed: measure doorbell latency and complete windows, not just
  combined kernel time. First isolate launch fusion with the original top-k
  arithmetic; test sorted lists and winner-only exponentials independently.
  Keep Q8 work parallel across blocks rather than serializing 80 input blocks
  behind the route warp; no global completion counter is needed for independent
  route/Q8 outputs. Preserve the existing QFUSE/QDEDUP skip rules and skip the
  later expert quantizer only after the fused producer actually succeeds.
  First three prototypes evaluated; none adopted. Original-selection fusion
  passes 468 kernel/sanitizer cases (expanded exact controls later pass 624),
  16 retrieval requests with byte-identical adopted-reference logits, and
  recorded 480 fused captures. Both async and synchronous full-workload ABBAs
  completed 28 requests with matched budgets, but outputs/acceptance vary within
  arms and no successful-task end-to-end gain is established. All first-baseline
  edits failed; original tasks and failures are retained. Async residency,
  timing-based draft windows and deferred CPU behavior remain quality/control
  questions. The QFUSE/QDEDUP queue did not run after its parity gate failed.
  Sorted lists pass exact output and sanitizers, but are 31–37% slower than
  original fusion at the observed 1/2/8-row shapes in a same-run comparison.
  Winner-only exponentials change expert IDs in 78/624 cases and weight bytes
  in 308/624; Q8/guards and nonfinite fallback match, sanitizers report no errors.
  Its roughly 0–4% kernel gain does not justify the unguarded routing change.
  Exact source, complete reproducible patch, experimental executable and all
  results are archived. Production source is restored to the adopted baseline;
  the temporary build must be rebuilt before new engine work. C02's remaining
  qualification gates are still open for any future variant; move current work
  to measuring and reducing retained graph memory.
  See `bench/volta-turing/router-q8-fusion.md` for measured scope and limits.
- [ ] C03 — Audit remaining gate/up-to-down conversions and repeated activation
  quantization after the existing mode-8 fusion. Investigate only uncovered
  format/shape paths, including packed-byte DP4A/LUT readers
  ([#982](https://github.com/1CatAI/1Cat-vLLM/pull/982),
  [#970](https://github.com/1CatAI/1Cat-vLLM/pull/970)).
- [ ] C04 — Exploratory: screen the revised arithmetic-compatible IQ2 nibble
  layout, preserving decoded weights, FP16 SiLU boundaries and reduction order.
  Charge expanded bytes against cache residency and measure acceptance; the
  first acceptance-changing revision stays rejected
  ([#1011](https://github.com/1CatAI/1Cat-vLLM/pull/1011)).
- [ ] C05 — Profile narrower padded grouped-MoE prefill tactics and selective
  Q5_K/Q6_K dequantize-plus-cuBLAS for large routed M. Admit only measured physical
  layouts, threshold neighbors and available scratch; retain small-M fallbacks
  ([#1205](https://github.com/1CatAI/1Cat-vLLM/pull/1205)).

#### Attention, QSA and GDN

- [ ] C06 — Trace Turing QSA dispatch and compare existing native scoring with
  cuBLAS plus exact top-k only on material large-prefill shapes. Check actual
  kernel admission; 1Cat's gain over its Triton fallback does not transfer
  ([#1229](https://github.com/1CatAI/1Cat-vLLM/pull/1229)).
- [ ] C07 — Screen narrower pre-Ampere QSA prefill tiles after proving which
  source path executes; account for the corrected historical attribution
  ([issue #441](https://github.com/1CatAI/1Cat-vLLM/issues/441),
  [#469](https://github.com/1CatAI/1Cat-vLLM/pull/469)).
- [ ] C08 — Adapt register-resident PV accumulation and launch-bound relaxation
  together to Strata's D256 attention; compare each change separately too.
  The D128 V100 issue is a design reference, not a compatible patch
  ([issue #1132](https://github.com/1CatAI/1Cat-vLLM/issues/1132)).
- [ ] C09 — Validate sm70/sm75 accumulator maps, grouped query/KV heads, strides,
  and ragged/shuffled pages. Bound planner/hash scratch to actual device shared
  memory and check compiled kernel availability before dispatch; retain a safe
  fallback ([#1236](https://github.com/1CatAI/1Cat-vLLM/pull/1236)).
- [ ] C10 — For GDN buffer aliasing, verify loop-end barriers and final FP32
  recurrent state across chunk boundaries and speculative commits. Select by
  actual shared-memory availability and compare high-register variants only
  where occupancy is already constrained
  ([#1228](https://github.com/1CatAI/1Cat-vLLM/pull/1228),
  [#1226](https://github.com/1CatAI/1Cat-vLLM/pull/1226)).

#### Copies, KV policy and temporary memory

- [ ] C11 — Audit whether host KV initialization can skip zeroing only after
  proving all reads follow writes, including padding, reuse and graph replay.
  Check pin registration and per-stream ownership/fences; reuse existing staged
  prefill and balanced chunks ([#1203](https://github.com/1CatAI/1Cat-vLLM/pull/1203)).
- [ ] C12 — Screen occupied-range cold-expert DMA and QSA DMA for a few large
  contiguous runs, keeping mapped gathers for fragmented/small selections.
  Calibrate local thresholds rather than copying the upstream four-run/512-KiB
  rule ([#1205](https://github.com/1CatAI/1Cat-vLLM/pull/1205)).
- [ ] C13 — Inspect small-miss copy-plan parallelism and avoid KV value reads
  that cannot be consumed. Measure complete cache resolution with existing K8V4
  storage and safe fallback for unsupported/alignment cases
  ([#1204](https://github.com/1CatAI/1Cat-vLLM/pull/1204)).
- [ ] C14 — Compare independent draft/target hot-cache budgets with a fixed
  total memory budget; include shared-prefix reuse and disjoint-context churn.
  Record hit rate, PCIe bytes, acceptance and expert-cache displacement
  ([#1204](https://github.com/1CatAI/1Cat-vLLM/pull/1204)).
- [ ] C15 — Prototype bounded prefill-history staging windows with a numerically
  checked softmax/LSE merge. Compare against existing streaming and include
  window boundaries, FP16 draft history and encoded target history
  ([#1242](https://github.com/1CatAI/1Cat-vLLM/pull/1242)).
- [ ] C16 — Conditional long-context work: partition exact top-k beyond 256K,
  align draft/target RoPE and capacity, and test that speculation remains active
  at the configured boundary. Do not raise model capacity as part of a speed A/B
  ([#1242](https://github.com/1CatAI/1Cat-vLLM/pull/1242)).
- [ ] C17 — Profile PLE convolution without padded history and release HC
  projection temporaries after last use. Check state commit, SiLU, null/empty
  slots and graph lifetime; measure whether savings recover cache slots or
  improve chunk size ([#1205](https://github.com/1CatAI/1Cat-vLLM/pull/1205)).

#### Verifier, graph and numerical validation

- [ ] C18 — Measure graph padding inflation and full verifier rounds (target,
  sampling and state commit), separately from draft time. Exercise mixed
  prefill/decode, concurrency, graph shrink/eviction/recreation, changing inputs
  and low-memory fallback
  ([#1211](https://github.com/1CatAI/1Cat-vLLM/pull/1211),
  [issues #1220](https://github.com/1CatAI/1Cat-vLLM/issues/1220),
  [#1042](https://github.com/1CatAI/1Cat-vLLM/issues/1042),
  [#1070](https://github.com/1CatAI/1Cat-vLLM/issues/1070)).
- [ ] C19 — Profile versioned per-layer QSA key reuse, bounded staging,
  compact recurrent state and bounded expert leases; count persistent bytes
  before considering any verifier cache
  ([#1211](https://github.com/1CatAI/1Cat-vLLM/pull/1211)).
- [ ] C20 — Profile grouped expert projections, mixed dense fusion, grouped Q8
  draft-output warps, whole-batch HC and reuse of canonical HC RMS statistics.
  Gate each on an observed Strata gap; upstream complete-verifier targets remain
  unmet despite operator wins ([#1211](https://github.com/1CatAI/1Cat-vLLM/pull/1211)).
- [ ] C21 — Check expert-ID width beyond 255, ragged/empty groups, owner changes,
  stale key versions and barrier-counter wrap in any changed grouping/cache path
  ([#1211](https://github.com/1CatAI/1Cat-vLLM/pull/1211)).
- [ ] C22 — Use per-window traces to identify gaps in existing adaptive lookup
  policy before revisiting width/chain decisions; include structured output,
  changing context and accepted-token cost
  ([#366](https://github.com/1CatAI/1Cat-vLLM/pull/366); Strata PR #3631).
- [ ] C23 — Extend numerical qualification with real activation outliers and
  FP64 scale references, FP32 attention accumulation, fixed reduction order,
  accepted recurrent-state tails and request-local sampling parameters. These
  are regression patterns to check, not diagnosed Strata bugs
  ([#666](https://github.com/1CatAI/1Cat-vLLM/pull/666)).
- [ ] C24 — Require WMMA alignment checks, poisoned scratch, changing-input
  replay, shuffled pages, ragged tails and natural EOS. For retrieval comparisons
  hold chunking/capacity fixed, distinguish cold/warm prefixes and inspect logit
  margins before attributing differing text to state corruption
  ([issue #98](https://github.com/1CatAI/1Cat-vLLM/issues/98),
  [issue #1003](https://github.com/1CatAI/1Cat-vLLM/issues/1003)).

#### Later hardware/format work and deferred experiments

- [ ] C25 — On V100, investigate compact IQ codebook/LUT layouts and direct
  register-fragment decoding; validate lossless permutations/inverses and format
  admission, retaining original GGUF storage. Preserve source attribution and
  licenses for any reused code
  ([#742](https://github.com/1CatAI/1Cat-vLLM/pull/742),
  [#848](https://github.com/1CatAI/1Cat-vLLM/pull/848),
  [#959](https://github.com/1CatAI/1Cat-vLLM/pull/959)).
- [ ] C26 — Future checkpoint support only: evaluate FP8/NVFP4/MXFP4 native
  packed linear/MoE dispatch separately on sm70 and sm75. Preserve scale precision
  and fallbacks; do not requantize the current IQ GGUF benchmark to claim a win
  ([#804](https://github.com/1CatAI/1Cat-vLLM/pull/804),
  [#742](https://github.com/1CatAI/1Cat-vLLM/pull/742),
  [#1157](https://github.com/1CatAI/1Cat-vLLM/pull/1157)).
- [ ] C27 — Future multi-GPU only: investigate shared pinned host KV for replicated
  heads, query-row partition plus compact-index all-gather, root-only/banked
  index staging and lossless FP16 peer-partial packing. Check ownership, mixed
  request fallback, collective costs and epoch wrap; these are not single-2060
  optimizations ([#1242](https://github.com/1CatAI/1Cat-vLLM/pull/1242),
  [#1205](https://github.com/1CatAI/1Cat-vLLM/pull/1205),
  [#1211](https://github.com/1CatAI/1Cat-vLLM/pull/1211)).
- [ ] C28 — Future general-attention/mixed-GPU work: independently build, load
  and test the current worker's sm75 FA2 backend and sm70 counterpart; preserve
  capability-aware fallback ([#797](https://github.com/1CatAI/1Cat-vLLM/pull/797)).
- [ ] D01 — Deferred: resident producer/consumer expert pools, including smaller
  worker pools. Retry only after a profile identifies how to overcome the
  recorded slowdown ([#1078](https://github.com/1CatAI/1Cat-vLLM/pull/1078)).
- [ ] D02 — Deferred: expanded IQ3 register-pair packets (+43% weight traffic)
  and the acceptance-changing first IQ2 layout. Require a new memory/quality
  argument; C04 concerns only the revised compatible IQ2 design
  ([#1068](https://github.com/1CatAI/1Cat-vLLM/pull/1068),
  [#1121](https://github.com/1CatAI/1Cat-vLLM/pull/1121),
  [#1011](https://github.com/1CatAI/1Cat-vLLM/pull/1011)).
- [ ] D03 — Deferred: deeper dense prefetch, byte-expanded weights and uniform
  format specialization. Lower register counts alone did not justify these in
  the upstream dense-reuse campaign
  ([#1142](https://github.com/1CatAI/1Cat-vLLM/pull/1142)).
- [ ] D04 — Deferred: HC tile tuning/row partition with weight transport, the
  rejected fused indexer, larger custom all-reduce buffers and OOM-prone larger
  chunks. Require fresh path-specific evidence, not the aggregate prefill gain
  ([#1205](https://github.com/1CatAI/1Cat-vLLM/pull/1205)).

### Additional leads from local measurements

- [x] L01 — Reuse the immutable MTP expert host snapshot across prompt-room
  loans, after auditing every writer and invalidation path. In
  `src/core/mtp.cpp`, `experts_lend` retains its pinned allocation but copies
  all 675 MiB from GPU to host on every loan; `experts_reclaim` restores those
  bytes in place. A valid cached snapshot could avoid the repeated D2H copy.
  Keep the idle/chain/ownership checks, graph addresses and reclaim copy intact;
  invalidate on any reload/change of weights and cover shared drafters, failed
  copies and repeated lend/reclaim cycles with byte hashes and model checks.
  Measure an independent opt-in eager allocation/snapshot during startup too:
  the first loan includes `cudaMallocHost`, which can stall under host pressure.
  Charge its pinned bytes and startup time, and retain lazy allocation by default.
  Evidence: the isolated attention candidate's first 4K loan took 1379.3 ms,
  and the second baseline's took 28299.6 ms, versus roughly 107 ms for ordinary
  loans. The benchmark's first request is 1023 tokens, below ROOM_MIN=1024, so
  4K triggers the first pinned allocation; quality's exact 1024-token request
  triggers it earlier. This is a transfer/allocation lead,
  not proof that the attention kernel caused the outlier. Keep first-loan and
  repeated-loan TTFT separate, and hold chunks/cache slots fixed.

L01 implementation status (2026-10-22): `STRATA_MTP_EXPERT_SNAPSHOT=1` now
reuses the first successful snapshot; unset/zero keep the original D2H copy.
Ownership/idle/chain guards and in-place reclaim remain. Load attempts invalidate
the snapshot; failed D2H never marks it valid. CUDA memcheck passes unset/0/1
with injected allocation/copy/idle failures, shared-owner rejection, size checks,
and five destructive loan/reclaim cycles followed by graph replay with exact
bytes. Full-model ABBA passes all 16 tasks and byte-identical first-token
logits against the adopted reference at every context. A new timing prompt
failed on baseline (summary/cap instead of marker); preserve that failure.
Corrected timing uses the qualified audit-marker template with cold prefixes.
All 24 requests pass; both ABBA pairs improve repeated TTFT by 2.10–2.96%
at 2K and 1.08–1.59% at 4K, with lower complete-request latency. Memory
budgets match. Committed as `6e4f8498` and adopted locally with
`STRATA_MTP_EXPERT_SNAPSHOT=1`; the installed release passed first/repeated
loan requests with exact answers (955.1 ms then below 0.1 ms loan time).
The separate eager-startup prototype built and passed lifecycle/memcheck
unset/0/1. Its full ABBA returns identical output tokens and budgets. Ready-to-
first-token improves 45.70–49.31%, but total start-to-answer improves 10.71%
in one pair and regresses 16.49% in the other. It moves an allocation stall;
no repeatable overall gain is demonstrated. Do not adopt eager priming locally.
Exact prototype and results are retained; eager-only source edits are restored.
See `bench/volta-turing/mtp-expert-snapshot-eager.md` and
`Strata-data/bench/2026-10-21-volta-turing/mtp-expert-snapshot-eager/`.
