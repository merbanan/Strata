# Volta and Turing inference: numbered backlog

Updated 2026-10-24. This is the authoritative task list, sorted by correctness, current RTX 2060 SUPER work, measured opportunity, and hardware dependencies. Parent order is research priority, not the order of already-running jobs. No running test was cancelled or reprioritized by this edit.

**Execution resumed:** tasks 1 → 2 → 3 → 13 were completed in the earlier requested sequence. Task 14.5's full-model gate is now complete and rejects adoption on the RTX 2060 SUPER; continue through the remaining backlog. Local RTX 2060 SUPER results do not substitute for V100/T4 runtime qualification.

Status: **Built** = host/compiler checks only; **Queued/Running** = GPU work outstanding; **Partial/Pending** = incomplete; **Held** = current candidate failed promotion gates; **Conditional/Hardware/Future/Deferred** = retained scope with an explicit prerequisite; **Complete** = stated investigation finished; **Adopted** = qualified and locally enabled. Ongoing requirements apply to every new candidate.

Previous lists had 110 implementation entries and 31 benchmark entries, including duplicates and stale statuses. Every entry is mapped in [the migration index](bench/volta-turing/todo-index.json). Detailed evidence, source links and historical caveats remain in [implementation history](TODO-volta-turing-history-20261022.md) and [benchmark history](../Strata-data/bench/2026-10-21-volta-turing/TODO-history-20261022.md). The [execution queue](../Strata-data/bench/2026-10-21-volta-turing/TODO.md) tracks running/queued work. Existing C01–C28/D01–D04/L01 references remain valid through the index and histories.

1. **QSA FP16 correctness — highest priority**

   1. **Built:** Correct the halving scorer's pooled-key reads; preserve FP32 behavior and FP32 tail keys. CUDA sm70/sm75 builds pass; HIP remains untested.
   2. **Complete:** Independent GPU review reproduced the original wrong-storage result and verified eight original/fallback/corrected control arms: 480 graphs and 1440 changed-input replays.
   3. **Complete:** Exact scores/top-k IDs, causal tails, guards and memcheck/racecheck/synccheck passed on RTX 2060 SUPER. This is synthetic kernel coverage, not engine qualification.
   4. **Complete diagnostic:** Corrected halving versus the FP16-aware fallback finished all 76 original task requests under fixed draft costs and controlled adaptation. Independent task and allocation reviews pass: 11/19 qualified per leg, matching startup budgets and per-request loans, 19/19 baseline-repeat tokens and 17/19 candidate-repeat tokens. Changed 32K candidate continuations prevent an exact-continuation throughput claim.
   5. **Complete:** Scorer-only and scorer-plus-selection FP32/FP16 ABBA finished with 19,200 samples, exact post-timing parity and independent raw-log review. At FP16 context 32K, corrected halving gives about 1.79x scorer-only and 1.32x scorer-plus-selection in both pairs; these are synthetic GPU-call ratios, not full inference speedups.
   6. **Complete, held:** The full one-switch production-policy ABBA finished all 76 requests on RTX 2060 SUPER. Independent review qualifies 11/19 per leg; startup and per-request loan records match across all four legs, including proven window-only reads for short requests 4/5. Nine successful tasks have identical output tokens across all four legs, but none has identical non-timing logged work in both pairs. Policy-level candidate/base geometric means: decode 0.984x/1.007x, prompt processing 0.997x/0.996x, TTFT 0.996x/0.996x, wall 0.987x/1.003x. No repeatable inference speedup; do not enable QSA halving. The isolated scorer gain remains a kernel-only result.

2. **Reliable baseline and task qualification**

   1. **Complete:** Review the IQ2 continuation run: all 76 requests reviewed; four warmups pass, all 72 substantive responses fail qualification.
   2. **Complete diagnostic:** In the controlled QSA comparisons, 10 of 18 substantive tasks qualify per arm; all eight fresh-code/prose tasks fail independent review. The original prompts, caps and failures were preserved across both repeats.
   3. **Complete diagnostic:** Corrected QSA halving and its FP16-aware fallback produce identical tokens and task qualifications on all 19 first-pair requests. The improvement over the older IQ2 workload cannot be attributed to QSA because draft costs, adaptation, PCIe fraction, deferred CPU and source binary also changed. The controlled candidate's 32K tail chunks (754/755 tokens) remain eligible for a time-measured CPU prefill share despite `STRATA_PREFILL_CPU_GATE=0`; this is a possible contributor to those long-context differences, not a proven cause. In the production candidate repeat, the original 46-token fresh-code prompt diverges after eight generated tokens even though both runs read it entirely through verifier windows. CPU prefill share therefore cannot explain that short-prompt divergence. The completed production-policy ABBA showed no repeatable inference gain.
   4. **Complete diagnostic, cause unresolved:** The first controlled pair matches all 19 token sequences and every non-timing DONE field. The baseline repeat matches 19/19; the candidate repeat matches 17/19, with 32K fresh-code/prose divergence after 354/148 common tokens. All task qualifications and startup/loan records match, including proven short window-only reads. The production candidate repeat matches 11/19 token sequences: all 11 qualified warmup/edit tasks match, while all eight failed code/prose outputs differ; only one request retains identical non-timing DONE fields. Startup and per-request loan records still match between the two production candidate legs. The original 46-token fresh-code prompt diverges after only eight generated tokens despite verifier-window-only prefill. The production baseline repeat also changes its original fresh-code/prose outputs (1790/0 common tokens), so short-output variation is not unique to the QSA candidate. Four cold production-policy short-prompt runs have identical first logits; the fourth continuation diverges after 2318 tokens, showing full request history is not required and first-token logits do not identify the cause. Four cold fixed-policy 32K candidate runs have byte-identical first logits, full tokens and non-timing DONE fields for both original long-context tasks, so the earlier fixed 19-task 32K variation depends on conditions absent from this cold subset. The numerical cause is not proven; task and exact-work gates remain required for later experiments.
   5. **Complete for QSA baseline:** The independent controlled and production-policy workload reviews each qualify 11/19 requests per leg (warmup plus ten edits); all eight fresh-code/prose tasks fail in every leg. Nine successful production tasks have identical tokens across four legs, and all four legs have matching startup/loan records. Production timing does not show a repeatable decode or prompt-processing gain. IQ2_XS requalification finished under the fixed baseline and did not qualify for adoption.

3. **IQ2_XS gate/up fusion qualification — C03**

   1. **Complete:** Audit missing format/shape paths, real pack formats, existing SwiGLU/Q8 fusion and VRAM/PCIe dispatch.
   2. **Complete:** Build opt-in mode-8 and normal-mode-4 variants; synthetic parity, changed-input graphs, sanitizers and repeated kernel timings pass.
   3. **Complete:** Review 16 model retrieval requests, first logits and corrected allocation traces; these pass.
   4. **Complete:** Independently verify model dispatch: 640 fused submissions through graph replay when enabled, zero when disabled; all eight retrieval outputs/reference logits match. Counts do not establish timing or active groups.
   5. **Complete:** Independently review original/v2/compact kernel parity with weights from 30 real experts: all 6000 comparisons pass; activations remain synthetic.
   6. **Complete diagnostic:** Four ABBA legs produced identical tokens and first-logit arrays for four selected prompts. Independent review passed. Startup budgets and loan counts match after proving two short prompts per leg used only verifier windows; the original loan-only auditor's false result is retained. Draft-policy/deferred-CPU confounds and the 256-token caps remain.
   7. **Complete, held:** Requalified the original 19-case, 76-request full workload under the corrected fixed baseline on RTX 2060 SUPER. All four legs independently qualify 11/19 tasks, preserve the original prompts/caps, pass saved-token/protocol review and match startup/per-request allocation records (including proven window-only short reads). Ten successful edits have identical output across four legs, but no task has identical logged work in both pairs. Candidate/base geometric means: decode 1.001x/0.976x, prompt processing 1.001x/0.997x. No repeatable full-model speedup. Evidence: `Strata-data/bench/2026-10-21-volta-turing/iq2xs-requalified-fixed-v1/reviews/`.
   8. **Complete, not adopted:** The strict and whole-policy timing gates reject promotion; keep `STRATA_IQ2XS_GU_FUSED` disabled locally. The exact tested patch and source are archived under `expert-conversion-audit-v1/iq2xs-gu-v2/` for future hardware or policy work.

4. **Compact IQ2 storage and compatible layouts — C04/C25**

   1. **Complete:** Audit revised upstream IQ2 arithmetic and storage expansion; expanding the entire set adds about 12.34 GiB, not a measured resident allocation.
   2. **Built:** Implement the lossless compact IQ2_XS codebook with original packed weights; CPU exhaustive checks and sm70/sm75 builds pass.
   3. **Complete:** GPU parity, changed-input graphs and sanitizers pass; compact-versus-v2 ABBA timing is reviewed. Populated VRAM calls lose at T=1/4/8; do not integrate this prototype.
   4. **Pending:** If the compact table wins, integrate separately and measure cache residency, full inference and acceptance.
   5. **Screen complete; selective hot-cache integration conditional:** The compatible signed-nibble codec round-trips all 65,536 table/sign combinations and 384,000 real blocks. The sm75 reader is bitwise equal on nine shapes and 1.51–1.76x faster in the original isolated row-dot timing; CUDA sanitizers pass and sm70 compiles. A two-run pinned-host copy-plus-dot screen remains bitwise equal but loses at all nine 1/3/30-expert by 1/4/8-token shapes because the converted bank is 1.95x larger. Each 512-expert layer would add 437.5 MiB. Profile actual expert reuse before selective integration; grouped GU/SwiGLU/down, prefill, cache displacement, overlapped PCIe and full-model quality/timing remain open. Do not adopt on this kernel result alone. See [screen](bench/volta-turing/iq2-nibble-layout.md).

5. **Measurement and correctness requirements — apply to every candidate**

   1. **Partial:** An opt-in per-window acceptance/dispatch trace is built in `generate.cpp`; serial windows report counter deltas and pipeline windows explicitly mark dispatch as overlapped. Runtime parity, overhead and route-frequency analysis remain pending while the service is stopped to free RAM. Pinned-stage per-stream fences from #3483 remain to evaluate. See [trace notes](bench/volta-turing/window-trace.md).
   2. **Ongoing:** Use same-day ABBA comparisons at short/4K/16K/32K with identical model, chunks, ring, residency, memory budgets and prefix state.
   3. **Ongoing:** Record accepted tokens/s, prefill/TTFT, draft/verify time, acceptance, PCIe traffic, graph/temporary bytes, cache slots, clocks and variance; separate profiling from serving timing.
   4. **Ongoing:** Check exact bytes where expected; otherwise teacher-forced scores/KL and task quality. Run kv1188 and appropriate CUDA sanitizers.
   5. **Pending:** Extend coverage to real activation outliers, FP64 scale references, FP32 attention accumulation, fixed reduction order, accepted recurrent-state tails and request-local sampling.
   6. **Ongoing:** Test expert IDs above 255, empty/ragged groups, owner/version changes, counter wrap, WMMA alignment, poisoned scratch, shuffled pages and changing-input replay.
   7. **Ongoing:** Distinguish cold/warm prefixes, inspect logit margins, preserve natural EOS and test threshold/chunk/capacity boundaries.
   8. **Ongoing:** Build touched CUDA/HIP/SYCL backends where available; explicitly record unavailable runtime/compiler coverage.
   9. **Ongoing:** Commit and locally opt in only qualified improvements, smoke-test the installed release and retain rollback/evidence. A rejected experiment is a completed investigation, not an adopted optimization.

6. **Expert dispatch by format and shape**

   1. **Complete:** Run the 90-case mode-0/4/8 screen and mode-8 engine ABBA; no consistent overall decode gain was established.
   2. **Complete, held:** Eight fresh v19 default/0/4/8 legs with fixed 1,024 MiB reserve match cache slots, chunks and pinned RAM. Mode 0/8 retain exact output hashes and draft counts; neither establishes a repeatable material decode or prefill gain. One mode 4 leg changes output and draft work. First-request 4K prefill is confounded by 0.68–13.92 s prompt-room lending, explaining the earlier 18.14 s mode-8 outlier as the same kind of setup stall. Keep default dispatch. See [screen](bench/volta-turing/dispatch-v19-screen.md) and [outlier review](bench/volta-turing/mode8-4k-outlier.md).
   3. **Pending:** Use measured production route frequencies and full-call costs to select opt-in format/shape cells; include GU/epilogue/down and input conversion costs.
   4. **Pending:** Validate numerics, teacher-forced output, acceptance and accepted tokens/s before promotion.

7. **Host transfers and KV copy work — C11–C13**

   1. **Pending:** Compare mapped-host expert reads with bounded DMA during concurrent compute, including small decode misses and expert-cache displacement.
   2. **Partial, K8V4 range screen complete:** On RTX 2060 SUPER, a known contiguous 1–256-page prefix staged by the existing DMA path passes exact byte checks and is cheaper than cold reset + streamed resolution in an isolated graph, but the two paths have different planning prerequisites and converge at larger ranges. Concurrent compute/PCIe, occupied-range cold experts, fragmented selections and full prefill/decode remain unmeasured; no threshold or dispatch change is adopted. See [screen](bench/volta-turing/kv-transfer-screen.md).
   3. **Isolated screen complete; inference gate pending:** An opt-in one-CTA-per-K/V/scale-run copy passes four-format streamed parity, K8V4 targeted memcheck/racecheck and sm70/sm75 compilation. K8V4 reset + resolve + copy ABBA is about 1.22x faster for one cold miss and 1.15x for four, with little difference at 64 misses. This is a synthetic GPU-call result; route frequency, concurrent compute/PCIe, full-model quality, accepted tokens/s and cache displacement remain unmeasured. Keep `STRATA_KV_COPY_RUNS` disabled. See [screen](bench/volta-turing/kv-copy-runs.md).
   4. **Applicable resolver audit complete; no new change:** Strata's streamed copy already uses the placed-miss count, so cache hits and unplaced blocks cause no host K/V copy; attention masks nonresident rows before K/V loads. An invalid-host-pointer hit test passes all four formats and targeted K8V4 memcheck on both default and split-run copy kernels. An over-capacity request is outside the production slot contract and was not qualified; skipping V for valid zero-weight attention rows would be a separate numerical experiment. The upstream direct-resolver shortcut does not justify a new Strata opt-in. See [audit](bench/volta-turing/kv-unused-values.md).
   5. **Complete for CUDA streamed decode on RTX 2060 SUPER:** Host-KV zeroing was already omitted in `qsa_state_zero`. A poisoned-host parity run passes all four formats, 306 eviction batches per format, reset/reuse with a partial last block, ring restore and 4/4 changed-input graph replays per format. This validates the tested CUDA read-after-write path; it is not a new speedup. HIP host-DMA prefill, full-model continuation, sanitizers and V100/T4 runtime remain outside this check. See [evidence](bench/volta-turing/kv-host-zeroing.md).
   6. **Source audit complete; runtime qualification pending:** The CUDA host allocation, reset, serial staging, opt-in prefetch release/ready fences and drain paths have been traced. No missing fence was found. Session-pinned buffers persist until process exit, so future in-process reinitialization needs explicit cleanup. RTX 2060 SUPER prefetch overlap, cancellation/relayout at runtime and HIP host DMA still need tests before any adoption. Reuse the existing staging pool and chunk balancing. See [audit](bench/volta-turing/kv-stage-lifetime-audit.md).

8. **KV cache policy and bounded history — C14/C15**

   1. **Partial, synthetic policy screen complete:** K8V4 parity emitted 306 selected-page batches; a CPU CLOCK replay predicts 70,472 misses versus 70,502 measured on GPU. Ideal LRU saves 0.34% of simulated misses, while a simplified 16-probe/64-hand screen adds 4.85%. Constructed shared-prefix reuse ties all policies; constructed disjoint-context churn saves 3.86% of misses with the probe screen. These are no model traces or GPU policy timings. Collect real shared-prefix/disjoint model traces, then qualify a GPU prototype before adopting any replacement change. Keep CLOCK and K8V4. See [screen](bench/volta-turing/kv-policy-replay.md).
   2. **Partial source audit and accounting; experiment pending:** The target has a CLOCK hot cache sized by `--kv-resident`; the INT8 draft has a modulo ring sized from `--mtp-window` plus round headroom, not an independent hot-cache policy. A smaller draft ring would change readable history and acceptance. Target RAM bytes now use the actual KV format, and draft resume-restore payload has a separate counter. Hold draft window and total VRAM fixed while comparing target budgets, KV misses/PCIe bytes, expert displacement, acceptance and time. A separate draft residency policy would be required for a genuine two-hot-cache split. See [audit](bench/volta-turing/kv-budget-audit.md).
   3. **Partial bounded staging prototype; production pending:** A CUDA sm75+ LSE hook and test-only bounded pinned-bounce staging pool now cover FP16 and K8V4. One-slot, empty/partial-page and sparse 32K page-remapping cases pass against unsliced output; 128-slot sparse screens need 15 windows and transfer 5.94 MB (K8V4) or 14.91 MB (FP16). Targeted memcheck passes; ordinary output hashes stay intact. The prototype synchronizes every window, keeps full test pools allocated and has no production caller or measured inference speedup. Integrate a capped planner with safe copy/compute overlap, handle live INT8 draft, then qualify quality, PCIe, expert displacement and speed in full inference. See [audit](bench/volta-turing/kv-bounded-history-audit.md).

9. **PLE and temporary-memory lifetimes — C17**

   1. **Partial direct-I/O screen:** The real IQ3_XXS PLE table takes 209.7 us per synthetic token for 1,000 direct, uncached 16-row gathers on this host, with only 0.5 MiB of the GGUF resident afterward. This isolates table-read cost but does not establish its share of inference because table I/O can overlap layer work. Record GPU PLE phases and complete prefill/decode windows before naming a bottleneck. See [screen](bench/volta-turing/ple-direct-io-screen.md).
   2. **Complete, existing path verified:** Both legacy and native PLE convolution already read the nine-row history directly, without a padded buffer. On RTX 2060 SUPER, an independent padded-history CPU reference agrees with the single-token legacy path and native 2/3/4/8-token windows (worst absolute error 4.77e-7); the native batch results, final history and snapshots remain bitwise equal to the serial path. This verifies arithmetic, not a new speedup or full-model timing. Run `verify_batch_parity` for the synthetic gate.
   3. **Partial, earlier reuse measured:** `STRATA_PF_HC_REGION=1` already reuses the prefill HC projection scratch after each half's last reader, saving 63,360 bytes per token outside the shared region in the normal BF16 path. The original RTX 2060 SUPER IQ3_XXS 15.5K-prompt screen grew the auto chunk from 5,120 to 6,400 and prompt speed from 508.6 to 548.2 tok/s; recent benchmark configs enable this opt-in. The separate `STRATA_HC_REQ8` and `STRATA_PF_HC_FAST` conversion scratches can each persist at 6.25 MiB on this geometry. Measure safe release/reuse, cache slots and full request timing before changing those lifetimes. See [audit](bench/volta-turing/hc-scratch-lifetime-audit.md).
   4. **Partial:** A padded-history CPU reference checks SiLU convolution for both PLE paths; native 2/3/4/8-token windows retain bitwise-equal results, committed history and snapshots. The legacy PLE block plus history advance matches eager output/history bitwise across two changed-input CUDA graph replays on RTX 2060 SUPER. Fixture-free hash parity passes independent NULL/EOS/zero-token-ID oracle cases. Synthetic PLE tables now pass empty batches and an out-of-range row in a live batch, for direct and mapped I/O across all seven formats. Graph lifetimes of the persistent HC conversion scratches remain to qualify before changing those buffers.

10. **Verifier and adaptive drafting — C19/C20/C22**

   1. **Partial existing-path screen:** The default QSA scorer already reads each pooled key once for a 2–8-query call. On RTX 2060 SUPER at 32K context, its isolated score call is 1.70x/1.81x/1.88x faster than the per-query scorer at 2/4/8 queries with matching hashes; this is an existing gain, not new inference work. A cache across verification windows would need per-layer/state ownership and append versions; collect real traces and full-round timing before implementing it. Bounded K/V staging remains open under item 8.3. See [audit](bench/volta-turing/qsa-key-reuse-audit.md).
   2. **Partial accounting:** Current 36-layer FP32 GDN state is 112.219 MiB persistent VRAM; a two-byte state has a 56.109 MiB upper saving before conversion costs, about 33 *average* IQ2_XS cache slots in the October 21 log, not a measured increase. The verifier's 16-token GDN `qkv_L_` and `h_L_` commit rows take another 45.0 MiB of its reported 127.2 MiB arena. Expert staging is already capped at 16 blobs (eight per pipeline group). Prototype compact state only with accepted-prefix/state parity; trace actual staging occupancy, slot ownership/version and complete-round timing before changing leases. See [audit](bench/volta-turing/verifier-state-lease-audit.md).
   3. **Pending:** Profile grouped expert projections and mixed dense fusion only where Strata has an uncovered operation.
   4. **Complete investigation; no adoption:** The installed MTP draft head is native Q5_K and runs with `STRATA_MMVQ_GROUP=16`. The earlier Q8_0 one/four-row screens did not cover that route. A grouped-warp prototype preserves non-grouped Q5_K output bits and passes sanitizer, but loses to installed group-16 by 18–44% in speed ratio at 2/4/6/8 columns, with different output hashes; its patch is archived. Cross-read RMS reuse is invalid because the two mixers normalize different residuals with different weights. Whole-batch HC reads already run by default for both MLP and final head on CUDA; RTX 2060 SUPER real-geometry and graph parity pass. No change is adopted. See [Q8 screen](bench/volta-turing/mtp-draft-head-q8-screen.md), [grouped-warp screen](bench/volta-turing/mtp-group-warp-screen.md) and [HC audit](bench/volta-turing/mtp-hc-rms-audit.md).
   5. **Pending:** Identify specific width/chain-policy gaps with per-window traces, structured outputs, changing context and accepted-token costs.
   6. **Pending:** Require complete verifier-round improvement, including target work, sampling and state commit; do not duplicate existing adaptive drafting.

11. **Dense weight reuse for T=9–16 — C01**

   1. **Conditional:** Proceed only if lookup profiling shows material T=9–16 work.
   2. **Pending:** Decode each packed weight once while preserving independent FP32 accumulators and each row's reduction order.
   3. **Pending:** Include quantization/interleave and complete pipeline costs in comparisons against group16; retain the slow large-Q5_K-head fallback.
   4. **Later:** Keep M=17–32 FP8/QPN8 reuse in the future-format work; do not expand verifier capacity merely to exercise a kernel.

12. **QSA prefill optimization — C06/C07**

   1. **Complete:** A real 4,096-token RTX 2060 SUPER prompt admitted `qsa_block_scores_tc` on all 192 scoring batches (12 QSA layers, 16 batches of 256 queries each) with the installed FP16-key/SIMT selector settings. The verifier tail was a separate one-token window. Decode and verifier call `qsa_block_scores` directly; the separate `native_qsa_score` API has no production call site. See [route audit](bench/volta-turing/qsa-prefill-score-route.md).
   2. **Conditional:** Compare native scoring with cuBLAS plus exact top-k only on material large-prefill shapes; preserve ties and causal tails.
   3. **Complete screen, held:** `STRATA_PA_K8V4_CH16=1` gives 1.16–1.19x isolated attention-call gains and passes FP64 parity, graph replay, sanitizers and experimental sm70 compilation. Four fresh full-model IQ3_XXS/K8V4 legs with matched cache/loan budgets show prefill regressions: paired candidate/base 0.972x/0.984x at 4K and 0.960x/0.965x at 16K. Each arm repeats its output exactly, but CH16 changes text and draft work across arms; decode is not comparable as equal work. Keep the installed opt-in off. The uncovered 2–8-query shared-key reuse investigation and corrected upstream attribution remain open as separate work. See [screen](bench/volta-turing/k8v4-pa-ch16.md).
   4. **Partial, Q4_0 full-model prefill gain:** The existing tensor-core prompt-attention path can be selected on sm75 with `STRATA_PA_Q4_TURING=1`. RTX 2060 SUPER FP64, changed-input graph, empty-selection and CUDA memcheck/racecheck/synccheck checks pass; sm70 translation-unit compilation passes. Isolated ABBA attention-call ratios are 3.06x at 4K, 2.95x at 16K and 2.67x at 32K. Four-leg full IQ3_XXS/Q4_0 inference gives paired prefill gains of 1.163x/1.182x at 4K and 1.333x/1.339x at 16K. A separate 16/16 retrieval-marker quality ABBA passes, with identical same-arm first logits and cross-arm top-1 equality; first-token KL is at most 0.000729. A four-leg teacher-forced screen scores 2,048 fixed continuation positions with exact same-arm repeats but mixed cross-arm results: 13/128 changed top-1 choices on 4K code and 1.078x perplexity on 16K code. A 28-request task ABBA qualifies four edits in every leg with identical cross-arm tokens and 1.110x/1.117x geometric-mean paired prefill gains; new code/prose diverge and need independent review. Cross-arm draft work differs and decode is not equal-work. The installed K8V4 model does not exercise this path, so keep the opt-in off there. See [screen](bench/volta-turing/q4-prompt-attn-turing.md).
   5. **Pending:** Independently trace and evaluate shared-key reuse for 2–8-query prompt-attention calls; distinguish it from the completed CH16 tile screen and from verifier scoring's existing within-call reuse. Preserve causal tails and the FP32 reduction boundary, then require a full-model gate.

13. **Further attention tuning — C08/C09: opt-in V layout adopted on RTX 2060 SUPER**

   1. **Complete:** Profiled D256 K8V4 p.v and tested an opt-in transposed V stage. Static SASS removes 32 shared loads and 32 permutes; paired isolated attention gains are 1.09–1.31x, with byte-identical outputs. The `acc[8][4]` map and exact FP16 rounding stay unchanged. Full-model 32K prefill improves +2.11%/+2.21% in ABBA; 16K is modest and variable across four pairs (+0.98%/-3.85%/+0.87%/+1.64%). No decode gain is claimed. See [task-13 evidence](bench/volta-turing/k8v4-v-transpose.md).
   2. **Complete:** Both layouts use 168 registers/thread and no stack/spills; shared memory is 29,968/31,120 bytes, and the RTX 2060 SUPER occupancy API allows two blocks/SM for each. No launch-cap change.
   3. **Complete on sm75; hardware-limited on sm70:** Grouped D256 heads, ragged/shuffled pages, empty selections, graph replay, exact logits, sanitizers, all-format and mode-3 fallback checks pass on RTX 2060 SUPER. The separate sm70/sm75 accumulator maps and sm70/sm75/sm80 translation-unit builds are checked; actual V100/T4 runtime and other head geometries require their hardware/model. The candidate adds no scratch allocation and retains existing host admission.
   4. **Conditional, not opened:** Direct Q4 code/scale arithmetic would change the tested FP16 rounding boundary; retain it as a separate numerical experiment, outside this adopted layout change.

14. **GDN prefill tuning — chunked C10 and Turing key-head**

   1. **Partial, Turing layouts rejected:** Standalone CUDA screens fit the existing chunked scan into 55,296 or 59,904 bytes of shared memory with one K buffer and either eight value columns or direct-global Q reads. Both are 1.5–1.7x slower than the current Turing column-split recurrence; neither is integrated. The distinct #1228 fused matrix-load/P-A schedule remains untested. See [screen](bench/volta-turing/gdn-turing-shared-screen.md).
   2. **Partial:** Both standalone layouts match output and final FP32 state within 6.47e-7/5.59e-7 of the existing recurrence at 33/128/512/2048 tokens, and the 128-token FP64 comparison passes. Speculative commits and cross-chunk engine state remain untested.
   3. **Partial:** The two sm75 layouts pass actual shared-memory admission and compile for sm70/sm75/sm80; each uses 168 scan registers per thread without spills. Runtime V100 and occupancy under production competition remain untested.
   4. **Partial:** At 2048 tokens the standalone preparation costs 0.71 ms but the scan costs 4.89/5.16 ms, versus 3.77 ms for Turing's current column-split recurrence. Complete input preparation/output, expert-cache displacement and full prefill timing remain unmeasured; keep the earlier slow kernels rejected.
   5. **Complete, held:** `STRATA_GDN_KEYHEAD_TURING=1` gives a bitwise 1.33–1.34x isolated GDN-call gain on RTX 2060 SUPER. A four-leg matched full-model 4K/16K ABBA has identical request hashes, output text, accepted work, expert-cache slots and RAM/file reads. At 16K, paired candidate/base prefill is 0.996x/0.999x and decode is 0.935x/0.942x; 4K prefill is variable, with low VRAM warnings. There is no full-model gain, so keep the installed opt-in off. V100/T4 runtime remains untested. See [evidence](bench/volta-turing/gdn-turing-keyhead.md).

15. **Graph memory and reuse — C18**

   1. **Partial:** Extend measured solo-graph lifetimes/free-memory deltas to batch graphs and arena ownership; distinguish owned bytes from allocator deltas.
   2. **Complete:** Prototype and test bounded solo-graph LRU. Caps 2/3 were not adopted; archive their exact patches, quality results and regressions.
   3. **Partial:** A standalone 4,307-node same-size update succeeded on RTX 2060 SUPER; changed kernel parameters and an output pointer survived 16 alternating changed-input updates (2.05 ms mean update versus 19.28 ms fresh instantiate in the follow-up). A 4,306-node update failed as expected. Explore actual verifier batch state, common padded topology and reuse before adoption.
   4. **Pending:** Include recurring preparation/recapture cost in lookup-policy measurements.
   5. **Pending:** Test padded CPU/GDN/KV state, mixed prefill/decode, concurrency, eviction/recreation, changing inputs and low-memory fallback.
   6. **Pending:** Measure full rounds and graph-padding inflation separately from draft time.
   7. **Conditional:** Seek lookup-16 memory savings without shrinking prompt chunks or raising reserve; 75–100 MiB remains a target, not an established production saving.

16. **Existing T=5–8 wide kernels — hold promotion**

   1. **Complete:** Port wide kernels, fused quantization/interleave and scratch sizing; add opt-in sm70/sm75 overrides and host dispatch tests.
   2. **Complete:** Run 33-shape parity/sanitizer/timing and real dense/head profiling; select measured cells and retain slow-head fallbacks.
   3. **Complete:** Run lookup-8/16 and speculative full-engine ABBA; no repeatable overall gain was established.
   4. **Held:** Qualify group16-versus-wide arithmetic and successful edit/code/prose tasks before reopening promotion; preserve the isolated archived patch.

17. **Fused IQ prefill and scratch — hold promotion / C05**

   1. **Complete:** Build the detached v14 port with the IQ2_S admission fix; kernel references and sanitizers pass, but this is not adopted.
   2. **Complete:** Review teacher-forced scoring and all 76 tasks; numerical differences and failed substantive tasks remain recorded.
   3. **Held:** Resolve model quality and match actual MMQ/fused chunks, ring, residency, per-request loans and prefix reuse before interpreting timing.
   4. **Partial:** Group-tail scratch prototype passes boundary logits and instrumented full-engine memcheck; qualify normal planner behavior, long contexts and race safety before adoption.
   5. **Partial:** Review existing auto/64/128 tile results; use measured per-format/routed-M dispatch rather than a global override.
   6. **Conditional:** Explore consistent smaller-block layouts and a one-stage barrier design; a THREADS-only edit does not cover all rows.
   7. **Partial synthetic check:** `PF_DM_GROUP`'s expert-order accumulation and the baseline router-order combine each match an independent FP32 reference bitwise on RTX 2060 SUPER. Their outputs differ at 6,905/10,240 values; ordinary-column max absolute error is 4.77e-7, while deliberate cancellation reaches 2. Empty experts, a short last group and scratch reuse pass memcheck/racecheck. Full-model quality, CPU/streamed fallback and matched-budget timing remain; keep the switch disabled. See [check](bench/volta-turing/pf-dm-group-order.md).
   8. **Conditional:** Screen narrow padded tactics and large-M Q5_K/Q6_K dequantize-plus-cuBLAS, including physical layout, scratch limits and threshold neighbors.
   9. **Held:** Require a full-engine prefill win; isolated layer gains do not qualify promotion.

18. **Router/Q8 fusion — hold promotion / C02**

   1. **Complete:** Audit QFUSE/QFUSE_HC/QDEDUP and prototype exact original-selection fusion for single/multi-token routing with the correct expert Q8 buffer.
   2. **Complete:** Reject sorted-list selection for measured slowdown and winner-only exponentials for changed routes/weights; preserve full-softmax arithmetic.
   3. **Held:** Finish relevant QFUSE/QDEDUP interaction coverage only if a new variant warrants reopening; preserve skip rules, rounding and nonfinite/tie behavior.
   4. **Held:** Requalify task outputs and acceptance under controlled execution; measure CPU-doorbell delay and complete windows before another adoption attempt.

19. **V100/Volta qualification — requires hardware / C25**

   1. **Hardware:** Run V100 separately before selecting sm70 defaults; also qualify T4 independently of the 2060 SUPER.
   2. **Pending:** Compare merbanan/v100 e24e2b2e, upstream PF_WMMA2 and QPN register-fragment scheduling as design references.
   3. **Pending:** Extend the Q2 approach to IQ4_NL down and IQ2_S gate/up, then IQ3 if useful; retain small-batch DP4A.
   4. **Pending:** Validate lossless prepacking/LUT permutations and inverse mappings while retaining original GGUF bytes and source licenses.
   5. **Hardware:** Measure registers, spills, parity/numerics, occupancy and full inference on V100; compilation alone is insufficient.

20. **Conditional long-context, format and multi-GPU work — C16/C26–C28**

   1. **Conditional:** Beyond 256K, test partitioned exact top-k, draft/target RoPE/capacity agreement and speculation at boundaries; do not raise capacity inside a speed A/B.
   2. **Future formats:** Evaluate native packed FP8/NVFP4/MXFP4 linear/MoE independently on sm70/sm75 with scale precision and fallbacks preserved; do not requantize this benchmark to claim a win.
   3. **Future formats:** Evaluate M=17–32 FP8/QPN8 reuse only for an appropriate checkpoint/workload.
   4. **Multi-GPU:** Test shared pinned KV for replicated heads, query-row partitioning and compact-index all-gather.
   5. **Multi-GPU:** Test root-only/banked index staging and lossless FP16 peer-partial packing, including ownership, epochs, mixed-request fallback and collective costs.
   6. **Future attention:** Independently build/load/test worker-selected sm75 FA2 and sm70 counterparts with capability-aware fallback; this does not replace current quantized sparse attention.

21. **Deferred research — reopen only with new evidence / D01–D04**

   1. **Deferred:** Resident producer/consumer expert pools, including smaller worker pools; explain how to overcome the measured slowdown.
   2. **Deferred:** Expanded IQ3 register-pair packets with 43% more weight traffic, and the acceptance-changing first IQ2 expansion.
   3. **Deferred:** The losing IQ3 pair-reader without a new measured reason.
   4. **Deferred:** Deeper dense prefetch, byte-expanded weights and uniform format specialization; lower registers alone are insufficient.
   5. **Deferred:** HC tile/row partition with transport, the rejected fused indexer, larger custom all-reduce buffers and OOM-prone chunks.
   6. **Deferred:** Layer-major prefill rewrites until new copy-wait profiles justify them; reconcile historical T4 percentages with exact runs.
   7. **Deferred:** Generic BF16 tensor-core caching because of memory cost.
   8. **Deferred:** Lossy routing/tail skipping; inspect the near-full-cache regression before reconsideration.

22. **Completed foundation and adopted improvements**

   1. **Complete:** Investigate Strata and 1Cat code/history plus open/closed PRs/issues; retain source snapshots, discussions and relevant open diffs.
   2. **Complete:** Create the isolated v20 worktree, resolve GPU access and record hardware/clocks/model/configuration/revisions.
   3. **Complete:** Establish the same-day RTX 2060 SUPER baseline and retain raw evidence, build/resource logs and validation reports.
   4. **Complete:** Build sm75 engine/matvec tests, compile the changed matvec unit for sm70, pass dispatch/invalid-TMAX tests and prepare GPU runners.
   5. **Adopted:** Exact-rounding K8V4 register staging, including parity, graph/sanitizer, retrieval/logit, full-engine timing and installed-release smoke checks; local opt-in remains enabled.
   6. **Complete:** Fix the INT8 V2T warp race; validate both switch states, all-format regression and sm70/sm75/sm80 compilation. V2T remains opt-in without a claimed engine speedup.
   7. **Adopted:** Immutable MTP expert-snapshot reuse with ownership/reload/failure checks, exact restoration/graphs, retrieval/logit ABBA, matched-budget timing and installed-release smoke checks.
   8. **Complete:** Test separate eager snapshot allocation; reject local adoption because end-to-end startup gains did not repeat. Preserve the patch and results.
   9. **Adopted locally:** Task-13 transposed K8V4 V staging is committed as `c62eb333` and enabled in the installed IQ3_XXS RTX 2060 SUPER release. The installed-release probe, live binary/switch check and rollback launcher are recorded in [task-13 evidence](bench/volta-turing/k8v4-v-transpose.md). The global default remains off.
