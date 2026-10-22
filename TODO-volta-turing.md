# Volta and Turing inference: numbered backlog

Updated 2026-10-22. This is the authoritative task list, sorted by correctness, current RTX 2060 SUPER work, measured opportunity, and hardware dependencies. Parent order is research priority, not the order of already-running jobs. No running test was cancelled or reprioritized by this edit.

**Current requested execution scope:** complete tasks 1 → 2 → 3 → 13, then stop. Preserve all other leads in this backlog without starting new work on them. Already-running support tests are allowed to finish; their results may inform the named tasks. Local RTX 2060 SUPER validation comes first; V100/T4 hardware work remains separate.

Status: **Built** = host/compiler checks only; **Queued/Running** = GPU work outstanding; **Partial/Pending** = incomplete; **Held** = current candidate failed promotion gates; **Conditional/Hardware/Future/Deferred** = retained scope with an explicit prerequisite; **Complete** = stated investigation finished; **Adopted** = qualified and locally enabled. Ongoing requirements apply to every new candidate.

Previous lists had 110 implementation entries and 31 benchmark entries, including duplicates and stale statuses. Every entry is mapped in [the migration index](bench/volta-turing/todo-index.json). Detailed evidence, source links and historical caveats remain in [implementation history](TODO-volta-turing-history-20261022.md) and [benchmark history](../Strata-data/bench/2026-10-21-volta-turing/TODO-history-20261022.md). The [execution queue](../Strata-data/bench/2026-10-21-volta-turing/TODO.md) tracks running/queued work. Existing C01–C28/D01–D04/L01 references remain valid through the index and histories.

1. **QSA FP16 correctness — highest priority**

   1. **Built:** Correct the halving scorer's pooled-key reads; preserve FP32 behavior and FP32 tail keys. CUDA sm70/sm75 builds pass; HIP remains untested.
   2. **Queued:** Reproduce the original wrong-storage result and verify eight original/fallback/corrected control arms.
   3. **Queued:** Check exact scores/top-k IDs, causal tails, guards, changed-input graph replay and memcheck/racecheck/synccheck.
   4. **Queued:** Compare corrected halving against the FP16-aware fallback on all 76 original task requests, with fixed draft costs and controlled adaptation.
   5. **Queued:** Measure scorer-only and scorer-plus-selection timing in FP32/FP16 ABBA; the harness builds and seven host parser checks pass.
   6. **Pending:** Review task quality, per-request allocations and production-policy performance before committing and adopting the correction.

2. **Reliable baseline and task qualification**

   1. **Complete:** Review the IQ2 continuation run: all 76 requests reviewed; four warmups pass, all 72 substantive responses fail qualification.
   2. **Pending:** Diagnose failed edits, incomplete code and ignored instructions in both baseline and candidate; preserve the original prompts, caps and failures.
   3. **Pending:** Distinguish the QSA storage defect from asynchronous residency, learned draft costs and deferred CPU decisions.
   4. **Pending:** Establish repeatable tokens, prefix reuse, acceptance and work counts under controlled execution; then test normal serving behavior separately.
   5. **Pending:** Qualify baseline and candidate independently before reporting successful-task throughput; retain meaningful independent tests, not only generated tests.

3. **IQ2_XS gate/up fusion qualification — C03**

   1. **Complete:** Audit missing format/shape paths, real pack formats, existing SwiGLU/Q8 fusion and VRAM/PCIe dispatch.
   2. **Complete:** Build opt-in mode-8 and normal-mode-4 variants; synthetic parity, changed-input graphs, sanitizers and repeated kernel timings pass.
   3. **Complete:** Review 16 model retrieval requests, first logits and corrected allocation traces; these pass.
   4. **Complete:** Independently verify model dispatch: 640 fused submissions through graph replay when enabled, zero when disabled; all eight retrieval outputs/reference logits match. Counts do not establish timing or active groups.
   5. **Complete:** Independently review original/v2/compact kernel parity with weights from 30 real experts: all 6000 comparisons pass; activations remain synthetic.
   6. **Queued:** Run the blocking-adaptation/fixed-PCIe diagnostic; retain its remaining draft-policy/deferred-CPU confounds.
   7. **Held:** The original full-workload timing gate declined because continuations differ. Requalify under the corrected baseline before a new performance comparison.
   8. **Pending:** Adopt fusion only after controlled full-model quality, matched budgets and repeatable accepted-token throughput pass.

4. **Compact IQ2 storage and compatible layouts — C04/C25**

   1. **Complete:** Audit revised upstream IQ2 arithmetic and storage expansion; expanding the entire set adds about 12.34 GiB, not a measured resident allocation.
   2. **Built:** Implement the lossless compact IQ2_XS codebook with original packed weights; CPU exhaustive checks and sm70/sm75 builds pass.
   3. **Complete:** GPU parity, changed-input graphs and sanitizers pass; compact-versus-v2 ABBA timing is reviewed. Populated VRAM calls lose at T=1/4/8; do not integrate this prototype.
   4. **Pending:** If the compact table wins, integrate separately and measure cache residency, full inference and acceptance.
   5. **Pending:** Screen the revised compatible IQ2 nibble layout selectively; preserve decoded weights, scale/SiLU boundaries and reduction order. Keep the first incompatible revision deferred.

5. **Measurement and correctness requirements — apply to every candidate**

   1. **Pending:** Evaluate per-window acceptance/dispatch tracing from Strata #3631 and pinned-stage per-stream fences from #3483.
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
   2. **Pending:** Finish the v19 default-versus-explicit-0/4/8 comparison, including mode 4's missing large-down fallback; investigate the anomalous 4K prefill result.
   3. **Pending:** Use measured production route frequencies and full-call costs to select opt-in format/shape cells; include GU/epilogue/down and input conversion costs.
   4. **Pending:** Validate numerics, teacher-forced output, acceptance and accepted tokens/s before promotion.

7. **Host transfers and KV copy work — C11–C13**

   1. **Pending:** Compare mapped-host expert reads with bounded DMA during concurrent compute, including small decode misses and expert-cache displacement.
   2. **Pending:** Measure occupied-range cold-expert and QSA DMA for large contiguous runs; retain mapped gathers for fragmented/small selections and calibrate local thresholds.
   3. **Pending:** Increase small-miss copy-plan parallelism only where complete cache-resolution time improves.
   4. **Pending:** Skip KV value reads only when the values cannot be consumed; preserve alignment and unsupported-case fallbacks.
   5. **Pending:** Consider omitting host-KV zeroing only after proving reads follow writes across padding, reuse and graph replay.
   6. **Pending:** Audit pin registration, initialization, stream ownership and fences; reuse existing staged prefill and chunk balancing.

8. **KV cache policy and bounded history — C14/C15**

   1. **Pending:** Compare probe-LRU under shared-prefix reuse and disjoint-context churn while retaining K8V4 encoded storage.
   2. **Pending:** Compare independent draft/target hot-cache budgets at fixed total memory; measure hit rates, PCIe bytes, acceptance and expert displacement.
   3. **Pending:** Test bounded history staging with checked softmax/LSE merging, FP16 draft history and encoded target history, including window boundaries.

9. **PLE and temporary-memory lifetimes — C17**

   1. **Pending:** Profile PLE separately; verify the actual bottleneck rather than assuming the HDD issue reported upstream.
   2. **Pending:** Test PLE convolution without padded history.
   3. **Pending:** Release HC projection temporaries after last use; measure recovered cache slots or larger usable prompt chunks.
   4. **Pending:** Verify SiLU, state commit, empty/null slots and graph lifetimes for both changes.

10. **Verifier and adaptive drafting — C19/C20/C22**

   1. **Pending:** Profile versioned per-layer QSA key reuse and bounded staging; existing multi-query reuse must not be duplicated.
   2. **Pending:** Measure compact recurrent state and bounded expert leases, including persistent bytes and ownership/version changes.
   3. **Pending:** Profile grouped expert projections and mixed dense fusion only where Strata has an uncovered operation.
   4. **Pending:** Screen grouped draft-output Q8 warps, whole-batch HC and reuse of canonical HC RMS statistics.
   5. **Pending:** Identify specific width/chain-policy gaps with per-window traces, structured outputs, changing context and accepted-token costs.
   6. **Pending:** Require complete verifier-round improvement, including target work, sampling and state commit; do not duplicate existing adaptive drafting.

11. **Dense weight reuse for T=9–16 — C01**

   1. **Conditional:** Proceed only if lookup profiling shows material T=9–16 work.
   2. **Pending:** Decode each packed weight once while preserving independent FP32 accumulators and each row's reduction order.
   3. **Pending:** Include quantization/interleave and complete pipeline costs in comparisons against group16; retain the slow large-Q5_K-head fallback.
   4. **Later:** Keep M=17–32 FP8/QPN8 reuse in the future-format work; do not expand verifier capacity merely to exercise a kernel.

12. **QSA prefill optimization — C06/C07**

   1. **Pending:** After the storage correction, trace actual native scoring admission and distinguish prefill from verifier/decode routes.
   2. **Conditional:** Compare native scoring with cuBLAS plus exact top-k only on material large-prefill shapes; preserve ties and causal tails.
   3. **Pending:** Screen narrower pre-Ampere tiles and only uncovered 2–8-query shared-key reuse; account for corrected upstream performance attribution.

13. **Further attention tuning — C08/C09**

   1. **Pending:** Profile the remaining p.v loads/scales and launch bounds on D256 attention. The adopted K8V4 kernel already keeps its p.v accumulator in registers (`acc[8][4]`); any further register use must preserve measured occupancy and exact numerics.
   2. **Pending:** Audit register pressure/spills where shared memory already limits occupancy; avoid global launch-cap changes.
   3. **Pending:** Validate separate sm70/sm75 accumulator maps, grouped heads, strides, ragged/shuffled pages, device scratch admission and compiled-kernel fallback.
   4. **Conditional:** Treat direct Q4 code/scale arithmetic as a separate numerical experiment; preserve the adopted exact-rounding path.

14. **Chunked GDN shared-buffer design — C10**

   1. **Pending:** Prototype native CUDA with no transposed-K copy, shared A/P lifetime reuse and loop-end synchronization, following the newer #1228 design.
   2. **Pending:** Verify final FP32 recurrent state across chunk boundaries and speculative commits.
   3. **Pending:** Admit by actual shared-memory capacity; test high-register variants only where occupancy is already constrained.
   4. **Pending:** Measure complete preparation, recurrence and output costs plus displaced expert-cache slots; prior 4–7x-slower kernels remain rejected.

15. **Graph memory and reuse — C18**

   1. **Partial:** Extend measured solo-graph lifetimes/free-memory deltas to batch graphs and arena ownership; distinguish owned bytes from allocator deltas.
   2. **Complete:** Prototype and test bounded solo-graph LRU. Caps 2/3 were not adopted; archive their exact patches, quality results and regressions.
   3. **Pending:** Explore same-size updates, common padded topology and batch reuse; direct cross-size updates failed CUDA's equal-node-count requirement.
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
   7. **Pending:** Check PF_DM_GROUP combine order and scratch/grouping interactions.
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
