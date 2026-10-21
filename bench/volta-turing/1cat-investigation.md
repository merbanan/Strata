# 1Cat-vLLM: ideas for Strata on Volta and Turing

Research snapshot: 2026-10-21. Main was
[`6f3557c973b4602905037e086fd4acba9b0d867f`](https://github.com/1CatAI/1Cat-vLLM/commit/6f3557c973b4602905037e086fd4acba9b0d867f).
This report distinguishes their measurements from our RTX 2060 SUPER tests.
No 1Cat kernels or models were executed locally and no upstream changes were published.

## Coverage and reproducibility

The isolated `1Cat-vLLM-research` clone contains the full commit graph, with
historical blobs fetched on demand. Indexed all 7,728 tracked paths, 20,594
commits reachable from main, and 21,782 subjects across the initially fetched
refs. The architecture/quantization search selected 1,682 file paths and
1,493 commit subjects. These counts include inherited vLLM history.

Downloaded all 1,057 PR records: 689 merged, 237 closed without merging,
131 open. Indexed all 183 standalone issues: 98 open, 85 closed. The unique
issues/PR union has 1,240 entries. The issues endpoint stopped at page 11;
fetching ascending pages recovered the older entries, with overlap deduplicated.
PR and issue state refers to this snapshot, not a permanent classification.

Saved 33 selected PR bodies for focused review, examined relevant issue
discussions and design records, and inspected kernel/runtime code. Fetched source
diffs for open PRs #1236, #1229, #1228, #1226, #1242, #1204, #1078 and #959.
This is a repository-wide inventory and focused technical review, not a claim
to have read every inherited source line or every comment in every PR.

Raw evidence is under `Strata-data/bench/2026-10-21-volta-turing/1cat-research/`:
`snapshot.json`, `all-files.txt`, `history.txt`, `relevant-history.txt`,
`pr-index.tsv`, `issue-index.tsv`, API JSON, selected discussions and PR patches.

## Highest-value experiments for Strata

### 1. Share weight decoding across verification rows, without expanding weights

Merged [#1142](https://github.com/1CatAI/1Cat-vLLM/pull/1142) replaces three
independent M8 tiles at M20 with one weight pass feeding three accumulators.
Their paired V100 projection sum fell 2.9573 to 2.3808 ms, with unchanged output
bits on the measured shapes. This is an operator result, not a model speedup.

Code: `csrc/sm70_turbomind/ops/gguf_dense_weight_major_sm70.cuh` and
`gguf_dense_segments_sm70.cuh`; integration commit `97c12d1903a0`.
`body_weight_major` decodes a group outside the token-tile loop, keeps independent
FP32 accumulators, and retains each accumulator's reduction order. The code
uses Volta fragment mappings; it is not a Turing drop-in.

Strata already has interleaved dense kernels for small token counts. Our v20
extension to T=5..8 tests the same reuse principle through integer dot products.
The next distinct experiment is reuse across T=9..16 lookup verification,
with two token groups and no extra resident weight bank. Benchmark against
the existing group16 route, including quantization and interleave cost.
Do not enable large Q5_K head shapes merely because smaller projections win.

### 2. Fuse router selection with an activation quantizer that is actually still needed

Merged [#1147](https://github.com/1CatAI/1Cat-vLLM/pull/1147) adds a native
top-10 selector and fused selection/Q8_1 operator, but leaves model dispatch
unchanged. On their V100, 48-layer projection/selection/quantization totals
improve about 8% at M5 and 43% at M20; the standalone M5 selector loses 5%.

Code: `csrc/sm70_turbomind/ops/sm70_router.cuh`, `sm70_router.cu`,
`gguf_q8_1.cuh`; merge `ec60883ad718`. One warp in the quantizer's first
block selects routes while independent blocks quantize activations. It avoids
a global completion counter. Local sorted lists merge only the winning heads.

Strata's `verify.cpp` calls `native_router_top10_multi` separately from expert
activation quantization. However, `STRATA_QFUSE`, `STRATA_QFUSE_HC` and
`STRATA_VERIFY_QDEDUP` already remove some quantization work. Measure coverage
before adding another fusion. Their selector consumes FP16 logits; Strata's
router consumes FP32 logits, so copying the packed FP16 sorting keys would
change expert selection. Preserve Strata's tie ordering, nonfinite behavior,
normalization and quantizer rounding. Test exact route IDs and Q8 bytes first.

### 3. Revisit Turing chunked GDN using shared-buffer lifetime reuse

Open [#1228](https://github.com/1CatAI/1Cat-vLLM/pull/1228), head
`9c291f494279`, adds a fused SM75 TileLang GDN prefill kernel. The existing
SM70 variant needs 86,016 bytes of shared memory. The new variant removes
the transposed K copy, uses a transposed matrix load, aliases the P tile with
the no-longer-needed A tile, and inserts a loop-end barrier before reuse.
It selects the variant using actual available shared memory.

Code in the PR: `flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py`.
The PR tests outputs and final recurrent states against FLA within 2e-3;
this is not bitwise equivalence to Strata. It is a materially different
memory design from the previously rejected Strata chunked GDN experiments.
Prototype the buffer schedule in native CUDA before considering a TileLang
dependency. Keep FP32 recurrent state and measure the complete preparation,
recurrence and output chain, including scratch lost from the expert cache.

Open [#1226](https://github.com/1CatAI/1Cat-vLLM/pull/1226), head
`4eae43fa5273`, also offers a useful diagnostic: adding high-register variants
to the FLA autotuner cuts a Turing recurrence microbenchmark from 6.91 to
3.01 ms. The selected configurations still spill. Strata does not use this
Triton kernel; the transferable action is to inspect ptxas resource usage
and remove occupancy hints that buy no occupancy, not to set every kernel
to 255 registers.

### 4. Share QSA index keys across verification queries

Merged [#1127](https://github.com/1CatAI/1Cat-vLLM/pull/1127), merge
`46931f857f0a`, scores 2..8 queries against each compressed-key stripe once.
The scorer-only C1 comparison improves 19.8219 to 19.4180 ms/round with
unchanged teacher logits/natural outputs in the reported check. Larger gains
combine it with a different attention reader and must not be assigned to
the scorer alone.

Code: `vllm/models/qwen4_exp/nvidia/ops/qsa_shared_key.py` and its native
operator `csrc/sm70_turbomind/ops/qsa_indexer_sm70.cu`. Strata already has batched QSA scoring in `qsa_select.cu`, including
tensor-core paths, plus an SM7x FP32 fallback in `native_qsa_score.cu`.
Trace the actual verifier path first. If it rereads keys per query on SM7x,
test a shared-key SIMT variant that preserves each dot-product order before
changing precision to FP16 MMA. Compare exact top-k sets, ties and causal tails.

Open [#1229](https://github.com/1CatAI/1Cat-vLLM/pull/1229), head
`7485f44cb599`, admits existing cuBLAS scoring and exact top-k on Turing.
Its large speedups replace a poor Triton fallback in 1Cat; Strata already has
native scoring/selection, so the same multiplier is not transferable.

### 5. Tune attention by actual architecture, layout and shared-memory budget

Open [#1236](https://github.com/1CatAI/1Cat-vLLM/pull/1236), head
`b1a646d80b18`, is particularly useful source material. It fixes accumulator
row mappings for Turing, shrinks a planner hash table from 8,192 entries
(96 KiB) to 5,120 (60 KiB), and extends grouped page4 prefill to more query
and KV head layouts. It checks compiled kernel availability and memory
requirements instead of merely accepting a compute-capability number.

Code: `flash-attention-v100/include/fused_mma.h`,
`kernel/flash_decode_paged.cu`, and the QSA Python dispatcher in that PR.
Volta's fragment row mapping differs from Turing's lane/4 and lane/4+8
mapping. This matters when scaling softmax accumulators in registers.
Strata's `qsa_prompt_attn.cu` already distinguishes the two MMA layouts and
has Turing register staging for its INT8 path. Apply this evidence to the
planned K8V4 extension, preserving its Q4-to-half rounding.

[Issue #1132](https://github.com/1CatAI/1Cat-vLLM/issues/1132) proposes keeping
the D128 PV accumulator in registers and relaxing an impossible two-block
occupancy hint. Its reference patch reports 8–12% long-prompt prefill gains
on one V100. The register change alone regresses under the old cap. It is
unmerged issue evidence, D128-specific, and not a direct replacement for
Strata's D256 sparse attention. Use it to audit spill/occupancy tradeoffs.

Merged [#797](https://github.com/1CatAI/1Cat-vLLM/pull/797) also packages a
separate SM75 FlashAttention-2 library selected for the current worker device.
Inspect `vllm/vllm_flash_attn/flash_attn_interface.py` for capability-aware
loading. The merger's validation was CPU-only; it explicitly did not repeat
the earlier author's Turing hardware tests. This is useful for a general
FP16 attention backend or mixed-GPU stages, rather than replacing Strata's
specialized quantized sparse attention directly.

[Issue #441](https://github.com/1CatAI/1Cat-vLLM/issues/441) led to merged
#469's narrower pre-Ampere QSA tiles. Later comments correct the original
claim about which source path was active. Its historical headline gains
should not be presented as still-unimplemented main-branch improvements.

### 6. Measure DMA staging under concurrent model work

Merged [#1203](https://github.com/1CatAI/1Cat-vLLM/pull/1203), main's merge,
reports that SM-driven zero-copy reads of cold experts slowed simultaneous
model kernels, even when an isolated copy benchmark looked acceptable.
Copy-engine DMA into about 68 MB of staging per GPU improved the full
prefill. It also balances prefill chunks and avoids zeroing host-backed KV.

Code: `csrc/gguf_expert_cache.cpp`,
`vllm/model_executor/layers/quantization/gguf_expert_cache.py`, and
`benchmarks/flashnext_prefill/HANDOFF.md`; key commit `4db646d45`.
Strata already stages prefill experts and balances chunks in the baseline
(`STRATA_STAGER_BATCH=1`, `STRATA_PREFILL_EQUAL=1`). Do not duplicate these.
The new experiment is a paired concurrent-compute/copy test for the remaining
mapped-host decode miss path, followed by a small-miss DMA threshold if it
wins. Count staging bytes, lost cache slots and stream fences. A second large
bank is especially unattractive on the 8 GB 2060 SUPER.

### 7. Keep KV history encoded and make cache policy sensitive to reuse

Open [#1204](https://github.com/1CatAI/1Cat-vLLM/pull/1204), head
`46f5c8c197ed`, keeps original E4M3 bytes/scales in the hot cache and decodes
to existing FP16 operands. Its churn screen improves 17–20%, while shared
selection cases lose 3–10%. Open [#1242](https://github.com/1CatAI/1Cat-vLLM/pull/1242)
adds probe-LRU eviction, shared host KV, bounded prefill staging windows and
partitioned exact top-k beyond 256K. These PRs contain larger campaigns and
dependencies, not isolated drop-in patches.

Strata already has K8V4 encoded pools and a resident KV tier. Borrow the
cache-policy experiments and bounded staging ideas, not E4M3 as an automatic
format replacement. Record hit rate, actual PCIe bytes and lookup acceptance
on disjoint-context and shared-prefix workloads. Shared host KV across TP
ranks has little immediate value for our single GPU.

### 8. Native packed formats and Volta-specific tensor-core layouts

Merged [#742](https://github.com/1CatAI/1Cat-vLLM/pull/742) integrates
NVFP4/MXFP4 skinny MoE kernels, adapted from
[v100-skinny](https://github.com/dnv2003/v100-skinny), with device routing and
in-place weight permutation. [#804](https://github.com/1CatAI/1Cat-vLLM/pull/804)
adds Turing linear dispatch. The QPN layout decodes weights directly into
Volta MMA register operands rather than keeping a full FP16 copy.

Code: `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu`,
`nvfp4_qpn2_sm70.cu`, `fp8_qpn8_sm70.cu`, and
`vllm/model_executor/kernels/linear/`. The primary kernel lineage is MIT;
the surrounding vLLM integration has Apache-2.0 notices. Vendored native
GGUF fallbacks separately identify llama.cpp commit `002a12ad2550` and the
vLLM GGUF plugin bridge in `vllm/third_party/gguf_native/PROVENANCE.txt`.

This is useful for the planned V100 register-fragment experiment and future
checkpoint-format support. Our current IQ3_XXS/IQ4/Q2 GGUF pack is not NVFP4.
Prefer original-byte IQ readers or a lossless layout permutation over
requantization. Open [#959](https://github.com/1CatAI/1Cat-vLLM/pull/959)
explores compact original-byte IQ3_XXS storage but has incomplete model
qualification. Strata already retains packed expert bytes, so its storage
savings against 1Cat's expanded canonical representation do not transfer.

## Results that argue against tempting approaches

* [#1078](https://github.com/1CatAI/1Cat-vLLM/pull/1078), open research:
  resident producer/consumer expert scheduling loses to the separate chain,
  including the smaller worker-pool revision. Fewer launches are insufficient.
* Closed-unmerged [#1068](https://github.com/1CatAI/1Cat-vLLM/pull/1068), with
  merged record [#1121](https://github.com/1CatAI/1Cat-vLLM/pull/1121): expanding
  IQ3_S to the register-pair packet adds about 43% weight traffic and does not
  consistently improve the four measured ranks. Do not port that expansion.
* Merged [#1011](https://github.com/1CatAI/1Cat-vLLM/pull/1011): the first IQ2
  nibble expansion sped kernels but hurt emitted-token latency as acceptance
  fell. Its revised arithmetic-compatible version produces a much smaller
  whole-model gain and consumes extra memory. Measure bytes and acceptance.
* Merged [#982](https://github.com/1CatAI/1Cat-vLLM/pull/982) fuses intermediate
  quantization into gate/up. Strata's fused expert mode already covers related
  work. Their operator savings are not evidence to force mode 8 everywhere.
* Their dense-reuse report rejects deeper prefetch, byte expansion and uniform
  format specialization despite lower register counts in some variants.

## Validation lessons from issues and history

[#98](https://github.com/1CatAI/1Cat-vLLM/issues/98) traced historical paged
prefill nondeterminism to an inadequately aligned WMMA shared-memory matrix;
checking output once would miss it. Include shuffled pages, ragged tails,
poisoned scratch, changing-input graph replay and natural termination.

[#1003](https://github.com/1CatAI/1Cat-vLLM/issues/1003)'s latest investigation
still calls long-context retrieval flips inconclusive. Chunk boundaries,
model capacity and profiled allocation changed between configurations.
Keep these fixed in an optimization comparison and measure logits/quality
when reduction order changes. Do not claim a state bug from differing text.

[#1220](https://github.com/1CatAI/1Cat-vLLM/issues/1220) reports graph-capture
OOM; #1042/#1070 discuss graph-shape and concurrency regressions. Together
with 1Cat's recorded extra packed-weight budgets, these strengthen the case
for Strata's planned graph-memory budget and cache-slot accounting.

The fork's history evolves from pre-Ampere attention support and packed
QPN kernels, through native GGUF integration, integer small-batch readers,
measured weight reuse, and finally constrained-memory expert/KV streaming.
There is considerable already-merged and superseded work. Consult current
dispatch plus a PR's measurement contract, not just its title or closure.

NVIDIA's [Turing tuning guide](https://docs.nvidia.com/cuda/archive/11.7.0/turing-tuning-guide/index.html)
documents 32 resident warps, a 64K-register file and 64 KiB shared memory
per SM, unlike Volta's larger warp/shared-memory capacity. Build and tune
sm_75 separately; running a Volta binary is not evidence of optimal Turing
tensor-core utilization. All proposed defaults still require hardware tests.
