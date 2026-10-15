# The V100 tree (merbanan/Strata `v100`)

Strata `main` (296b59d) plus what measured faster on a Tesla V100-PCIe-32GB (sm_70) beside a Ryzen 9 7950X, with
`strata-q36` on Qwen3.6-35B-A3B (unsloth `Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf`, every routed expert in VRAM). Built
with `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70`. `q36_bench.sh` runs the checks below.

## In the tree

| Change | Before | After |
|---|---|---|
| Decode attention: a K/V head's query heads in one block (`k_attn_part_g`, 810d1ea; `STRATA_Q36_ATTN_GQA=0` the old) | 16K prompt: tg256 56.4 tok/s; 16K greedy run 56.4 tok/s | **84.5** / **90.9** tok/s, the same tokens (md5) |
| ... its key rows in float4 loads (same summation order) | short tg256 99.6-99.7, 16K 84.0-84.6, 16K greedy 90.4-90.7 tok/s | **104.1** / **94.0-94.2** / **97.5-97.8** tok/s, the same tokens |

Together: decode after a 16K prompt 56.4 -> 94.1 tok/s (+67%), after a short one 99.6 -> 104.1 (+4.5%); the prompt speed is unchanged (pp512 / pp16384 ~1,685 / ~1,516 tok/s). Strata main measured 94.3 tok/s short on the same file and card before either change.

## Measured, not taken

| Candidate | Result on this V100 |
|---|---|
| `--no-token-graph` (the graph off) | short 99.5 = 99.5 tok/s, 16K 67.5 against 84.9 with the graph (before the attention change the graph lost at short prompts: 93.9 vs 103.5 at --max-context 8192) |
| `STRATA_SM70_TABLE=1` (PR 1401's V100 GEMV table) | 93.1 against 94.3 tok/s |
| `STRATA_Q36_NO_MMQ=1` (the native grouped prompt experts) | pp512 / 4K / 16K 1,147 / 1,124 / 849 against MMQ's 1,685 / 1,650 / 1,515 tok/s |
| #1909 (all groups in the fused GU / LDS down launches) | short 99.5, 16K 84.2 tok/s against 99.0-99.7 / 84.1-84.7 - noise; same tokens |
| #2058 `STRATA_EXPDEC=1` (grouped gate/up with q8_1 rows in shared memory) | 99.1 / 84.4 tok/s - noise; same tokens |
| `STRATA_EXP_MODE=8 / 4 / 7` (Volta's mode 8, Turing's 4, the AMD default 7) against 0 | 99.8 / 99.6 / 99.7 against 99.4 tok/s short, 84.8 / 84.4 / 84.6 against 84.4 at 16K - noise, same tokens |
| #2357 (the multi-block argmax on sm_70) | strata-q36 has its own two-stage multi-block argmax already |
| #2358 (fused GR norm), #2356 (pf_wmma2 for IQ2_S / IQ4_NL), #1822 (in-place prompt MMQ), #2321 / #1961 (route-resident, DFlash2 runtime) | the Flash-Next path / other expert types - not in strata-q36 |

## The kernel suite on the V100 (kernel_suite.sh, on this tree)

| Check | Result |
|---|---|
| native_grouped_parity, native_multi_parity, mmvq_il_parity (T 1-4, rows 1/2/4 and the V100 table), mmvq_multi_parity (multi_exact), gdn_parity, gdn_rec_parity, iq_multi_parity, quantize_act_parity, router_top10_parity, router_dot_parity, rope_parity, sampler_parity, gemm_bf16_parity, shared_expert_parity, elementwise_parity, bf16_gemv_parity | 0 failures (bitwise where they check bits) |
| iq_parity (10 expert types, fixtures from tools/iq_fixture.py) | 0 failures: dequant exact, mmvq rel 4.6-5.7e-3 (ncols 1-8) |
| gemm_f32_parity | FAILED on sm_70: the opt-in STRATA_DENSE_F32 dense projections are bit-compared with cuBLAS's own Sgemm, whose algorithm choice on cc 7.0 differs for these shapes (T >= 256 / 512 / 4096) - not used by strata-q36 |
| gdn_rec_parity's chunked prompt recurrence | not on this card (needs sm_80+, 93.7 KB a block) |
| q8_gemv_bench | 666 GB/s at 2 columns, 623 at 3, 547 at 4, 434 at 6 |
| mmq_expert_bench | 20-30 TOPS (q8_0 / iq3_s / iq4_nl / nvfp4) |
| mmvq_il_parity --bench | the V100 table equal to the multi kernel within 1% (e.g. Q3_K 6144x2560 T=3: 37.1 vs 37.8 us) - as the end-to-end -1.3% |
| native_expert_parity, native_expert_bench, expert_parity | need Flash-Next shards / packs - not on this machine |

Every reviewed PR and its verdict: [STRATA_PRS.md](STRATA_PRS.md).

## Where a V100 still loses

The prompt: MMQ (dp4a) on the routed experts is ~70% of a 4K prompt's GPU time. A Volta-specific expert GEMM (FP16
tensor cores on dequantized tiles, or packed multi-token dp4a as Maya's strata-dense does at 2,333 tok/s on the same
file) is the next step.
