# The V100 tree (merbanan/Strata `v100`)

Strata `main` (296b59d) plus what measured faster on a Tesla V100-PCIe-32GB (sm_70) beside a Ryzen 9 7950X, with
`strata-q36` on Qwen3.6-35B-A3B (unsloth `Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf`, every routed expert in VRAM). Built
with `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70`. `q36_bench.sh` runs the checks below.

## In the tree

| Change | Before | After |
|---|---|---|
| Decode attention: a K/V head's query heads in one block (`k_attn_part_g`, 810d1ea; `STRATA_Q36_ATTN_GQA=0` the old) | 16K prompt: tg256 56.4 tok/s; 16K greedy run 56.4 tok/s | **84.5** / **90.9** tok/s, the same tokens (md5) |

Short prompts are unchanged (pp512 / tg256 1,685 / 99-100 tok/s), the prompt speed too (pp16384 ~1,517 tok/s).

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

Every reviewed PR and its verdict: [STRATA_PRS.md](STRATA_PRS.md).

## Where a V100 still loses

The prompt: MMQ (dp4a) on the routed experts is ~70% of a 4K prompt's GPU time. A Volta-specific expert GEMM (FP16
tensor cores on dequantized tiles, or packed multi-token dp4a as Maya's strata-dense does at 2,333 tok/s on the same
file) is the next step.
