# The V100 tree (merbanan/Strata `v100`)

Strata `main` (296b59d) plus what measured faster on a Tesla V100-PCIe-32GB (sm_70) beside a Ryzen 9 7950X, with
`strata-q36` on Qwen3.6-35B-A3B (unsloth `Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf`, every routed expert in VRAM). Built
with `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70`. `q36_bench.sh` runs the checks below.

## In the tree

| Change | Before | After |
|---|---|---|
| Flash-Next: grouped experts in mode 4 (#2069's Turing layout) by default on sm_70 (35c9285) | modes 0 / 8: 56.8 / 57.2 avg | 57.5 avg (+1.2%), the same tokens |
| Flash-Next: #2357 + #2358 on by default on sm_70 (e584be8) | decode 56.0-56.2 avg | 56.8-56.9 avg (+1.3-2%), the same tokens |
| Decode attention: a K/V head's query heads in one block (`k_attn_part_g`, 810d1ea; `STRATA_Q36_ATTN_GQA=0` the old) | 16K prompt: tg256 56.4 tok/s; 16K greedy run 56.4 tok/s | **84.5** / **90.9** tok/s, the same tokens (md5) |
| ... its key rows in float4 loads (same summation order) | short tg256 99.6-99.7, 16K 84.0-84.6, 16K greedy 90.4-90.7 tok/s | **104.1** / **94.0-94.2** / **97.5-97.8** tok/s, the same tokens |

Together: decode after a 16K prompt 56.4 -> 94.1 tok/s (+67%), after a short one 99.6 -> 104.1 (+4.5%); the prompt speed is unchanged (pp512 / pp16384 ~1,685 / ~1,516 tok/s). Strata main measured 94.3 tok/s short on the same file and card before either change.

## Flash-Next (Strata's main engine) on the V100

Qwen3.8-Flash-Next GSQ-RCO Q2_0 (ISTA-DASLab, 66.4 GB, two shards; pack from `tools/iq_pack.py`), `strata --serve`
with `--spec 2 --prefill auto --kv int8 --vram-reserve-mib 2048 --expert-cache auto` (the tables below up to 35c9285 used `--prefill 512`) (the default 700 MiB reserve left
too little for the verify windows' graph capture here), `fn_bench.py`: three 128-token greedy answers and a 6K prompt.

| Build | Decode (tok/s) | 6K prompt | Tokens |
|---|---|---|---|
| main / v100 before e584be8, two rounds | 56.6 / 56.3 / 55.1, 56.8 / 56.5 / 55.3 | 729-731 tok/s | md5 c7a041f7db35 |
| + #2357 (STRATA_ARGMAX_ROWS_SM70=1) | 56.4 / 56.6 / 55.5 | 728 | same |
| + #2358 (STRATA_GR_NORM_FAST=1) | 57.1 / 56.8 / 55.6 | 728 | same |
| **v100 (e584be8: both, default on sm_70)**, two rounds | **57.6 / 57.3 / 56.1, 57.5 / 57.2 / 56.0** | 729 | same |
| v100 with both set to 0 | 56.1 / 56.2 / 54.3 | 728 | same |

Measured on Flash-Next and not taken (same tokens in every run): #2221 STRATA_Q2_WARP=1 57.3 / 57.0 / 55.8, #2091 57.4 / 57.1 /
55.9, #2222 STRATA_FUSE_WAIT=2 STRATA_QIL=1 57.7 / 57.4 / 56.2, #1949 57.4 / 57.1 / 55.9, STRATA_NO_MMVQ_ROWS2=1 (#2447's switch)
57.8 / 57.3 / 56.0, against v100 57.5 / 57.2 / 56.0 and 57.2 / 56.9 / 55.7. With mode 4 (35c9285) the tree runs 58.2 / 57.8 /
56.7 tok/s; strata-q36 on it 103.9 / 94.2, md5 75a302c75b7c.

### Prompt chunks (`--prefill`), 2026-10-15

`--prefill 512` (what the runs above used) keeps every chunk below `stream_all_min` (1024 tokens), so the whole prompt
takes the staged small-chunk path (8 expert slots, the CPU share on, ~10 rows an expert). `--prefill auto` streams the
experts in chunks of up to 8K. `STRATA_PREFILL_TIMING=1`, fn_bench.py (`FN_PREFILL`, `FN_RESERVE`, `FN_CTX`, `FN_FILL`):

| `--prefill`, reserve, context | 6K prompt (tok/s) | 16.5K prompt | decode (tok/s) | tokens |
|---|---|---|---|---|
| 512, 2048 MiB | 665 | 658 | 58.0 / 57.7 / 56.5 | c7a041f7db35 (8K ctx), c4ce9bff1fe8 (20K) |
| 2048, 2048 MiB | 1,165 | 1,168 | 57.9 / 57.3 / 54.9 | same |
| 4096, 2048 MiB | 1,319 | - | 58.2 / 57.9 / 56.7 | same |
| **auto, 2048 MiB** | **1,527** | - | 58.1 / 57.7 / 56.5 | **same** |
| auto, 3072 MiB | 1,528 | 1,582 | 57.8 / 57.7 / 55.8 | differ (another expert-cache mix) |

The phase split at 6K moves from experts + dequant 57% (512: dequant 13.8%, host grouping 4.8%) to 47% (auto: dequant
4.4%); QSA attention is 10.4% and GDN 14% at auto. With the 2048 MiB reserve, auto fits the verify-graph capture at an
8K context. fn_bench.py now defaults to `--prefill auto`.

Not applicable to this file: #2356 (pf_wmma2) - its experts are all Q2_0 (native_experts.txt: type 42), not IQ2_S /
IQ4_NL. Volta's mode-8 experts, the mmvq_il table and the GR up kernel are main's sm_70 defaults already
(docs/NVIDIA_V100.md).

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
