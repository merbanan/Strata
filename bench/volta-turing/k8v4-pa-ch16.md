# K8V4 Turing prompt-attention tile: 16 versus 32 cells

The existing Turing prompt-attention kernel walks selected KV cells in 32-cell chunks. A 16-cell variant uses the same query, K/V storage and MMA layout, with half as many K/V rows in shared memory and one 16-cell p.v step. `STRATA_PA_K8V4_CH16=1` permits it only for K8V4 on sm75 when the query count times KV-head count exceeds the current GPU's SM count. Smaller calls keep the 32-cell kernel. The default remains 32 cells; Volta and other formats retain their existing paths.

The RTX 2060 SUPER `qsa_prompt_attn_parity` fixture used shuffled page IDs, signed Q4 V scales, an FP64 reference, a zero-width selection and changed-input graph replay. The table is per isolated attention call. The 4K variant comparison alternated CH32/CH16/CH16/CH32 processes, with eight calls per internal ABBA arm. The 16K and 32K results are same-shape CH32 and CH16 ABBA runs with eight calls per arm.

| Context | Queries | CH32 (ms) | CH16 (ms) | CH32/CH16 |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 64 | 2.078/2.079 | 1.790/1.787 | 1.16x |
| 16,384 | 128 | 4.246 | 3.560 | 1.19x |
| 32,768 | 128 | 4.256 | 3.598 | 1.18x |

The extra short/batch screen found that query count, more than selection width, controls this tile's crossover. With 16 queries the forced CH16 build loses at 512 cells (0.176 versus 0.133 ms) and at 4K (0.707 versus 0.523 ms); with 20 queries it wins at 512 cells (0.203 versus 0.255 ms) and at 4K (0.818 versus 1.015 ms). At 32 queries it wins even with 256 cells (0.116 versus 0.133 ms). The RTX 2060 SUPER has 34 SMs and this geometry launches two blocks per query, so the opt-in admits CH16 only when the grid has more than 34 blocks. The first admitted counts, 18 and 19 queries, win by 1.24–1.26x at both 512 and 4K. After this admission was added, 16-query runs at both 512 and 4K produced the exact CH32 output hashes with the opt-in set. The threshold is a screen-driven guard for this GPU, not T4 or V100 runtime qualification.

All tested shapes passed the fixture's FP64 bound. At 4K, repeated CH32 outputs matched byte for byte, as did repeated CH16 outputs; CH32 versus CH16 differed by at most 4.768e-6 absolute over 393,216 floats. The distinct result is expected from the different online-softmax chunk boundary; the FP64 comparison remains the numerical gate. CH16's FP64 maximum absolute error was 6.33e-6 at 4K, 6.96e-6 at 16K and 4.9e-6 at 32K. At 1,500/33, CUDA memcheck, racecheck and synccheck all reported zero errors with graph replay and empty selection. The translation unit compiles for sm70 with `STRATA_EXPERIMENTAL_SM60=1`; no V100 runtime is available.

Logs and cross-process output files from the isolated screen are under
`/run/media/benjamin/TMP/strata-codex-20261023/k8v4-pa-{a0,b1,b2,a3}.*`,
`k8v4-pa-optin-{16384,32768}.log` and `k8v4-ch16-{grid,admit,default}-*.log`,
with sanitizer logs named `k8v4-pa-ch16-*.log`. At that point the full model
was left unloaded to preserve host RAM; the call-level gain alone did not
qualify serving adoption.

## Full-model screen on 2026-10-24

The current v20 sm75 binary ran the installed IQ3_XXS/K8V4 configuration in
base/candidate/candidate/base order. The only config difference was
`STRATA_PA_K8V4_CH16=0` versus `1`. Each fresh server passed the 156-token
kv1188 probe, then processed the same 4,096- and 16,384-token prompts with
256 output tokens. All four legs had 1,403 expert-cache slots and borrowed
1,185 for prefill. Request hashes and prompt sizes match.

| Prompt | Base prefill tok/s | CH16 prefill tok/s | Paired CH16/base | Base decode tok/s | CH16 decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 495.1 / 480.2 | 481.4 / 472.7 | 0.972x / 0.984x | 28.22 / 27.78 | 29.21 / 28.94 |
| 16K | 730.9 / 724.1 | 701.5 / 698.9 | 0.960x / 0.965x | 35.48 / 33.55 | 35.73 / 32.67 |

Both prefill comparisons regress. Each arm repeats its own text exactly, but
CH16 changes text in both prompts: it offers/accepts 188/139 versus base
192/132 drafts at 4K and 170/141 versus 177/132 at 16K. Thus decode rates
are not equal-work speedups. The isolated attention-call gain does not carry
through this model configuration. Keep CH16 disabled; full-model task quality
and teacher-forced scoring are unnecessary for adoption unless a new variant
establishes a prefill benefit. Configs, responses and logs are under
`/run/media/benjamin/TMP/strata-v20-ch16-abba/full-20261024a/`.
