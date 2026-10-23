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

Logs and cross-process output files are under `/run/media/benjamin/TMP/strata-codex-20261023/k8v4-pa-{a0,b1,b2,a3}.*`, `k8v4-pa-optin-{16384,32768}.log` and `k8v4-ch16-{grid,admit,default}-*.log`, with sanitizer logs named `k8v4-pa-ch16-*.log`. A full IQ3_XXS model comparison must check exact logits and task quality, matched-budget prefill, decode and accepted-token timings, and installed-release smoke. The model remains unloaded to preserve host RAM; this call-level gain does not qualify local serving adoption yet.
