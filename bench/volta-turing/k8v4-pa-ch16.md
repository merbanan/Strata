# K8V4 Turing prompt-attention tile: 16 versus 32 cells

The existing Turing prompt-attention kernel walks selected KV cells in 32-cell chunks. A 16-cell variant uses the same query, K/V storage and MMA layout, with half as many K/V rows in shared memory and one 16-cell p.v step. `STRATA_PA_K8V4_CH16=1` selects it only for K8V4 on sm75. The default remains 32 cells; Volta and other formats retain their existing paths.

The RTX 2060 SUPER `qsa_prompt_attn_parity` fixture used shuffled page IDs, signed Q4 V scales, an FP64 reference, a zero-width selection and changed-input graph replay. The table is per isolated attention call. The 4K variant comparison alternated CH32/CH16/CH16/CH32 processes, with eight calls per internal ABBA arm. The 16K and 32K results are same-shape CH32 and CH16 ABBA runs with eight calls per arm.

| Context | Queries | CH32 (ms) | CH16 (ms) | CH32/CH16 |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 64 | 2.078/2.079 | 1.790/1.787 | 1.16x |
| 16,384 | 128 | 4.246 | 3.560 | 1.19x |
| 32,768 | 128 | 4.256 | 3.598 | 1.18x |

All tested shapes passed the fixture's FP64 bound. At 4K, repeated CH32 outputs matched byte for byte, as did repeated CH16 outputs; CH32 versus CH16 differed by at most 4.768e-6 absolute over 393,216 floats. The distinct result is expected from the different online-softmax chunk boundary; the FP64 comparison remains the numerical gate. CH16's FP64 maximum absolute error was 6.33e-6 at 4K, 6.96e-6 at 16K and 4.9e-6 at 32K. At 1,500/33, CUDA memcheck, racecheck and synccheck all reported zero errors with graph replay and empty selection. The translation unit compiles for sm70 with `STRATA_EXPERIMENTAL_SM60=1`; no V100 runtime is available.

Logs and cross-process output files are under `/run/media/benjamin/TMP/strata-codex-20261023/k8v4-pa-{a0,b1,b2,a3}.*` and `k8v4-pa-optin-{16384,32768}.log`, with sanitizer logs named `k8v4-pa-ch16-*.log`. A full IQ3_XXS model comparison must check exact logits and task quality, matched-budget prefill, decode and accepted-token timings, and installed-release smoke. The model remains unloaded to preserve host RAM; this call-level gain does not qualify local serving adoption yet.
