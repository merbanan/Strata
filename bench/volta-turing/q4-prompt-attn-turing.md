# Q4_0 prompt attention on Turing

The existing mode-4 tensor-core prompt-attention implementation was gated to sm80+ because its sm75 MMA pair had not been parity checked. `STRATA_PA_Q4_TURING=1` now admits it on sm75. The default remains the old prompt-attention path; `STRATA_PROMPT_ATTN_Q4=0` can force the old path even with the new opt-in. sm70 remains excluded.

On RTX 2060 SUPER, `qsa_prompt_attn_parity` format 2 compares the old and new paths to an FP64 reference, checks their outputs and runs changed-input graph replay plus a zero-width selection. The fixture includes shuffled page IDs and signed Q4 scales. Each ABBA arm used ten calls. Timings are per isolated attention call, not full inference.

| Context | Queries | Old (ms) | New (ms) | Old/new |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 64 | 4.787 | 1.564 | 3.06x |
| 16,384 | 128 | 8.440 | 2.859 | 2.95x |
| 32,768 | 128 | 8.505 | 3.186 | 2.67x |

At 1,500/33, 4,096/64, 16,384/128 and 32,768/128, both implementations passed their FP64 bounds. The largest new-versus-old difference was 1.27e-5 absolute, or 1.7e-6 of output scale, at 32K. The 4K ABBA arm pairs were old 4.790/4.784 ms and new 1.565/1.563 ms. The 16K/32K old arms drifted lower in their second legs; the ratios above use the full ABBA averages. The result supports the call-level opportunity but needs a full model gate.

Evidence logs are under `/run/media/benjamin/TMP/strata-codex-20261023/q4-turing-{4k,16k,32k}-abba.log`. The current installed IQ3_XXS model uses K8V4 KV, so this Q4_0 path has no bearing on its measured prefill or decode speed. A Q4_0 checkpoint needs exact outputs, accepted-token and task-quality checks, matched-budget full inference ABBA, and release smoke before enabling this feature for serving.

## Q4_0 full-model screen (2026-10-24)

Four fresh RTX 2060 SUPER IQ3_XXS/Q4_0 legs ran base/candidate/candidate/base with identical configuration except `STRATA_PA_Q4_TURING=0/1`. Each leg used 4K and 16K prompts and 256 generated tokens. Paired prompt-processing rates were 1.163x/1.182x at 4K (base 422.75/413.19, candidate 491.63/488.35 tok/s) and 1.333x/1.339x at 16K (base 526.67/524.75, candidate 702.03/702.40 tok/s). Both arms repeat their own text, but outputs and draft work differ between arms; decode ratios therefore are not equal-work comparisons. The 16K candidate decode ratios were 0.968x/0.977x. Raw configurations, responses, logs and summary are under `/run/media/benjamin/TMP/strata-v20-q4attn-abba/full-20261024a/`.

A separate cold-server quality ABBA passed the same four retrieval markers at 1K, 4K, 16K and 32K in all four legs (16/16). Each arm's first-token FP32 logits repeat bitwise across its two legs. Candidate versus base keeps the top-1 token at all lengths, has 9/10 top-10 overlap and first-token KL of 0.000118, 0.000729, 0.000152 and 0.000085 respectively. These first-token checks do not establish full-continuation or broad task quality. Evidence is under `/run/media/benjamin/TMP/strata-v20-q4attn-abba/quality-20261024a/`.

The full-model prefill signal is substantial, but the Q4_0 opt-in remains off pending continuation/task qualification and serving smoke. The installed K8V4 profile cannot benefit from this Q4_0 path.

At 1,500 context and 33 queries, changed-input graph replay and empty selection passed under CUDA memcheck, racecheck and synccheck; all three reported zero errors. The touched CUDA translation unit also compiles for sm70, though sm70 runtime stays on its original path. Sanitizer timings are instrumentation results and are excluded from the table.
