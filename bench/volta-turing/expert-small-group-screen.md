# Turing expert dispatch with real small-group shapes

2026-10-24, RTX 2060 SUPER (sm75), installed IQ3_XXS/K8V4 model. This follows the [window-level dispatch inventory](expert-route-opportunity.md). The opt-in host-planner trace `STRATA_EXP_ROUTE_TRACE=1` records the actual resident and PCIe distinct-group/entry counts, layer formats and verifier group width after the pool decides its routing. It does not time a GPU kernel. The auditor in `audit-expert-route.py` drops capture/prefill planner rows and requires a complete 48-layer set for each committed serial window, matching the known pack formats and split width. The raw four-leg trace A/B and both audits are under `/run/media/benjamin/TMP/strata-v20-exp-route/full-20261024a/`.

The two traced legs have identical per-format, per-width cells. Both 4K and 16K requests retain identical text, request hashes, generated tokens, accepted/offered drafts and expert-cache work across the four trace-off/on/on/off arms. The 4K trace has 124 windows and 226 split verifier groups; 16K has 125 and 225. The actual resident entry totals (39,924/64,209) exactly match the independent window trace's VRAM hits, and PCIe distinct-group totals (5,239/2,540) exactly match its PCIe expert count. These cross-checks exclude graph-capture planner calls from the counted execution.

| Prompt | Group width | Resident calls | Resident groups/call | Resident entries/call | PCIe groups/call | Zero-resident calls |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 1 | 6,720 | 2.56 | 2.56 | 0.26 | 26.2% |
| 4K | 2 | 4,032 | 3.98 | 5.25 | 0.83 | 15.9% |
| 4K | 4 | 96 | 7.77 | 16.47 | 1.32 | 1.0% |
| 16K | 1 | 7,104 | 4.46 | 4.46 | 0.08 | 7.6% |
| 16K | 2 | 3,696 | 6.57 | 8.80 | 0.54 | 3.5% |

Each resident call also has a PCIe-share call, but the strided PCIe launch takes mode 0 by design. Unlike the earlier 16-group isolated screen, these requests often have only 2–7 active resident groups. The complete expert benchmark with original GGUF weights, GU, SwiGLU/Q8 epilogue, down, and a new `--with-input` Q8 conversion option was screened at G=2/4/8, one entry per group, and G=4 with two entries per group. The input-inclusive G=4 mode-8/default ratios were 1.16–1.27x for IQ3_S/IQ2_S at one entry per group and 1.69–1.86x at two; the tested outputs were bitwise equal. All ten representative format pairs passed bitwise output comparison. These are GPU-call results with synthetic activations and populated VRAM weights, not accepted-token speedups.

The current v20 global `STRATA_EXP_MODE=8` four-leg base/candidate/candidate/base gate kept exact 4K/16K output, request and logged work. At 16K, paired candidate/base prefill was 0.997x/0.999x and decode 0.982x/0.986x. At 4K, decode was 1.020x/1.021x, while prompt-room lending stalls make prefill ineligible. This global switch is not a safe general default for these requests. In both traced legs the logged CPU pool duration exceeded the GPU stage duration in all 498 measured windows; median pool-minus-GPU differences were 1.68/1.49 ms at 4K and 1.11/1.44 ms at 16K. This is consistent with GPU-call gains being hidden by overlapping CPU expert work, although these stage timers do not prove the exact critical path.

`STRATA_EXP_SM75_GU8_SELECT=1` is an isolated candidate for IQ3_S (type 21) and IQ2_S (type 22) resident gate/up when the group capacity is at most 20. It preserves the existing down mode and mode-0 PCIe calls. Unset or `0` retains current dispatch. A forced-mode-4 versus selected-path G=4 screen with input conversion passed bitwise output comparison for ten representative format pairs; IQ3_S/IQ2_S complete-call ratios were 1.20–1.39x at one entry per group and 1.64–1.81x at two.

## Selective full-model gate

The four-leg base/candidate/candidate/base K8V4 gate used the same clean v20 binary, pack, fixed 131K capacity, 20,480 resident KV cells, prompt chunks, 1,403 expert-cache slots and 39.60 GiB pinned expert complement. `STRATA_EXP_SM75_GU8_SELECT` was the only config difference. Every measured 4K and 16K request had identical hash, text, 256 generated tokens, offered/accepted drafts and logged expert-cache work across all four legs. The probe passed. Raw configs, logs and results are under `/run/media/benjamin/TMP/strata-v20-exp-route/select-full-20261024a/`.

| Prompt | Paired prefill candidate/base | Paired decode candidate/base | Decision |
| ---: | ---: | ---: | --- |
| 4K | 0.686x / 0.370x | 1.003x / 0.982x | Prefill setup is confounded; no decode gain. |
| 16K | 0.999x / 0.996x | 1.068x / 0.957x | Decode changes sign between pairs. |

The 4K candidate legs spent 4,277.5/14,692.1 ms lending prompt room, versus 619.9/691.5 ms in the baselines. All four legs warned of less than 512 MiB free VRAM. The candidate's 16K decode time itself changed from 7,146.5 to 7,568.5 ms, despite exact output and work; the first-pair gain does not repeat. Do not enable this selector locally or claim an inference speedup. This finishes the format/shape opportunity screen on the present RTX 2060 SUPER configuration. A later candidate would still need teacher-forced/task quality and accepted-token throughput gates, and V100/T4 runtime remains untested. The runner stopped all services and released GPU memory.
