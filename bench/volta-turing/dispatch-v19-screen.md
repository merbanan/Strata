# v19 expert dispatch screen on RTX 2060 SUPER

2026-10-23. Eight fresh engine legs ran in symmetric order: default, explicit mode 0, 4, 8, then 8, 4, 0, default. Each leg read the same uncached 4K and 16K synthetic prompt and generated 256 tokens. The only config difference among modes was `STRATA_EXP_MODE`; all used the same v19 binary, 1,024 MiB VRAM reserve, 1,137 expert cache slots, 4,608-token prompt chunks, and 39.63 GiB pinned expert complement. No leg logged a low-VRAM warning. The [runner](dispatch-v19-screen.sh), [audit](audit-dispatch-v19.py), [raw summaries](../../../Strata-data/bench/2026-10-21-volta-turing/dispatch-v19-20261023/summary.json) and [work ledger](../../../Strata-data/bench/2026-10-21-volta-turing/dispatch-v19-20261023/audit.json) preserve the eight legs.

Default, mode 0, and mode 8 have identical output hashes, offered/accepted draft counts, and RAM blob counts for both prompts in both repeats. One of the two mode 4 legs differs in output and draft work at both lengths; its other leg matches. Thus mode 4 timing does not pass the equal-work gate. The standalone 90-case expert microbench had bitwise equality, but that did not make these complete continuations repeat identically.

| Shape | Mode 0 vs default, paired repeats | Mode 8 vs default, paired repeats |
| --- | ---: | ---: |
| 16K prefill | −0.230%, −0.106% | −0.243%, −0.043% |
| 16K decode | +0.009%, +2.571% | −0.138%, −1.337% |
| 4K decode | −2.786%, +3.098% | −2.817%, +5.170% |

Mode 0 has only a near-zero first 16K decode change and a +2.57% second change, so no repeatable material decode gain is established. Mode 8 loses at 16K decode in both repeats. Both modes' 4K decode pairs cross zero. Do not change the default dispatch or adopt a format/shape override from this screen.

First-request 4K prefill is ineligible for mode comparison: the prompt-room loan includes a one-time 675 MiB pinned host allocation and device-to-host snapshot copy. Its elapsed time ranges from 682 ms to 13,918 ms here; the v19 mode-8 outlier in the earlier comparison was 18,143 ms. The second 16K request in every leg lends room in about 107 ms. A future 4K comparison should prewarm the long-prompt loan before the measured request and separately time the allocation and copy. This screen does not test V100 runtime or provide a current v20 inference result.
