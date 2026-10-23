# MTP draft-head grouped-warp screen

The installed IQ3_XXS RTX 2060 SUPER run enables `STRATA_MMVQ_GROUP=16` and reports an experimental native Q5_K head. Its 40,525-row draft subset takes 68.0 MiB, matching 2,560 inputs at 1,760 Q5_K bytes per row. The earlier [Q8_0 row-count screen](mtp-draft-head-q8-screen.md) was a synthetic format screen, not the installed draft-head dispatch.

The archived [prototype](mtp-group-warp-prototype.diff) assigns one output row to each of four warps in a CTA. Each lane replays the existing four virtual-warp partial sums in their original order, then applies the same warp reduction. It supported Q8_0 and Q5_K at the 2,560→40,525 shape through default-off switches. The Q5_K test override was placed before `STRATA_MMVQ_GROUP=16` to compare against the actual installed route. That override is **rejected and not present in the source tree**.

Without `STRATA_MMVQ_GROUP`, the Q5_K prototype was bitwise equal to single-column calls across widths 1, 2, 3, 4, 5, 6 and 8: 1,175,225 outputs compared, zero differences and no non-finite outputs. The negative-control layout differed on 487,479 finite outputs. CUDA memcheck, racecheck and synccheck reported no errors; the changed translation unit compiled for sm70. Fully warmed (1,000 warmup calls per width) isolated Q5_K timings favored the prototype by roughly 8–13% over that **non-grouped** layout in two opposite-order ABBA sequences. The Q8_0 prototype also matched baseline hashes and showed an isolated call gain, but the installed head is Q5_K.

The installed group-16 baseline is substantially faster. The table below uses 1,000 warmup and 100 timed calls per width, one 71.32 MB synthetic Q5_K draft-head matrix, and alternating baseline/candidate/candidate/baseline processes. Values are GPU-event microseconds per call; the first and second entries are the paired legs.

| Columns | Group-16 baseline (µs) | Grouped-warp prototype (µs) | Baseline / prototype |
| ---: | ---: | ---: | ---: |
| 2 | 242.5 / 242.9 | 294.4 / 294.5 | 0.82x |
| 4 | 292.0 / 288.3 | 478.5 / 475.3 | 0.61x |
| 6 | 375.8 / 383.7 | 673.9 / 673.9 | 0.56x |
| 8 | 505.1 / 509.2 | 871.9 / 878.6 | 0.58x |

The prototype's output hashes also differed from group-16 at every width. Its exact equality to the non-grouped kernel therefore does not establish production output parity. The direct configured comparison rejects both a speedup and adoption. No full-model request or task-quality run is warranted for this slower candidate while the model remains unloaded to preserve host RAM.

Raw logs are under `/run/media/benjamin/TMP/strata-codex-20261023/mtp-q5k-{warps,group16}-*.log`. `q8_gemv_bench --type Q5_K --shape MTP --warmup-iters 1000 --hash` now reproduces this format and timing method. `mmvq_multi_parity` includes Q5_K and Q8_0 MTP shapes and accepts `MMVQ_CASE` for targeted checks. The installed group-16 route remains unchanged.
