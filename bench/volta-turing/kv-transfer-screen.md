# Known-range K8V4 KV staging versus cold streamed resolution

2026-10-24, RTX 2060 SUPER, CUDA 12.4, sm75. The model service was stopped. `kv_stream_parity` adds a test-only `STRATA_KV_TRANSFER_BENCH=1` screen for the existing `kv_stage_from_host` DMA path. It captures 16 repeated copies of a known contiguous prefix, launches that graph 64 times per sample, and reports the median of three GPU-event samples. The same process's `STRATA_KV_COPY_BENCH=1` screen captures 32 repeated cold map resets and `kv_stream_resolve` calls, with 128 graph launches per sample. Both use the same pinned host K8V4 copy and VRAM pools; the second run checked every staged K, scale and V byte after each DMA size. All full parity checks also passed.

| Pages | Encoded bytes | Cold reset + resolve + mapped copy, µs/call (two runs) | Known-range DMA stage, µs/call (two runs) |
| ---: | ---: | ---: | ---: |
| 1 | 3,264 | 10.509 / 10.817 | 5.964 / 6.459 |
| 4 | 13,056 | 11.608 / 11.858 | 6.673 / 6.186 |
| 16 | 52,224 | 16.571 / 16.366 | 11.912 / 11.831 |
| 64 | 208,896 | 39.573 / 39.722 | 35.233 / 35.333 |
| 256 | 835,584 | 135.076 / 135.111 | 129.241 / 129.295 |

The operations have different prerequisites: DMA is usable here because the host already knows a contiguous prefix and its destination, as in staged prefill. The streamed decode path receives GPU-selected blocks and must resolve residency first; moving that plan to the host would add synchronization and planning costs absent from the DMA number. Neither arm includes concurrent expert copies or compute, cache displacement, actual prefill, or accepted-token throughput. The fixed-prefix graph repeatedly copies the same bytes, so these measurements do not establish a cold bandwidth threshold. No new transfer policy or local opt-in is justified. Retain the existing prompt staging and mapped decode path until complete workload measurements show a gain.
