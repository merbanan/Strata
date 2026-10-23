# Split-run streamed KV copy: isolated screen

2026-10-24, RTX 2060 SUPER (sm75), CUDA 12.4. The model service stayed stopped. The default `kv_stream_resolve` copy kernel uses one CTA per missed page and copies that page's K, scale and V runs in sequence. `STRATA_KV_COPY_RUNS=1` selects an experimental kernel with one CTA per run. It keeps the same 96-CTA launch, the same map and the same encoded bytes. Unset or `0` uses the original kernel.

The standalone `kv_stream_parity` test passed for FP16, INT8, Q4 and K8V4 with the opt-in set: 306 selected-query batches per format, zero overflows, bitwise resident/streamed attention, ring restoration, reused host copy and 4/4 changed-input graph replays. Targeted K8V4 CUDA memcheck reported zero errors, and racecheck reported zero hazards/warnings. The sm75 target built and the changed translation unit compiled for sm70. These checks cover the tested shapes, not all hardware.

With `STRATA_KV_COPY_BENCH=1`, the same test captures 32 repetitions of **reset + resolve + copy** for a fixed selection of 1, 4, 16 or 64 cold pages. It launches the graph 128 times per sample and reports the median of three GPU-event samples as microseconds per complete call. This includes map reset and resolution, not just the copy kernel. Each arm ran in a separate process. K8V4 was screened ABBA; the other formats had one baseline/candidate pair.

| K8V4 misses | Baseline A1/A2, µs | Split runs B1/B2, µs | Paired interpretation |
| ---: | ---: | ---: | --- |
| 1 | 10.789 / 10.491 | 8.769 / 8.677 | about 1.22x faster isolated call |
| 4 | 11.578 / 11.755 | 10.191 / 10.122 | about 1.15x faster isolated call |
| 16 | 16.132 / 16.367 | 15.682 / 15.682 | about 1.04x faster isolated call |
| 64 | 39.565 / 39.596 | 39.201 / 39.141 | about 1.01x, little practical difference |

The single paired screens of INT8, FP16 and Q4 also passed parity. Their one-miss baseline/candidate times were 11.605/11.329, 14.299/12.657 and 11.976/9.344 µs; these pairs are exploratory and cannot establish repeatability. The benchmark selects pages from a fully populated host copy and does not exercise concurrent expert transfers, cache displacement or the natural distribution of misses during inference. GPU clocks were not fixed. The complete full-model decode/prefill, accepted-token and memory gates remain open. Keep the option disabled locally and do not claim an inference speedup.
