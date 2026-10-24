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

The single paired screens of INT8, FP16 and Q4 also passed parity. Their one-miss baseline/candidate times were 11.605/11.329, 14.299/12.657 and 11.976/9.344 µs; these pairs are exploratory and cannot establish repeatability. The benchmark selects pages from a fully populated host copy and does not exercise concurrent expert transfers, cache displacement or the natural distribution of misses during inference. GPU clocks were not fixed.

## Full-model inference gate

The four-leg base/candidate/candidate/base run on 2026-10-24 used the IQ3_XXS model with K8V4, 131072 maximum context, 20480 resident KV cells, and identical settings except `STRATA_KV_COPY_RUNS=0/1`. Each leg warmed up, ran a 4096-token prompt and a 16384-token prompt, and generated 256 tokens per measured request. Raw results and engine logs are under `/run/media/benjamin/TMP/strata-v20-kv-copy-runs/full-20261024a/`.

All four legs had identical requests, output text, generated tokens, offered/accepted drafts, cache hit rates, RAM expert reads, and file reads for both prompt lengths. The 4K requests offered/accepted 192/132 drafts; 16K requests offered/accepted 177/132. Expert-cache allocation was 1403 slots in all legs, with 1185 prompt-lendable slots and a 39.60 GiB pinned RAM complement. No GPU compute process remained after the runner stopped.

| Prompt | Base A1/A2 prefill tok/s | Candidate B1/B2 prefill tok/s | Paired prefill ratio | Paired decode ratio |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 479.93 / 484.02 | 504.50 / 361.52 | 1.051x / 0.747x | 0.996x / 1.010x |
| 16384 | 728.18 / 728.72 | 726.77 / 728.31 | 0.998x / 0.999x | 0.982x / 1.015x |

The 4K candidate repeat spent 3839.8 ms lending prompt room, versus 616.1 ms in its first candidate leg and 1021.3/948.7 ms in the two baselines. Startup warned of less than 512 MiB free VRAM in every leg. Therefore the 4K prefill spread is dominated by variable setup, while the stable 16K result shows no full-model prefill gain. Decode has no repeatable gain. The isolated copy-call improvement did not translate into this inference workload. The workload does not report a direct distribution of cold KV misses, and broader task quality, concurrent PCIe stress, V100/T4 runtime, and 32K+ prompts remain unqualified. Keep the local option disabled; this is a completed RTX 2060 SUPER screen, not an adopted optimization.
