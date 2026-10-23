# PLE direct table I/O screen on the RTX 2060 SUPER host

The October 24, 2026 screen used the IQ3_XXS second GGUF shard (the real
27 GiB PLE table) and `ple_reader_test --direct-only`. Direct mode did not
keep a mapping of the table and disabled its row cache. The input was a fixed
seed of random row IDs, 16 per synthetic token. No model weights were loaded.

| Requests | Rows | Direct gather | SSD reads | Read latency p50 / p99 | Blocked |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1,600 | 208.4 us/token | 1,600 | 121 / 197 us | 20.130 ms |
| 1,000 | 16,000 | 209.7 us/token | 16,000 | 121 / 198 us | 202.845 ms |

The 1,000-token run reported 0% row-cache hits and 67.0 MB of SSD reads.
`fincore` found only 524,288 bytes of this shard resident after the run;
system available RAM remained about 60 GiB. This confirms the direct table
read's isolated cost on this host. It does not establish that table I/O is
the bottleneck in inference: layer work may overlap the read, and GPU PLE
compute, prefill phase timing, decode windows and request-level throughput
were not measured in this screen. Keep the full-model PLE timing gate open.
