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
were not measured in this screen. The complete-model screen below closes that
local opportunity check.

## Complete-model opportunity screen (October 24)

The earlier direct-I/O measurement is deliberately cold and uncached. Real-model
prefill traces show much smaller exposed PLE work. On the same IQ3_XXS pack,
the Q4_0 prefill GPU timer attributed 67/8,028 ms at 4K, 276/29,845 ms at 16K,
555/57,444 ms at 32K, 1,147/118,483 ms at 64K, and 2,278/241,349 ms at
131K to PLE: 0.8–1.0% of the GPU timeline. The installed K8V4 profile at
4K attributed 67/6,687 ms (1.0%) to PLE, with about 0 ms of exposed host
PLE wait. These are phase profiles, not speed A/B tests. The raw Q4_0 traces
are in `/run/media/benjamin/TMP/strata-v20-select-route/profile-*`; the K8V4
trace is in `/run/media/benjamin/TMP/strata-v20-ple-profile/serve-profile-decode-only-20261024a`.

For decode, a verifier counter now separates direct PLE table gathering from
the existing host-stage total. Opt-in `STRATA_VERIFY_PROFILE=1` also brackets
the layer-1 PLE GPU stage around both token groups; `STRATA_DECODE_TIMING=1`
drains prompt-tail samples before decode. A real 4,096-token prompt followed
by 64 generated tokens completed in 31 verifier windows on RTX 2060 SUPER,
using the installed IQ3_XXS/K8V4 config, 20,480-cell target residence,
1,024 MiB VRAM reserve, and direct PLE I/O. The final instrumented engine log
reports 0.371 ms/window summed over both PLE token groups, or 11.50 ms across
the decode request. Host PLE gathering totals 0.329 ms. The instrumented
decode took 2,167.5 ms, so even eliminating the measured PLE GPU stage plus
host gather would save at most about 0.55% of that measured time before
accounting for overlap. The GPU stage sum includes its PLE-adjacent writes and
profile stamps, so it is an upper bound on the replaceable PLE kernels, not
their exact standalone runtime. The profiler disables some verifier fusions;
its request time is not a serving-speed baseline.

The direct-engine 4K/64-token run without GPU markers took 2,117.1 ms for
decode and attributed 0.088 ms/round of 35 rounds to host PLE gathers (about
3.1 ms total). The service-protocol profile has a different amount of cached
table work and yielded 0.329 ms; both are small against full decode. Three
service-protocol profiles produced the same 64-token output hash
`deea630e7a7775ea` and the same 33/42 accepted/offered drafts; the direct
engine's tokens differ later, so its timing is not paired as equal work with
the service run. The final decode-only log separates four prompt-tail verifier
windows from all 31 decode windows. Raw direct and service logs, exact commands,
binary hashes and exit codes are under
`/run/media/benjamin/TMP/strata-v20-ple-profile/`.

This local screen gives no reason to optimize direct PLE I/O or PLE GPU
compute ahead of the larger measured decode and prefill phases. Another drive,
model or PLE cache policy needs its own trace before changing that priority.
