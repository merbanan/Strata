# KV replacement policy replay screen

2026-10-24, RTX 2060 SUPER. The model service remained stopped. The K8V4 `kv_stream_parity` harness wrote 306 GPU-selected page batches through the test-only `STRATA_KV_SELECTION_TRACE` option; all parity and graph-replay checks passed. The trace is from the parity harness's synthetic indexer-shaped selections, not real inference. The JSONL trace is at `/run/media/benjamin/TMP/strata-codex-20261023/kv-policy-k8v4.jsonl` and is deliberately outside git.

`kv-policy-replay.py` streams a trace through three CPU policies at the harness's 4,828-slot capacity: a second-chance CLOCK approximation, ideal global LRU, and a **simplified** 16-probe/64-hand LRU screen inspired by upstream PR #1242. It protects every page in the current selection from eviction. The probe model does not reproduce upstream Triton scheduling or Strata GPU timing. Each K8V4 miss represents 3,264 host bytes.

| 160/306-call workload | GPU misses | CLOCK misses | Ideal LRU misses | Probe-LRU screen misses |
| --- | ---: | ---: | ---: | ---: |
| Parity trace, 306 calls | 70,502 | 70,472 | 70,234 | 73,888 |
| Shared 1,024-page prefix, four rotating contexts, 160 calls | — | 7,644 | 7,644 | 7,644 |
| Four disjoint rotating contexts, 160 calls | — | 53,520 | 53,520 | 51,452 |

The CLOCK approximation differs from the measured GPU count by 30 misses (0.043%), so its aggregate prediction is close on this trace. Ideal LRU saves 238 misses (0.34%) there; the probe screen increases misses by 3,416 (4.85%). On the constructed shared-prefix pattern all three policies tie. On the constructed disjoint pattern the probe screen saves 2,068 misses (3.86%) versus CLOCK. Synthetic cases are policy stress tests, not measured model behavior. Miss counts alone omit resolver time, PCIe overlap, expert displacement and accepted-token throughput.

This does not justify adopting probe-LRU. The next gate is a real streamed-decode page-selection trace for shared-prefix reuse and disjoint-context churn, followed by a GPU policy prototype with identical outputs and whole-inference timing. Keep the existing CLOCK policy and K8V4 storage.

## Real prompt selections, still not streamed decode

The existing `STRATA_QSA_DUMP` hook captured the last 2,047 queries of a real
32,768-token IQ3_XXS/K8V4 prompt on RTX 2060 SUPER, QSA layer 0. The dump
includes step widths and positions; `kv-policy-replay.py --qsa-dump` validates
every record and converts selected cells to four-cell page IDs. It found
positions 30,720–32,766, 513–514 unique pages/query, and 7,762 distinct pages
over the trace. Consecutive queries have mean page-set Jaccard overlap 0.545.
The binary dump SHA-256 is
`66a7d4fe2a3605820a46680533c465c45edeacf8210154627066b5f0411e23b7`.

| Hypothetical sequential-query replay, 5,120 slots | Misses | Host bytes if K8V4 |
| --- | ---: | ---: |
| CLOCK approximation | 25,553 | 83,404,992 |
| Ideal global LRU | 23,848 | 77,839,872 |
| Simplified 16-probe/64-hand screen | 49,798 | 162,540,672 |

The LRU oracle saves 1,705 simulated misses (6.7%) over CLOCK, while the probe
screen adds 24,245 (94.9%). These are **what-if** results on real model prompt
selections. The prompt path stages KV and does not call `kv_stream_resolve` for
each query, so there is no GPU miss count to validate against and no inference
speed claim. In particular, this is not the requested real streamed decode
trace or a shared-prefix/disjoint-context request screen. The raw request,
config, 16.8 MB dump, parser report, command and engine log are under
`/run/media/benjamin/TMP/strata-v20-kv-policy-real/prompt32-20261024a/`.
The process exited and released the model's pinned RAM and GPU allocation.

The next gate remains actual decode resolver selections under shared-prefix
reuse and disjoint-context churn. The synthetic and prompt screens do not
justify replacing CLOCK with the simplified probe policy.

Reproduce the CPU screens with `python3 bench/volta-turing/kv-policy-replay.py --selftest`, `--synthetic shared-prefix`, `--synthetic disjoint-context`, or the JSONL trace path. Replay the real prompt dump with `--qsa-dump PATH --slots 5120 --block-bytes 3264`. The GPU selected-page trace option is only in `kv_stream_parity` and writes to the supplied file; use a separate file per format.
