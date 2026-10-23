# KV replacement policy replay screen

2026-10-24, RTX 2060 SUPER. The model service remained stopped. The K8V4 `kv_stream_parity` harness wrote 306 GPU-selected page batches through the test-only `STRATA_KV_SELECTION_TRACE` option; all parity and graph-replay checks passed. The trace is from the parity harness's synthetic indexer-shaped selections, not real inference. The JSONL trace is at `/run/media/benjamin/TMP/strata-codex-20261023/kv-policy-k8v4.jsonl` and is deliberately outside git.

`kv-policy-replay.py` streams a trace through three CPU policies at the harness's 4,828-slot capacity: a second-chance CLOCK approximation, ideal global LRU, and a **simplified** 16-probe/64-hand LRU screen inspired by upstream PR #1242. It protects every page in the current selection from eviction. The probe model does not reproduce upstream Triton scheduling or Strata GPU timing. Each K8V4 miss represents 3,264 host bytes.

| 160/306-call workload | GPU misses | CLOCK misses | Ideal LRU misses | Probe-LRU screen misses |
| --- | ---: | ---: | ---: | ---: |
| Parity trace, 306 calls | 70,502 | 70,472 | 70,234 | 73,888 |
| Shared 1,024-page prefix, four rotating contexts, 160 calls | — | 7,644 | 7,644 | 7,644 |
| Four disjoint rotating contexts, 160 calls | — | 53,520 | 53,520 | 51,452 |

The CLOCK approximation differs from the measured GPU count by 30 misses (0.043%), so its aggregate prediction is close on this trace. Ideal LRU saves 238 misses (0.34%) there; the probe screen increases misses by 3,416 (4.85%). On the constructed shared-prefix pattern all three policies tie. On the constructed disjoint pattern the probe screen saves 2,068 misses (3.86%) versus CLOCK. Synthetic cases are policy stress tests, not measured model behavior. Miss counts alone omit resolver time, PCIe overlap, expert displacement and accepted-token throughput.

This does not justify adopting probe-LRU. The next gate is a real model page-selection trace for shared-prefix reuse and disjoint-context churn, followed by a GPU policy prototype with identical outputs and whole-inference timing. Keep the existing CLOCK policy and K8V4 storage.

Reproduce the CPU screens with `python3 bench/volta-turing/kv-policy-replay.py --selftest`, `--synthetic shared-prefix`, `--synthetic disjoint-context`, or the JSONL trace path. The GPU trace option is only in `kv_stream_parity` and writes to the supplied file; use a separate file per format.
