# QSA scoring route on RTX 2060 SUPER

The production prompt selector in `src/prefill/prefill.cpp` first calls `qsa_block_scores_tc`; it falls back to `qsa_block_scores` when that call declines admission or `STRATA_SELECT_OLD` is set. On pre-sm80 CUDA, `qsa_block_scores_tc` admits its FP32 tiled SIMT scorer only with `STRATA_SELECT_SIMT=1/2`; the installed test configuration uses `STRATA_SELECT_SIMT=1` and `STRATA_IDX_F16=1`. The function's FP16-key branch uses the compatible SIMT reader. This is distinct from the separate `native_qsa_score` API, which has no production call site in this tree.

`STRATA_SELECT_ROUTE_TRACE=1` logs each prompt selection batch after the dispatch decision. A direct-engine 4,096-token English prompt, followed by a one-token verifier tail, ran on the RTX 2060 SUPER. Its engine log records 192 prefill scoring batches, all routed through `qsa_block_scores_tc`: 12 QSA layers, 16 batches of 256 queries per layer, active-block counts 65–1,025. Coverage records exactly 4,096 batched-prompt tokens and one verifier-window token. The opt-in trace is off by default and is not used for performance timing. Raw manifest, configuration, coverage and engine log are under `/run/media/benjamin/TMP/strata-v20-select-route/trace-20261024a/`.

Decode calls `qsa_block_scores` with one query from `src/core/layer.cpp`. Verification calls it from `src/core/verify.cpp`, using either one query per slot or a shared-state query batch. Neither path calls the prefill `qsa_block_scores_tc` admission function. The real prompt route is therefore already the pre-sm80 tiled SIMT scorer on this RTX 2060 SUPER configuration. A cuBLAS comparison under TODO 12.2 should target this route and material prompt shapes, including the existing exact top-k and causal tail work.

## Scorer opportunity screen

The existing `STRATA_PREFILL_TIMING=1` phase timer profiled real single-request Q4_0 prefill at the same 131,072-token configured capacity. The `qsa select` phase contains both scoring and exact top-k, so its share is an upper bound on any scorer-only replacement. These are instrumented phase measurements, not speed A/B results:

| Prompt tokens | GPU prefill timeline | QSA select | Share | QSA attention share |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 8,028 ms | 28 ms | 0.4% | 27.1% |
| 16,384 | 29,845 ms | 245 ms | 0.8% | 36.6% |
| 32,768 | 57,444 ms | 730 ms | 1.3% | 39.4% |
| 65,536 | 118,483 ms | 2,460 ms | 2.1% | 39.7% |
| 131,000 | 241,349 ms | 8,836 ms | 3.7% | 39.3% |

Even a zero-cost scorer and top-k would improve the 131K GPU timeline by at most about 3.8%; a cuBLAS implementation would still retain top-k and conversion/launch work. The 131K point is near this configuration's 131,072-token capacity. On this RTX 2060 SUPER model, the conditional cuBLAS comparison does not clear a material-opportunity gate. Reconsider for a different device or model only after a real phase profile shows a substantially larger selection share. Raw logs and coverage are under `/run/media/benjamin/TMP/strata-v20-select-route/profile-{20261024a,large-20261024a,131k-20261024a}/`. The failed `profile-128k-20261024a` attempt requested 131,074 prompt tokens plus one generated token and was rejected by the 131,072-token context bound before inference.
