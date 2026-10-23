# Streamed host-KV read-after-write check (RTX 2060 SUPER)

Strata already omits clearing the authoritative pinned host KV when a CUDA streamed state is allocated or reset. `qsa_state_zero` clears resident VRAM and calls `kv_stream_reset`, which invalidates the residency map; subsequent appends write host cells before a selected block is copied back. This check tests that ordering for the CUDA streamed decode path. It does not change production code or measure speed.

The standalone `kv_stream_parity` harness fills newly allocated host KV with `0xa5`, then compares streamed and resident attention bit for bit. Each format processes 40,000 cells with a smaller slot pool, 306 selected-query batches and forced evictions. It also restores a ring from the host copy, resets the map without clearing host memory, writes a fresh 133-cell sequence and attends the partial last block. Finally it captures reset, append, resolve and attention in one CUDA graph and replays it four times with changed K/V, query and selected position. The graph checks one selected cell per replay.

Built on 2026-10-23 with CUDA 12.4 for sm75, Release, `STRATA_BUILD_TESTS=ON`; only `kv_stream_parity` was built. Build and temporary files are under `/run/media/benjamin/TMP/strata-codex-20261023/strata-v20-kv-parity-build`. The model service remained stopped. Each invocation below ran as a separate process, so pinned allocations were released on exit:

```text
kv_stream_parity --format k8v4
  306 batches, 702021 lookups, 70488 misses, overflow 0, failures 0
  ring restore identical; reused host copy identical; graph replays 4/4 identical; PASS
kv_stream_parity --format fp16
  306 batches, 672800 lookups, 69602 misses, overflow 0, failures 0
  ring restore identical; reused host copy identical; graph replays 4/4 identical; PASS
kv_stream_parity --format int8
  306 batches, 703691 lookups, 71300 misses, overflow 0, failures 0
  ring restore identical; reused host copy identical; graph replays 4/4 identical; PASS
kv_stream_parity --format q4
  306 batches, 680672 lookups, 69183 misses, overflow 0, failures 0
  ring restore identical; reused host copy identical; graph replays 4/4 identical; PASS
```

This covers poisoned unwritten cells, a partial final page, reset/reuse, eviction, ring restoration and changed-input graph replay for the four CUDA storage formats on this GPU. It does not cover the HIP `STRATA_KV_HOST_DMA` prefill route, a full-model continuation, CUDA sanitizer runs, or V100/T4 runtime. Those remain separate checks. No inference speedup or new opt-in follows from this validation.
