# KV host allocation and staged-prefill ordering audit

Source review on 2026-10-24, v20. This is a CUDA code-path audit, not a runtime speed or race claim.

| Area | Source path and finding |
| --- | --- |
| Host allocation | `qsa_state_init` uses `cudaHostAllocMapped | cudaHostAllocPortable` for each streamed/ring layer's authoritative KV and obtains its device address with `cudaHostGetDevicePointer`. An allocation failure aborts session initialization; there is no pageable fallback. The host KV is intentionally left uninitialized. |
| New sequence | `qsa_state_zero` clears resident VRAM and resets the streamed residency map on the caller's stream. It does not clear the authoritative host KV. The four-format poisoned-host CUDA parity test covers eviction, partial pages, reuse and changed-input graph replay; see `kv-host-zeroing.md`. |
| Serial staging | `kv_stage_from_host` queues each format's contiguous encoded runs with `cudaMemcpyAsync` on the compute stream. Attention and appends on that stream follow the copy. It stages through the existing pool and identity table; there is no second KV pool. |
| Opt-in prefetch | `STRATA_KV_PREFETCH` defaults off. Before overwriting the staging pool, `kv_prefetch_after` records `kv_released` on the compute stream after the previous layer's last attention/output work. The nonblocking copy stream waits for it, stages the next layer's immutable prefix, and records `kv_ready`. The compute stream waits for `kv_ready` before the next layer uses the pool. The pending ordinal prevents a second prefetch into the same pool. |
| Drain and reuse | `run_impl` drains a pending copy on early return; normal completion, `idle`, `relayout`, and `Prefill::release` synchronize the copy stream before borrowed storage can be rebound or freed. |
| Host-write route | On CUDA, `STRATA_KV_HOST_DMA` is disabled at compile time; appends write the host copy directly on the compute stream. The opt-in HIP route queues unstage DMA on the same stream after append. K8V4's folded K/V path continues direct host writes even if the HIP variable is set. |

No missing CUDA staging fence was found in this source path. This review does not verify overlap or profitability on the RTX 2060 SUPER, and it does not exercise early returns, relayout, or HIP at runtime. Keep prefetch disabled until a same-policy full-model comparison passes output and memory checks.

`session_release` currently releases the registered RoPE table but not the `QsaState` array or its pinned `host_step`, `host_pos`, and KV allocations. In the current engine these are created during startup and persist until process exit; stopping the service releases them. Any future in-process session reconstruction needs explicit ownership and cleanup before it can safely reuse this path. This is a lifetime limitation, not evidence of growing memory use across requests in the current engine.
