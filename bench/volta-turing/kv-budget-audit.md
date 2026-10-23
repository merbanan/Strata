# Draft and target KV budget audit

2026-10-24, source audit only. The model service remained stopped to preserve host RAM. This audit prepares TODO 8.2; it is not a performance comparison.

`--kv-resident` sets one target hot-cache capacity for every QSA layer. In `kv_plan` (`src/core/layer.cpp`), a positive value is clamped to at least 20,480 cells, rounded to pages, and becomes mode 1 (the CLOCK streamed cache) when smaller than context. Every target layer also gets a full pinned host copy. `kv_stream_counters` reports its lookups and misses.

The MTP draft has different behavior. `MtpDrafter::load` (`src/core/mtp.cpp`) derives its ring capacity from `--mtp-window` plus `4 * max_t + 64` cells. With streaming enabled, `kv_plan` selects mode 2: a modulo-addressed ring over a pinned host copy. On resume, `kv_ring_restore` copies the most recent ring range. It has no CLOCK hit/miss policy to tune independently, and the drafter attends only to its configured window. K8V4 applies to target layers; the draft state remains INT8 when K8V4 is selected.

These are already separate allocations but not two adjustable hot-cache budgets for the same lookup semantics. Reducing draft slots below the attention window would change what the drafter can read; changing `--mtp-window` would alter draft acceptance. Increasing draft slots beyond the window might affect restore traffic, but that needs a separate ring-range experiment and cannot be interpreted as a cache hit-rate comparison. Reassigning draft ring VRAM to target slots also changes expert-cache space unless the total allocation is fixed explicitly.

The serve log's target KV RAM estimate previously multiplied every miss by a fixed 4,224 bytes. The accounting patch uses `kv_block_bytes` for each layer's actual format (3,264 bytes per K8V4 page here) and reports cumulative draft ring restore calls and payload bytes separately. The latter counts only explicit `kv_restore` refills, not the draft's ordinary host KV writes. These counters prepare a comparison; they do not alter allocation or inference dispatch.

Verification: a one-job CUDA build of the `strata` executable completed on RTX 2060 SUPER using the build and temporary directories under `/run/media/benjamin/TMP/strata-codex-20261023`. No model was loaded, so this verifies compilation and accounting paths by inspection, not the values of a live request.

The next useful experiment is to retain one draft window and one target format, compare target `--kv-resident` values with the expert cache held to the same total VRAM budget, and record target KV misses and bytes, draft restores, expert slots and misses, accepted tokens, output quality, and prefill/decode time. A true independent draft hot-cache budget would first require a new draft residency policy with a defined way to access its full window. No allocation or policy change is adopted from this audit.
