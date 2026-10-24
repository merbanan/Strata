# Draft and target KV budget audit

2026-10-24, source audit and RTX 2060 SUPER 32K full-model budget comparison. The service was stopped after the benchmark.

`--kv-resident` sets one target hot-cache capacity for every QSA layer. In `kv_plan` (`src/core/layer.cpp`), a positive value is clamped to at least 20,480 cells, rounded to pages, and becomes mode 1 (the CLOCK streamed cache) when smaller than context. Every target layer also gets a full pinned host copy. `kv_stream_counters` reports its lookups and misses.

The MTP draft has different behavior. `MtpDrafter::load` (`src/core/mtp.cpp`) derives its ring capacity from `--mtp-window` plus `4 * max_t + 64` cells. With streaming enabled, `kv_plan` selects mode 2: a modulo-addressed ring over a pinned host copy. On resume, `kv_ring_restore` copies the most recent ring range. It has no CLOCK hit/miss policy to tune independently, and the drafter attends only to its configured window. K8V4 applies to target layers; the draft state remains INT8 when K8V4 is selected.

These are already separate allocations but not two adjustable hot-cache budgets for the same lookup semantics. Reducing draft slots below the attention window would change what the drafter can read; changing `--mtp-window` would alter draft acceptance. Increasing draft slots beyond the window might affect restore traffic, but that needs a separate ring-range experiment and cannot be interpreted as a cache hit-rate comparison. Reassigning draft ring VRAM to target slots also changes expert-cache space unless the total allocation is fixed explicitly.

The serve log's target KV RAM estimate previously multiplied every miss by a fixed 4,224 bytes. The accounting patch uses `kv_block_bytes` for each layer's actual format (3,264 bytes per K8V4 page here) and reports draft ring restore calls and payload bytes separately. The latter counts only explicit `kv_restore` refills, not the draft's ordinary host KV writes. Counters reset with request state; the old source comment saying they were process cumulative was corrected. No draft ring restores were logged in the measured 32K requests. These counters do not alter allocation or inference dispatch.

Verification: a one-job CUDA build of the `strata` executable completed on RTX 2060 SUPER using the build and temporary directories under `/run/media/benjamin/TMP/strata-codex-20261023`. No model was loaded, so this verifies compilation and accounting paths by inspection, not the values of a live request.

## Full-model budget screen

The raw ABBA run is `/run/media/benjamin/TMP/strata-v20-kv-budget/full32-20261024a` (`summary.json`, per-leg `results.json` and `engine.log`). It uses the installed IQ3_XXS/K8V4 pack, 32,768 prompt tokens, 256 generated tokens, one draft window and fixed 1,024 MiB VRAM reserve, 5,120 prefill chunk, 40 GiB host resident budget and all other switches. The only config difference is `--kv-resident 20480` (base) versus `32768` (candidate). Requests match byte-for-byte across legs and have zero prefix reuse.

| Leg | Target slots per QSA layer | Expert slots | Host target KV read | Expert RAM blobs | Drafts offered/accepted | Prefill tok/s | Decode tok/s | Prompt-room loan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base 1 | 20,480 | 1,137 | 150.5 MiB | 100,557 | 172/134 | 520.7 | 27.23 | 13.44 s |
| Candidate 1 | 32,768 | 1,071 | 152.2 MiB | 107,524 | 198/137 | 461.3 | 25.64 | 1.28 s |
| Candidate 2 | 32,768 | 1,071 | 150.0 MiB | 105,532 | 186/136 | 459.6 | 25.52 | 1.53 s |
| Base 2 | 20,480 | 1,137 | 150.5 MiB | 100,557 | 172/134 | 635.5 | 26.57 | 2.11 s |

Target KV hit rates were 97.44%/97.60%/97.55%/97.44%. The larger target budget displaced 66 expert slots and did not reduce host target KV bytes. Base results repeated text and draft work exactly; candidate results did not, and cross-arm text differs. Thus the paired 0.886x/0.723x prefill ratios are a regression screen, not a precisely controlled kernel comparison; the first baseline's 13.44-second prompt-room loan especially contaminates its prefill time. Candidate decode ratios of 0.942x/0.961x are not equal-work comparisons. The second pair had similar 1.53/2.11-second loans and still showed a large prefill regression. Do not increase the installed target budget to 32,768 on this workload.

A 64K or larger model workload could produce a different miss curve and would need its own matched test. A true independent draft hot-cache budget would first require a new draft residency policy with a defined way to access its full window. No allocation or policy change is adopted from this screen.
