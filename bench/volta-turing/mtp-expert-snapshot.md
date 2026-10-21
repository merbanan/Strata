# Reusing the MTP expert snapshot

Validated on RTX 2060 SUPER / IQ3_XXS, 2026-10-22. GPU lifecycle, full-model
quality and matched serving ABBA pass. Eligible for local opt-in adoption; the
global default remains off.

`STRATA_MTP_EXPERT_SNAPSHOT=1` retains the first successfully copied host
snapshot across prompt-room loans. Unset or zero still copies GPU to host on
every loan. No extra host or device allocation is introduced: the existing
pinned 675 MiB buffer was already retained until drafter destruction. The
first loan still allocates and copies it. Eager startup allocation is a separate
pending experiment, not part of this change.

## Writer and lifetime audit

`MtpDrafter::load` allocates and fills `experts_` from `experts.bin`. The shared
load path aliases the owner's weights and does not write them. `record_rest`
reads the weights for MoE. The public `experts_base()` is used only by
`generate.cpp` to give prompt-room storage to the prefill path. Lending is
rejected for `--batch`, whose drafters share the owner's weights.

The room overwrites device weights only between successful lend and reclaim.
Before reclaim, `generate.cpp` waits for the room's streams and disables it.
Reclaim copies the saved host bytes back to the same device address; captured
graph addresses never move. Decode entry points reject a lent owner. Failed
reclaim keeps it lent, allowing retry without pretending the weights are ready.

The host snapshot is written only by the loan's D2H copy and freed by the
owner's destructor. A new load attempt invalidates its identity, including a
failed load. A failed D2H never marks it valid, since part of the destination
may have changed. A snapshot-size mismatch fails before using it. The existing
ownership, idle and live-chain checks still run when a cached snapshot exists.
Future weight mutation/reload paths must invalidate the snapshot too; this
change does not introduce a general model-reload API.

## GPU validation

`mtp-snapshot-parity.cpp` calls the real MTP methods with eight expert blobs
(11,059,200 bytes), injects failures at CUDA calls through linker wrappers, and
replays a graph captured with the original expert address after each destructive
room use. The test alone compiles with `-fno-access-control`; production exposes
no new testing API. Tests cover allocation and D2H failure before first loan,
failed H2D restore/retry, idempotent calls, shared-owner rejection, live-chain
and idle guards, size mismatch, load-attempt invalidation, five repeated
loan/overwrite/reclaim cycles and byte equality through the captured graph.

CUDA memcheck passes with zero errors for flag unset, `0` and `1`. Default
runs make eight D2H calls, including injected failure and invalidation; the
cached run makes three, avoiding the five repeated copies. All restore bytes
match. This establishes lifecycle correctness for these cases, not model
quality or end-to-end speedup.

Rebuild the diagnostic with an existing CUDA Unix Makefiles engine build:

```sh
python Strata-v20/bench/volta-turing/build-snapshot-parity.py \
  /tmp/strata-v20-attention-build /tmp/mtp-snapshot-parity
STRATA_MTP_EXPERT_SNAPSHOT=1 compute-sanitizer --tool memcheck \
  --error-exitcode 1 /tmp/mtp-snapshot-parity
```

Repeat with the switch unset and set to `0`. This helper builds only the test;
it does not rebuild the engine or launch a GPU workload.

Evidence and exact build/link commands:
`Strata-data/bench/2026-10-21-volta-turing/mtp-expert-snapshot/`.
`manifest.json` records the compiled source and binary identity. Full-model
retrieval/first-logit ABBA also compares with the adopted K8V4-only release's
saved outputs. All 16 retrieval requests passed with natural termination;
all 248,320 first-token logits per request match byte-for-byte across the four
arms/repeats and adopted reference at 1K/4K/16K/32K (zero KL). These are
first-token/task checks, not broad model quality or full-continuation KL.
The dependent timing queue requires these checks before serving measurements.

`snapshot-compare.py BASE CANDIDATE OUT` runs ABBA with interleaved 2K/4K
retrieval requests. It saves first-loan timing separately, checks exact answers,
natural termination, cold prefixes, one loan/reclaim per request, unchanged
binaries and identical memory budgets. Later requests have unique prefixes and
run in the same order in each arm. It records both loan time and client TTFT/
complete-request latency, so a removed transfer alone cannot justify a model
speed claim. Quality instrumentation is excluded from these timing runs.

## Serving result

`mtp-expert-snapshot-timing-audit/` completed ABBA, six fresh retrieval requests
per leg, alternating 2K and 4K. All 24 answers are exact and terminate naturally.
The first loan in each leg is reported separately. Repeated-loan observations
per arm: four at 2K and six at 4K. Both paired comparisons improve TTFT and
complete-request latency:

| Repeated prompt | TTFT reduction, pair 1 / pair 2 | Complete-request reduction, pair 1 / pair 2 |
| --- | --- | --- |
| 2K | 99 / 141 ms (2.10% / 2.96%) | 85 / 103 ms (1.63% / 1.97%) |
| 4K | 82 / 120 ms (1.08% / 1.59%) | 98 / 134 ms (1.22% / 1.67%) |

Loan time falls from 107.1–107.2 ms to below the log's 0.1 ms precision.
Reclaim remains approximately 106 ms. Every leg retains 1,403 cache slots,
6,400-token chunks and 39.60 GiB pinned expert cache. The existing 675 MiB
snapshot allocation is unchanged. Both arms keep K8V4 staging enabled.
`adoption-audit.json` verifies full coverage, matching prompt hashes/configs,
cold prefixes, unchanged source/binary and both-pair latency improvements.

The first baseline allocation took 8,692.5 ms; this prototype does not eliminate
that allocation, and the first-loan variation is excluded from the speed claim.
No broad decode gain is claimed. Exact tested engine SHA-256:
`3a89e7337f70f6f778260e95800fc32bf17488ae341f04b8d19246de75448be2`.
The reproducible test-build helper also passes its cached-path memcheck run.

## Separate eager-allocation follow-up

A possible second switch can prime the snapshot after startup memory budgets
are chosen but before the engine emits READY (`generate.cpp`). That would move
the first pinned allocation out of the first request without changing graph
addresses. It must report added startup time and 675 MiB of pinned host bytes,
and preserve a working lazy fallback on allocation failure. Compare process
start-to-first-answer as well as ready-to-first-token; moving a stall alone is
not a total-latency gain. The existing lazy snapshot prototype does not perform
this priming. Any implementation must retain the same room/batch admission and
not mark the experts lent while the engine is advertised as ready.

The initial timing template placed a short marker at the end of the records.
Its baseline 4K response summarized the records and hit the 64-token cap, so
the harness failed before measuring any candidate. Preserve this failure under
`mtp-expert-snapshot-timing/`; it provides no successful-task speed result.
The replacement run, `mtp-expert-snapshot-timing-audit/`, uses the qualified
audit-marker template with early repeat identifiers to prevent prefix reuse.
It retains the same exact-answer and natural-stop gates.
