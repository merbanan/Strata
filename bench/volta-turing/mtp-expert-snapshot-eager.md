# Eager MTP snapshot priming: tested, not adopted

RTX 2060 SUPER / IQ3_XXS, 2026-10-22. The experiment correctly moved the
first pinned allocation before READY, but did not give a repeatable reduction
in total process-start-to-answer time. Local startup remains lazy. Snapshot
reuse from commit `6e4f8498` remains enabled.

The separate `STRATA_MTP_EXPERT_SNAPSHOT_EAGER=1` prototype calls
`experts_prime()` after memory budgets are selected, before READY and the
request watchdog. It requires snapshot reuse and an admitted prompt room.
Priming retains resident expert bytes and their addresses; it does not mark
the block lent. Allocation/copy failure retains lazy preparation. Unset/zero
does not prime. The source patch is preserved outside the active worktree in
`Strata-data/bench/2026-10-21-volta-turing/mtp-expert-snapshot-eager/`, including
base/candidate files, hashes, `proposal.patch`, build and sanitizer logs.

## Validation

The sm75 engine built. The actual MTP methods passed CUDA memcheck with reuse
unset, `0` and `1`, including priming/loan allocation and copy failures, retries,
resident/lent state, shared ownership, idle and live-chain guards, load
invalidation and five overwrite/reclaim cycles through a captured graph.
All checks restored exact bytes; memcheck reported zero errors.

`mtp-startup-compare.py` ran four fresh engine processes in ABBA order. Both
arms enabled K8V4 staging and snapshot reuse; only eager priming differed.
One qualified 4,096-token retrieval per process returned exactly the same 12
output token IDs and stopped naturally. Every process used 1,403 cache slots,
6,400-token chunks and 39.60 GiB pinned expert cache. Both eager processes
logged a successful 675 MiB snapshot before READY. No source or binary changed
during the run. The compared engine SHA-256 was
`da548449eef65d570b9449106604867c47ecc9cfce14bd4630732a63b7699b39`.

These are direct engine-protocol timings, excluding tokenizer preparation and
Python API startup. READY, first token and DONE were timestamped from the
protocol rather than a health-poll interval. Results and exact source/config
copies are in `mtp-expert-snapshot-eager-startup/` beside the experiment folder.

| Measurement (seconds) | Base 1 | Eager 1 | Eager 2 | Base 2 |
| --- | ---: | ---: | ---: | ---: |
| Process start to READY | 94.653 | 90.197 | 105.418 | 83.009 |
| READY to first token | 14.914 | 7.560 | 7.538 | 13.882 |
| READY to complete answer | 15.423 | 8.095 | 8.028 | 14.376 |
| Process start to complete answer | 110.076 | 98.292 | 113.446 | 97.385 |
| Startup priming | 0 | 0.917 | 13.688 | 0 |
| First-request loan | 7.400 | <0.0001 | <0.0001 | 6.280 |

Reclaim stayed at 106.0 ms. The existing 675 MiB snapshot is pinned earlier,
including for a service that never receives a room-eligible request. No extra
snapshot buffer is introduced, but that earlier host-memory use is a cost.

## Disposition

READY-relative first-token time fell 49.31% and 45.70%. Total start-to-answer
time improved 10.71% in one pair and regressed 16.49% in the other. Host
allocation and file-loading variability remain visible; moving the stall is
not evidence of faster inference computation. Telemetry began after the first
engine started loading, so early first-leg samples are missing.

Keep the tested prototype for a future explicit readiness-policy requirement,
but do not adopt it as an inference-speed improvement. The four eager-only
production/test edits were restored to `6e4f8498` after verifying their exact
archived hashes. The temporary build was subsequently reused for other experiments; the
installed immutable snapshot-reuse release is unchanged. No HIP, SYCL or
Volta runtime validation was performed for this prototype.
