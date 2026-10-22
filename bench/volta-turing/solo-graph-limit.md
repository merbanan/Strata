# Solo verifier graph limit

Experimental, RTX 2060 SUPER, CUDA 12.4, 2026-10-22. Not adopted.

`STRATA_VERIFY_SOLO_GRAPHS=N` limits retained solo verifier executables for a
non-pipelined single stage. The default `0` keeps the existing behavior. The
limit counts executables, not bytes; a resident and doorbell variant each count
as one. On a cache miss, the oldest graphs are destroyed before recording and
instantiating the new window. Existing graph hits update recency. State arenas,
commit graphs, batch graphs, cache slots, prompt chunk sizes and VRAM reserve
are outside this policy.

The motivation is the [retention experiment](graph-memory.md): an early limit
reduced metadata high-water memory in synthetic tests, while destruction did
not return it to globally free VRAM. No graph-allocation-pool trim is added.

`capture_all` pins all windows because the pipelined launch path requires them
to remain prepared. Layer splits, shared external compute streams and
always-publish mode bypass eviction. In-flight solo or batch windows refuse
eviction. Before destroying a graph the policy synchronizes its compute and
shared-expert streams, and propagates synchronization/destruction errors.

## Validation so far

The sm75 engine builds. `solo-graph-limit-probe.cpp` links the actual Verifier
implementation, using the compiler's `-fno-access-control` for test access.
It does not require production test hooks or implement another cache policy.
Example:

```sh
python3 bench/volta-turing/build-solo-graph-limit-probe.py BUILD /tmp/solo-probe
/tmp/solo-probe/probe
compute-sanitizer --tool memcheck --error-exitcode 97 /tmp/solo-probe/probe
compute-sanitizer --tool synccheck --error-exitcode 97 /tmp/solo-probe/probe
```

Plain execution, memcheck and synccheck pass with zero sanitizer errors. Each
run creates 37 graphs and checks 35 changed-input results. Checks cover
disabled behavior, cross-variant recency, pipeline bypasses, refusal while a
window is in flight, and repeated eviction/recreation over T=1..16. Synthetic
graphs write deterministic memory patterns; these checks do not establish
preservation of real model GDN/KV/CPU state or task quality.

The original seven-request workload completed in both arms with the same
rebuilt engine and lifecycle instrumentation. All 14 requests naturally stop,
and both engines exit cleanly. Warmup tokens match the adopted-release reference.

| Measurement | Limit 0 | Limit 2 |
| --- | ---: | ---: |
| Peak live solo verifier graphs | 5 | 2 |
| Verifier graph lifetimes | 5 | 174 |
| Evictions | 0 | 172 |
| Minimum free VRAM at verifier lifecycle observations | 25.125 MiB | 105.125 MiB |
| Expert cache slots | 1,403 | 1,403 |
| Prompt chunk tokens | 6,400 | 6,400 |
| Pinned expert RAM | 39.60 GiB | 39.60 GiB |
| Logged verifier device arena | 128.8 MiB | 128.8 MiB |

The observed minimum headroom improves by 80 MiB. All graph lifetimes end and
there are no recorded CUDA/metadata errors. The candidate recaptures every
window size: T=3/1/16/2/8 has 33/58/30/32/21 lifetimes, respectively. The
128.8 MiB arena is a separate persistent allocation in `Verifier::init`; the
limit does not resize or free it. Exact allocation ownership is not measured.

This is a single instrumented pair, not a throughput comparison. All six long
continuations differ; warmup alone is byte-identical. Existing asynchronous
adaptation and deferred CPU work are known possible causes of variation, but
that does not exonerate this change. Long-state correctness is still unproven.
Retrieval/first-logit ABBA completed: all 16 tasks pass at 1K/4K/16K/32K;
all raw first-logit files match the adopted-release reference byte for byte.
The full original-task control with adaptive swaps and deferred CPU work off
also completed all 28 requests. Strict continuation parity still fails,
including repeats within the baseline. Three long cases (original logging
edit, prose and instruction-first rename) have identical token sequences in
all four legs; acceptance still varies in the edit cases. The matched prose
pairs have identical acceptance and are 5.07%/5.14% slower in full-request wall
time with limit 2. This is a measured matched-output regression, not a claim
that the prose task is correct. The two-graph setting is not selected for local
adoption on this evidence.

A follow-up instrumented run with limit 3 completes all seven original requests
and destroys all graphs without recorded errors. Peak solo graphs are three,
with 53 lifetimes and 50 evictions. Minimum observed free VRAM is 75.125 MiB,
50 MiB above the baseline, at unchanged slots/chunks/pinned RAM/128.8 MiB arena.
This reduces recaptures relative to limit 2, but is not a speed measurement.
See `retention-comparison.json`.

A separate diagnostic binary now returns the existing `kShape` prior from
`DraftPolicy::cost_ms` for every size, removing measured timing from window
cost estimates. It keeps acceptance learning and the rest of the policy.
A 1,000-round CPU negative control uses equal acceptance histories and different
timing inputs: original policy changes 987 choices and one chain; the diagnostic
changes none. Its full original-task limit-0/2 ABBA completed all 28 requests
with natural stops and clean engine exits. All seven cases have identical
output tokens, text hashes, request hashes and non-timing DONE fields across
all four legs: 11,593 output tokens per leg. Each candidate leg captures 149
solo graphs and evicts 147, covering T=1/2/3/8/16; both baselines capture five.
Slots/chunks/pinned RAM match at 1403/6400/39.60 GiB. With deferred CPU work off,
the logged verifier arena is 127.2 MiB in both arms, and minimum logged free
VRAM is 33 versus 109 MiB (coarse capture observations).

This establishes observable continuation/work-count parity across recreation
for the tested workload with fixed draft costs, no adaptive swapping and no
deferred CPU work. It is not a complete internal-state comparison or a claim
of production-policy determinism. The diagnostic fixed-cost policy is not a
speed proposal. Source copies, link inputs, commands and binary hashes are
under `fixed-policy/`; `qualification.json` pins the complete audit. The
installed release is unchanged.

Independent review of these exact outputs retains all six long-task failures.
The original rename leaves the class unchanged and omits a method; both
instruction-first edits also omit a method; the original logging edit adds an
unrequested duplicate dictionary key. Fresh code fails the exact TTL deadline
and its supplied concurrency test assumes an entry cannot have been evicted.
The prose misstates fast recovery. Reviews/diffs are under `fixed-policy/reviews/`;
they apply only to matching output hashes. These diagnostic timings cannot be
reported as successful-task throughput.

A full cap-0/3 ABBA started after that diagnostic under `cap3-timing/`.
It restores the original adaptive swapping, deferred CPU work and learned
window costs in both arms. It uses the archived production candidate, not the
fixed-cost diagnostic binary. The only arm difference is the graph count limit;
all seven tasks and four legs are retained. Configuration preflight passes;
input hashes and predecessor process identity are saved. The queue checks that
the preceding diagnostic completed all 28 requests and that the GPU is idle
before starting. It retains a failed parity result for review rather than
promoting it to a correctness pass.

Production-config timing results and model-state/quality qualification remain
necessary before adoption. HIP, SYCL, V100 and multi-GPU runtime coverage have
not been performed.

Evidence: `Strata-data/bench/2026-10-21-volta-turing/solo-graph-limit-v1/`.
`engine/manifest.json` pins the initial source patch and rebuilt executable;
`policy-validation.json` records all three probe runs. The installed release
remains snapshot `6e4f8498`.


The first controlled queue was stopped while waiting, before launching any
model: the old comparison runner still assumed `STRATA_ROUTE_Q8_FUSE=0/1`.
The runner now accepts an explicit feature and values, and has a device-free
configuration preflight. Graph-limit 0/2 and original router 0/1 configurations
pass; mismatched VRAM budgets and instrumentation are rejected. The corrected
queue passed preflight before waiting for retrieval qualification. Original
queued inputs and cancellation status are retained for traceability.


Independent review of the diagnostic outputs retains the task failures. Every
edit response in both arms fails the requested-only AST transformation. Both
fresh-code implementations pass basic LRU/get/put/delete, per-entry TTL/update
and a finite concurrency check, but keep an entry accessible at its exact TTL
deadline. Expired entries can also evict a live entry; that preference was not
specified, so it is recorded as a limitation rather than an additional gate.
The prose responses have congestion-control inaccuracies: baseline slow-start
ACK growth and candidate fast-recovery growth. The review uses
[RFC 5681 sections 3.1–3.2](https://www.rfc-editor.org/rfc/rfc5681.html#section-3.1).
Review records are pinned to each output hash under `diagnostic-reviews/`.
None of these diagnostic timings qualifies as a successful-task speedup.


One timing-accounting lead remains: in `generate.cpp`, `graph_fits()` calls
`Verifier::prepare` before `round0`, while `DraftPolicy::observe` receives the
interval starting at `round0`. Lookup graph recreation is therefore excluded
from its learned round cost. MTP misses captured inside `Verifier::run` occur
after that timer starts. With frequent eviction this asymmetry may affect
window selection; that performance effect has not been measured. Include it
when evaluating the bounded cache rather than changing the default policy.
