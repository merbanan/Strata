# CUDA graph memory investigation

RTX 2060 SUPER, CUDA 12.4, IQ3_XXS, 2026-10-22. Diagnostic measurements on the
installed snapshot release `6e4f8498`; no engine memory policy adopted.

`trace-graph-memory.cpp` records graph lifetimes, top-level node counts,
successful host submissions and device free-memory observations. It adds no
GPU synchronization. Enable with `STRATA_GRAPH_MEMORY_LOG=path` or `-` for
stderr, and load the compiled interposer with `LD_PRELOAD`. Example build:

```sh
c++ -O2 -std=c++17 -Wall -Wextra -Werror -shared -fPIC \
  bench/volta-turing/trace-graph-memory.cpp -o /tmp/trace-graph-memory.so -ldl -pthread
python3 bench/volta-turing/analyze-graph-memory.py engine.log --output summary.json
```

The interposer's GPU control passes 624 exact-output cases both disabled and
enabled, recording 78 graph lifetimes, peak two live, and eight submissions per
lifetime. All graphs are destroyed and all memory queries succeed. The model
runner retains all seven archived requests (warmup, four edit variants, fresh
code and prose). Warmup tokens match the reference and the engine exits cleanly.
The diagnostic does not qualify task quality or measure serving throughput.

## Model observations

There are 21 executable lifetimes, peak 21 simultaneously retained, and 8,544
successful graph submissions. All are destroyed; no recorded API or metadata
errors. The five large verifier graphs are:

| Window tokens | Top-level nodes | Submissions | Free VRAM decrease during instantiate |
| ---: | ---: | ---: | ---: |
| 3 | 4,307 | 1,143 | 30 MiB |
| 1 | 2,300 | 79 | 16 MiB |
| 16 | 4,440 | 593 | 34 MiB |
| 2 | 4,300 | 54 | 32 MiB |
| 8 | 4,346 | 28 | 32 MiB |

These observations sum to 144 MiB. The remaining 16 graphs have 5–70 nodes;
three instantiations coincide with another 2 MiB each and the others with zero.
Every destruction shows zero immediate increase in device free memory. Before
the large graph destructions, only 25.125 MiB is free. Later buffer teardown
releases memory, but this cannot be attributed to individual graphs.

Free-memory deltas are not exact graph ownership: pooling, queued operations
and other users can affect them. Host submission success does not independently
prove GPU completion. Node counts do not flatten child graphs. Exit snapshots
can precede global destructors and are not leak evidence. Batch graphs and
arena allocations still need separate coverage.

`Verifier::instantiate_evicting()` currently evicts batch graphs on allocation
failure. Solo verifier executables remain until destruction. An earlier limit
may reduce the high-water mark even if late destruction does not return memory
to ordinary allocations; this needs a controlled recreation experiment before
selecting an engine policy.

CUDA's graph-memory trimming API addresses memory reserved by allocation nodes;
it is not a documented general-purpose trim for executable metadata. See the
[CUDA graph memory documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#physical-memory-footprint).
The standalone probe therefore records those allocator counters separately
and tests destruction, synchronization and recreation before and after trim.

Artifacts: `Strata-data/bench/2026-10-21-volta-turing/graph-memory-v1/` contains
the frozen runner/configuration, interposer, original logs, source hashes,
probe summary and model summary. `graph-retention-v1/` holds the follow-up
standalone experiment. Neither runner changes the installed release.


## Standalone retention and recreation

`run-graph-retention.py OUT` builds `graph-retention-probe.cu` and runs fresh
processes for limits 1/2/3/5, twice in opposite limit order, with and without a
fixed allocation leaving 512 MiB headroom. Each process captures sequential
kernel graphs of 4,307/2,300/4,440/4,300/4,346 nodes, with 128-byte parameters.
These approximate node counts, not model graph topology or kernel behavior.
The deterministic workload checks 60 changed-input cases and 180 launches per
process against independent CPU integer sums. All 16 processes pass (960 cases,
2,880 launches). Each process destroys all graphs twice, recreates them in the
same context, and separately tries the graph-allocation-pool trim API.

| Maximum live executables | Peak decrease in free VRAM | Creations per process |
| ---: | ---: | ---: |
| 1 | 28 MiB | 60 |
| 2 | 60 MiB | 60 |
| 3 | 90 MiB | 51 |
| 5 | 136 MiB | 15 |

Every repetition, headroom condition and recreation phase gives these memory
values. Destruction plus device synchronization does not increase free VRAM;
recreation does not grow the established high-water mark. Graph-allocation
pool counters stay at zero, and trimming that pool changes nothing. This
supports trying an early cap: the two-graph case saves 76 MiB in this synthetic
experiment, but does more captures/instantiations. It does not establish an
engine memory saving, safe state recreation or a throughput improvement.

Evidence: `graph-retention-v1/matrix/run.json`, `summary.json` and per-process
JSON logs contain commands, hashes, phase measurements and exact-output checks.
The initial three-process trial is also retained. No sanitizer claim is made
for this standalone retention probe.


## Whole-graph update feasibility

The rebuilt default-path diagnostic confirms different top-level node counts
for every observed solo window: T=1/2/3/8/16 has 2300/4300/4307/4346/4440 nodes.
All 20 ordered pairs differ. CUDA 12.4 documents a topology-change failure
when the updating graph and executable have different top-level node counts.
Therefore direct `cudaGraphExecUpdate` replacement between these five existing
shapes cannot remove the recreation cost. This is an inference from measured
counts and the [CUDA 12.4 update contract](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART__GRAPH.html),
not a mutating experiment on the running model.

A common padded topology would need a separate redesign and state/padding
qualification. Same-size recaptures, batch layouts and individual-node updates
are not ruled out by this result. Evidence:
`solo-graph-limit-v1/graph-update-topology-audit.json`.

### Same-size update probe, 2026-10-24

`graph-same-size-update.cu` captures a 4,307-node sequential kernel graph on
the RTX 2060 SUPER, instantiates it, then captures an equal-node-count graph
with different kernel parameters. `cudaGraphExecUpdate` succeeds and the
updated executable passes two changed-input replays. A fresh executable of
the second graph also passes. The negative control has 4,306 nodes: the update
returns `cudaErrorGraphExecUpdateFailure`, and the original updated executable
still passes replay afterward. Each replay checks all 32 output lanes against
an independent host sum.

In this single unloaded-GPU run, initial instantiate took 18.49 ms, same-size
update 2.08 ms, and fresh instantiate of the second graph 19.98 ms. These
host times show API feasibility, not steady-state inference speed: this graph
contains simple kernels rather than verifier graph topology, pointers, CPU/GDN
state, KV state or batch layouts. A model recapture, state validation and
full-round timing are required before engine adoption.
