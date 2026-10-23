# QSA verifier key-reuse audit

`Verifier::run` appends each proposed row's indexer key before scoring. For
ordinary windows, it then calls `qsa_block_scores` once with all queries in
the group; batch recurrence instead gives each row its own `QsaState` and
scores separately. The scorer's default `STRATA_SCORES_MULTI=1` path already
loads each pooled key block once for up to eight queries. The same call then
selects blocks, resolves streamed K/V and runs attention. A proposed
multi-query shared-key kernel would duplicate this existing dispatch.

On October 24, 2026, `qsa_select_bench` ran on the RTX 2060 SUPER with
synthetic keys/queries, 32,768-cell context, 262,144-cell capacity and 50
repetitions per leg. `STRATA_SCORES_HALVE=0` isolated the current multi scorer
from its separate halving option. Times are score-kernel calls only:

| Queries | Per-query scorer | Shared-key scorer | Ratio | Score hash |
| ---: | ---: | ---: | ---: | --- |
| 2 | 60.7 us | 35.8 us | 1.70x | equal |
| 4 | 119.4 us | 65.8 us | 1.81x | equal |
| 8 | 236.7 / 236.7 us | 126.2 / 126.1 us | 1.88x | equal |

The eight-query legs were run ABBA. This is a confirmation of an existing
default, not a new inference speedup. Across verification windows, indexer
keys are versioned by the per-layer append: only the prefix before the
newly appended cells could be reused. A cache would need the layer, state
owner, block count and append/version boundary, with invalidation on rollback
or a different request. It would also need to beat the current register/shared
reuse and avoid extra resident VRAM or host staging. No production trace or
full-round timing establishes that it would.
