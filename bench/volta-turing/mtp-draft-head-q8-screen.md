# MTP draft-head Q8 output-row screen

The October 24, 2026 RTX 2060 SUPER screen used the production `native_mmvq`
call on a synthetic Q8_0 matrix shaped like the 40,525-token draft head:
2,560 inputs and 40,525 outputs. The benchmark now accepts `--shape MTP`
and `--working-set-mib 128`, so it allocates one 110.23 MB matrix and leaves
the full model unloaded. Each leg used 30 GPU-event-timed calls per shape.

`STRATA_NO_MMVQ_ROWS2=0` keeps the existing two-output-row mapping;
`=1` forces one row per block. The first four sizes ran ABBA:

| Columns | Two rows, us | One row, us | Result |
| ---: | ---: | ---: | --- |
| 1 | 352.8 / 352.5 | 352.7 / 352.6 | tied |
| 2 | 446.5 / 446.3 | 455.9 / 455.7 | one row 2.1% slower |
| 4 | 664.6 / 664.4 | 678.0 / 677.8 | one row 2.0% slower |
| 6 | 885.9 / 872.4 | 892.4 / 889.4 | one row slower, noisy |
| 8 | 1110.0 | 1125.2 | one row 1.4% slower |

This rejects the existing one-row override as a Turing draft-head speedup.
It does not test a new grouped-warp layout, real draft activations, sampler
time or complete draft/verify rounds. No feature is enabled from this result.

A second ABBA screen tried an opt-in four-output-row mapping in the exact Q8
multi-column kernel, with the same one-matrix working set and 30 timed calls.
`--hash` read back every output and computed a float-bit FNV hash after timing.
For each column count, the four-row and existing two-row hashes matched.

| Columns | Two rows, us | Four rows, us | Result |
| ---: | ---: | ---: | --- |
| 1 | 352.8 / 352.7 | 352.7 / 352.7 | tied |
| 2 | 446.6 / 446.5 | 441.7 / 441.9 | four rows 1.1% faster |
| 4 | 664.6 / 664.6 | 689.3 / 689.6 | four rows 3.7% slower |
| 6 | 878.9 / 877.9 | 929.7 / 916.2 | four rows 4.4–5.9% slower |
| 8 | 1078.4 / 1078.5 | 1208.6 / 1083.4 | unstable, no useful gain |

The four-row production switch was removed after this screen. The small
two-column kernel gain does not offset the losses at common wider calls, and
complete draft/verify timing was not run. The benchmark's optional `--hash`
check remains available for subsequent exact-output experiments.
