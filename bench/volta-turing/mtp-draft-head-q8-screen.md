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
