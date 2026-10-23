# Turing chunked GDN shared-memory screen

The existing 32-token chunked GDN scan asks for 93,696 bytes of dynamic
shared memory per block and admits sm80+ only. Its standalone CUDA parity
source now has two `--turing-screen` modes on sm75. These are test-only
layouts; neither changes the production recurrence or enables
`STRATA_GDN_CHUNKED` in the engine.

- Mode 1 uses eight value columns per scan block and one K tile instead of
  the original double buffer. It stages the next K/Q only after the current
  K's final state update. Its dynamic shared memory is 55,296 bytes.
- Mode 2 keeps sixteen columns, uses one K tile, and reads Q directly from
  the input rather than a shared Q tile. Its dynamic shared memory is
  59,904 bytes.

Both scan instantiations use 168 registers per thread and have no local
stack/spill bytes in the sm75 object. The unchanged preparation kernel uses
64 registers and both modes allocate the same persistent 25.6 MB T/P/gamma
scratch as the original chunked path. The source compiles for sm70, sm75 and
sm80; runtime checks were on the RTX 2060 SUPER only.

Both modes pass output and final FP32 state comparisons to the current
`gdn_rec_kh_kernel` at 33, 128, 512 and 2,048 tokens. The worst tested
max-absolute-difference divided by max-absolute-reference is 6.47e-7 for
output and 5.59e-7 for state. At 128 tokens, the mode's FP64-reference
errors are 4.76e-7 output and 5.19e-7 state. This is numerical parity,
not bitwise equality or a speculative-commit check.

The recurrence timings below are one-layer CUDA-event medians of 20
alternating runs. They include both preparation and scan but exclude the
preceding input projections and following output norm.

| Tokens | Turing default columns | Key-head reference | Mode 1 | Mode 2 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.31 ms | 0.23 ms | 0.52 ms | 0.53 ms |
| 512 | 1.22 ms | 0.89 ms | 1.83 ms | 1.94 ms |
| 2,048 | 3.77 ms | 2.77 ms | 5.72 ms | 5.92 ms |

At 2,048 tokens, separate event medians attribute 0.711 ms to preparation
and 4.890 ms to mode 1 scan; mode 2 takes 0.711 and 5.157 ms. The scan
dominates and both layouts are 1.5–1.7 times slower than Turing's existing
column-split kernel.
Their extra CTA waves and the retained register-tiled products outweigh
the shared-memory reduction on this 34-SM card. Neither layout should be
integrated or enabled. The upstream #1228 fused TileLang design remains a
separate research lead; this screen does not reproduce its matrix-load or
P/A buffer schedule.

Raw outputs are under `/run/media/benjamin/TMP/strata-codex-20261023/` as
`gdn-turing-mode1.log` and `gdn-turing-mode2.log`.
