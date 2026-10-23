# Compatible IQ2_XS signed-nibble layout screen

2026-10-23, RTX 2060 SUPER (sm75). This is a selective storage and kernel feasibility screen, not a full inference result or an adopted format.

The converter reads the original GGUF IQ2_XS blocks and writes a 144-byte aligned block: the same FP16 scale, the same eight subscales, six padding bytes, and 128 bytes containing two six-value signed selectors per byte. It checks all 65,536 grid/sign combinations and round-trips every one of the 384,000 blocks from 30 saved real gate/up experts across ten layers back to the original canonical 16-bit codes. The 30-expert gate/up bank grows from 28,416,000 to 55,296,000 bytes (+26,880,000 bytes). At 512 experts per layer this would add 458,752,000 bytes (437.5 MiB) for each IQ2_XS GU layer; ten such layers would add 4,375 MiB. Converting all 29 IQ2_XS layers would add about 12.34 GiB. The RTX 2060 SUPER has only 8 GiB of VRAM, so whole-model conversion is infeasible here.

The standalone row-dot reader preserves Strata's DP4A chain, integer local-scale formula, FP32 accumulation order and warp reduction. Outputs are bitwise identical for all nine tested shapes: 1, 3 or 30 experts times 1, 4 or 8 tokens, using saved real weights and synthetic Q8 activations. ABBA CUDA-event medians show 1.51–1.76x isolated row-dot speedup; for 30 experts and eight tokens, 3.058 ms falls to 1.732 ms. The [timing review](../../../Strata-data/bench/2026-10-21-volta-turing/iq2-nibble-v2/review.json) records all raw ranges, and [the CPU converter report](../../../Strata-data/bench/2026-10-21-volta-turing/iq2-nibble-v2/report.json) records source hashes and per-layer checksums. Memcheck, racecheck and synccheck pass the one-expert T=1/8 screen; sm70 and sm75 builds have no spills. There is no V100 runtime result.

The reader is not wired into `native_expert_grouped`, cache slots, host streaming or prefill MMQ. In particular, the deployed expert cache uses raw packed GGUF blob offsets, and replacing one GU part changes the blob stride and cache capacity. The standalone benchmark keeps both banks resident, which excludes conversion, extra PCIe traffic, eviction, SwiGLU/Q8, down projection and model output. These limits prevent an inference speed or adoption claim. Keep the implementation opt-in if integration is later attempted, convert only selected hot experts, and require an allocation-matched full-model ABBA before enabling it locally.

## Pinned-host cold copy and compute screen

On October 24, the same saved 30-expert banks were timed on the RTX 2060
SUPER with `--cold`. Each arm copies its gate/up bank from pinned host memory
to its GPU buffer and immediately runs the row-dot kernel on the same stream.
Both arms use the same Q8 activations. Two ABBA runs covered 1/3/30 experts
and 1/4/8 tokens; every output comparison was bitwise equal. Values below
are medians in microseconds across both runs' eight samples per arm/shape.

| Experts | Tokens | Original warm | Nibble warm | Original copy + dot | Nibble copy + dot |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 18.2 | 12.3 | 166.3 | 297.9 |
| 1 | 4 | 57.3 | 33.9 | 205.2 | 321.5 |
| 1 | 8 | 109.1 | 63.3 | 258.3 | 345.6 |
| 3 | 1 | 44.2 | 27.1 | 477.9 | 863.2 |
| 3 | 4 | 161.4 | 94.8 | 595.9 | 930.3 |
| 3 | 8 | 312.7 | 180.3 | 747.5 | 1017.1 |
| 30 | 1 | 390.9 | 222.4 | 4660.0 | 8512.5 |
| 30 | 4 | 1552.7 | 884.3 | 5824.2 | 9178.9 |
| 30 | 8 | 2413.5 | 1339.6 | 6650.4 | 9643.5 |

The converted bank copies 1.95 times as many bytes. For one expert, its
copy-plus-dot call is 1.34–1.79 times slower, depending on token count. If
the copy is paid once and each subsequent call hits a resident converted
expert, the isolated timings imply roughly 23 additional one-token uses, or
two additional eight-token uses, before the one-expert conversion breaks
even. This is a serial transfer screen; production can overlap DMA with other
compute, and neither cache displacement nor conversion CPU time is included.
It rules out promoting this layout for ordinary cold misses on this GPU;
profiling actual expert reuse is the prerequisite for a selective hot cache.
Raw runs: `/run/media/benjamin/TMP/strata-codex-20261023/iq2-nibble-cold-screen.csv`
and `iq2-nibble-cold-screen-repeat.csv` in the same directory.
