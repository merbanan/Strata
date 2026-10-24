# Expert dispatch opportunity from real verifier windows

2026-10-24, RTX 2060 SUPER, IQ3_XXS/K8V4, with `--spec-split`. This is a source-derived dispatch inventory, not a GPU activity or timing trace. Run:

```sh
python3 bench/volta-turing/expert-route-opportunity.py \
  ../Strata-data/packs/iq3_xxs/native_experts.txt \
  /run/media/benjamin/TMP/strata-v20-window-trace/full-20261024a/2-candidate/engine.log \
  --requests 3 4 --spec-split
```

The retained output is `/run/media/benjamin/TMP/strata-v20-window-trace/expert-route-opportunity.json`. Request 3 is the 4K/256-token continuation and request 4 the 16K/256-token continuation. The trace was hash-audited under [window tracing](window-trace.md), and the pack's `native_experts.txt` lists 48 layers in ten format pairs. The model has 9/10/6/13/10 gate/up layers of types 16/17/18/21/22 and 18 IQ4_NL (type 20) plus 30 Q2_0 (type 42) down layers.

| Request | Window widths | Split verifier group widths | Resident calls per GU or down projection | Resident down mode 4 / mode 0 | PCIe calls per GU or down projection |
| --- | --- | --- | ---: | ---: | ---: |
| 4K | 1:22, 2:17, 3:84, 8:1 | 1:140, 2:84, 4:2 | 10,848 | 10,752 / 96 | 10,848 |
| 16K | 1:25, 2:23, 3:77 | 1:148, 2:77 | 10,800 | 10,800 / 0 | 10,800 |

`Verifier::post` calls `native_expert_grouped` once for resident experts and once for the PCIe share per layer and verifier group. With `K=10`, a group of width 1/2/4 has capacity 10/20/40. On sm75, default mode 4 applies to the resident GU projection; the down projection switches to mode 0 only above 30 capacity slots. The PCIe call has `grid_groups=4`, smaller than its capacity, so its effective mode is 0. This is why a global `STRATA_EXP_MODE=4` changes the PCIe route too and cannot be inferred from the default resident-cell results. The one 8-token 4K window splits into two width-4 groups, yielding 96 resident down calls in mode 0, not 48.

The figures are *possible call sites* in captured verifier graphs. They do not count active expert groups, weight bytes, GPU execution cost, graph replay count or input quantization. A zero-group PCIe call can still launch its configured grid. The host window trace alone cannot tell which format pair incurred cache hits or misses, and its wall counters cover whole windows. These counts therefore cannot select an opt-in cell or establish a speedup. Next, collect GPU active-group/entry histograms by format, source tier and width on the same requests; measure complete GU + SwiGLU/Q8 + down graph calls with those real shapes and cold-expert traffic. Only then choose cells for the numerical and full-model gates in 6.4.
