# Turing INT8 attention stage reuse

The existing opt-in `STRATA_PA_V2T=1` kernel reused its V tile and scales before
all lanes had completed their p.v reads. RTX 2060 SUPER racecheck reproduced
25 reported hazards (warnings, exit 1) at context 65, queries 33, with graph
replay and an empty selection. Numeric parity alone passed despite the race.

A warp barrier before writing the next stage fixes the single-stage Turing
path. The sm80 double-buffered path is unchanged. Arithmetic is unchanged;
this fix does not qualify V2T for default selection or establish model speedup.
The locally adopted K8V4 release uses its separate, already validated kernel.

Validation on RTX 2060 SUPER, CUDA 12.4, 2026-10-22:

- Racecheck: zero hazards, errors or warnings at context 4096, queries 33.
- Memcheck and synccheck: zero errors on the same graph/empty-selection case.
- Context/query cases 65/33, 1500/128 and 4096/256: exact output SHA-256 equality
  before/after for both default INT8 and opt-in V2T, checked separately. The
  existing V2T-versus-default arithmetic difference is retained.
- All-format suite passes INT8, FP16 and K8V4; pure Q4 skips sm75 as before.
- sm75 engine builds; serial sm70 and sm80 translation-unit compiles pass.
  HIP/SYCL and other GPU runtime checks were not available.

Evidence: `Strata-data/bench/2026-10-21-volta-turing/int8-v2t-race/` in the
surrounding workspace, including the original failing log, command, fixed
source, build logs, validation manifest, output files and hashes.

```sh
STRATA_PA_V2T=1 STRATA_PA_K8V4_STAGE=0 QA_GRAPH=1 QA_EMPTY=1 \
  compute-sanitizer --tool racecheck --error-exitcode 1 \
  Strata-v20/build/qsa_prompt_attn_parity 4096 33 1 1
```

Repeat with `--tool memcheck` and `--tool synccheck`. The parity executable now
accepts explicit format `1` (INT8) as well as `3` (K8V4), so the sanitizer can
exercise this path without unrelated formats. Omit the format to run all.
