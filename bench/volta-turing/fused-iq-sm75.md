# Native IQ prefill on Turing

Experimental port, prepared 2026-10-22. Not built, measured or adopted on the
current stack yet. Target: RTX 2060 SUPER, CUDA 12.4, IQ3_XXS.

The v14 change `540953d7ac292af48f5749473099b54bf55eada3` applies to the current
`src/prefill/moe_fused_iq.cu` without offsets or fuzz. A detached source copy and
patch are saved under
`Strata-data/bench/2026-10-21-volta-turing/fused-iq-sm75-v1/`. Production source
and the installed engine have not changed for this experiment.

The opt-in `STRATA_PF_FUSED_SM75_IQ=1`, alongside the existing fused/SM75
switches, admits the IQ gate/up and IQ4_NL down kernels on Turing. It sets up
two warps per row for every covered format and four where shared memory fits.
IQ2_S needs 67,840 bytes at four warps, beyond this card's 64 KiB limit, so a
layer containing it uses two. This does not enable a Volta implementation.

The old report (`2026-10-18-fusediq/TODO.md`) measured 1.2–1.67x isolated layer
speedups but no engine gain: 545.3 versus 549.1 prefill tokens/s. That comparison
also changed prompt chunks from 6656 to 6400 tokens. Its first-token KL of
0.35 against v13 is retained as an unresolved numerical result, not a quality
pass. Fused and MMQ activation quantization/accumulation differ; their outputs
are not expected to be byte-identical.

## Queued validation

Compilation and GPU work wait for the graph-cache full-model timing run to
finish. The queue pins its predecessor's PID/start time and requires all 28
requests plus an idle GPU. It does not stop or replace that run.

- Compile the detached native IQ object and the existing independent reference
  test, with an optional canonical pair-order output dump.
- Compare the existing Q2 path's output bytes against the original library with
  the new switch unset, zero and one. Grouping storage order is normalized before
  comparison.
- Require all 11 supported format pairs to run against the double-precision
  reference and MMQ. Skips are failures for this experiment.
- Run memcheck, racecheck and synccheck on IQ3_XXS/IQ4_NL.
- Repeat full-layer timing at 2048, 6400 and 8192 tokens, including grouping,
  quantization and both expert products.

The builder reuses hashed, unmodified libraries and places the candidate object
before the archive; only its native IQ member is replaced. It strips CMake's
link dependency-output option so artifact links cannot overwrite build metadata.
Commands, source and library hashes are in `build-prepared.json`.

A separate checkout at `/tmp/strata-v20-iq-prefill` is based on `5e4d56b7` and
contains only the 31-line IQ admission patch. Its full sm75 engine build is
queued after all microchecks pass, with a separate build directory and pinned
source hashes (`engine-prepared.json`). It excludes the uncommitted solo-graph
cache experiment. Build completion alone will not trigger adoption or bypass
model qualification.

## Model gates still open

After the kernel tests pass, build an isolated engine and compare with identical
chunk size, expert residency, ring and MTP-loan settings. Current baseline logs
show 6400-token chunks, a 96-slot ring and 303 additional loaned slots. The ring
can be fixed with `STRATA_PREFILL_RING`; verify actual allocations in both arms.
Do not infer equality from identical configuration alone: the fused layout
changes scratch allocation and the automatic planner can choose other budgets.

`audit-prefill-allocation.py` compares observed startup slots/chunks/rings,
maximum borrowed slots, pinned RAM and MTP room, plus each request's loan
counts. Use `STRATA_TRACE=1` in separate diagnostic runs; do not use their
timings as throughput measurements. Missing trace coverage, incomplete prefill,
multi-device lending or chunk-reduction warnings cannot pass. Example:

```sh
python3 bench/volta-turing/audit-prefill-allocation.py base/engine.log candidate/engine.log \
  --expected-requests 4 --output allocation-comparison.json
```

Nine parser/gate checks pass, including mismatched rings, maximum/request loans,
prompt shapes, missing/incomplete traces and reduced chunks. A real untraced
baseline compared with itself correctly fails completeness. These checks use
synthetic request traces and are recorded in `allocation-audit-validation.json`;
they are not new GPU measurements. Equal counts do not prove equal expert
identities or owned bytes. Source audit: the existing loan trace prints the
requested chunk (`want`), so a logged reduction to `want_p` requires review.

Retain short/4K/16K/32K prompts and the full editing, code and prose workload.
Measure first-token and teacher-forced numerics plus task quality before treating
any timing as an adoption result. A kernel win alone is insufficient.
