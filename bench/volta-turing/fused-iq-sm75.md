# Native IQ prefill on Turing

Experimental port, 2026-10-22. Kernel validation passed; model qualification in progress; not adopted. Target: RTX 2060 SUPER, CUDA 12.4, IQ3_XXS.

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

## Validation

The graph-cache full-model timing run finished before compilation/GPU work
began. v1 builds and passes all 11 independent format references. Q2 output
bytes match the original library with the new switch unset, zero and one.
Memcheck then reports two CUDA API errors from probing IQ2_S's oversized
four-warp shared-memory request; the reference calculation itself passes. The
engine build correctly declined that failed gate.

v2 adds a device-capacity check, including static shared memory, before calling
`cudaFuncSetAttribute`. It keeps the kernel arithmetic and two-warp fallback.
The corrected source and rerun are under `fused-iq-sm75-v2/`; all v1 evidence
is retained. All 11 references and exact Q2 controls pass. Memcheck and
synccheck report zero errors; racecheck reports zero hazards, errors or warnings
on IQ3_XXS/IQ4_NL at 256 tokens. Repeated full-layer timings (grouping,
quantization and both products included) are 1.634/1.635x at 2048 tokens,
1.311/1.311x at 6400, and 1.228/1.222x at 8192. These are isolated layer gains;
IQ output is not byte-identical to MMQ. See `micro-validation.json`.

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

A separate checkout at `/tmp/strata-v20-iq-prefill`, based on `5e4d56b7`,
contains the admission patch plus resource-capacity fix. Its complete sm75
engine build passed after the microchecks. Binary SHA-256:
`e153886094b55c9172d8723a39dcc65f3ab1ef4b332e272fcff88287aeee1cf0`.
Pinned source/build evidence is in `engine-prepared.json` and
`engine-queue.json`; the graph-cache and router prototypes are absent.

## Original-planner model diagnostic

All 16 retrieval requests in the 1K/4K/16K/32K ABBA pass. Both baseline legs'
first logits match the adopted-release reference exactly. Each arm repeats
exactly. Baseline-to-fused first-token KL is 0, 0.0040404, 0.0172764 and
0.1086496, with unchanged top-1 on these four retrieval prompts. The 1K case
keeps MMQ; candidate logs confirm fused execution at longer prompts. This is
not a broad model-quality pass. See `model-diagnostic-audit.json`.

The allocation audit fails equality: startup chunks are 6400/6144 tokens,
maximum borrowed slots 1185/1192, pinned expert RAM 39.60/39.57 GiB. Cache slots
(1403), startup ring (96) and MTP room (675 MiB, 303 extra slots) agree. Actual
request loans are baseline/fused 317/317, 859/956, 1076/1132 and 1185/1192.
The 16K request uses 5632-token chunks in both arms, but still borrows different
numbers of slots. All traces are complete with no chunk-reduction warning.
No throughput claim uses this instrumented comparison.

`prefill-teacher.py` uses the existing `pin=P` protocol cut and short-read
windows to score a fixed 128-token tail after batched prefill. It verifies the
exact prefix/tail read paths, target IDs and score count, keeps within-arm
repeats, and reports full-vocabulary target NLL and top-1 agreement. Its top-64
lists do not establish full-distribution KL. Corpus repetitions are explicit;
these numerical probes do not replace full editing/code/prose task review.
The 4K code smoke test passed: exactly 4096 batched prefix tokens and 128
scored window positions, with fused execution confirmed. The full sweep is
running: five corpus segments, four context lengths, ABBA (80 requests and
10,240 scored positions planned). See `teacher-harness-validation.json` and
`teacher-queue.json`.

## Model gates still open

Compare the isolated engine with identical
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

## Scratch lead retained for the next experiment

`moe_bufs` reserves ungrouped MMQ fallback rows for `stream_all_min()-1` even
with group scratch enabled. The larger fused layout retains `m.group_cap=0`
for its final short chunk; `run_impl` does not recarve on every chunk. Reducing
the reserved bytes alone would therefore be unsafe. Any proposed reduction
must also select bounded group scratch for those small MMQ tails, restore zero
for fused chunks, and preserve all bounds/unsupported-layer checks. Test every
tail boundary and the multi-chunk transition before any engine comparison.
This is a source lead, not an implemented or measured memory saving.
