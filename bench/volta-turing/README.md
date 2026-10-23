# RTX 2060 SUPER first

Work list: ../../TODO-volta-turing.md. Baseline v19 e3a32c9c, candidate v20.
The active source now isolates the K8V4 attention change. The unsuccessful
wide-matvec engine experiment is preserved with exact source copies and
`wide.patch` in `Strata-data/bench/2026-10-21-volta-turing/wide-experiment-preserved/`.
The patch applies to v19; apply it in a separate experimental worktree before
reproducing the wide-specific build/tests below. No wide defaults were adopted.

GPU access was restored on 2026-10-21. The full 33-shape, T=1..8 parity suite
passes on the RTX 2060 SUPER, and compute-sanitizer reports zero errors. Two
pipeline timing runs are saved under
Strata-data/bench/2026-10-21-volta-turing/20261021-215005-micro.

Build (CUDA 12.4, sm_75; use temporary storage because the workspace disk is nearly full):

```sh
cmake -S Strata-v20 -B /tmp/strata-v20-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP="$PWD/Strata-v19/build/_deps/strata_llamacpp-src"
CCACHE_DIR=/tmp/strata-v20-ccache cmake --build /tmp/strata-v20-build -j6 --target mmvq_il_parity strata
ctest --test-dir /tmp/strata-v20-build -R '^mmvq_il_(tables|wide_override|wide_disabled)$' --output-on-failure
```

The active attention worktree's local, untracked `build` symlink now points to
`/tmp/strata-v20-attention-build`. The earlier `/tmp/strata-v20-build` contains
the archived wide experiment. Keep build symlinks out of commits and recreate
the selected build after clearing temporary files.

```sh
bash Strata-v20/bench/volta-turing/run-2060.sh micro
bash Strata-v20/bench/volta-turing/run-2060.sh modes
bash Strata-v20/bench/volta-turing/run-2060.sh wide
```

The runner checks GPU access and identity before launching anything, runs the
existing correctness probe before engine timing, and keeps the current v19
configuration as the baseline. It uses the existing local systemd benchmark
harness on port 18080. Run with that port free. Each invocation writes to a new
timestamped directory under Strata-data/bench/2026-10-21-volta-turing.

Wide row overrides are experimental, not recommended settings. `wide1` and
`wide2` deliberately screen uniform choices; use the repeated microbench results
to create selective per-format/per-shape overrides before choosing defaults.
The `--pipeline` microbench includes quantization and interleave cost. Production
may reuse an interleaved input across several projections, so also measure
matvec-only timing and full-engine results before judging a marginal cell.

The recorded v19 configuration uses STRATA_MMVQ_GROUP=16, which changes float
summation order. The runner therefore uses `--group16-bench` to time that reference,
while bitwise checks always use group=0 with the exact layout. Passing these checks
does not establish bitwise equality to the configured v19 engine: teacher-forced
numerical and acceptance/quality checks remain required for any winning candidate.

`STRATA_MMVQ_IL_ROWS` now accepts ncols 5–8. Unset retains the old dispatch.
`STRATA_MMVQ_IL_WIDE=0` disables wide table overrides. The test-only tune API can
still force wide kernels. `STRATA_QIL_WIDE=1` separately enables fused wide
quantization in verification; leaving it unset preserves existing selection.
No Blackwell tuning tables were imported. The wide kernel launch follows upstream
75fabbd94152; rows=4 aliases rows=2 for wide windows.

Use `select-wide.py --base CONFIG --out CANDIDATE LOG1 LOG2` to select rows from
complete repeated group16 pipeline measurements. It requires every measured
shape in a dispatch cell to beat the reference by at least 3% in both runs;
unmeasured cells remain disabled. The first 2060 SUPER selection enables 52/100
cells. The large Q5_K head is slower and stays on the existing kernel.

`compare.sh LABEL BASE CANDIDATE [TARGETS] [PAIRS]` uses the existing context-length
benchmark, gates on the kv1188 probe, and stops on missing/failed runs.
`lookup-compare.sh LABEL BASE CANDIDATE` runs the existing editing, fresh-code and
prose workload in ABBA order. Results include generated tokens, acceptance,
suffix-window counts and saved outputs; kernel timing alone does not establish
an engine speedup.

The first extended quantizer check found a difference between the expert
quantizer (`quantize_q8_1_rows`) and the fused dense quantizer at Q5_K 6144x2560,
T=2. The dense engine's reference is `native_quantize_q8_1`, so the test now
compares that standalone producer with its fused counterpart. That comparison
passes for all cases. The original failed log is retained for provenance.

GPU parity covers fused quantization bytes and matvec outputs for T=1..8, formats
IQ4_XS/Q4_K/Q5_K/Q6_K/Q3_K, real dense shapes, the head and odd output rows.
`--tables-only` runs dispatch checks without a device. CUDA errors and empty case
filters fail instead of reporting a successful parity run. HIP/SYCL and V100
runtime behavior still require separate validation.

Compiled resource usage (CUDA 12.4, optimized builds): the 40 new sm_75 kernel
instances report zero local and stack bytes. One-row variants use 80–168 registers,
two-row variants 116–248. The sm_70 translation unit also compiles; its maximum
register count is 252, and three variants have 8–16 stack bytes. These are compiler
resource reports, not measured speed or numerical parity. Raw reports are in
Strata-data/bench/2026-10-21-volta-turing/resources-sm75.json and resources-sm70.json.

The [1Cat-vLLM investigation](1cat-investigation.md) records the additional
Volta/Turing research, including open PR source, closed/rejected experiments,
existing Strata overlap and new entries in the repository TODO list.

RTX 2060 SUPER mode-8 full-engine ABBA, 2026-10-21:

| Prompt | Default decode, two runs | Mode 8 decode, two runs |
| --- | --- | --- |
| 4K | 33.64 / 32.39 tok/s | 34.24 / 30.37 tok/s |
| 16K | 36.54 / 30.56 tok/s | 33.05 / 32.75 tok/s |

Both paired delta ranges cross zero. Do not promote mode 8 from the faster
expert microbenchmarks. The second candidate's 4K prefill is an unexplained
outlier (158 versus 478–489 tok/s); the later [request-level review](mode8-4k-outlier.md)
found 18.14 s spent lending prompt room versus 0.70–0.87 s in the other legs.
Retain the raw data and investigate why that lending operation stalled.
At 16K prefill, medians are 669.7/673.3 tok/s. All arms retain 1,403 cache slots,
6,400-token prompt chunks and 39.60 GiB pinned expert RAM. Results and exact
configs are in `mode8-abba-2155` under the artifact directory above.

Use `analyze-lookup.py DIRECTORY --source PATH/TO/src_edit.py` after a complete
lookup ABBA. It checks requested edits using Python ASTs, detects unclosed code
fences and capped generations, and reports paired throughput and output hashes.
Generated fresh-code tests are not executed by this static analyzer. The first
baseline run copied both edit files without the requested changes and capped
the fresh-code response; those timings do not qualify successful task throughput.

The completed wide lookup ABBA (`wide-lookup-abba-2208`, 16 requests) gives:

| Workload | Default median tok/s | Wide median tok/s | Paired change range |
| --- | ---: | ---: | ---: |
| Edit rename | 49.18 | 49.37 | -0.50% to +1.35% |
| Edit logging | 52.78 | 51.99 | -1.54% to -1.46% |
| Fresh code | 37.32 | 35.84 | -4.91% to -3.07% |
| Prose | 33.17 | 28.69 | -14.58% to -12.50% |

These are observations from two runs per arm, with different generated outputs,
acceptance and baseline drift; they do not isolate a causal kernel slowdown.
The 8- and 16-token graphs execute successfully. All eight edit responses omit
the requested change, so neither arm passes the editing quality gate. Candidate
fresh-code outputs are complete and pass their generated test suites (16/15).
Baseline fresh-code results are one capped/incomplete response and one complete
response with 19 generated tests passing and one failing. Those self-generated
tests and the small sample do not establish a model-quality improvement.

Conclusion for this experiment: retain the implementation and microbenchmark
selection as opt-in research; no wide or expert-mode default is promoted. The
benchmark services and telemetry monitor were stopped after completion.
Raw timings, outputs, AST checks, test logs and hashes are preserved in the
artifact directory. V100 runtime, HIP/SYCL and teacher-forced quality validation
remain untested.

`qualify-edits.py URL OUT --source SOURCE` retains the original edit requests,
allows up to 4096 output tokens, and fails on omitted edits.
`--instruction-first` is a separate prompt-placement diagnostic; it does not
replace the original workload. `profile-prefill.py` retains the failed local
Nsight attempt for diagnosis; no usable Nsight report or timing claim resulted.
