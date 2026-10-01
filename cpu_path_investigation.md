# Strata CPU expert path: investigation (static analysis, no tests run)

Scope: algorithmic and CPU-code-path optimization for the `pruned256` no-MTP deployment on the
Ryzen 9 3900X (Zen 2, AVX2, no AVX-512) + RTX 2060 SUPER. Per instruction, **nothing was built,
benchmarked or profiled**; everything below is from reading code, configs and existing logs.
Claims are tagged **[verified]** (read directly from code/logs) or **[estimate]** (reasoned, needs
measurement).

## 1. Corrections to the handoff's premises

1. **The GPU hit rate is ~40-58%, not 8-14%.** [verified] The handoff divided slots by experts
   (1675/12288). But the routing-frequency profile plus the adaptive tier (`adapt_every=4`,
   `adapt_swaps=96`, on in serve) put the *hot* experts in VRAM. The serve logs show a decode hit
   rate of 28.6% on the first request, then 39-47% cumulative, and 58.6% on the 393-token
   benchmark. The CPU therefore handles roughly half the routed entries, not 90%.
2. **The "CPU is saturated" evidence is invalid.** [verified] `mpstat` shows 100% on CPU0-11
   because of spin-waits:
   - Parked pool workers spin on `_mm_pause` for up to 20 ms before sleeping
     (`ExpertPool::kSpinBeforeSleep`, `src/kernels/cpu/pool.cpp:487`), and one layer is ~1.6 ms.
   - The host thread spins on the GPU doorbell (`while (*seq < want) _mm_pause()`,
     `src/core/verify.cpp:1006`).

   So all 12 physical cores read 100% whether they're computing or waiting. Similarly,
   `nvidia-smi` "utilization" only means a kernel was resident during the sample.
3. **Per layer, CPU and GPU alternate; they don't run fully in parallel.** [verified] Per layer in
   a verify window, `Verifier::run` (`src/core/verify.cpp:998-1043`):
   1. The GPU runs the mixer (attention/GDN) and the router, then rings `seq`.
   2. The host runs the pool on the missed experts while the GPU computes the VRAM hits and any
      PCIe-fetched experts.
   3. The host raises `flag`, and the GPU combines and moves on.

   The CPU idles during step 1, and the GPU idles during step 2 once its hits are done.
4. **PCIe runs at gen3 x8, not x16.** [verified] (`nvidia-smi`: width current 8, max 16.) That's
   why the startup probe measured 6.7 GB/s, which set `pcie_frac` to 0.14 (the default is 0.55).
   The second (AMD) card is probably splitting the lanes. This is hardware, but it directly limits
   how much CPU work can be offloaded over PCIe.

### Rough budget [estimate]

- **Time per layer:** ~31 tok/s at ~2.43 tokens per window (e.g. 1214/1698 suffix drafts
  accepted) gives ≈ 12.7 windows/s. That's ≈ 79 ms per window, or **≈ 1.65 ms per layer**.
- **CPU work per layer:** a window of up to 3 tokens × 10 experts = 30 entries. At ~1.5 tokens
  per distinct expert that's ~20 distinct experts, and ~50% misses leaves **~10 CPU experts**.
  At 1.38 MB each, that's ≈ 14 MB.
- **CPU time per layer:** at a realistic 30-38 GB/s DDR4 stream rate, **≈ 0.4-0.5 ms**.
- **Kernel capacity:** the AVX2 kernel does roughly 7-8 cycles per 18-byte block at NT=1
  (see §3.1). That's ~9-10 GB/s per core, so 12 threads have ~2.5-3x more compute than DRAM can
  feed. When every thread is busy, the pool should be **DRAM-bound, not compute-bound**.

So the CPU pool is plausibly only **~25-40% of per-layer wall time**. The rest would be GPU
mixer/router/combine work plus the doorbell round trips. **This must be measured before picking
optimizations** (§5). If it holds, even a perfect CPU path caps out at about +35-65% end to end,
and overlap and hit-rate levers are worth more than kernel speed.

## 2. What the CPU actually runs [verified]

- The pack is native (`native_experts.txt`, gate/up/down types 42/42 = Q2_0 in GGUF 18-byte
  blocks). `run_split_multi_native` → pool modes 5/6 → `q2_rows_any` →
  `q2_0_gguf_rows_multi_avx2` (`src/kernels/cpu/q2_avx2.cpp`). The AVX-512 code in `expert.cpp`
  is never reached on this CPU.
- Per layer, the host does the following (`expert_source.cpp:977-1198`, `pool.cpp:730-759`):
  1. Plan the GPU/PCIe/CPU split.
  2. Quantize each token's activation (`act_quant_q8_1_avx2`).
  3. Build jobs, one per distinct missed expert, with nt = tokens routed to it (1-3 with
     `--spec 2`).
  4. **Phase 5:** 36 equal row ranges (3 × 12 threads) over all experts' gate/up rows. Gate rows,
     then up rows, then scalar `std::exp` SiLU.
  5. **Barrier.** The host serially quantizes every expert×token intermediate.
  6. **Phase 6:** 36 row ranges over the down rows.
  7. **Barrier.**
- Each `run_phase` is wait-parked → publish → drain → wait-done → wait-parked. That's 4 barrier
  crossings per layer, ×48 layers per window.
- `pool.cpp` is compiled for baseline x86-64 (no `-mavx2`). The global flags are
  `-O3 -DNDEBUG` with no `-march`. Only `q2_avx2.cpp` and `iq_avx2.cpp` get AVX2 flags.
- The expert arena (15.8 GiB, `cudaHostRegister`ed) is on **4 KB pages**. The hugetlb pool is 0,
  and the fallback `mmap` in `src/core/pinned.cu:207` never calls `madvise(MADV_HUGEPAGE)`, while
  THP is in `madvise` mode on this system. `AnonHugePages` is only 4 MB system-wide.

## 3. Opportunities, ranked by expected value / risk

### Tier A: no numerical change, small code or config changes

**A1. Transparent huge pages for the expert arena.** Add `madvise(p, bytes, MADV_HUGEPAGE)`
right after the 4 KB fallback `mmap` in `pinned.cu:207`, before the arena is touched (and
optionally for the `--mmap-experts`/shared-file paths).
- **Why it helps:** each expert is 1.38 MB, which is 338 4-KB pages. The pool streams ~14 MB of
  scattered blobs per layer, which thrashes Zen 2's 2048-entry L2 DTLB. Also, the L2 stream
  prefetcher stops at 4 KB page boundaries, so each worker's stream re-trains every 4 KB. 2 MB
  pages fix both.
- **Risk:** low. It's ignored if THP is off. Compaction may make load slower, and with ~16 GB
  "available" it may only partly succeed; `AnonHugePages` in `/proc/meminfo` shows how much it
  got.
- **Expected gain:** [estimate] a few percent up to ~15% on the pool's achieved bandwidth, and
  more if the pool is currently far below the DRAM peak.
- **Alternative:** a reserved hugetlb pool (`vm.nr_hugepages=8100`, ~15.8 GiB), which the code
  already tries first. That's hard on a machine using 44 GB.

**A2. Give the expert cache more VRAM.** [verified] The log shows `502 MiB of VRAM free with
everything loaded` on top of the `700 MiB reserved`. At 1.38 MB per slot, ~300-350 more slots
(+20%) look feasible via `--vram-reserve-mib`. Lowering `--max-context` from 131072 (KV is
reserved for it) could free more if 128K isn't needed.
- **Why it helps:** every hit removes a whole 1.38 MB CPU stream, and moves the hottest remaining
  misses to the GPU.
- **Risk:** the 700 MiB reserve exists because a 128K IQ3_S run once paged the driver
  (`generate.cpp:2317`). Keep a margin, and check for a display compositor/desktop on this GPU.

**A3. Try the existing split-window pipeline, `--spec-split`.** [verified that it exists and is
exact]
- **What it does:** it splits a T≥2 window into two token groups, so group A's CPU experts run
  while the GPU runs group B's mixer (`verify.cpp` `record_window`, `G=2`).
- **Upstream measurement:** "exact, ~7% slower", on the author's machine.
- **Why it might differ here:** on this machine the CPU phase is a larger share of each layer and
  the CPU sits idle during the GPU mixer (§1.3). That's exactly what the split overlaps, so the
  result may be different here. Try it with one config flag.

**A4. PCIe x16.** If the board splits x8/x8 because of the RX 580, removing it or moving it
doubles the host→device rate. The startup probe then raises `pcie_frac` automatically, which
offloads more misses from the CPU and speeds up cache refills. (Hardware, not code.)

**A5. `--pool-workers 23` (SMT).** [estimate] If the pool is DRAM-bound, SMT threads can't add
bandwidth. They mainly add more outstanding misses, which helps only if a single core can't keep
enough misses in flight (which huge pages already help with). Cheap to try, but low expected
value.

### Tier B: CPU code-path changes (same results, or only a change in rounding order)

**B1. A "bit-plane" AVX2 Q2_0 kernel, which removes the unpack and the scalar work.** The current
kernel (`row_multi` / `row_multi_b256`) does the following:
- Per 64 codes, a 14-op interleave unpack.
- Per token per 32-value chunk:
  - `maddubs`, `madd`, `cvt`;
  - a scalar `d*scale` multiply, broadcast, and `fma`;
  - a scalar correction FMA chain, `corr += d*(hx+hx)`.
- Two dependent vector FMAs per block per token, a 10-cycle loop-carried chain on Zen 2
  (FMA latency 5).

Proposed kernel:
- **Pre-permute each token's int8 activation once per layer (row-independent).** Plane
  k ∈ {0..3} holds `x[64b + 4i + k]`, i = 0..15, so `(codes >> 2k) & 3` lines up with its
  activation bytes without any interleave. For two blocks in one ymm: 3 shifts + 4 ANDs for 128
  codes (vs ~16 ops now).
- **Integer sum across the 4 planes:** 4× `maddubs` → 3× `paddw` → 1× `pmaddwd`. int16 can't
  saturate: |pair| ≤ 2·3·127 = 762, and 4 planes gives ≤ 3048.
  - After `pmaddwd`, int32 lanes 0-1 of each 128-bit half are chunk 2b and lanes 2-3 are chunk
    2b+1, so the per-chunk activation scale is a precomputed per-token pattern vector
    `(s,s,s',s' | …)`.
- **Q2_0 offset in the integer domain:** `psubd` a precomputed per-token `(sum,0,sum',0,…)`
  vector, i.e. Σ(c−1)q, exact. This deletes the scalar `corr` chain entirely, and it's what
  ggml's reference does (`chunk_dot_oracle`).
- **Per block pair per token:** `acc = fma(Dpat*Spat_t, cvt(isum), acc)`, where
  `Dpat = (d_b×4, d_{b+1}×4)` is built once per block pair and shared by all tokens.

Op count per 128 weights (2 blocks):

| | Shared | Per token |
|---|---:|---:|
| Current | ~16 | ~30, incl. 8 int-multiply uops and 4 dependent FMAs |
| Proposed | ~12 | ~12, incl. 5 int-multiply uops and 1 FMA |

- **Effect:** roughly **2x fewer uops at NT=1**, a 4x shorter FMA dependency chain, and no
  scalar↔vector traffic.
- **Numerics:** the integer part is exact; only the FP summation order changes. Speculative
  decode still equals plain decode if every NT variant uses the new kernel (the same argument
  `expert.cpp`'s ZMM kernel makes). It's not bit-identical to today's CPU output. It also needs a
  parity test against `s2_expert_scalar(quant_acts=true)`.
- **Where it pays:** where the pool isn't DRAM-bound, i.e. the phase tails, layers with few
  misses, and down rows (below). It also frees SMT/power headroom. If §5 shows the pool already
  near DRAM peak, this drops in priority.

**B2. Remove the mid-layer barrier with per-expert dataflow.** Replace "phase 5 → barrier → host
quantizes all intermediates → phase 6" with a per-expert atomic counter of finished gate/up tasks.
The thread that finishes an expert's last gate/up task quantizes that expert's intermediate and
makes its down tasks claimable, all in a single epoch. This:
- removes 2 of the 4 barrier crossings per layer;
- removes the host's serial quantize loop;
- lets expert A's down rows overlap expert B's gate/up tail.

[estimate] 10-30 µs per layer, i.e. ~1-2% per window. Cheap, with no numerical change.

**B3. A multi-row down kernel.** Down rows are only 10 blocks (180 B) each, so the per-row setup,
NT horizontal reductions and stores are a big fraction of the work. Process 4 rows per iteration
(shared activation loads, one transposed `hadd` reduction for 4 rows × NT). This combines well
with B1. No change to the math if each row's lane order is kept.

**B4. Vectorized SiLU.** In mode 5 (`pool.cpp:619-621`), `std::exp` runs scalar in a TU built
without AVX: ~20k exps per layer, [estimate] ~1-1.5% per window. Move the SwiGLU into
`q2_avx2.cpp` with an AVX2 exp (or compile `pool.cpp` with `-mavx2 -mfma` behind the runtime
check it already relies on). Watch out: a different `exp` rounds differently. That's fine
consistency-wise, but it isn't bit-identical.

**B5. Fuse the gate and up passes.** Mode 5 calls `q2_rows_any` twice (all gate rows of the
range, then all up rows). Interleaving gate row r / up row r in one kernel reuses the per-token
activation/scale vectors while they're in registers and removes the two temporary buffers. Small
gain.

### Tier C: algorithmic changes that alter model output (need a quality eval, e.g. the EvalScope setup)

**C1. Weight-thresholded CPU experts (dynamic top-k on misses).** With top-10 of 256, the last
few routed experts often carry small normalized gate weights. Dropping *CPU-bound* entries below
a threshold ε (and renormalizing, or not) cuts CPU bytes proportionally, while hits stay free.
First log the per-entry weight distribution of missed experts to size the gain. Known
MoE-inference technique; quality has to be checked.

**C2. Cache-aware routing tie-break.** When a missed expert's router logit is within δ of the
best resident expert just outside the top-10, swap it in. Higher effective hit rate at a
controlled deviation from exact routing. Riskier than C1.

**C3. Deeper speculation.** With ~71% draft acceptance (suffix drafter), `--spec 3` puts more
tokens behind each expert read (nt per expert rises, distinct experts per token fall), at extra
GPU verify cost. MTP gives better drafts but costs ~957 MiB of expert-cache VRAM. It's a
trade-off to measure, not a clear win. (Output is unchanged for greedy decoding; only speed
changes.)

### Considered and not recommended

- **Prefetching the next layer's predicted experts:** the experts are already RAM-resident, and
  Zen 2's L3 is a 16 MB-per-CCX victim cache, so there's nowhere useful to prefetch a
  multi-MB working set.
- **Quantizing activations per 64 instead of per 32**, which would allow one `madd` per block:
  it changes the activation contract that `expert.hpp` warns about, for a gain B1 gets exactly.
- **A kernel rewrite before measuring:** see §1. The kernel is probably not the aggregate limit.

## 4. Suggested order

1. Measure (§5). That decides whether Tier B is worth doing at all.
2. Try A1 (THP), A2 (VRAM reserve) and A3 (`--spec-split`) as A/Bs. Each is a one-line or flag
   change.
3. B2 + B4 (small, safe).
4. B1 + B3 if the measurements show the pool below the DRAM roof or phase tails dominating.
5. C1 only with a quality eval next to it.

## 5. Measurement plan (for when testing is allowed)

The engine already has the right instruments; they're printed by `strata generate`, not
`serve` (`generate.cpp:5620-5780`):
- `verify window: wait for rings X pool Y host Z commit W ms/round`: the GPU vs CPU split per
  round. This settles §1.
- `pool multi: gate/up, quantize, down ms/round; N GB/s over the rows phases`: the achieved pool
  bandwidth. Compare it with a STREAM-style peak for this box (dual-channel DDR4; the RAM speed
  isn't known here).
- `dispatch: plan / activation quantize / jobs / run`, and `pool phases: wait-park / drain /
  re-park`: host overhead and barrier cost.
- `CPU experts X distinct / Y routed per layer`, and `pcie experts`.

Run the same args as `strata-q2_0-pruned256-nomtp.json` through `generate` with a fixed prompt
(the 393-token benchmark), after the EvalScope run finishes, since it's currently using the
server.

## 6. B1 implemented: bit-plane AVX2 kernel (2026-09-30)

Code: `src/kernels/cpu/q2_avx2.cpp` (`row_bp`, `bitplane_image`), plus new `ActQ` fields `qp`,
`psum`, `pscale` and `bp_pairs` in `include/strata/kernels/cpu/expert.hpp`.
- The new kernel is the default.
- `STRATA_Q2_LEGACY=1` selects the old kernel (also callable as
  `q2_0_gguf_rows_multi_avx2_legacy`).
- The engine builds (`build-main/strata`). The deployed `engine/strata` was not replaced.

Benchmark: a standalone A/B harness (scratchpad `bench_q2.cpp`, compiled against the engine's
`q2_avx2.cpp`) using random GGUF-layout Q2_0 weights at the model's geometry.

**Caveat:** it ran while the EvalScope server was generating on the same 12 cores. Old and new
runs were interleaved so they saw the same load, but absolute numbers are depressed and the
multi-thread DRAM figures are noisy.

**Parity** (against an exact double-precision reference on the same int8 activations):

| | legacy | bitplane |
|---|---:|---:|
| gate rows, rms rel | 2.7e-7 | 1.0e-7 |
| down rows, rms rel | 1.6e-7 | 6.8e-8 |

- Width invariance (a token alone vs in a 4-token group is bitwise equal): passes for both.
- Whole expert, bitplane vs legacy: 4.0e-7 rms rel, i.e. last-bit rounding-order differences
  only. The new kernel is *closer* to exact, because the -1 offset is now subtracted exactly in
  the integers.

**Single thread, L2-resident weights (compute-bound):** 2.3-2.8x faster at every NT; the ratio
is stable across runs.

| rows | NT=1 | NT=2 | NT=3 | NT=4 |
|---|---:|---:|---:|---:|
| gate (40 blocks) | 2.76-2.82x | 2.79-2.82x | 2.34-2.36x | 2.39-2.40x |
| down (10 blocks) | 2.53-2.55x | 2.59-2.60x | 2.24-2.34x | 2.26x |

**12 threads streaming 531 MB of experts from DRAM (the pool's regime)**, two runs:

| NT | run 1 (legacy → bitplane, GB/s) | run 2 |
|---|---:|---:|
| 1 | 19.4 → 20.3 (1.05x) | 32.9 → 29.1 (0.88x) |
| 2 | 16.9 → 22.8 (1.35x) | 27.5 → 27.9 (1.01x) |
| 3 | 13.7 → 21.0 (1.53x) | 27.0 → 33.3 (1.23x) |

How to read this:
- At NT=1 the legacy kernel is already DRAM-bound, so the change is noise.
- At NT≥2 the legacy kernel falls off, because compute per byte rises with tokens. The new kernel
  stays at the bandwidth ceiling.
- Verify windows route ~1.5 tokens per distinct expert, so the realistic pool gain lies between
  the NT=1 and NT=2 rows. That's tens of percent at most, on the ~25-40% of wall time the pool
  takes.

Re-run on a quiet machine before trusting the DRAM numbers.

**Still to do before deploying:**
1. Run the engine's own `native_expert_parity` (needs its fixture).
2. Do an end-to-end A/B: `strata generate` with and without `STRATA_Q2_LEGACY=1` on the
   393-token prompt (`pool multi ... GB/s` and tok/s).
3. Copy `build-main/strata` to `engine/strata`.

All three need the EvalScope run to finish first.

## 7. GPU-side PRs and levers (2026-10-01)

Survey of the 40 open upstream PRs, checked against how *this* deployment runs:
- **Pack:** native GGUF Q2_0 experts.
- **GPU:** Turing sm_75, 8 GB.
- **Drafting:** suffix drafts, no MTP.
- **Prefill:** fixed `--prefill 512`.

The key code fact: for a native pack, the GPU cache hits run through `native_expert_grouped` →
`native_gu_kernel<42>` / `native_down_kernel<42>` in `src/kernels/cuda/iq_kernels.cu`
(`verify.cpp:683-690`), **not** `moe_grouped_s2`.

| PR | What | Applies here? | Exact? | Status in `gpu-opt-probe` worktree |
|---|---|---|---|---|
| **#242** | IQ/native kernels decode each weight part once for all columns/entries | **Yes.** It covers type 42 (Q2_0) in `native_expert_grouped` (our GPU hits) and `iq_mmvq` | bitwise | applied cleanly |
| **#258** | `fused_gr` down kernel TILE=1280 on sm_75 | **Yes.** hc down projection, all 48 layers (BF16 `hc_*`, 1.2 GiB read per pass). The author measured -30% kernel time on an RTX 2070 | bitwise | applied; conflict resolved (kept upstream's pre-Volta per-block shared-memory rule, sized by the new tile) |
| **#284** | the verify commit no longer `cudaStreamSynchronize`s (~0.45 ms/round) | **Yes**, single GPU. Without MTP the host goes straight to suffix drafting and the next launch | same tokens | applied; conflict resolved onto our serve changes (the sync at request start kept) |
| **#186** | hc read in 2 kernels, stream-split | Yes, but opt-in (`STRATA_GR_V3=1`) | **not** bitwise (summation order) | applied cleanly; default path unchanged, so it's an A/B knob |
| #241 | faster `moe_grouped_s2` / per-hit Q2_0 kernels | **No.** Those kernels serve the canonical Strata pack, not a native one | bitwise | not applied |
| #282 | `--prefill auto` chunks up to 32768 | No: fixed `--prefill 512`, and 1675 slots can't lend a 16K+ chunk | - | not applied |
| #285, #294 | load time / arena pinning (Windows, multi-GPU) | No | - | - |
| #280, #281, #283 | numerics fixes (RoPE table, FP16 saturation) | correctness, not speed | changes output | not applied |

### Related levers found along the way

- **Drafting mode.** In no-MTP serve, a window has more than one token only when the suffix
  drafter matches (`generate.cpp:4729`: `T = use_mtp ? S_mtp : 1`). Across 188 logged eval
  requests it accepted 21,557 of 28,842 drafts, so code workloads do get multi-token windows. The
  MTP config (889 MiB VRAM, 1000 vs 1675 cache slots) has no completed-request data to compare
  against. It needs a direct A/B, especially now that the new CPU kernel gains most at NT≥2.
- **Existing switches worth an A/B here:**
  - `STRATA_DECODE_TIMING=1`: per-request decode split in serve (wait for the GPU / CPU pool /
    host / plan / actq / jobs / run). This is the measurement §5 asks for, without leaving serve.
  - `STRATA_VERIFY_PROFILE=1`: GPU stage stamps per layer.
  - `STRATA_VERIFY_DEVICE_PLAN=1`: exact; neutral upstream.
- **Weight-format idea (changes output):** the four `hc_*` projections are BF16 (1.2 GiB,
  resident in VRAM, read every forward pass). Q8_0 would halve their read, and free ~600 MiB,
  which is ~430 more expert-cache slots. It needs a quality check.

### Combined build

Worktree `../Strata-gpuopt`, branch `gpu-opt-probe`: 85ad3e9 + #242 + #186 + #284 + #258 + the
bit-plane CPU kernel.
- Built with `-DSTRATA_BUILD_TESTS=ON` (sm_75), exit 0. The only new warning is an unused
  `TQ3` constant from #186.
- Binary: `../Strata-gpuopt/build/strata`, with parity tests in the same dir: `iq_multi_parity`
  (#242), `gr_parity`, `native_expert_parity`.
- **Nothing GPU-side has been run:** the ninfer phase of the EvalScope comparison is on the GPU
  (7.4 of 8 GB used).

Test plan once the GPU is free:
1. `iq_multi_parity` and `gr_parity`, to confirm bitwise equality on sm_75.
2. The same 393-token prompt through `serve` with `STRATA_DECODE_TIMING=1`, A/B each change by
   its switch:
   - `STRATA_COMMIT_SYNC=1` (#284)
   - `STRATA_Q2_LEGACY=1` (CPU kernel)
   - `STRATA_GR_V3=1` (#186, opt-in)
   - #242 and #258 against the `engine/strata` baseline.

## 8. Measured results (2026-10-01, idle machine, RTX 2060 SUPER at PCIe gen3 x8)

Binaries are in `Strata-data/abtest/bin/`:
- `strata-base`: 85ad3e9 + the scoring patch only.
- `strata-new`: the `gpu-opt-probe` branch.

Raw results: `Strata-data/abtest/{logits,speed}/`.

### Quality instrument

`--dump-logits` writes nothing for native packs (the spec loop never dumps), so I added
`STRATA_TF_DUMP=path`. It runs a fixed text teacher-forced through the real verify windows
(sizes 1..5, average 3 tokens), commits the text whatever the model predicted, and writes every
window's head logits.
- Text: 3,072 tokens = Python code (textwrap.py) | prose (docs/DETAILS.md) | a chat in the
  model's template.
- Script: scratchpad `quality/metrics.py` computes top-1 agreement, KL(base‖var), and the
  perplexity of the actual text with a paired standard error.

Baseline window breakdown:
- 33.1 ms waiting for the GPU + 28.3 ms CPU pool (25.7 GB/s) + 1.05 ms commit per 3-token round.
- The pool is ~45% of the window.
- An idle-machine streaming test tops out at 34-35 GB/s with 12 threads, and 24 threads add
  nothing.
- So the pool is memory-bound at ~75% of the ceiling. **The kernel isn't the limit**, which is
  why the 2.5x faster bit-plane kernel shows no end-to-end gain.

### Exactness

`strata-new` with `STRATA_Q2_LEGACY=1 STRATA_SCALAR_SILU=1` vs `strata-base`: **bitwise-identical
logits at all 3,070 positions.** So these change nothing in the output:
- #242, #258, #284;
- the fused single-batch pool;
- THP;
- the residency/skip plumbing when off.

Serve output can't be compared as text: base vs base already differs, because the adaptive tier
swaps on its own thread at timing-dependent points.

### Quality of the output-changing variants

| variant | top-1 | KL mean | Δ ppl (nats/token, ±SE) | CPU pool ms/window |
|---|---:|---:|---:|---:|
| bit-plane CPU kernel (rounding only; the noise floor) | 97.65% | 7.8e-3 | −0.0010 ± 0.0024 | 27.5 (base 28.3) |
| #186 GR v3 | 97.79% | 8.2e-3 | −0.0031 ± 0.0024 | 25.2 |
| CPU skip w<0.02 | 97.65% | 8.4e-3 | −0.0032 ± 0.0024 | 24.2 |
| CPU skip w<0.04 | 97.43% | 1.0e-2 | −0.0038 ± 0.0026 | 23.3 |
| CPU skip w<0.06 | 96.55% | 1.5e-2 | −0.0031 ± 0.0038 | 20.9 |
| residency bias β=0.2 | 97.07% | 1.1e-2 | +0.0003 ± 0.0033 | 21.3 |
| residency bias β=0.5 | 97.07% | 1.6e-2 | +0.0050 ± 0.0036 | 17.4 |
| residency bias β=1.0 | 95.90% | 3.0e-2 | +0.0088 ± 0.0061 (chat +0.021 ± 0.011) | 14.8 |

### Speed (serve, 4 fixed greedy prompts × 400 tokens; baseline mean of 3 runs = 31.6 tok/s, run-to-run noise ±1)

| config | decode tok/s | vs base | 1.5K-token prompt tok/s |
|---|---:|---:|---:|
| baseline | 31.6 | – | 165 |
| new build | 33.2 (32.9-33.7 across lossless switches) | +5% | 165 |
| new + `--max-context 65536` (2,087 slots) | 33.6 | +6% | 169 |
| new + `--max-context 32768` (2,295 slots) | 34.7 | +10% | 172 |
| new + `--vram-reserve-mib 350` (1,940 slots) | 33.6 | +6% | 168 |
| new + `--prefill auto` | 33.4 | +6% | **485 (2.9x)** |
| new + MTP (spec 2; 942 slots) | 34.35 | +9% | 158 |
| new + MTP spec 4 | 33.4 | +6% | 158 |
| new + `--spec 4`, `--spec-split`, `STRATA_GR_V3`, `STRATA_VERIFY_DEVICE_PLAN` | 32.1-33.2 | no gain | – |
| new + `--pool-workers 23` | 20.4 | **−35%** | – |
| THP on vs off | same decode; load 100-140 s vs 21 s | – | – |
| **best lossless:** new + prefill auto + 64K ctx | **34.25** | **+8%** | **475** |
| **best lossy:** + `STRATA_ROUTE_RES_BIAS=0.5` | **35.8** | **+13%** | 477 |

THP is now opt-in (`STRATA_THP=1`).

### Task accuracy

| config | HumanEval (164) | GPQA-Diamond (20) |
|---|---:|---:|
| baseline (q4_0 KV, 2026-09-30 eval) | 92.68% (152) | 25% |
| best lossy: new build + `--prefill auto` + 64K ctx + `STRATA_ROUTE_RES_BIAS=0.5` | **93.29% (153)** | – |

The residency-biased routing (β=0.5) costs nothing measurable on HumanEval (+1 problem, within noise), which
matches its small teacher-forced cost.

### GPU parity on sm_75 (`gpu-opt-probe` build)

- `gr_parity`: 0 failures, after the STRATA_HC_Q8 kernel edits (the bf16 path is unchanged).
- `iq_multi_parity` (#242): 0 failures. Q2_0/IQ4_NL, Q2_0/Q2_0 and Q2_0/IQ4_XS `native_expert_grouped` are bitwise
  equal to the old kernels.

### VRAM reserve

With the default `--vram-reserve-mib 700`, 502 MiB is still free after everything is loaded. At 350 MiB, 154 MiB
is free and the cache has 1,940 slots instead of 1,675. Upstream's reason for the 700 MiB default is that a 128K
IQ3_S run that left 30 MiB free made the driver page and stalled a request. **Recommended: 400 MiB, which leaves
~200 MiB free and gives ~+215 slots.**

### KV-cache options (upstream docs)

- q4_0 is +8-12% perplexity on long documents (bench/results/2026-09-27-kv-q4).
- `k8v4` (8-bit K, rotated Q4_0 V, 816 B/cell) cannot stream.
- `--kv-resident N` keeps only N cells per QSA layer in VRAM and the full KV in pinned RAM. Attention reads the same
  values, and the freed VRAM goes to expert slots.

## 9. VRAM investigation: where the 8 GiB goes, and how to fit more experts

Expert slot = 1.38 MB, so 100 MB ≈ 72 slots. At ~56% hit rate, +600 slots measured ≈ +5% decode (§8).

### Ledger (deployment config: 128K q4_0 KV, reserve 700)

Static accounting; the STRATA_TRACE startup ledger is queued in batch 4.

| consumer | size | notes |
|---|---:|---|
| other processes on the card | 184 MiB | two GNOME Showtime background services (`showtime --gapplication-service`) |
| CUDA context, cuBLAS | ~300 MiB | CUDA 12.4: lazy module loading already on |
| dense arena (canonical) | 1,356 MiB | **1,200 of it the BF16 `hc_*` projections**; router 60, indexer projections 37, ple_value 12, ... |
| native projections | 1,376 MiB | attn_qkv 429, ssm_out 303, attn_gate 235, attn_q 158, attn_output 124, shared experts 101 |
| output head (Q5_K) | 417 MiB | |
| KV cache (12 QSA layers, q4_0, 128K) | ~906 MiB | int8 1,660; k8v4 1,284; fully resident unless `--kv-resident` |
| indexer pooled keys (fp32) | ~201 MiB | always resident, also when streaming |
| RoPE tables | 34 MiB | shared |
| GDN states, verify buffers, graphs | ~150 MiB | |
| VRAM reserve | 700 MiB | **502 MiB of it never used** |
| expert cache | 2,160 MiB | 1,675 slots |

### Levers, best value first

| lever | VRAM freed | ≈ slots | output change | status |
|---|---:|---:|---|---|
| close the two Showtime services (or keep them off the NVIDIA card) | 184 MiB | +133 | none | user action |
| `--vram-reserve-mib 400` (~200 MiB headroom left) | ~300 MiB | +215 | none | measured (350: +265 slots) |
| KV streaming `--kv-resident 20480` instead of a full 128K KV | 0.6-1.2 GiB vs resident | +400-900 | none (same values) | int8/q4_0 measured; **k8v4 streaming implemented today** |
| `--kv-resident 20480` instead of 32768 | ~156 MiB (int8), ~120 (k8v4) | +85-110 | none; more page traffic beyond 20K context | queued |
| `STRATA_HC_Q8=1` (BF16 hc → int8 + fp32/32 scales) | ~525 MiB | +380 | small (to be measured) | implemented, queued |
| `--max-context` below 128K | KV scales with it | – | none | measured (32K: +620 slots, +5%) |
| indexer keys fp32 → fp16 | ~100 MiB | +72 | small | not implemented (~10 kernels) |
| output head Q5_K → IQ4_XS / Q4_K | 80-100 MiB | +60-70 | small | needs a requantized head GGUF |
| pruned-vocabulary head (e.g. 64K most-used tokens on GPU) | ~300 MiB | +220 | yes, rare tokens unavailable | idea only (the `--draft-vocab` machinery exists) |

## 10. Batches 2-5: KV options, Q8 hc, fp16 indexer keys, max-cache (2026-10-01)

### Implemented today (branch `gpu-opt-probe`)

- `STRATA_HC_Q8=1`: the hc projections as int8 + fp32 scale per 32. 675 MiB instead of 1,200, **+397 slots**.
  Quality is at the noise floor (top-1 97.56%, Δppl −0.003 ± 0.0025). `gr_parity` passes.
- **K8V4 KV streaming** (`--kv k8v4 --kv-resident N`): a `kKvHybrid` block format, aliased host/stage pools for the
  hybrid appends, and snapshots. `kv_stream_parity` k8v4: bitwise, 0 failures. `kv_hybrid_parity` and
  `conversation_snapshot_test` (1,781 checks) pass.
- `STRATA_IDX_F16=1`: the indexer's pooled keys in FP16, native packs only. **+73 slots**. Short-text quality is noise
  (top-1 98.08%, Δppl −0.002 ± 0.002); long-context results below. The snapshot test fixture needed its pooled sizes
  made format-aware (fixed; the rerun is pending a GPU window).
- `STRATA_TF_FROM` and long-text scoring: 38K tokens prefilled, the next 2,048 teacher-forced through verify windows.

### Task accuracy (GPQA-Diamond 20 / HumanEval 164)

| config | GPQA | HumanEval |
|---|---:|---:|
| baseline (q4_0 KV) | 25% | 92.68% |
| best lossy (rb 0.5) | – | 93.29% |
| k8v4 at 128K | 30% | 91.46% |
| int8 KV + streaming at 128K | 25% | 91.46% |

All of these are within noise: GPQA-20 is ±10 points, HumanEval ±2 problems.

### Long-context quality (40K context; reference int8 KV, ppl 1.0819)

| config | top-1 | Δ ppl |
|---|---:|---:|
| **q4_0 (current deployment)** | 97.56% | **+0.048 ± 0.013 (+4.9%)** |
| k8v4 | 97.90% | +0.033 ± 0.011 |
| q4_0 + fp16 idx keys | 97.61% | +0.030 ± 0.012 |
| k8v4 + fp16 idx keys | 97.61% | +0.029 ± 0.011 |
| int8 streamed 32K | 98.34% | +0.007 ± 0.008 |
| k8v4 streamed 20K | 97.95% | +0.010 ± 0.008 |
| **max-cache** | 98.24% | **+0.015 ± 0.008** |

q4_0 KV is measurably worse at long context. 8-bit K is clearly better. Differences between the 8-bit variants are
partly expert-placement noise (±1-2%).

### Speed (serve; baseline 32.2-32.3 tok/s in these runs)

| config | slots | decode | prompt (1.5K / 60K tokens) |
|---|---:|---:|---|
| baseline | 1,675 | 32.2 | 165 / 160 tok/s |
| int8 streamed 32K + pf auto + reserve 400 | 2,253 | 34.7 (+7%) | 516 / 613 |
| + Q8 hc | 2,650 | 35.25 (+9%) | 518 / – |
| k8v4 streamed 20K + Q8 hc | 2,808 | 35.7-36.3 (+11-12%) | 509-515 / 586 |
| **max-cache (+ fp16 idx keys)** | **2,882** | **35.5 (+10%); 33.4 at 60K (+9.5%)** | **510 / 624 (3.9x)** |
| max-cache + rb 0.5 | 2,882 | 36.4 (+13%) | 510 / – |

- Streaming costs nothing even at 60K context: it is faster than resident KV, because of the extra slots.
- The engine warns at 204 MiB free (reserve 400). **Deploy with `--vram-reserve-mib 500`.**
- My earlier +20-30% decode estimate was too high. The miss-rate curve flattens, and the ~18 ms/token GPU part now
  dominates. Measured: +10% lossless, +13% with rb 0.5, plus 3-4x faster prompts.

## 11. Start scripts (2026-10-01)

Both use `engine-opt/strata`, branch `gpu-opt-probe` at 0a846da. Settings shared by both:
- `--kv k8v4 --kv-resident 20480 --prefill auto --max-context 131072`
- env `STRATA_HC_Q8=1 STRATA_IDX_F16=1 STRATA_ROUTE_RES_BIAS=0.5`

| card | script | reserve | measured |
|---|---|---:|---|
| 8 GB | `run-q2_0-pruned256-maxcache-8gb.sh` | 400 MiB | 2,882 slots; 36.4 tok/s decode; 624 tok/s on a 60K-token prompt. The engine warns "LOW" at ~204 MiB free; no stalls in 3.5 h of evals |
| 6 GB | `run-q2_0-pruned256-maxcache-6gb.sh` | 450 MiB | 6 GB simulation (reserve 400): 1,261 slots, 33.9 tok/s, 324 tok/s on 60K tokens |

Fixed after the evals: `STRATA_IDX_F16` with serve's conversation cache. The incremental snapshot prefix and the
checkpoint restore of the spare pooled row assumed FP32 keys. `conversation_snapshot_test` passes with FP32 and FP16
keys (1,781 checks each). The evals ran without these fixes but never exercised them (each request starts fresh).
