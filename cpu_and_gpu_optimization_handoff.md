# Strata CPU/GPU optimization handoff

Context for a fresh Claude instance continuing performance work on this Strata deployment.
Everything below is verified by actual measurement on this machine, not vendor claims.

## Hardware / environment

- **GPU**: NVIDIA GeForce RTX 2060 SUPER, 8GB VRAM, compute capability 7.5 (Turing). Also present
  but unused: an AMD RX 470/480/570/580-class card (Strata is NVIDIA-only; porting to it would mean
  writing a new HIP/ROCm backend from scratch, not a quick patch - not recommended, see "Dead ends"
  below).
- **CPU**: AMD Ryzen 9 3900X, 12 physical cores / 24 logical threads (SMT2). **No AVX-512** (Zen 2) -
  confirmed via `/proc/cpuinfo`. Has AVX2 and FMA.
- **RAM**: 60GB total. Currently ~44GB used system-wide (other desktop apps + the running model),
  ~16GB "available". This machine runs a lot of other things concurrently (browser, other Claude
  Code/agent sessions, etc.) - don't assume all 60GB is free.
- **Repo**: `/run/media/benjamin/BHOME/projects/hy3/Strata`, a git clone of
  `https://github.com/Niko1221/Strata` (a llama.cpp-derived, from-scratch C++/CUDA engine for a
  large sparse-MoE model, "Qwen3.8-Flash-Next", 512 experts/layer x 48 layers, top-10 routing).
- Current HEAD: `85ad3e9` on `main`, built from: upstream `main` (commit `30ec18e`, their v0.1.30)
  merged with our local work, plus two cherry-picked open PRs. Build dir: `build-main/`. Deployed
  binary: `engine/strata` (copied from `build-main/strata`; `engine/BUILD.json` tracks the source
  hash so `setup.py` won't redundantly rebuild it).

## Models in play

Three Strata-format model variants exist locally, all derived from the same base checkpoint
(`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, 2-bit "Q2_0" quantization):

1. **Q2_0** (original, unpruned): 24,576 total experts (512/layer x 48). Pack:
   `Strata-data/packs/q2_0`. ~31.6 GiB resident when loaded.
2. **Coder (IQ1_M)**: ISTA-DASLab's own release with half the experts *removed* per layer (256/layer,
   12,288 total) and the remainder requantized to a 3.5bpw i-quant mix. Slower than Q2_0 on this
   hardware (see "Findings" below) because the CPU i-quant kernel path is heavier per-weight than
   Q2_0's own dedicated AVX2 kernel, and this CPU has no AVX-512 to make up the gap.
3. **pruned256** (our own artifact, `Strata-data/packs/q2_0-pruned256`): built by
   `tools/prune_experts_from_mask.py`, which reads ISTA-DASLab's own `rco-allocation.txt` for the
   Coder release (it literally documents the exact per-layer retained expert indices, "so the mask
   can be reapplied to the original weights") and applies that *same* 256-of-512 expert mask to the
   original **Q2_0** weights - i.e. Coder's expert *selection*, but keeping Q2_0's quantization
   instead of Coder's heavier i-quant mix. Half the RAM of full Q2_0 (~15.8 GiB), same per-weight CPU
   kernel cost as Q2_0. This is the model actively used for the current benchmark and for real
   opencode usage on this machine.
   - The pruning script and the remapped expert-cache profile
     (`data/expert-profile-q2_0-pruned256.bin`, built by translating the *real* Q2_0
     routing-frequency profile through the same mask rather than starting cold) are both committed
     to this working tree (not yet upstream - they're a local tool, not a PR).
   - GGUF mechanics worth knowing if you touch this again: the expert axis is the **outermost,
     contiguous** axis in the raw tensor bytes for both the quantized expert tensors and the BF16
     router (`gguf-py`'s `GGUFReader` exposes `.data.shape` with experts as axis 0 for both), so
     pruning is a plain numpy gather, no block-level surgery needed. `split.count`/`split.no`/
     `split.tensors.count` metadata must be preserved unchanged (pruning resizes tensors, not the
     tensor *count*, so the two-shard pairing check still passes) - stripping it breaks the engine's
     shard-pairing validation ("additional shard must match the architecture-validated first shard's
     split metadata").

Config/launch files per variant (pattern: `strata-<tag>.json` + `run-<tag>.sh`, both at repo root):
`strata-q2_0.json`, `strata-coder-iq1_m.json`, `strata-q2_0-pruned256.json` (with MTP),
`strata-q2_0-pruned256-nomtp.json` (without MTP - **this is the one in active use**).

## What's been done

### 1. Turing (sm_75) compatibility - now upstream, no longer our patch

We originally wrote our own `STRATA_VOLTA_BUILD` CMake/runtime patch to get this card supported at
all (upstream at the time hard-refused anything below sm_80). While doing follow-up research we
found that **upstream independently fixed the exact same two bugs** (same files, same root cause)
in the meantime - officially merged as "Turing port" (PR #87, `aad5bb15`). We merged upstream's
239-commit `main` and took their versions of `CMakeLists.txt`, `src/core/device.cu`,
`src/kernels/cuda/fused_gr.cu`, and `src/kernels/cuda/native_qsa_score.cu` wholesale (theirs is
strictly better: per-device caching for multi-GPU, HIP-aware, more efficient kernel-launch pattern
than our version). **No special build flag is needed anymore** - `sm_75` builds natively:
```
cmake -S . -B build-main -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc
cmake --build build-main -j<N>
```
Verified: `strata-device --selftest` passes, `qsa_parity` passes (0 failures), real generation
produces correct output before and after.

### 2. Optional MTP in `serve` mode (our own contribution, not upstream)

The `serve` command hard-required `--mtp DIR` (the speculative-decode draft model, ~957MiB VRAM).
The plain single-shot `generate` command already supported running without it (`use_mtp` threaded
through every call site, falling back to the suffix/prompt-lookup drafter or plain decode), but
`serve` never got that treatment. We ported the same pattern into `serve` in
`src/program/generate.cpp` (8 call sites: bind, prefill (x2, including a newer upstream
`draft_kv`/batched-prefill path we had to merge around), kv_restore, set_prompt_len,
set_max_drafts, the draft-window-size calc, the suffix-lookup guard, the final draft call, plus the
newer coupled-draft-sampling hook). Verified correct output and **more GPU-resident experts** as a
result (freed VRAM goes to the expert cache): 1000 -> 1678 slots in one A/B test. This survived the
upstream merge cleanly (re-applied on top of the new call sites, including the new multi-GPU
layer-split and coupled-draft-sampling code that landed upstream in the interim).

### 3. Cherry-picked two open upstream PRs (not yet merged upstream, but reviewed and tested here)

- **PR #257** - `q2_avx2: two-block 256-bit unpack for rows of 1-2 tokens (bit-exact)`. Bit-exact
  (verified by the PR author via memcmp, and by us via real generation before/after). Targets
  *exactly* our bottleneck: the AVX2-only (no AVX-512) Q2_0 CPU expert kernel. The PR author's own
  test rig was nearly identical to ours (AVX2-only i7-8700K + RTX 2070/sm_75 + this same Q2_0
  model), so their numbers were expected to transfer.
- **PR #270** - `qsa_prompt_attn: the f16 tensor-core path runs on Turing (sm_75) as two m16n8k8
  steps`. Main previously compiled the tensor-core prompt-attention kernel to require sm_80
  (tf32/`mma.m16n8k16`) and fell back to a slow FP32 path on Turing. This PR reuses Turing's
  `mma.m16n8k8` (two steps instead of one, same register mapping) so Turing gets the tensor-core
  path too. Author measured 2.6-3.2x kernel-level speedup on a Quadro RTX 8000 (also sm_75).

**Combined measured effect** (identical 393-token benchmark prompt, before vs. after merge +
both cherry-picks, same `pruned256` model, no-MTP config):

| | prefill tok/s | decode tok/s |
|---|---:|---:|
| Before (our original hand-patch only) | 145.7 | 20.6-21.5 |
| After (upstream merge + PR #257 + PR #270) | 150.5 | **29.3** |

That's a **real, measured +36% decode improvement** and a modest +3% prefill improvement. Both
`qsa_parity` (covers PR #270's territory) pass; PR #257's kernel was validated via real generation
correctness rather than a standalone parity binary (the standalone `expert_parity`/
`native_expert_parity` tests need a `pack/full/experts.bin` fixture path this tree doesn't have
populated - they're not wired to run against our actual pack locations).

## The actual bottleneck (verified, not guessed)

This card's 8GB VRAM can only hold a small slice of experts resident:
- **pruned256** (12,288 total experts): ~1000-1700 GPU-resident depending on context/cache config -
  roughly **8-14% hit rate**.
- **Q2_0** (24,576 total experts): ~900-1200 resident - roughly **4-5% hit rate**.

The rest is served by an 11-worker CPU expert pool (11 physical cores + 1 host-loop core = all 12
physical cores; the pool deliberately does **not** use the 12 SMT sibling threads).

Measured live during actual generation (`mpstat -P ALL`):
- CPU0-11 (physical cores, pool workers): **100% each**.
- CPU12-23 (SMT siblings of the same 12 cores): **5-15%, essentially idle**.
- GPU: **93-100%** utilized at the same time.
- System-wide average CPU%: ~52% - this is what makes it *look* like there's spare capacity in a
  naive monitor; there isn't. SMT siblings share execution ports with their physical core and mostly
  can't add throughput to an already-saturated AVX2 vector workload; they *might* help on a
  genuinely memory-bandwidth-bound section (untested here - see "Ideas not yet tried").

**Conclusion: both GPU and CPU are near-simultaneously saturated, but the CPU expert pool is the
structural ceiling**, because only a single-digit-to-low-teens percentage of experts fit in VRAM
regardless of how fast the resident-expert GPU kernels get. This is why PR #258 (see below) measured
"within noise" on hardware just like this one, despite being a real 30% kernel-level GPU speedup.

## Investigated but deliberately NOT merged

From the upstream PR queue at time of writing (`https://github.com/Niko1221/Strata/pulls`):

- **PR #258** (`fused_gr: TILE=1280 kernel specialization for sm_75`): real GPU kernel speedup
  (-30% kernel time) but the **author's own measurement on hardware nearly identical to ours**
  (i7-8700K + RTX 2070 + Q2_0) found **zero end-to-end benefit, explicitly because decode is
  CPU-pool-bound there too**. Also currently shows merge conflicts against the now-merged Turing
  port (both touch `fused_gr.cu`). Skip unless the CPU bottleneck is resolved first.
- **PR #241** (`Faster grouped and per-hit Q2_0 expert kernels, bitwise identical`): targets the
  GPU-resident Q2_0 expert compute path specifically. Tested only on RTX 5090 (sm_120, not our
  arch) with an IQ3_S model (not Q2_0) - the author explicitly could not test Q2_0 end-to-end
  ("none was available for this run") and found no measurable end-to-end gain even on their own
  IQ3_S test. Plausibly positive for us but unvalidated on both our architecture and our quant
  format; lower priority than the CPU-side work.
- **PR #186** (`decode: hyper-connection read in 2 kernels, stream-split`): tested on dual RTX 3090
  (Ampere) with IQ3_S, +7% decode there. NOT bitwise identical (different summation order - author
  says their exactness gate accepts it, but it's a real numerical change, not just a repack). Touches
  `fused_gr_read_multi`, the same function our Turing/PR#258 work lives in - would need careful
  re-integration, and the evidence base is the least relevant to our exact setup of the three.
- **AMD RDNA4/HIP PRs, multi-GPU session-carve, Pascal sm_60**: not applicable (single NVIDIA
  Turing GPU here).

## Benchmark in progress (context, don't be surprised by leftover state)

An EvalScope comparison (`GPQA-Diamond` x20 samples + full `HumanEval`, `reasoning_effort: low`,
`max_tokens: 4096`, `timeout: 600`, `--eval-batch-size 1`) is/was running: `pruned256` (no-MTP)
first, then auto-switches to `ninfer-serve` running a *different* model (`Qwen3.6-35B-A3B`, a
separate engine entirely - see `/run/media/benjamin/BHOME/projects/hy3/ninfer-35b-dram`) for the
same two benchmarks, then restores Strata. Orchestration script:
`Strata-data/evalscope/run_comparison.sh`. Results land in
`Strata-data/evalscope/strata-pruned256/reports/` and
`Strata-data/evalscope/ninfer-qwen36-35b/reports/`.

**Two real bugs hit and fixed while setting this up, worth knowing about if you touch EvalScope
again:**
1. Default `eval_batch_size` is 8 for remote API eval types, but this engine only serves one request
   at a time - concurrent requests caused an hours-long silent stall (retry loop, zero progress,
   server kept redundantly reprocessing). Fix: `--eval-batch-size 1`.
2. GPQA-Diamond answers at "low" reasoning effort still run long (measured ~2400 completion tokens
   average, ~200-260s/question) - the default HTTP client timeout is too short for that, causing an
   identical silent-retry stall. Fix: `"timeout": 600` in `--generation-config`.
3. (Orchestration-script-specific, already fixed in the committed script) `ss -ltn | grep :8080` is
   not a valid readiness check - the port opens long before the model finishes loading and the app
   layer actually serves requests. Use a real HTTP probe (a tiny actual chat-completion request)
   instead.

GPQA-Diamond result so far (pruned256, no-MTP, 20 samples, low effort): **25% accuracy**. HumanEval
was in progress at last check.

## Ideas not yet tried (if you're picking this up to go further)

1. **`--pool-workers 23`** (all logical CPUs instead of just the 12 physical ones): untested. Given
   the workload is partly memory-bandwidth bound (streaming large expert weight blobs from RAM), SMT
   siblings *might* help hide memory latency even with ALU ports contended - this is a genuine
   unknown, not a known-negative, and it's a zero-risk one-flag experiment.
2. Re-examine PR #241 (Q2_0 GPU kernels) specifically with our `pruned256` or `Q2_0` pack loaded -
   the author wanted exactly this test and couldn't get it.
3. The upstream "low-RAM resident-experts" mode (`--resident-experts`, merged in the same 239-commit
   batch) doesn't help *this* model (fits in RAM fine) but would matter if you go back to testing
   full Q2_0/IQ3_XXS/IQ3_S under heavy desktop multitasking (we hit a genuine OOM earlier in this
   session under exactly that condition).
4. The "conversation snapshot / RAM cache" work (PR #189, merged upstream) may speed up
   repeated-turn opencode-style sessions further; not benchmarked here.
5. AVX-512 is a hard hardware ceiling on this CPU (Zen 2) - not fixable in software. If more raw CPU
   throughput is ever needed, that's a hardware decision, not a Strata one.
6. We did **not** attempt PR #258/#241/#186 integration given the evidence above; if VRAM ever grows
   (different GPU) the CPU-bound conclusion should be re-checked before writing them off.

## Quick reference: reproducing the speed comparison yourself

```
# stop whatever's running on :8080 first, then:
cd /run/media/benjamin/BHOME/projects/hy3/Strata
STRATA_VOLTA_BUILD=1 ./run-q2_0-pruned256-nomtp.sh &   # env var is now a harmless no-op, safe to drop
# wait for "ready:" in Strata-data/serve-nomtp.log, then:
curl -s -m 180 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-pruned256","messages":[{"role":"user","content":"'"$(python3 -c "print('Explain how a hash table works, then implement one in Python with get/set/delete. '*20)")"'"}],"max_tokens":400}' > /dev/null
grep "prompt.*tok/s" Strata-data/serve-nomtp.log | tail -1
```
