# Flint support work

The requested model is `unsloth/flint-GGUF/flint-UD-Q3_K_XL.gguf`, revision
`20468edfc38ab43d6914cc88b08cbecb05d0fc62`, 18,494,977,472 bytes.
Its SHA-256 is
`91b8d85ab6ff7fb095d9a972f9afa6f3fa2b9acb642867a079f3e013ea3c350c`.
The local copy is on `/run/media/benjamin/TMP/strata-flint/` to avoid filling
the project drive. The downloaded digest matches the repository's linked
digest.

This GGUF reports architecture `flint`, 49 layers, width 2,048, 32 query
heads and four KV heads of width 128. Two leading FFNs are dense (width
6,144); the following 47 layers have 256 routed experts of width 512 with
top-6 routing and a shared FFN of width 1,024. Its boolean per-layer metadata
marks sliding versus full attention and RoPE versus no RoPE. Sliding attention
has a 513-token window; every attention layer has Q/K head RMSNorm and a
learned sink for each query head. The tokenizer uses `tokenizer.ggml.pre=flint`
and the file carries its own chat template.

The initial Strata change recognizes this family and checks every one of the
file's 779 tensors against its expected shape. It deliberately rejects an
inference request because the existing Flash-Next and Qwen3.6 drivers would
compute different layers. A native Flint driver still needs:

1. Bring the now oracle-checked Python Flint tokenizer into the serving/native
   path and apply the GGUF's chat template.
2. Per-layer sliding/full KV state, conditional RoPE, Q/K RMSNorm and sinks,
   with decode and prompt attention checked against a reference.
3. The two dense FFNs and the 47 top-6 MoE layers, including Flint's router
   weighting and shared branch.
4. Hybrid weight placement for the 8 GiB RTX 2060 SUPER, exact-logit and
   generated-text checks, then PP/TG and quality benchmarks.

The native grouped expert path now has an opt-in `-DSTRATA_FLINT_EXPERTS=ON`
build option for Flint's IQ3_S and Q4_K down rows. With this option on, a
real-weight parity check on the RTX 2060 SUPER passed layers 2, 14, 20 and 47
at the model's 2048/512 expert geometry, covering IQ4_XS, Q4_K and IQ3_S
downs. GPU output differed from a float reference by 1.03–1.14% relative;
CPU output by 1.48–1.75%, and all four checks passed the existing 3% gate.
Run it with `native_expert_parity --flint <flint.gguf> 2 14 20 47`.
The `native_expert_parity_flint_formats` synthetic CTest also passes on the
RTX 2060 SUPER without requiring the model file.
An opt-in sigmoid 256/top-6 router now matches Flint's
`flint.expert_gating_func=2` and normalized selected weights. The top-6 plus
ungated shared-branch combine has a vectorized path. `native_multi_parity`
passes sigmoid routing and bitwise combine checks, including a misaligned
fallback. Neither kernel is connected to a full Flint driver yet.
This enables the expert arithmetic needed by a future native driver; it does
not yet connect those kernels to Flint inference or measure a cache speedup.

Current upstream `ggml-org/llama.cpp` implements Flint in
[`src/models/flint.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/flint.cpp).
That runtime is the reference for the first benchmark, not a Strata-native
performance result. The model's [GGUF repository](https://huggingface.co/unsloth/flint-GGUF)
documents the architecture and generation settings.

## First RTX 2060 SUPER reference benchmark

Official llama.cpp revision `9ae73e6`, CUDA 12.4, 12 of Flint's 49 layers
offloaded to the 8 GiB RTX 2060 SUPER, 12 Ryzen 9 3900X CPU threads, FP16 KV,
default batch/ubatch 2048/512. The standalone benchmark command was:

```sh
llama-bench -m flint-UD-Q3_K_XL.gguf -p 512 -n 64 -r 3 -ngl 12 -t 12 -o jsonl
```

The first two-repetition process measured PP512 at 59.71 then 115.57 tok/s;
the file was cold in the host page cache. A repeat process measured PP512 at
208.64, 211.98 and 212.95 tok/s. TG64 measured 15.70/16.29 tok/s in the
first process and 15.24/15.17/15.27 tok/s in the repeat. Thus a quick warm
reference is about **211 tok/s PP512 and 15.2 tok/s TG64**. These are separate
`llama-bench` cases, not a serving request and not a Strata-native result.
The first-pass PP number should not be mixed with the warm throughput.

An upstream `llama-cli` single-turn smoke test at context 2048 generated
"The capital of Sweden is Stockholm." The reference binary and raw JSONL,
stderr and CLI logs are under `/run/media/benjamin/TMP/strata-flint/`. Both
benchmark and CLI processes exited; the GPU returned to 2 MiB used.

## Expert placement screen

The 12-layer baseline leaves most attention work on the CPU. To mirror
Strata's separation of dense weights and routed experts, a second llama.cpp
screen put all 49 non-expert layers on the GPU (`-ngl 49`) and selectively
kept expert tensors on the CPU (`-ncmoe N`). This is static *layer* placement,
not Strata's per-expert cache; it supplies a stronger reference for the native
cache work. All arms use the same model, CPU thread count, 512/64 token cases
and two repetitions. The baseline row uses the three warmed PP and TG samples
from the repeat run above.

| Placement | PP512 tok/s | TG64 tok/s |
| --- | ---: | ---: |
| `-ngl 12`, default expert placement | 208.64–212.95 | 15.17–15.27 |
| `-ngl 49 -ncmoe 49` (all experts on CPU) | 196.84–202.05 | 29.57–30.99 |
| `-ngl 49 -ncmoe 40` (nine tail MoE layers on GPU) | 230.64–232.71 | 31.81–33.38 |
| `-ngl 49 -ncmoe 36` (13 tail MoE layers on GPU) | 248.21–248.84 | 33.78–38.00 |
| `-ngl 49 -ncmoe 32` (17 tail MoE layers on GPU) | 270.37–275.15 | 35.96–36.95 |
| `-ngl 49 -ncmoe 31` (18 tail MoE layers on GPU) | 277.25–281.42 | 36.22–37.35 |

`-ncmoe 30` (19 tail layers) failed to allocate a 313.25 MiB CUDA prompt
compute buffer. The fastest fitting `-ncmoe 31` configuration is about 1.32x
warm PP and 2.42x TG versus the initial placement, using mean rates. Its
single-turn CLI run again answered Stockholm. This improves an upstream
reference configuration; Strata-native expert residency, task quality and
longer-context memory admission are still unmeasured. Keep this setting
opt-in and leave GPU headroom for other applications.

## Pascal fork investigation and wider prefill workspaces

[SirDebugALot's Pascal fork](https://github.com/SirDebugALot/Strata-Qwen3.6-35B-A3B-Pascal-Optimized)
reports its largest long-prompt gains from layer-major prompt windows, packed
expert MMQ and reusing decode-only VRAM for prompt workspace. It refills a
prompt-informed GPU expert cache after prefill, then overlaps resident GPU
experts with CPU misses and limits cache swaps. Strata's main path already has
native packed experts, a borrowed expert-cache region for prompt buffers,
adaptive residency and CPU/GPU miss work. The fork's bounded layer-major
window and explicit phase switch are worth testing in a native Flint driver.
The fork's MTP work does not transfer to this Flint GGUF, which has no MTP
layer. Its Pascal launch geometry is not assumed best on Turing.

The fork's workspace idea has an immediate, measurable effect in the upstream
Flint reference. The table below is **llama.cpp**, not Strata; `-ncmoe N`
places whole expert layers on CPU, and does not create a per-expert cache.
All rows are same-day runs of llama.cpp `9ae73e6`, RTX 2060 SUPER, 12-core
Ryzen 9 3900X, `-ngl 49`, FP16 KV, prompt 2048, generation 128,
`-b 2048`, eight CPU threads. Each confirmed row has two repetitions.

| CPU expert layers | Microbatch | PP2048 tok/s | TG128 tok/s |
| ---: | ---: | ---: | ---: |
| 32 | 512 | 267.05–269.00 | 36.67–37.64 |
| 32 | 1024 | 467.82–473.84 | 36.43–36.84 |
| 36 | 2048 | 713.41–718.57 | 32.15–34.32 |

The middle row is a useful balanced opt-in profile: **1.76x** the matched
PP2048 rate while TG stays close. The wide row is **2.67x** the matched PP
rate but gives up about 10% TG because four more expert layers run on CPU.
`-ncmoe 31 -ub 1024` failed to reserve a 626.51 MiB CUDA prompt buffer;
32 CPU expert layers fit. A one-off 2048-microbatch run measured 709.67 PP
before the repeated confirmation. The CPU thread screen at 512/64 with 8,
12 and 16 threads moved PP only 271.78–275.18 and TG 36.43–38.15; it was
much smaller than the workspace effect.

Run the balanced reference with `llama-bench -m flint-UD-Q3_K_XL.gguf -p 2048
-n 128 -r 2 -ngl 49 -ncmoe 32 -t 8 -b 2048 -ub 1024 -o jsonl`.
Use `-ncmoe 36 -ub 2048` for prefill-focused measurement at this context.
An upstream `llama-cli` smoke run with the balanced settings at context 2048
generated "The capital of Sweden is Stockholm."
These settings still require a context- and memory-fit check in a serving
request. A native Flint implementation can avoid the static placement
tradeoff by releasing decode cache slots for wide prefill, then refilling
hot experts before generation. Measure prompt-to-decode transition cost and
continuation correctness before claiming combined throughput.

## Opt-in 160K K8V4 context profile

`tools/run_flint_2060_160k.sh` configures the upstream llama.cpp reference
runtime for the RTX 2060 SUPER with `-c 160000 -ctk q8_0 -ctv q4_0` (K8V4),
flash attention, 49 GPU layers, 40 CPU expert layers and a 1024-token
microbatch. Flint's GGUF declares a 262144-token maximum. Example:

```sh
STRATA_FLINT_LLAMA_CLI=/path/to/llama-cli \
  tools/run_flint_2060_160k.sh /path/to/flint-UD-Q3_K_XL.gguf \
  -n 64 -p 'What is the capital of Sweden? Answer in one sentence.'
```

This is an explicit opt-in launcher for llama.cpp until native Flint inference
exists. A short-prompt CLI smoke test with the same 160000-token context and
K8V4 cache answered Stockholm; the launcher run displayed 43.2 prompt tok/s
and 30.9 generation tok/s for this brief request. The 512-token microbatch
version peaked at 6733 MiB GPU usage. The 1024-token version also initialized
and generated;
full 160K filled-context throughput, memory and quantized-KV quality remain
unmeasured.

A quick `llama-cli` benchmark of this exact launcher configuration used a
2529-token repeated-text prompt (`llama-tokenize --show-count`) and 128 output
tokens. One RTX 2060 SUPER run displayed **466.2 prompt tok/s** and **29.5
generation tok/s**. The full 160K context was selected, but only the short
prompt was populated; this does not measure throughput near the context limit.

The same llama.cpp checkout has an experimental per-expert GPU MoE cache
(`--moe-cache-mib` in `llama-cli`). For a controlled benchmark, the local
`llama-bench` checkout on the TMP drive was patched to read
`LLAMA_BENCH_MOE_CACHE_MIB` into its context parameters; this is a benchmark
switch, not a Strata change. On the same RTX 2060 SUPER with all 49 expert
layers on CPU, prompt 2048, generation 128, eight CPU threads and 1024
microbatch, its first-pass results were:

| GPU MoE cache | PP2048 tok/s | TG128 tok/s |
| ---: | ---: | ---: |
| Off | 349.74 | 28.86 |
| 1 GiB | 340.50 | 17.86 |
| 2 GiB | 349.77 | 21.37 |

This cache is slower for Flint on this first-pass test. The code uses SLRU
with uploads on misses and skips caching large prompt batches. A likely cause
is PCIe upload cost and churn across 47 top-6 layers; the benchmark did not
measure hits or transfer time, so that remains a hypothesis. Do not enable
this cache from these results. The Pascal fork's bounded swaps and
prompt-informed refill are the stronger decode-cache design to test once a
native Flint driver exists.

The same `-ngl 49 -ncmoe 31` placement also completed PP2048 at
278.46/278.99 tok/s and TG128 at 38.20/36.81 tok/s. The same-day initial
placement measured PP2048 at 206.59/215.20 and TG128 at 16.05/16.84
tok/s, so the mean-rate gains are 1.32x PP and 2.28x TG at this longer
case. These still do not measure a continuous serving request.

## Tokenizer boundary

`tools/strata_tokenizer.py` now implements Flint's ordered number, CJK and
letter/punctuation pre-tokenization passes. A single combined regex changed
three of 66 preliminary cases around whitespace next to CJK; the ordered
implementation matches upstream `llama-tokenize` on 117 plain text cases,
including 100 seeded random strings, and six special-token checks with and
without special parsing. The local tokenizer regression suite passes. This
validates Python token IDs on those cases, not the server chat template or a
full-model Strata output path.

## Strata code review: next speed work

These are ranked implementation leads, not measured Strata-native gains.
The runtime still refuses Flint in `model_family_runs` and its Flash-Next
`--native` path requires a PLE tensor Flint does not contain.

1. **Build a Flint driver around reusable native kernels.** The Qwen3.6 driver
   (`src/qwen35/q36_engine.cu`) has expert grouping, CPU/GPU split, MMQ and
   CUDA graphs, but its mixer, shared-expert gate and state differ. Flint
   needs 49 attention layers with sinks, per-layer RoPE and sliding/full
   selection, two dense FFNs and 47 top-6 MoE layers. Check full logits and
   continuations against llama.cpp before tuning. The native expert
   arithmetic is already parity-tested.
2. **Loan more VRAM to prefill, then restore experts for decode.** The
   measured 1024/2048 microbatch gains identify prompt workspace as a large
   lever. Strata's prompt path already borrows expert-cache slots, but
   `src/prefill/prefill.cpp` fixes width 2560, top-10/512 and FF 640. Adapt
   the layout to Flint and test bounded layer-major windows: its stream-all
   schedule builds a sequence of nonresident experts for every chunk, so
   larger windows can reuse each transferred weight across more tokens.
3. **Use a 513-cell KV ring in Flint's 35 sliding layers.** Fourteen layers
   use full context. FP16 K/V with four 128-wide KV heads costs 2048 bytes
   per layer-token. At context 2048, all-full KV is 196 MiB versus about
   91 MiB with sliding rings; at context 8192, 784 versus 259 MiB. These
   are capacity estimates, not measured allocator savings. Spend the released
   VRAM on the prefill workspace or expert cache, and validate sink and RoPE
   behavior against the reference.
4. **Connect and measure Flint's sigmoid router and top-6 combine.**
   Flint applies sigmoid to 256 router logits, selects six experts and
   normalizes their weights. The opt-in `native_router_sigmoid_top6` and
   vectorized six-expert shared combine now pass kernel parity; wire them into
   the native driver and measure decode. Flint adds its shared branch without
   Qwen3.6's sigmoid shared gate.
5. **Make cache decisions from Flint routing and transfer cost.** The
   tested llama.cpp SLRU cache lost decode speed at 1–2 GiB. Reuse Strata's
   bounded adaptive swaps, CPU/GPU overlap and prompt-informed refill, but
   first measure held-out routing hits, upload MiB, CPU miss time and GPU
   waits. `tools/make_profile.py` now accepts Flint's 49x256 shape and excludes
   routed experts from dense layers 0–1; the binary profile round trip is
   tested. Build one with `--n-layer 49 --n-expert 256 --dense-lead 2 --no-base
   --out PROFILE`. The shipped 48x512 base profile does not match Flint, so
   `--no-base` is required unless a 49x256 base is provided. `ExpertCache`
   storage accepts dynamic dimensions, but a native Flint driver must still
   connect routing traces and cache residency. Compare each cache change with
   static placement.
6. **Keep packed MMQ coverage per layer.** `moe_mmq.cu` recognizes Flint's
   IQ3_S, IQ4_XS and Q4_K formats (Q4_K requires the K-quant build option).
   The Qwen3.6 driver enables MMQ only if every layer passes its format and
   shape check; one unsupported layer falls back for the whole model. A Flint
   driver should dispatch per layer and compare with grouped experts on
   `sm75` at prompt widths 512, 1024 and 2048.

Performance gates: same-day PP512, PP2048, PP8192 and TG128 at fixed prompt
token IDs; peak VRAM and host RAM; cache hits and uploaded bytes; logit and
continuation parity; and prefill-to-decode transition time. The short
512-token prompt has not gained from a wider microbatch and remains a
separate target.
