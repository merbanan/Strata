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
