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

1. A tokenizer and chat-template path that matches the Flint GGUF.
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
