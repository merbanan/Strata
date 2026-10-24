# Opt-in key-head GDN on the RTX 2060 SUPER

The existing CUDA `gdn_rec_kh_kernel` shares Q/K staging across the three
value heads of one key head. Strata previously admitted it only on sm80+
because those cards support `cp.async`; its sm75 build already has a plain
copy fallback. `STRATA_GDN_KEYHEAD_TURING=1` now permits sm75 when the CUDA
occupancy API says all 64 recurrence blocks can reside concurrently. The
global default and `STRATA_GDN_KEYHEAD=0` override remain unchanged. This
is a production dispatcher opt-in, but it is not enabled in the installed
release pending a full-model gate.

The RTX 2060 SUPER has 34 SMs; the key-head kernel permits two blocks per SM,
enough for its 64-block grid. The sm75 object uses 199 registers per thread,
25,984 bytes of static shared memory per block and no local stack/spill bytes.
The changed production CUDA unit also compiles for sm70; there is no V100/T4
runtime result. The opt-in adds no persistent scratch or model-weight copy.

The standalone bitwise screen compares the key-head kernel with Turing's
current `gdn_rec_cols_pipe_kernel` at 1, 2, 3, 7, 8, 9, 31, 33, 257,
1,000, 4,099 and 8,192 tokens. Every recurrence output, final FP32 state
and FP16 output matches bitwise. An actual `strata::prefill::gdn_recurrence`
ABBA across separate baseline/opt-in processes also has identical state,
recurrence-output and FP16-output hashes at 128, 512 and 2,048 tokens.

| Tokens | Baseline production call, A1/A2 | Opt-in, B1/B2 | Call-level ratio |
| ---: | ---: | ---: | ---: |
| 128 | 0.346 / 0.346 ms | 0.260 / 0.260 ms | 1.33x |
| 512 | 1.342 / 1.341 ms | 1.003 / 1.003 ms | 1.34x |
| 2,048 | 5.183 / 5.190 ms | 3.902 / 3.867 ms | 1.33–1.34x |

Each call includes the recurrence and FP16 output norm; 20 CUDA-event
samples follow four warmups per shape. A 129-token production CUDA graph
replayed after changed Q/K/V and gates has matching hashes in both switch
states, and the second replay differs from the first as expected. The
opt-in graph passes CUDA memcheck, racecheck and synccheck.
Setting `STRATA_GDN_KEYHEAD=0` alongside the new opt-in returns production
timing to the baseline and preserves the baseline output hashes.

This establishes an exact, repeatable GDN-call speedup on this GPU. At the
time of that isolated screen, the full model was left unloaded to honor the
host RAM constraint. Raw logs are under
`/run/media/benjamin/TMP/strata-codex-20261023/` as
`gdn-turing-keyhead-8192.log`, `gdn-keyhead-prod-{a1,b1,b2,a2}.log` and
`gdn-keyhead-graph-{base,candidate}.log` and
`gdn-keyhead-prod-override.log`.

## Full-model gate on 2026-10-24

The current v20 source was rebuilt for sm75. Four fresh server processes used
the installed IQ3_XXS/K8V4 configuration with one change:
`STRATA_GDN_KEYHEAD_TURING=0` for base and `=1` for candidate. The run order
was base/candidate/candidate/base, with 4,096- and 16,384-token prompts and
256 output tokens per leg. Each leg passed the 156-token kv1188 correctness
probe. Request hashes, generated text, prompt/output tokens, draft
offers/accepts, expert RAM/file reads, hit rates and PCIe shares match across
all four legs. The GPU expert cache had 1,403 slots and the prompt path
borrowed 1,185 slots in every leg.

| Prompt | Base prefill tok/s | Candidate prefill tok/s | Paired candidate/base | Base decode tok/s | Candidate decode tok/s | Paired candidate/base |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 503.6 / 452.7 | 503.3 / 486.6 | 0.999x / 1.075x | 30.45 / 27.57 | 27.84 / 27.13 | 0.914x / 0.984x |
| 16K | 729.2 / 728.2 | 726.2 / 727.5 | 0.996x / 0.999x | 35.92 / 34.29 | 33.59 / 32.31 | 0.935x / 0.942x |

The 16K prefill result is flat and decode regresses in both pairs. The 4K
prefill result varies between pairs; server logs warned of only 63–209 MiB
free VRAM during request setup, so the 4K variation is not an adoption signal.
The isolated GDN-call gain does not improve this full-model configuration.
Keep the opt-in disabled locally. The complete configs, response text,
per-request work fields and server logs are in
`/run/media/benjamin/TMP/strata-v20-gdn-abba/full-20261024a/`.
