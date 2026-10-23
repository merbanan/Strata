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

This establishes an exact, repeatable GDN-call speedup on this GPU. It does
not establish a full-model prefill gain or unchanged task quality under an
actual request. The full model remains unloaded to honor the current host
RAM constraint. Before local adoption, run matched prefill/decode ABBA,
inspect chunk selection and accepted-token quality, then smoke-test the
installed binary and retain a rollback switch. Raw logs are under
`/run/media/benjamin/TMP/strata-codex-20261023/` as
`gdn-turing-keyhead-8192.log`, `gdn-keyhead-prod-{a1,b1,b2,a2}.log` and
`gdn-keyhead-graph-{base,candidate}.log` and
`gdn-keyhead-prod-override.log`.
