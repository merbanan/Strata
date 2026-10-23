# K8V4 V-layout follow-up (task 13)

Tested on RTX 2060 SUPER (sm75), CUDA 12.4, IQ3_XXS, 2026-10-23. The
existing `STRATA_PA_K8V4_STAGE=1` gate still restricts this kernel to actual
sm75 devices. `STRATA_PA_K8V4_V_TRANSPOSE=1` selects a second, local opt-in
specialization; its global default is off. Both arms use one binary. sm70 and
sm80 retain their prior dispatch paths.

The V stage now optionally stores FP16-rounded Q4 values by dimension and cell,
so p.v reads adjacent cells as one 32-bit pair. The original layout remains the
other specialization. The Q4-to-half rounding boundary, p.v MMA sequence and
`acc[8][4]` register accumulator are unchanged. Static SASS has 112 rather than
144 `LDS` and 96 rather than 128 `PRMT` instructions in this kernel, with four
additional `IMAD` instructions. These are static counts, not dynamic hardware
counters. `static-sass-counts.json` and the frozen baseline binary/source hashes
are under `Strata-data/bench/2026-10-21-volta-turing/k8v4-task13-v1/`.

CUDA reports 168 registers per thread and no stack/local spills for both
specializations. Dynamic shared memory rises from 29,968 to 31,120 bytes; the
occupancy API reports two 128-thread blocks per RTX 2060 SUPER SM for both
(25% theoretical occupancy). No launch-bound or global cap was changed.

The isolated attention screen passes byte-for-byte output equality with graph
replay, changed queries, empty/ragged selections and shuffled pages at
33/1500/4096/16384/32768 contexts and a full 4096-query case. Relative to the
original K8V4 stage specialization, paired kernel gains are 1.09x, 1.19x,
1.17x, 1.31x and 1.13x across those five shapes; the full-query 4096 case is
1.19x in both orders. Direct mode-3 fallback comparisons also match output
bytes. Memcheck, racecheck, synccheck, and the all-format CUDA suite pass;
Q4-only attention remains correctly skipped on sm75. The changed translation
unit compiles for sm70, sm75 and sm80. V100 and Ampere runtime performance was
not tested.

The first full-model one-switch ABBA, fixed 1,403 expert slots and 6,400-token
prompt chunks, shows 32K prefill +2.11%/+2.21% and TTFT -2.07%/-2.15% in the
two pairs; complete-request elapsed falls 1.85%/1.64%. At 16K the first two
pairs give +0.98% and -3.85% prefill. A second 16K ABBA gives +0.87% and
+1.64%. Thus 32K is repeatable; 16K is small and variable. No decode speedup
is attributed to the prompt-attention change. Both campaigns passed input,
budget, cold-prefill and output-text audits. Their artifacts are
`k8v4-task13-transpose-abba/` and `k8v4-task13-transpose-16k-confirm/` under
the same benchmark root.

The current-baseline versus current-candidate quality comparison produced
identical first-token logits (KL 0 and maximum absolute difference 0) and the
expected retrieval answers at 1K/4K/16K/32K. An older saved K8V4 reference
uses a different build and has different logits; it is not a valid same-binary
layout comparison. The current comparison is in `k8v4-task13-v1/quality-base/`.

The sm70 path already uses a separate `acc[2][2][8]` map and the sm75 stage
uses `acc[8][4]`. The new specialization preserves the sm75 map and adds no
device scratch allocation. Host admission still requires D256, the compiled
grouped-head geometry, valid selection/page pools and an actual sm75 device;
otherwise the existing kernel or caller fallback runs. No V100 runtime claim
follows from the sm70 compile check. Direct Q4 code/scale arithmetic remains a
separate numerical experiment because it would change the exact FP16 rounding
boundary tested here.

The committed source is `c62eb333`. The rollbackable local release is
`Strata-data/releases/v20-k8v4-vlayout-c62eb333/`; its installed engine has the
same SHA-256 as the full-model benchmark binary. The installed `kv1188` smoke
probe passed counting and colors, and the live service reported `loaded=true`
with `STRATA_PA_K8V4_STAGE=1` and `STRATA_PA_K8V4_V_TRANSPOSE=1`. The local
launcher is `Strata/start-current.sh`; `Strata/start-v20-snapshot.sh` retains
the previous release and config for rollback. The release adoption manifest
and `k8v4-task13-v1/installation-smoke.json` record the exact files and hashes.
