# MTP hyper-connection RMS reuse audit

In `MtpDrafter::draft`, the MLP hyper-connection read computes normalization statistics for the residual after the attention write, using `mlp_hyper_connection.hc_norm.weight` (the first `fused_gr_read_multi` call in `src/core/mtp.cpp`). The MoE output is then combined and written back to that residual. The final head mixer reads this **new** residual with `hyper_connection_mixer.hc_norm.weight` (the later `fused_gr_read_multi` call, or `gr_read` fallback).

The two RMS calculations therefore have different inputs and different normalization weights. Reusing the MLP read's `rs_` for the head read would change the computation. In the final fused read, the normalizer's `rs` is already passed to its own down/up work; the code does not launch a second final-head RMS pass to remove. The Q8_1 quantization after the mixer uses an absolute-maximum scale, not an RMS scale.

This closes the cross-read RMS-reuse lead without a code change. A separate whole-batch HC/layout optimization could still reduce launch or memory cost, but it must preserve the two distinct normalization steps and pass exact-output and full-round gates.
