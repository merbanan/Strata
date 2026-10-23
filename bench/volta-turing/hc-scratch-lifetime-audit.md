# HC projection scratch lifetime audit

The prefill path already implements the main reuse proposed by TODO 9.3. With
`STRATA_PF_HC_REGION=1`, `Prefill::carve` puts `xn16`, `lo`, `lo16`, and `gated`
in the region shared with the GDN/QSA/MoE sets. Each half's HC read finishes
before that half writes its region set. `Prefill::bytes_needed_impl` prices the
same layout. For the normal BF16 geometry, `hc_set_bytes` is 63,360 bytes per
token; at T=5,120 this removes 309.4 MiB from the buffers outside the region.
The region's maximum may already be set by a different kernel set, so these
figures are layout accounting, not a measured allocator delta.

The original commit, `5226e859`, reports an RTX 2060 SUPER, IQ3_XXS,
15.5K-token prompt screen: the auto chunk increased from 5,120 to 6,400 and
prefill from 508.6 to 548.2 tok/s when HC region reuse alone was enabled.
The same commit reports a combined `HC_REGION + EMB_REUSE_ACCOUNT` A/B with
14.3% and 15.8% higher prefill at 16K and 32K; those percentages cannot be
attributed solely to HC region reuse. The October 21 IQ2_XS benchmark configs
also set `STRATA_PF_HC_REGION=1` and `STRATA_EMB_REUSE_ACCOUNT=1`. No fresh
full-model A/B was run for this audit because the service is unloaded to keep
host RAM available.

Two conversion scratches remain separate from the chunk layout:

- `bf16_proj` keeps a static `STRATA_HC_REQ8` BF16 expansion buffer and grows
  it to the largest matrix. A 320 x 10,240 HC down/up matrix takes 6.25 MiB.
- `hc_f16_scratch` keeps one `STRATA_PF_HC_FAST` FP16 conversion buffer per
  device, also 6.25 MiB for the down/up matrix.

Both features are enabled in the October 21 benchmark configs. Their buffers
are reused across layers and prompts. Releasing them after the last projection
requires a stream completion fence and a recapture/lifetime check if any
captured work retains their addresses. The next qualification should compare
actual free VRAM and expert-cache slots before/after release, plus full
prefill and decode timing under the same chunk/ring/budget. A 12.5 MiB upper
bound on these two buffers alone is too small to infer a useful chunk increase.
