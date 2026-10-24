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

Both features are enabled in the October 21 benchmark configs, but enabling
them does not imply both buffers are allocated. Their buffers are reused across
layers and prompts. Releasing either after the last projection would require
completion of its stream and ownership across any concurrent prompt users;
captured work would additionally retain its address. The 12.5 MiB sum above
was a source-level upper bound, not the actual installed-route allocation.

## Actual installed-route allocation (October 24)

An opt-in `STRATA_HC_SCRATCH_TRACE=1` logs the fast-HC admission and each
conversion-scratch allocation. The production IQ3_XXS/K8V4 configuration has
both `STRATA_HC_REQ8=1` and `STRATA_PF_HC_FAST=1`. A real 4,096-token prompt
on RTX 2060 SUPER used one 4,095-token prefill chunk, admitted fast HC, and
allocated one 6.25 MiB FP16 conversion buffer. No BF16 expansion allocation
or decode Q8-view scratch allocation was logged. The run completed its prompt
at 506.9 tokens/s; this trace is not an A/B speed comparison. Raw config,
binary hash, command, exit status and log are at
`/run/media/benjamin/TMP/strata-v20-hc-scratch/trace-20261024a/`.

The startup expert cache has 1,137 slots before the FP16 buffer is allocated.
This configuration uses the cache's fixed `cudaMalloc` arena, not
`--vram-elastic`; `ExpertCache::open` fixes the slot count, and ordinary prompt
completion has no grow step. Freeing the 6.25 MiB conversion buffer at the end
of a prompt would therefore release VRAM to the driver but create no expert
slot or auto-chunk increase in this configuration. A future prompt would need
the same buffer again; safe release would also need concurrent-prefill ownership
tracking. We do not add a recurring free/reallocation path without a capacity
or complete-inference gain to test. The existing HC region reuse remains the
measured and enabled prefill improvement. Reopen this buffer-lifetime question
if a different model makes the BF16 buffer active or an elastic cache can
actually claim the freed memory.
