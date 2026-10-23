# Prefill down-group accumulation order and scratch check

`STRATA_PF_DM_GROUP=1` uses the existing `STRATA_PF_GROUP_SCRATCH` layout to
reuse the down-projection rows after each streamed expert group. A per-token
accumulator replaces the full chunk's 10 down rows per token. The production
guard requires streamed MMQ, no fused layer, no CPU rows, and no native
NVFP4 tail scales. Chunks that take a different path must fit all rows in
`Dm`, as checked before launch.

The direct combine sums each token's ten routed rows in router-slot order.
The grouped combine walks experts in group order and sums each expert's rows
before the next group reuses the scratch. For a given token, the two orders
have the same real-number sum but different FP32 rounding. The group kernel
uses a barrier between experts; within one expert, each token has at most one
row, so distinct warps do not update the same accumulator element.

The new `prefill_dm_group_parity` synthetic check runs the production combine,
row-weight, group-accumulation and epilogue kernels on the RTX 2060 SUPER. It
uses 4 tokens, 10 routed experts per token, 35 possible experts, three empty
experts, groups of three and a final group of two. The group scratch is
overwritten between groups on the same stream. Independent CPU FP32 `fma`
references in router order and expert order both match their respective GPU
outputs bitwise, with zero errors over 10,240 values. The two GPU paths differ
at 6,905 values; outside one deliberately cancellation-heavy column, the
largest absolute difference is 4.76837158e-7. The cancellation case reaches
an absolute difference of 2. CUDA memcheck and synccheck report zero errors;
racecheck reports zero hazards. This verifies the tested arithmetic and scratch reuse,
not full prefill quality or timing.

Earlier v13 IQ3 screens reported a larger auto prompt chunk (7,424 tokens)
and +17.3%/+18.9% prefill at 16K/32K with `PF_DM_GROUP`, versus
+14.3%/+15.8% from the accepted HC region and embedding accounting changes;
decode after a 4K prompt was 2.1 tok/s lower and the output was not bitwise
equal. Those historical runs do not qualify this switch for adoption. It
remains disabled in the installed config. A full-model comparison must check
task quality, exact work, accepted tokens/s, scratch capacity and CPU/streamed
fallbacks under matched chunk and cache budgets.
