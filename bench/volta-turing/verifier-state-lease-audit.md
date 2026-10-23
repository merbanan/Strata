# Verifier recurrent-state and expert staging audit

`session_bytes` allocates a distinct FP32 recurrence and convolution state for
each of the 36 GDN layers. In the current 48-layer geometry, one layer has
`128*48*128 + 10240*3 = 817152` floats, or 3.117 MiB. All 36 layers use
112.219 MiB of persistent VRAM. A hypothetical two-byte representation of
every element could save at most 56.109 MiB before alignment, conversion
scratch or metadata. The October 21 IQ2_XS run reported 1,403 cache slots in
2,367 MiB, so that upper bound is about 33 slots at its *average* slot size;
it is not a measured slot increase.

The verifier separately stores the convolution input (`qkv_L_`) and history
(`h_L_`) for every GDN layer and proposed token until commit. At its logged
16-token maximum, those two FP32 arrays occupy 45.0 MiB of the 127.2 MiB
verifier arena. Accepted-prefix commit reads these rows, while rejected rows
must not enter state. A compact layout must preserve that ordering, exact
ownership and the batch-slot cases; a half-precision edit would alter the
recurrence and needs output-quality and acceptance checks.

Expert staging is already bounded to `kStagingBlobs = 16` maximum-size blobs
per verifier arena; two pipeline groups split it into eight each. The plan
publishes counts and source pointers for a group, then the copy/compute stream
waits for its ready flag before using the staging bytes. The resident cache
uses host and device residency tables; prompt loans, shrink and swaps can
change the resident set between windows. Any proposed lease across windows
needs a slot owner and version, plus a completion fence before reuse.

The full model remains unloaded to preserve host RAM. Next gates are a
same-day full-model trace of actual stage occupancy and cache displacement,
then a compact-state prototype with state/commit parity and complete
verification-round timing. No new speedup is established by this accounting.
