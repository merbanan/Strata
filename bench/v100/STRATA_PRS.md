# Strata PRs reviewed for the V100 tree

Every PR of the 400 most recent in Niko1221/Strata (2026-10-15), classified by the files it changes against what
 calls (iq_kernels' grouped experts, native_mmvq incl. the packed Q8_0 GEMVs, native_router_topk, fused_gdn,
prefill/gemm and moe_mmq, the CPU expert pool). The ones that could touch it were applied or switched on and measured
with q36_bench.sh on the V100 (Qwen3.6-35B-A3B UD-Q4_K_M; v100 baseline short 99.0-99.8 / 16K 84.1-84.8 tok/s).

| PR | State | Title | Verdict |
|---|---|---|---|
| #2497 | open | serve: an engine program that cannot be started is a 503, not a dropped connection | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2495 | open | sycl: images on the SYCL port (CPU encoder, --vision cpu) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2494 | open | Adding new model/{model id} endpoint to allow use of Strata with Visual Studio | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2493 | open | bench: community report - RTX 5060 Ti 16 GB, Coder IQ1_M, 204800-token context (0.1.44.1) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2492 | open | gfx12: empty workgroups past the full-die fill stall in the decode matvecs (grid_pad) | another backend (SYCL / HIP / Metal / Vulkan) |
| #2491 | open | fix(setup): don't require tensor-data alignment for GGUF shards without … | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2490 | open | setup: Alchemist on xe also gets the Level Zero v2 adapter (UR_LOADER_USE_LEVEL_ZERO_V2=1; A770 decode +7%, pr | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2489 | open | kernel benches: host-timed (graph + barrier-event profiling hung on an A770 under xe); native_grouped_parity - | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2488 | open | sycl/A770: faster hyper-connection read on Arc A-series (gr_bench 88/123/151 -> 62/83/100 us at 1/4/6 tokens) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2487 | open | sycl/A770: large GRF for the grouped expert kernels (bit-identical; Alchemist default) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2486 | open | sycl: Alchemist expert cache in segments, so borrowed prompt GEMM operands are not staged (A770 prompt +8-12%, | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2485 | open | sycl: pin the RAM-resident expert copy in 2 GiB chunks when one pinned allocation is refused (Arc A770; prompt | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2484 | open | Community benchmark: 2x RTX PRO 4500 Blackwell, Swift IQ3_XXS, engine 0.1.44.1 (official suite, --long) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2483 | open | sycl: images on the Intel port, and the setup wiring for it | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2481 | open | serve: keep console writes outside shared state locks | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2480 | open | serve: send the role chunk with the first response data | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2479 | open | Community report: RTX 3080 Laptop 8 GB, i7-11800H, 32 GB RAM, Linux (0.1.44.1) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2476 | open | serve: skip file lookahead when experts remain resident | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2475 | open | Metal: the GGUF reader refuses malformed headers with a message instead of crashing or exiting (#1919) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2474 | open | Community benchmark: RTX 3090, Xeon W-2235, 30.9 GiB RAM, Q2_0 at 262144 context | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2473 | open | MCP: opt-in cached image previews and authenticated downloads | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2472 | open | bench: community report, RTX 4070 12 GiB Thunderbolt 4 eGPU + i9-12950HX, Flash-Next IQ3_S, 32K on 0.1.44.1 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2471 | open | bench: community report - RTX 3090 + RTX 5060 Ti expert tier, UD-IQ4_XS, custom-line 0.1.43.1 engine | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2469 | open | serve: reject array-root JSON instead of extracting a nested object | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2468 | open | fix(sycl): permit Level Zero IPC in expert-parallel containers | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2466 | open | engine+serve: the confidence monitor (STRATA_CONFIDENCE=1, measurement only) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2460 | open | gfx906/HIP build broken on v0.1.44.1: two compile blockers (platform/hip_compat/cublas_v2.h missing three cubl | another backend (SYCL / HIP / Metal / Vulkan) |
| #2458 | open | engine: --dump-kv-select, the decode windows' KV block selections as a policy-bench trace (#2415) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2455 | open | serve: an all-cached image turn never wakes the elastic encoder | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2450 | open | setup: no KV streaming on one card that holds every expert and the whole KV cache (#2261) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2447 | closed | native_kernels: sm_86 dense Q8_0 GEMVs keep two rows per block (#2441) | sm_86 Q8_0 GEMV rows per block (closed, unmerged) - sm_86 only |
| #2443 | open | conversation cache: evict redundant holders without losing pinned prefixes | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2439 | open | docs: community benchmark — RTX 5080 + 4070 Ti unequal pair, both orders (+ helper cache arm) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2437 | open | SYCL: the IQ4_XS expert dot reads its codebook from registers | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2436 | open | SYCL: defer pipelined one-token commits to the host verdict (port of c81f9984) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2435 | open | serve: an elastic image encoder that cannot start is a 503; its idle stop is logged on a layer split too | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2434 | open | tools: the KV streaming page-pool policy bench - CLOCK against LRFU, LRU and Belady (#2415) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2433 | open | SYCL: a pipelined window still on the GPU is polled again, not failed | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2429 | open | Add Intel GPU monitoring on Windows via IGCL | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2428 | open | bench: RTX 5060 Ti 16 GB on Ubuntu 24.04, 0.1.44.1, Flash-Next IQ3_XXS | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2427 | open | HC_Q8: skip the layers' bf16 hyper-connection copies on a layer split too | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2425 | open | Layer split auto: STRATA_SPLIT_UNIFORM=1 restores the uniform miss cost | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2423 | open | serve: GET /metrics gives each request's PCIe expert bytes and its own KV streaming share, and a bandwidth sum | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2422 | open | web: the PCIe metric shows the current link generation, not the card's capability | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2421 | open | Video: the encoder reads the clip itself (ENCV - mtmd samples its frames at video_fps and merges them) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2419 | open | arm64: build + run Strata on NVIDIA DGX Spark (GB10, sm_121) - rebased onto v0.1.41, IQ3_S validated | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2414 | open | bench: community report, 2x RTX 5060 Ti 16 GB, Xeon E5-2690 v4, IQ3_XXS at 204,800-token context, engine 0.1.4 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2413 | open | docs/AMD_HIP.md: short GPU stalls while the kernel compacts memory, and how to keep them out of a speed compar | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2409 | open | fix: avoid CUDA OOM during conversation-cache restore | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2405 | open | bench: community report, RX 9070 XT 16 GB on Windows, IQ3_XXS, engine 0.1.44.1 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2404 | open | Community benchmark: RTX 4080 SUPER 16 GB, Ryzen 7 7800X3D, 64 GB, IQ3_S at 192K (0.1.44.1) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2400 | open | Add opt-in usage accounting, output continuation and authenticated remote API | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2399 | open | `--vram-elastic` with `--kv-grow`: VRAM takes its chunks where the K/V grows into (resubmits #2201) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2398 | open | HIP: virtual memory for `--kv-grow` and `--vram-elastic`, on only where a start-up check passes (resubmits #21 | another backend (SYCL / HIP / Metal / Vulkan) |
| #2397 | open | serve: allow JSON response_format with tools on /v1/chat/completions | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2395 | open | bench: community report, Tesla V100-PCIE-32GB + Tesla PG500-216 (2 x sm_70, layer split), Flash-Next GSQ-RCO I | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2394 | open | setup: Alchemist on xe gets UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 (A770 expert cache 1,906 -> 4,540 slots,  | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2392 | open | hip: hipBLASLt tuning table for gfx1101 (RX 7800 XT, hipBLASLt 1.2.0) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2390 | open | bench: community results, RTX 3090 + Qwen3.8-Flash-Next UD-IQ4_XS (engine 0.1.43.1) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2388 | open | metal: build supports_metal4() without the macOS 26 SDK | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2387 | open | bench: community report 2026-10-13 CPU-only Swift 1.5 IQ2_XS (Ryzen 9 3900X) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2385 | open | planner: rank auto layer splits for prompt pipelines | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2384 | open | Batch slots keep a conversation's turn checkpoints across slot -> solo -> slot moves (#1906) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2383 | open | bench: RTX 5080, Ryzen 7 9800X3D, Qwen3.8-Flash-Next IQ3_XXS at 131K, engine 0.1.43.1 (Windows) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2382 | open | setup: put the CUDA runtime DLLs (bin\x64) on the image job's search path on Windows | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2380 | open | setup: calibrate a dual config whose gpu is the string 0,1 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2379 | open | bench: community reports, 1x/2x/6x Tesla V100-PCIE-32GB scaling, Unsl… | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2376 | open | setup: give the --families probe the engine's ROCm DLLs on PATH (#2372) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2371 | open | fix(serve): coerce XML tool arguments consistently by schema | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2370 | open | add lifetime useage statistics to monitoring | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2369 | open | Add benchmark results for v0.1.44 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2366 | open | bench: community report - Apple M1 Max, 32 GiB, Q2_0 streaming | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2365 | open | setup: honour --source on a Mac, whose hand-over skips setup.py's parser (#2363) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2364 | open | bench: community report, 4x RTX 3090 layer split, stock 0.1.44 against 0.1.43 with a patch series, 1 to 16 use | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2362 | open | tools: poll the server address the config binds, not loopback (#2330) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2360 | open | telemetry: the Mac's Monitor cards - GPU load and VRAM from IOKit, th… | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2359 | open | serve: average the throughput gauges over the window between scrapes (#2341) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2358 | open | fused_gr: STRATA_GR_NORM_FAST=1 on CUDA too (opt-in): decode +0.5-0.6% on a V100 + PG500 layer split | fused_gr (Flash-Next hyper-connection) - not called by strata-q36 |
| #2357 | open | sm_70: the multi-block greedy pick (argmax_rows) as an opt-in, STRATA_ARGMAX_ROWS_SM70=1: decode +0.7-0.8% on  | argmax_rows on sm_70: strata-q36 already has its own two-stage multi-block argmax (k_argmax) |
| #2356 | open | pf_wmma: a second Volta prompt-expert kernel for IQ2_S gate/up and IQ4_NL down (STRATA_PF_WMMA2=1, opt-in): pr | pf_wmma2 for IQ2_S / IQ4_NL prompt experts in prefill.cpp - not strata-q36's path, not its expert types |
| #2355 | open | pf_wmma: set the tiles' shared-memory attribute on each card, so the second card of a layer split can launch t | pf_wmma tile attribute for a layer split's second card - prefill.cpp, not strata-q36 |
| #2352 | open | serve: preserve explicit false against configured reasoning defaults | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2351 | open | Suffix-reuse: conversation KV prefix mount, quant-safe splices, state checkpoint ladder | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2350 | open | mac docs: the two prompt opt-ins an M1 Max cannot run (measured) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2349 | open | moe_mm_parity: the default tile that did not exist, and --v3's v1 arg… | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2348 | open | metal: the M1 Max prompt kernels - a select with a device load in the… | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2345 | open | fix(metal): prompt buffer fallback when arena chunks are too small (#2323) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2343 | open | MCP: a picture a tool returns reaches the model as a picture (#2130) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2340 | open | sycl mmvq: Q6_K decode kernels no longer ALU-bound; Q6_K output heads as aligned planes; int4 activation loads | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2338 | open | metal: keep IQ3_XXS resident on 64 GB Macs (wrap a model bigger than maxBufferLength as overlapping buffers) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2337 | open | sycl serve: STRATA_SYCL_MEMLOCK=0 leaves out --ulimit memlock=-1 (unprivileged LXC, rootless Docker) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2336 | open | sycl serve: opt-in non-root engine container and read-only data mount | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2335 | open | sycl serve: keep STRATA_API_KEY out of the engine container | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2332 | open | Add community benchmarks: RTX 4090 Laptop, IQ3_XXS and IQ3_S, default vs calibrated | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2328 | open | Community benchmark: Tesla P40 (Windows Server 2022 VM), four models at 256K context | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2327 | closed | qwen35: widen F16 by hand so upload_f32 builds on CUDA 12.1 | qwen35 build fix for CUDA 12.1 (closed) - builds here with CUDA 12.x already |
| #2326 | open | batch-groups: read a cold prompt beside the decoding slots, stall 11.9 -> 0.8 s, same work 6.4% sooner (opt-in | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2325 | open | --batch: keep system-prompt roots apart from the conversations (--prompt-cache-roots, opt-in) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2321 | open | route-resident: support 256-expert top-8 models (Qwen3.6-35B-A3B) | route-resident in verify.cpp (Flash-Next decode) - strata-q36 routes with native_router_topk |
| #2320 | open | bench: Unsloth UD-Q4_K_XL two-host RTX 5090/3090 rerun and quality checks | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2319 | open | SYCL verify: batch windows keep the doorbell nodes under the lean window ("parallel" crashed on a B70) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2318 | open | KV cache: FP16 sliding window + attention sinks, and an optional 2-bit K / 4-bit V cache scheme (--kv q2k) [EX | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2317 | open | Monitor: one line per card on a layer split (load, VRAM, temperature, power, PCIe) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2312 | open | fix(calibrate): keep configured vision resident during tuning | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2311 | open | web: default temperature 1.0, as the model card and generation_config.json say | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2308 | open | docker: complete deployment bundle with a management console | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2304 | open | setup: write the gfx1103 exact speed switches for the opt-in install | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2303 | open | hip: a hipBLASLt tuning table for gfx1103 (Radeon 780M) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2302 | open | hip: gfx1103 gets the native dot4 and the gfx11 WMMA kernels | another backend (SYCL / HIP / Metal / Vulkan) |
| #2300 | open | bench: add hosted Codex quality baseline for the quant study | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2299 | open | prefill: STRATA_GDN_CHUNKED=1 on gfx11 WMMA (opt-in; gfx1151: recurrence 1.6x, a warm 4K-128K prompt 1.1-2.5%) | another backend (SYCL / HIP / Metal / Vulkan) |
| #2298 | open | community benchmark: RTX 4080 SUPER, Qwen3.8-Flash-Next IQ2_XS | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2296 | open | bench: run under UTF-8 and read CUDA from new nvidia-smi headers | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2294 | open | web: an optional interface language (zh-CN first), English unchanged | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2290 | open | Anthropic API: cache_creation_input_tokens, the prompt tokens read now and kept for the next request | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2288 | open | tools: validate GGUF headers and tensor extents | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2287 | open | setup: resume downloads when HEAD omits Content-Length | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2286 | open | fix(vision): retain image ownership through request preparation | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2285 | open | fix(serve): preserve late instruction roles across adapters | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2284 | open | fix(serve): preserve declared tool schemas through streamed aliases | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2283 | open | setup: opt-in gfx1010 (RX 5700 XT) plus a community bench | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2280 | open | bench: community report, RTX PRO 6000 Blackwell Workstation Edition 96 GB (sm_120), IQ3_S fully resident | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2278 | open | serve: add the legacy /v1/completions endpoint for raw-prompt clients (#2277) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2276 | open | bench: community report, RX 9060 XT (gfx1200) with 16 GB of RAM and a SATA SSD, Coder IQ1_M at 128K on 0.1.43. | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2273 | open | serve: an AMD card's load, VRAM, temperature and power on Windows | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2272 | open | Monitor: the PCIe traffic from the bytes the engine moved, not from a driver counter | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2271 | open | --shared-expert-arena: one engine loads the file, the others wait and attach | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2265 | open | Conversation cache: --conversation-cache-park-gain N (opt-in, default 0): a turn that grew a few hundred token | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2258 | open | Add opt-in browser model downloads and switching through official setup | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2254 | open | Add openai_logprobs/: OpenAI-API logprobs + allowed_token_ids for AnyJev, with a bilingual CN/EN launcher | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2253 | open | serve: the RAM gate releases the heap and stops evicting when it cannot help (rebased #1394) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2252 | open | docs: add Korean README translation | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2249 | closed | Add gentoo's gcc-config to pick proper gcc for CUDA compile | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2247 | open | --session-state-codec bf16: a saved/parked conversation's running state at half the bytes | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2243 | open | drop-file-cache: --drop-file-cache returns the model files' page cache after the startup reads (opt-in) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2242 | open | numa-placement: --numa-hot-node N - keep the GPU-side host pools on the GPU's NUMA node (Linux, dual-socket, o | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2240 | open | Remote stage: a worker for the SYCL build (Intel Arc), stacked on #1748 / #1887 (protocol 7); STRATA_LOGPOS_ID | another backend (SYCL / HIP / Metal / Vulkan) |
| #2238 | open | test: exercise uncached BPE paths in tokenizer parity checks | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2237 | open | bench: community report, RX 6900 XT (gfx1030) x1 and x2 on 0.1.43.1 - IQ3_S and UD-Q4_K_XL, stock and with the | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2234 | open | Batch decode: one launch for all rows in QSA attention and GDN recurrence (opt-in) | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2233 | open | docs(bench): RTX 3060 12GB / 32GB RAM Coder IQ1_M with Budget12 64K profile | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2231 | closed | HIP gfx1100: bit-exact decode and prompt-path speedups (+12% decode, KL 0) | another backend (SYCL / HIP / Metal / Vulkan) |
| #2230 | open | [HIP] Keep mixed-gfx prefill GEMM modes per device (#690, #703) | another backend (SYCL / HIP / Metal / Vulkan) |
| #2229 | open | sycl: enable vision (--vision cpu) on intel | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2226 | open | STRATA_HC_PERSIST=1 (opt-in, HIP): the verifier's hyper-connection read as one persistent kernel (gfx1030: rea | another backend (SYCL / HIP / Metal / Vulkan) |
| #2224 | open | pcie: the measured PCIe share default (PcieAuto) on gfx1100 too — gate-measured on a native pack | another backend (SYCL / HIP / Metal / Vulkan) |
| #2223 | open | verify: STRATA_NO_PCIE_GROUP=1 drops the empty PCIe-share stage with --pcie-frac 0 | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2222 | open | sm_75: fewer launches per decode layer: plan wait folded, fused quantize + interleave (bitwise) | sm_75 launch folding in the Flash-Next decode layer - not strata-q36 |
| #2221 | open | sm_75: Q2_0 matvec with one warp per four rows (bitwise) | sm_75 Q2_0 matvec - Q2_0 not in Qwen3.6 files |
| #2220 | open | sm_75: QSA decode attention for INT8 KV off the shared-memory pipe (bitwise) | another NVIDIA architecture's kernels |
| #2217 | open | Count clean active file cache as reclaimable under a cgroup limit | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2216 | open | bench: community report, one Arc A770 16 GB (xe) - main on the card and the A770 pull requests, same-day A/B | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2215 | open | cpu: one token's IQ3_XXS/IQ3_S/IQ2_S gate/up rows in ggml's exact bits, faster (same output; STRATA_CPU_SINGLE | CPU IQ rows - IQ types and the CPU pool, not used here |
| #2209 | closed | CUDA: keep CPU expert servicing active during instrumented graph launches | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2208 | open | docs/STRIX_HALO.md: STRATA_PF_SWITCH_MIN_T compares the chunk, not the prompt (a 4,096-token prompt is a 4,089 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2207 | open | sycl/A770: wide MMVQ kernels at 2-8 columns in large-GRF mode on Arc A-series (bit-identical; 1.3-7.4x kernel, | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2206 | open | bench: community report, 2x RTX 3090 24 GB (NVLink), IQ3_S, 204K, eng… | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2205 | open | sycl: with --expert-parallel-device, card N computes its own experts of each prompt chunk | another backend (SYCL / HIP / Metal / Vulkan) |
| #2204 | open | Monitor: token usage and cost per day, priced, with the folder's models merged or shown one by one | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2203 | closed | sycl/A770: fused GR read with more, smaller work-groups on Arc A-series (same sums; decode +3-5%; on top of #2 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2201 | closed | --vram-elastic with --kv-grow: VRAM takes its chunks where the K/V grows into | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2199 | closed | sycl: prompt GDN recurrence with one barrier per token on Alchemist (bit-identical; A770 kernel 2.1x, prompt + | another backend (SYCL / HIP / Metal / Vulkan) |
| #2198 | closed | Add Nix/OS install story | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2196 | closed | serve: the image encoder only while there are pictures ("elastic": true in the vision section) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2195 | closed | setup --update: an MTP folder shared by several configs keeps the draft subset of the most recently used one ( | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2193 | closed | sycl: fused GR read without the private GrMulti copy (GR_PICK); large GRF on Arc A-series only - A770 decode 5 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2191 | closed | --prefill-head-alias (opt-in, --serve, one GPU): the output head's VRAM holds the prompt buffers while a promp | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2190 | closed | sycl: size the expert cache from the DRM fdinfo when xe reports 0 free card memory (Arc A770, #1545) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2189 | closed | HIP: virtual memory for --kv-grow and --vram-elastic (#533) through vmm.hpp | another backend (SYCL / HIP / Metal / Vulkan) |
| #2187 | open | RTX 4070 ti 12GB / 96GB RAM Bench | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2185 | closed | tools/windows: enable_large_pages.ps1 helper to grant SeLockMemoryPrivilege | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2184 | closed | bench: a 15-question checkable quality suite for head-to-head comparisons | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2183 | closed | Windows HIP: dynamically resolve hipKernelNameRefByPtr and map auto PCIe mode to DMA | another backend (SYCL / HIP / Metal / Vulkan) |
| #2179 | closed | core: read the model's geometry from its GGUF metadata and refuse a shape the kernels are not compiled for | geometry checks (closed) - no speed change |
| #2178 | closed | sm_61 (Tesla P40): a wide dequant kernel, expert mode 8 and its matvec table by default (bitwise), the dense G | sm_61 (P40) tables (closed) - other arch |
| #2176 | closed | STRATA_ROUTE_PRIOR_MAP: the route prior per layer (opt-in); off in layers 0-11 keeps the speed with KL 0.141 - | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2175 | closed | serve: "env_to_user" (opt-in) moves opencode's <env> block into the first user message (new session's first re | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2174 | closed | pool: STRATA_POOL_UMWAIT - parked workers sleep with UMONITOR/UMWAIT instead of spinning (opt-in, -3.1% ms/win | pool UMWAIT (Intel instruction; this host is a Ryzen) |
| #2173 | open | Community benchmark: RTX 5070 Ti 16 GB, Threadripper 7960X, 64 GB RAM (IQ2_XS, IQ3_XXS, IQ3_S) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2172 | closed | setup: install the CUDA Toolkit on Debian 12/13 too, not Ubuntu only | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2170 | closed | serve: fix a race that makes a long prompt read give way to a short request one chunk late, or never; its TTFT | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2168 | open | serve: refuse a port another program holds on 127.0.0.1, and say which program | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2166 | open | bench: RTX 5090 + RTX 3090 two-host stage-RPC UD-Q4_K_XL results | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2165 | open | bench: RTX 5090 + Ryzen 7 9800X3D, Qwen3.8-Flash-Next IQ3_XXS @ 131k (Strata 0.1.43.1) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2164 | open | web: runtime i18n layer (English + Chinese), zero markup changes | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2163 | open | bench: 2x RTX 3080 Ti 12 GB community results (Flash-Next Q4_K_M abliterated, 196K ctx, layer split) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2161 | open | setup: an interrupted engine update is tried again instead of skipped for good (#2159) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2160 | open | Community benchmark: RTX 3090, Qwen3.8-Flash-Next IQ3_S | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2158 | open | Benchmark results for RTX 3080 Ti and Core Ultra 7 270K Plus | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2156 | open | bench: RTX 4090 community results (flash-next UD-IQ4_XS 204K/131K, Swift 1.5 IQ3_S) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2155 | open | serve: answer GET /models/sse (llama.cpp's model status feed) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2153 | open | docs: the xe expert tier has no SSD fallback, and --spec 0 is refused on native packs (#1796) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2152 | open | hip(RDNA4): a parity-gated fused-GR register-half down kernel for gfx1200/gfx1201 decode | another backend (SYCL / HIP / Metal / Vulkan) |
| #2151 | open | SYCL: preserve the borrowed default queue during teardown | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2150 | open | SYCL: opt in to 256 GRFs for fused hyperconnection up | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2148 | open | hip: add a gfx1150 hipBLASLt 100500 tuning table (Radeon 890M / Ryzen AI 9 HX 370) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2147 | open | setup: opt-in gfx1150 (Radeon 890M / 880M, Strix Point) as a unified-memory AMD card | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2146 | open | sm_75: the v2 prompt attention in one stage, three blocks per SM (bitwise, attention 1.66x; stacked on #2092) | another NVIDIA architecture's kernels |
| #2145 | open | prefill: the fused int8 prompt experts on Turing (sm_75), opt-in (+9 % prefill on RTX 2080 Ti) | another NVIDIA architecture's kernels |
| #2144 | closed | bench: community report, 2x Tesla P40 (PCIe 3.0 x8), i5-6500, 39 GiB RAM, IQ3_S, 131K, 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2142 | open | Community benchmark: mixed dual V100 + RTX PRO 4000 Blackwell on Windows | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2140 | open | serve: honour llama-server's thinking_budget_tokens (a llama.cpp client's thinking budget was silently ignored | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2139 | open | Community benchmark: RTX 3060 on Windows, Strata 0.1.44.1 calibrated settings through 260K input | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2136 | closed | setup: reject conflicting --calibrate options (#2036) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2135 | closed | bench: warn about expert profile contamination (#2088) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2132 | open | Community benchmark: Radeon AI PRO R9700 (gfx1201), ROCm 7.14.1, Ryzen 9 3900 (AVX2) — UD-IQ4_XS and the opt-i | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2131 | open | serve: the prompt path's loan has room for a short request that keeps 0.1.39's ring (#2090) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2122 | open | gitignore: local per-machine root files | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2119 | open | Add disk-backed conversation cache parking (--conversation-cache-storage disk) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2116 | open | Community benchmark: RTX 5090 D v2 (24 GB), IQ3_S (abliterated), engine 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2115 | open | setup: the SYCL setup on an integrated Arc (xe): native launcher, a RAM fit rule, no KV streaming | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2114 | open | sycl: run on an integrated Arc (Arc 140T, xe): free memory from /proc/meminfo, no prompt-slot borrowing, a nat | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2112 | open | Community benchmark: RTX 4060 Ti 16 GB, Coder IQ1_M | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2111 | open | strata_bench: read CUDA UMD version, storage and GPU power/PCIe detail | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2110 | open | sycl: STRATA_WINDOW_SYNC=1 by default on an integrated GPU (Arc 140T: 10 of 10 requests stalled, 20 of 20 pass | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2109 | open | --batch: request priority (a free slot and the next prompt read go to the highest priority waiting) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2106 | open | sycl verify: complete STRATA_VERIFY_EAGER for the commit path | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2105 | open | feat(mtp): support native Q4_0 and Q8_0 experts | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2103 | open | docs/TEST_REQUESTS: an Arc on Windows with and without the Level Zero platform (#1549) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2100 | open | bench: add RTX 3090 community benchmarks | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2098 | open | disk park: the profile tag carries the engine version, and the blob carries the tag | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2094 | open | Draft: immutable branching contexts with validated restore and serial benchmarks | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2093 | open | sm_75: the prompt path's top-k with one full read of the scores (identical ids) | another NVIDIA architecture's kernels |
| #2092 | open | sm_75: the v2 int8 prompt attention with register staging instead of cp.async (STRATA_PA_V2T=1) | another NVIDIA architecture's kernels |
| #2091 | open | sm_75: the hyper-connection's xn16 written as the FP16 its tensor-core GEMMs read (bitwise, prompt +3 %) | another NVIDIA architecture's kernels |
| #2089 | closed | CUDA sm_75: interleaved 2-4 column GEMV with a Turing rows table (CMP 50HX), opt-in STRATA_SM75_TABLE=1 | sm_75 GEMV table (closed); Volta's own table (STRATA_SM70_TABLE, PR 1401) measured 93.1 vs 94.3 tok/s |
| #2087 | open | Community benchmark: RTX 3090 / Qwen3.8-Flash-Next-UD-Q4_K_XL / Strata 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2086 | open | Community benchmark: RTX 3090 + RTX 5070 + RTX 5070 Ti layer split (Q2_0) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2083 | open | bench: community report, RTX 5060 Ti 16 GB + RTX 3060 12 GB unequal layer split (PCIe gen5 x8 + gen3 x4), IQ3_ | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2082 | open | fix: make conversation parking admission fail safely | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2081 | open | dequant: the prompt projections' BF16 dequant with 16-byte stores on sm_75 (bit-exact; STRATA_TC_SM75=0 keeps  | another NVIDIA architecture's kernels |
| #2080 | open | qsa_select: the FP32 tiled block scores below sm_80 (PR #1091) on by default; STRATA_SELECT_SIMT=0 keeps the w | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2079 | open | Add native Windows support for Intel SYCL GPUs | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2078 | open | prefill: refill the lent expert slots from the GGUF mapping's pieces (opt-in STRATA_REFILL_PARTS=1; gfx1151 ta | another backend (SYCL / HIP / Metal / Vulkan) |
| #2077 | open | serve: --prompt-cache-file <path>, the long prompt prefix saved to disk and restored across restarts (opt-in) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2076 | open | bench: 2x RTX 5060 Ti + i7-12700, IQ3_XXS high reasoning on 0.1.42 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2075 | open | QSA decode attention: PR #540's two-cells-per-step chunk kernel on every CUDA card and a bounded merge (STRATA | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2073 | open | PLE: --ple-io direct stays direct when the start-up probe measures the disk slow (STRATA_PLE_AUTO_MMAP=1 opts  | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2072 | open | sm_75: the decode QSA selection at long context, a halving scorer and a split top-k (bitwise, decode +6-12 % a | another NVIDIA architecture's kernels |
| #2071 | open | sm_75: a faster tiled FP32 block scorer (STRATA_SELECT_SIMT=2), and the case for SIMT on RTX 20 by default | another NVIDIA architecture's kernels |
| #2070 | open | sm_75: a hyper-connection up kernel for 3+ tokens and the staged down kernel in token pairs (bitwise) | another NVIDIA architecture's kernels |
| #2069 | open | sm_75: Turing tables for the decode matvecs and the grouped experts (bitwise, decode +6 %) | sm_75 tables / expert mode 4: mode 4 measured on the V100 via STRATA_EXP_MODE=4 - 99.6 / 84.4 tok/s, noise; mode 8 (Volta's) 99.8 / 84.8, noise |
| #2066 | open | bench: dual Quadro RTX 6000 results, upgrade A/B and NUMA comparison | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2064 | open | Community benchmarks: RTX PRO 1000 Blackwell 8 GB, Windows, Coder IQ1_M and IQ2_XS, Strata 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2063 | open | Add Windows ARM64 RTX Spark support | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2060 | open | Linux AMD: images on the GPU through Vulkan, on a chosen card or the integrated GPU (vision.device) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2059 | open | QSA decode: the window's block scores with the queries in registers and six shuffles per block (STRATA_QSA_ROW | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2058 | open | Verify window: the routed experts' gate/up with the group's q8_1 rows in shared memory (STRATA_EXPDEC=1, opt-i | tested on v100 (STRATA_EXPDEC=1): 99.1 / 84.4 tok/s - noise, same tokens, not taken |
| #2057 | open | serve: an opt-in "stream_all_tool_args" sends a tool call's non-string argument in pieces as the model writes  | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2056 | closed | hip: gfx103x creates the MTP drafter's side stream again (RX 6800, Windows: 7.9K prompt 584–596 → 798–808 tok/ | another backend (SYCL / HIP / Metal / Vulkan) |
| #2053 | open | Community benchmark: RTX 5070 Ti 16 GB / 64 GB RAM, v0.1.43 IQ2_XS and IQ3_XXS | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2050 | open | batch-groups: a cold prompt read no longer pauses every stream for its whole length, longest pause 12.2 -> 4.2 | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2048 | open | bench: 4× V100-16GB Q4_K_XL (sm_70 MMQ + bandwidth-proportional layer split) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2047 | open | Community benchmark: Radeon AI PRO R9700 + Ryzen 7 5700X (AM4), IQ3_S, HIP + WMMA, engine 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2045 | open | bench: RTX 3070 MTP control-decision latency and break-even | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2043 | open | serve: fix potential race in NoLeakedSampler during test runs | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2042 | open | STRATA_EXCHANGE_ROTATE: rotation for packs whose layers differ in expert block size (spares kept per size) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2041 | open | setup: agentionai's Signal 3.8 Flash Next as --family signal (AP-Q4_K_XL, AP-IQ4_XS), images with its own enco | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2039 | open | serve: preserve literal </tool_call> in prompts | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2035 | open | Add opt-in Windows Tray with managed lifecycle, profiles and updates | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2032 | open | community bench: 2x RTX 2080 Ti (sm_75, NVLink unused), IQ3_S, 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2031 | open | Add Q4_1 PLE dequantization support | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2030 | open | bench: RX 9060 XT 16 GiB, Flash-Next IQ3_S at 262K, engine 0.1.43 (Linux, HIP gfx1200) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2025 | open | docs: use Strata as the backend for ACP-capable editors (Zed / JetBrains) via acp-bridge | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2024 | open | bench: RTX 5090 desktop 1M-agent — 0.1.42 vs 0.1.43 interleaved A/B (overlap default pays +1.2/+2.4% decode; # | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2023 | open | Video input: bounded preparation progress and cancellation follow-up | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2022 | open | bench: community reports - RTX 2000 Ada 16 GB, Flash-Next IQ3_XXS at 262K, 0.1.42 and 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2020 | open | prefill: gfx11 F16 matrix-core routed experts (opt-in STRATA_PF_F16=1): UD-Q4_K_XL 32K +15.6%, UD-IQ4_XS 32K + | another backend (SYCL / HIP / Metal / Vulkan) |
| #2019 | open | prefill: q's split + norm + rope as one kernel (STRATA_QPREP=1); gate_attn in the prompt attention's epilogue  | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #2018 | open | prefill: VGPR-capped gfx11 fused expert kernels back on for gfx1151 on clang 23 (same bits, 2-15% less time pe | another backend (SYCL / HIP / Metal / Vulkan) |
| #2017 | open | prompt attention (gfx11 / gfx12 WMMA): issue a chunk's reads before its compute | another backend (SYCL / HIP / Metal / Vulkan) |
| #2016 | open | prefill: the QSA top-k radix digit by a cooperative scan (same ids; gfx1151 -5% to -49% at 16K-128K, RTX 3060  | another backend (SYCL / HIP / Metal / Vulkan) |
| #2015 | closed | Draft: immutable branching contexts with serialized execution and shared checkpoints | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2012 | open | Community benchmark: RTX 5090 + RTX 3090 (IQ3_S, 131K ctx, engine 0.1.43) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2011 | closed | STRATA_ADAPT_OVERLAP: opt-in on every backend (RTX 5090 Laptop, Linux CUDA: 84.6 vs 106.6 tok/s decode median  | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2009 | open | Add experimental Linux gfx900 setup with wave64 validation | another backend (SYCL / HIP / Metal / Vulkan) |
| #2008 | open | Community benchmark: 2x RTX 5060 Ti 16 GB (layer split), IQ2_XS, 262K, engine 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2006 | open | bench: RTX 3080 Laptop 16 GiB, Flash-Next Q2_0 on Windows | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2005 | open | Community benchmark: RTX 3080 Ti 12 GB, KVM guest with GPU passthrough, Docker (engine 0.1.42) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2004 | open | Community benchmark: RTX 4060 8 GB + 32 GB RAM (Coder IQ1_M, Flash-Next IQ2_XS low-RAM) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2003 | open | Docs/0.1.41 otel traces | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #2002 | open | sycl setup: name the Arc Pro B65 (8086:e222) in the Intel id table | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #2000 | open | bench: community report, RTX 3060 + RTX 4060 Ti (Windows, layer split), IQ3_XXS on 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1998 | open | bench: community suite 1 (--long) on RX 9070 XT, engine 0.1.43, IQ3_S | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1992 | open | calibrate: measure the shared-expert stream fork on AMD (STRATA_SH_STREAM) and keep it per PC (#1272) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1990 | open | Add opt-in external llama.cpp token engine | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1989 | open | STRATA_HC_HOST=1: hyper-connection down/up pairs in pinned host memory (opt-in, CUDA) | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #1987 | open | Community benchmark: RTX 4090 D 48 GB + Flash-Next IQ3_XXS (0.1.43), 4K/16K/32K fresh 2,314/3,333/3,528 tok/s | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1986 | open | Add community benchmark: Quadro RTX 8000 48 GB, UD-Q4_K_XL, 131K (v0.… | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1983 | closed | Linux CUDA: warm adaptive-overlap callbacks before host pinning (opt-in) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1981 | open | bench: community report, NVIDIA TITAN RTX 24 GB, IQ3_S at 262K, engine 0.1.43 (BF16_TC + SELECT_SIMT) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1979 | open | docs: Huihui abliterated UD-IQ4_XS, measured against Unsloth's and both OrcaRouter files | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1978 | open | bench/results: gfx906 IQ3_XXS (2x Instinct MI50 / Radeon VII), and the same run with --mmap-experts | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1976 | open | bench: RTX 5090 Laptop GPU follow-up for 0.1.43 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1974 | open | tools/strata_bench.py: keep the ROCm driver read on the ROCm path | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1973 | open | serve/web: UI translations for 12 more languages and a language picker (builds on the en / zh-CN patch from #1 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1972 | open | bench: community report - RTX 3070 8 GB, Ryzen 9 5900X, 96 GB: agent-coding runs, quantization and thinking ef | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1970 | closed | Docs/security review 2026 10 11 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1969 | open | Community rtx pro 4500 blackwell 600k | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1967 | open | Community benchmark: RTX 5080, Ryzen 7 7800X3D, 64 GB RAM (Windows) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1966 | open | serve: continue a final assistant message instead of restating it (#1950) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1965 | open | fix(bench): prevent UnicodeDecodeError on Windows | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1963 | closed | setup: --draft-vocab latin ships a draft subset for accented Latin-script answers | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1961 | open | Add opt-in experimental DFlash2 runtime | DFlash2 runtime for the Flash-Next path - not strata-q36 |
| #1959 | open | serve: --prompt-cache-spread, checkpoints kept spread back from the conversation's end (opt-in) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1955 | open | Tool calls: a boolean parameter written as True/False is a boolean | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1953 | open | serve: the keep-alive while a prompt is read is a real SSE frame, not a comment | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1949 | open | cpu(bug): the AVX2 dot kernels keep their accumulators in registers (2.1x router; +20-24% expert rows at 4 tok | CPU AVX2 dots for the hybrid CPU pool - not used when every Qwen3.6 expert is in VRAM |
| #1948 | open | Add files via upload | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1947 | open | bench: ROG Flow Z13 Windows HIP, Huihui IQ4_XS 256K tuning and placement | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1946 | open | fix(vision): scale GPU warm-up for image caps above 4096 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1943 | open | hip: opt-in exact gfx1100 QSA scorer for long prefill | another backend (SYCL / HIP / Metal / Vulkan) |
| #1942 | open | serve: the image encoder can run on a second PC over ssh (#1404) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1940 | open | Add Arc native Windows support | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1939 | open | setup: write run-<model>.bat and the build .bat with single CRLF line ends | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1937 | open | serve: parked conversations on the disk, optionally across restarts (--conversation-disk-dir, --conversation-d | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1934 | open | mtp_pack: Q2_0 scale search over 0.15 to 1.0 x max/w/, either sign | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #1933 | closed | serve: aggregate batch request timings | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1931 | open | fix(hip): add opt-in stability profile for reset reports | another backend (SYCL / HIP / Metal / Vulkan) |
| #1927 | open | bench: community report, RX 6900 XT (gfx1030) x1 and x2 on 0.1.42 - Unsloth UD-Q4_K_XL beside GSQ-RCO IQ3_S, s | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1925 | open | Add optional GPU electricity cost display to Power card | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1924 | open | docs: CONFIGURATION.md reference documentation for all flags and environment variables | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1921 | open | Community benchmark: 5070 Ti + 2x 5060 Ti (one on a USB4 eGPU), UD-IQ4_XS, engine 0.1.42, Ryzen 7 7700X, 46 Gi | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1918 | closed | serve: run Codex's client-side tool_search | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1916 | open | OpenAI logprobs on /v1/chat/completions for the CUDA/HIP engine | another backend (SYCL / HIP / Metal / Vulkan) |
| #1914 | open | bench: community report, RTX 5080 Laptop 16 GB, IQ3_XXS, engine 0.1.42 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1912 | open | --pool-affinity measure: on a hybrid CPU, time the all and p-cores pool layouts at start and take the faster ( | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1911 | open | Expert cache: STRATA_VRAM_FREE=device reads the whole card's free VRAM through NVML (opt-in, Windows + NVIDIA) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1909 | open | kernels: process all groups in fused GU and LDS down launches | tested on v100: 99.5 / 84.2 tok/s vs 99.0-99.7 / 84.1-84.7, same tokens - noise, not taken |
| #1905 | closed | arena: pin by need (STRATA_PIN_BY_NEED=1) and a PCIe share floor (STRATA_PCIE_MIN1=1), +5-7 % prompt read on W | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1904 | closed | adaptive tier: --adapt-async 1 in the arena mode (one GPU, copy-in only), x1.07-1.11 cold decode on an RTX 309 | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1901 | open | test: synchronize the keepawake service fixture | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1899 | open | serve: the thinking budget's continuation keeps the tokens emitted after the STOP, so it is not read again | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1897 | open | setup: build the HIP engine with the K-quant MMQ prompt kernels (UD-Q4_K_XL's experts on RDNA2, 633 -> 1,036 t | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1896 | open | resident RAM mode: a budget keeps the prompt path's lend region when it has room (#1683) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1895 | open | chat: syntax highlighting for fenced code blocks in streamed output | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1894 | open | bench: community report, RX 6900 XT (gfx1030): STRATA_ROUTE_TAIL_SKIP=7 on HIP, decode x1.24 (Unsloth UD-Q4_K_ | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1893 | open | docs: the three things 0.1.42 changed that no page told | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1892 | closed | Non-finite logits (#879): a guard with self-healing | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #1891 | open | logit_bias: normalize a long list once per request (65,929 entries: +175 ms -> +24 ms) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1890 | open | sycl: Windows/OpenCL support for the merged port, and a live free-VRAM figure for the expert cache (#1549) | another backend (SYCL / HIP / Metal / Vulkan) |
| #1889 | open | Verify window: the mixer's input-only work on graph branches (STRATA_SIDE_MIXER=1/2, opt-in) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1888 | open | QSA decode attention: split each query's cells across blocks on the int8 mma path (STRATA_QSA_SPLIT_ATTN=1, op | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #1887 | open | Remote stage: conversation checkpoints across the link (--prompt-cache), stacked on #1748 | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #1886 | open | serve: preserve literal </think> across reasoning parsing | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1883 | open | serve: on Windows the engine and the image encoder start without a console window of their own | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1882 | open | serve: /v1/models also gives the context size as `max_model_len` and `context_length` | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1881 | open | serve: a tool call whose `</tool_call>` is missing ends at its own `</function>` (two calls were merged into o | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1880 | open | setup: the images question now asks where the encoder runs (GPU / CPU) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1879 | open | bench: community report, Tesla V100-SXM2 16 GB (Linux, sm_70), IQ3_S:… | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1877 | open | monitor: check the AMD board name's capacity against the memory the driver reads | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1876 | open | Add opt-in cumulative GPU energy monitoring to the Power card | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1874 | open | docs/UNSLOTH_Q4: the first NVIDIA numbers for UD-IQ4_XS (2x2080 Ti, 46 GB RAM) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1873 | open | Community report: Strix Halo gfx1151 (GMKtec EVO-X2, 128 GB), UD-Q4_K_XL and IQ2_XS at a 262K context | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1871 | open | Dashboard: GPU load, VRAM, temperature, power and PCIe for AMD cards … | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1864 | open | Report pending swap and adaptive join times in serial decode timing | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1862 | open | serve: Ctrl+C prints "Exception ignored in: <_io.TextIOWrapper ...>" from the vision encoder's buffered QUIT | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1859 | open | weights: add an opt-in read-only safetensors source (no engine changes) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1856 | open | Host memory probe: accept the root cgroup v2 group ("0::/", WSL without systemd) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1855 | open | bench: community results on one MI50 32 GB (gfx906), 262,144 context | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1854 | open | Loop guard: catch a young sentence loop, splice medium effort on recovery | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1852 | open | serve: tokenise the prompt from the last shared prefix, not from the start | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1851 | open | Gemini api: add gemini api endpoint like openai and anthropic | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1850 | open | Community report: 10-file quant battery on RTX 5070 Ti 16 GB (ISTA vs Swift vs Gyro-S/M) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1849 | open | bench: RTX 4090 Laptop GPU follow-up, engine 0.1.41, defaults and --calibrate settings (IQ3_S, 131K and 262K) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1848 | open | Seamless CUDA 12 & 13 Docker Support | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1841 | closed | serve: an opt-in VRAM headroom for --expert-cache auto (#1549) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1840 | open | serve: a reply that ends inside its thinking with no answer is continued (#1814) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1839 | closed | fix(serve): preserve literal closing thinking tags | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1838 | open | Resident RAM mode: the RAM copy follows the conversation (`STRATA_RAM_ADAPT`, opt-in) | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1836 | open | serve: a Chinese / English language switch for the web app (adapted to v0.1.44.1) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1834 | open | Community benchmark: 2x RTX PRO 4500 Blackwell, Swift IQ3_XXS, engine 0.1.41 against 0.1.40.1 (single stream t | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1833 | open | Add optional isolated full-TeX chat math rendering | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1832 | open | Add locally bundled MathJax rendering to web chat | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1830 | open | Add locally bundled KaTeX rendering to web chat | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1829 | open | Add shared LaTeX/web math core without a rendering engine | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1828 | closed | sycl: register iq_parity and ple_parity like the top-level build | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1826 | closed | serve: probe the SYCL launcher's engine binary for --tail-role-token | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1825 | open | qsa_select: prompt-path block scores with a thread per query head (below sm_80, same bits) | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #1824 | open | cpu: Q2_0 rows on AVX-only CPUs (q2_avx1.cpp) instead of ggml-cpu's scalar dot | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #1823 | open | tokenizer: a long message's ids remembered whole, on top of the piece cache | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1822 | open | prefill: gate/up MMQ reads the experts in place (CUDA; patched copy of llama.cpp mmq.cuh) | in-place MMQ for prefill.cpp - strata-q36 calls moe_mmq on resident blobs already |
| #1821 | open | qsa_prompt_attn: int8 K/V prefetched in registers and sharing one buffer (Turing / sm_75) | another NVIDIA architecture's kernels |
| #1820 | open | qwen35moe follow-ups: VRAM cap + quality mode, MTP for F16 draft tensors, chunked GDN, 256/8 route-resident | a separate qwen35moe branch (+5,529 lines: VRAM cap, quality mode, chunked GDN) - superseded by strata-q36 in main |
| #1818 | open | fix: make CPU IQ dispatch independent of draft occupancy | kernels strata-q36 does not call (QSA, PLE, hyper-connections, verify, CPU pool) |
| #1817 | closed | sycl: use the source version for setup model checks | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1816 | open | Opt-in OTLP traces of every /v1 request (0.1.41), plus the docs pages 0.1.41 left uncovered | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1813 | open | Add exact speculative-window token parity and coverage gate | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1812 | closed | build: link CUDA MMQ target to the CUDA driver API | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1811 | closed | Community benchmark: RX 9070 16 GB (gfx1201) on Windows 11, IQ3_XXS, engine 0.1.41 | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1810 | open | feat: persist conversation checkpoints with RAM and disk restore | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
| #1808 | open | serve: a queue cap (max_queue) answers 429 + Retry-After instead of queueing without end | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1807 | open | config: forward - POSTs to a path answered by another server (embeddings, reranking) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1806 | open | config: user_args - engine options of the user's own that setup keeps | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1805 | open | setup: install jsonschema, so json_schema answers are checked against their schema | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1804 | open | structured: one-string-field json_schema answered as plain text (opt-in), progress in structured keep-alives | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1803 | open | serve: opt-in guard for a repeated phrase (repeat_stop_period), and name repetition stops (stop_cause) | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1802 | closed | bench: RX 9060 XT IQ2_XS and IQ3_XXS community results | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1801 | closed | fix(setup): discard damaged llama.cpp source archives | no engine code (docs, web, benchmark reports, tools, setup, serve front end) |
| #1800 | open | Support external PLE replacement for a missing split shard | engine plumbing (serve, caches, batching, splits, memory) - no strata-q36 code |
