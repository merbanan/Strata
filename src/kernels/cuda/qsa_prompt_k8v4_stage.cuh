// Opt-in sm75 K8V4 prompt attention. Included inside qsa_prompt_attn.cu's
// anonymous namespace; uses its MMA and swizzle helpers. K codes are staged in
// registers; Q4 V bytes stay packed there until the previous tile is consumed.
// Q4-to-half rounding and p.v arithmetic match the original mode-3 reader.
// q.k work is distributed across warps, retaining mode 3's ordered scale FMAs.
template<bool V_TRANSPOSE>
struct SmemK8V4Stage {
    int8_t kv[1][4][1][CH2][64];
    __half v[4][V_TRANSPOSE ? 72 : CH2][V_TRANSPOSE ? CH2 + 2 : 72];
    float sc[1][4][2][CH2];
    float p[16][CH2 + 1];
    float qmax[4], alpha[16], lsum[16], mrow[16];
};
static_assert(sizeof(SmemK8V4Stage<false>) <= 32 * 1024, "two shared-memory blocks per Turing SM");
static_assert(sizeof(SmemK8V4Stage<true>) <= 32 * 1024, "two shared-memory blocks per Turing SM");

template<bool V_TRANSPOSE>
__global__ void __launch_bounds__(THREADS) prompt_attn_k8v4_stage_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ steps, int n_kv_heads,
                                                                 int page_size, float scale_log2,
                                                                 float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    SmemK8V4Stage<V_TRANSPOSE>& S = *reinterpret_cast<SmemK8V4Stage<V_TRANSPOSE>*>(smem_raw);
    // the partial of (row r, cell c) of warp w: cell group c / 8's 512 bytes of w's K slice, [row][cell % 8]
#define PAK4_PART(w, r, c) reinterpret_cast<const float*>(&S.kv[0][w][0][0][0])[((c) >> 3) * 128 + (r) * 8 + ((c) & 7)]
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;

    // q: the power-of-two prescale over all 12 heads (as v1), then this warp's 64 dims as hi/lo A fragments
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    uint32_t qh[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = make_float2(0.f, 0.f);
            if (row < G) x = *reinterpret_cast<const float2*>(q + (size_t) row * HD + col);
            x.x *= qup;
            x.y *= qup;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    }

    // Pipeline cells two chunks ahead, pool rows and packed data one ahead.
    const int n_chunks = (n + CH2 - 1) / CH2;
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(ids + c) : -1; };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    // Turing (no cp.async): the warp's next pieces wait in registers and are stored into their stage after this
    // chunk's p.v (store_stage). K storage becomes score scratch after consumption.
    uint4 rk[4];
    uint32_t vbits[2][4];
    float vd[2];
    auto store_stage = [&](int st) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            *reinterpret_cast<uint4*>(&S.kv[st][warp][0][0][0] + swz(cell, pc * 16)) = rk[j];
        }
        // Lane owns one cell's two Q4 blocks. Prefetch packed bytes, then round
        // each product to FP16 exactly as launch<3> before storing the V tile.
#pragma unroll
        for (int b = 0; b < 2; ++b) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const unsigned byte = (vbits[b][j / 4] >> (8 * (j % 4))) & 255u;
                const __half lo = __float2half_rn((float) ((int) (byte & 15u) - 8) * vd[b]);
                const __half hi = __float2half_rn((float) ((int) (byte >> 4) - 8) * vd[b]);
                if constexpr (V_TRANSPOSE) {
                    S.v[warp][b * 32 + j][lane] = lo;
                    S.v[warp][b * 32 + j + 16][lane] = hi;
                } else {
                    S.v[warp][lane][b * 32 + j] = lo;
                    S.v[warp][lane][b * 32 + j + 16] = hi;
                }
            }
        }
    };
    auto issue = [&](long long r, int st, float& ksr, float& vsr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = __shfl_sync(0xffffffffu, r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
            (void) st;
            rk[j] = ok ? __ldg(reinterpret_cast<const uint4*>(p.k_q + off)) : make_uint4(0, 0, 0, 0);
        }
        ksr = r >= 0 ? __half2float(__ushort_as_half(__ldg(p.k_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
        vsr = r >= 0 ? 1.0f : 0.0f;
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const block_q4_0* blk = r >= 0
                ? reinterpret_cast<const block_q4_0*>(p.v_q4 + r * (HD / QK4_0) * sizeof(block_q4_0)) + warp * 2 + b
                : nullptr;
            vd[b] = blk ? __half2float(__ushort_as_half(__ldg(&blk->d))) : 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // Q4 blocks are only two-byte aligned; never issue uint4 loads.
                const uint16_t* src = blk ? reinterpret_cast<const uint16_t*>(blk->qs) : nullptr;
                vbits[b][j] = src ? (uint32_t) __ldg(src + 2 * j) | ((uint32_t) __ldg(src + 2 * j + 1) << 16) : 0u;
            }
        }
    };
    float ksn, vsn;
    issue(row_of(cell_of(lane)), 0, ksn, vsn);
    store_stage(0);
    S.sc[0][warp][0][lane] = ksn;
    S.sc[0][warp][1][lane] = vsn;
    long long r_next = row_of(cell_of(CH2 + lane));
    int cell_next2 = cell_of(2 * CH2 + lane);

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int st = 0, nst = 0, c0 = ci * CH2;
        const bool more = ci + 1 < n_chunks;
        if (more) issue(r_next, nst, ksn, vsn);
        r_next = row_of(cell_next2);
        cell_next2 = cell_of((ci + 3) * CH2 + lane);
        __syncwarp();
        const int8_t* K = &S.kv[st][warp][0][0][0];

        // q.k over this warp's 64 dims, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b[2];
                b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig)));
                b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig + 8)));
                mma16816(tg, qh[kk], b);
                mma16816(tg, ql[kk], b);
            }
            __syncwarp();   // every lane has read the group's cells: their 512 bytes take its partials
            float* pp = reinterpret_cast<float*>(&S.kv[0][warp][0][0][0]) + nt * 128;
            pp[gid * 8 + 2 * tig] = tg[0];
            pp[gid * 8 + 2 * tig + 1] = tg[1];
            pp[(gid + 8) * 8 + 2 * tig] = tg[2];
            pp[(gid + 8) * 8 + 2 * tig + 1] = tg[3];
        }
        __syncthreads();
        // Match mode 3's sequential scale FMAs, including rounding at each group.
        // Online softmax: row t/8, four cells per thread.
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                float score = 0.0f;
#pragma unroll
                for (int g = 0; g < 4; ++g)
                    score = fmaf(PAK4_PART(g, r, c), S.sc[0][g][0][c], score);
                x[j] = c0 + c < n ? score * qdown : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this warp's 64 dims (as v1)
        {
            float vmax = S.sc[st][warp][1][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH2 / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                            w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* h = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = j * 8 + gid;
                    uint32_t b[2];
                    if constexpr (V_TRANSPOSE) {
                        b[0] = *reinterpret_cast<const uint32_t*>(&S.v[warp][d][cA]);
                        b[1] = *reinterpret_cast<const uint32_t*>(&S.v[warp][d][cB]);
                    } else {
                        const __half2 h0 = __halves2half2(S.v[warp][cA][d], S.v[warp][cA + 1][d]);
                        const __half2 h1 = __halves2half2(S.v[warp][cB][d], S.v[warp][cB + 1][d]);
                        b[0] = *reinterpret_cast<const uint32_t*>(&h0);
                        b[1] = *reinterpret_cast<const uint32_t*>(&h1);
                    }
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
        // Independent thread scheduling can otherwise let a lane overwrite V
        // or its scales while another lane still reads them for the final MMA.
        __syncwarp();
        if (more) {
            S.sc[nst][warp][0][lane] = ksn;
            S.sc[nst][warp][1][lane] = vsn;
            store_stage(nst);
        }
        __syncwarp();
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(attn + (size_t) gid * HD + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<float2*>(attn + (size_t) (gid + 8) * HD + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

#undef PAK4_PART

bool launch_k8v4_stage(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                       int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    // Admission is restricted to actual sm75 devices; emulation overrides must
    // not select this kernel on Volta or substitute another architecture's tuning.
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess ||
        major != 7 || minor != 5) return false;
    static const bool transpose = [] {
        const char* e = std::getenv("STRATA_PA_K8V4_V_TRANSPOSE");
        return e != nullptr && e[0] == '1';
    }();
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        if (transpose)
            prompt_attn_k8v4_stage_kernel<true><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS,
                sizeof(SmemK8V4Stage<true>), st>>>(q + q0 * s.n_head * HD, pools, ids + q0 * cap,
                steps + q0 * kStepCount, (int) s.n_head_kv, (int) s.page_size, scale_log2,
                attn + q0 * s.n_head * HD, (int) cap);
        else
            prompt_attn_k8v4_stage_kernel<false><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS,
                sizeof(SmemK8V4Stage<false>), st>>>(q + q0 * s.n_head * HD, pools, ids + q0 * cap,
                steps + q0 * kStepCount, (int) s.n_head_kv, (int) s.page_size, scale_log2,
                attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (k8v4 stage): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
