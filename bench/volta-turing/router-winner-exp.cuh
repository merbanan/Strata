// Approximate C02 experiment, never an exact replacement by assumption.
// Select raw finite logits, exponentiate the ten winners, then accumulate
// their probabilities in the same winning lanes as the original router.
// Raw-logit ties can differ from rounded-softmax ties; the probe reports this.
__device__ __forceinline__ void row(const float* logits, int32_t* ids, float* weights,
                                   int n_tok, int tk) {
    if (tk >= n_tok) return;
    const int lane = threadIdx.x;
    float values[16];
    bool nonfinite = false;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = logits[(size_t) tk * 512 + lane + i * 32];
        nonfinite |= !isfinite(values[i]);
    }
    if (__any_sync(0xffffffffu, nonfinite)) {
        row_original(logits, ids, weights, n_tok, tk);
        return;
    }
    ids += (size_t) tk * 10;
    weights += (size_t) tk * 10;
    float maximum = 0.0f, selected_logit = 0.0f;
    int selected_expert = 0;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if (rank == 0) maximum = best;
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
        }
        if (rank == lane) { selected_logit = best; selected_expert = expert; }
    }
    const float probability = lane < 10 ? expf(selected_logit - maximum) : 0.0f;
    float selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        const int expert = __shfl_sync(0xffffffffu, selected_expert, rank, 32);
        const float value = __shfl_sync(0xffffffffu, probability, rank, 32);
        if ((expert & 31) == lane) selected_sum += value;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = probability * inverse_selected_sum;
}
