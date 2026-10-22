// C02 standalone experiment. Sort each lane's 16 probabilities once, then
// merge the 32 lane-local lists. Full softmax precedes this function; keep
// winning-lane accumulation and normalization identical to route_multi.
template<int Width, int Distance>
__device__ __forceinline__ void sorted_stage(float (&values)[16], int (&experts)[16]) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        constexpr int distance = Distance;
        const int peer = i ^ distance;
        if (peer > i) {
            const bool first = values[i] > values[peer] ||
                               (values[i] == values[peer] && experts[i] < experts[peer]);
            const bool exchange = (i & Width) == 0 ? !first : first;
            if (exchange) {
                const float value = values[i]; values[i] = values[peer]; values[peer] = value;
                const int expert = experts[i]; experts[i] = experts[peer]; experts[peer] = expert;
            }
        }
    }
    if constexpr (Distance > 1) sorted_stage<Width, Distance / 2>(values, experts);
}

__device__ __forceinline__ void sorted_selection(float (&values)[16], int32_t* ids,
                                                float* weights, int lane) {
    int experts[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) experts[i] = lane + i * 32;
    sorted_stage<2, 1>(values, experts);
    sorted_stage<4, 2>(values, experts);
    sorted_stage<8, 4>(values, experts);
    sorted_stage<16, 8>(values, experts);
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = experts[0];
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            ids[rank] = expert;
            selected_sum += best;
#pragma unroll
            for (int i = 0; i < 15; ++i) {
                values[i] = values[i + 1];
                experts[i] = experts[i + 1];
            }
            values[15] = -INFINITY;
            experts[15] = 0x7fffffff;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
