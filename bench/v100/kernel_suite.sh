#!/bin/bash
# bench/v100/kernel_suite.sh BUILD_DIR: the kernel parity checks and benches that cover strata-q36's kernels, on this
# card.  BUILD_DIR: a build configured with -DSTRATA_BUILD_TESTS=ON (sm_70: -DSTRATA_EXPERIMENTAL_SM60=ON).  iq_parity
# needs its fixtures first: PYTHONPATH=<llama.cpp>/gguf-py python tools/iq_fixture.py --out BUILD_DIR/logs/iq_fixture
cd "${1:?usage: kernel_suite.sh BUILD_DIR}"
for t in native_grouped_parity native_multi_parity mmvq_il_parity mmvq_multi_parity gdn_parity gdn_rec_parity iq_parity \
         iq_multi_parity quantize_act_parity router_top10_parity router_dot_parity rope_parity sampler_parity \
         gemm_f32_parity gemm_bf16_parity shared_expert_parity elementwise_parity bf16_gemv_parity q8_gemv_bench \
         mmq_expert_bench; do
  echo "== $t"; timeout 600 ./$t 2>&1 | tail -3; echo "exit ${PIPESTATUS[0]}"
done
echo "== mmvq_il_parity --bench"; timeout 600 ./mmvq_il_parity --bench 2>&1 | tail -6
