#include "ape.h"
#include "common.h"
#include "kernel.h"

namespace ape {
void test_gemm_fp32(int m, int n, int k, ape::ApeAlgo algo) {
    int width;
    switch (algo) {
    case APE_ALGO_AUTO:
        width = 8;
        break;
    case APE_ALGO_FP32F:
        width = 4;
        break;
    case APE_ALGO_FP32B:
        width = 6;
        break;
    case APE_ALGO_FP32T:
        width = 8;
        break;
    default:
        width = 0;
    }
    apeInit((m * k + k * n) * width);
    float *data_eval_a = 0, *data_eval_b = 0, *data_eval_c = 0;
    cudaSafeCall(cudaMalloc((void **)&data_eval_a, m * k * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&data_eval_b, k * n * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&data_eval_c, m * n * sizeof(float)));
    double *data_res_a = 0, *data_res_b = 0, *data_res_c = 0;
    cudaSafeCall(cudaMalloc((void **)&data_res_a, m * k * sizeof(double)));
    cudaSafeCall(cudaMalloc((void **)&data_res_b, k * n * sizeof(double)));
    cudaSafeCall(cudaMalloc((void **)&data_res_c, m * n * sizeof(double)));

    curandGenerator_t gen;
    curandSafeCall(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandSafeCall(curandGenerateUniformDouble(gen, data_res_a, m * k));
    curandSafeCall(curandGenerateUniformDouble(gen, data_res_b, k * n));
    curandSafeCall(curandGenerateUniformDouble(gen, data_res_c, m * n));

    ape::convert_fp64_to_fp32(data_eval_a, data_res_a, m * k);
    ape::convert_fp64_to_fp32(data_eval_b, data_res_b, k * n);
    ape::convert_fp64_to_fp32(data_eval_c, data_res_c, m * n);

    double alpha_res = 1.0, beta_res = 0;
    ape::apeGemmFP64(ape::APE_TRANS_N, ape::APE_TRANS_N, m, n, k, &alpha_res, data_res_a, m, data_res_b, k, &beta_res,
                     data_res_c, m, ape::APE_ALGO_CUBLAS);
    float alpha_eval = 1.0f, beta_eval = 0.0f;
    ape::apeGemmFP32(ape::APE_TRANS_N, ape::APE_TRANS_N, m, n, k, &alpha_eval, data_eval_a, m, data_eval_b, k, &beta_eval,
                     data_eval_c, m, algo);
    double max_error, mean_error;
    ape::compare_fp32_to_fp64(data_eval_c, data_res_c, m * n, max_error, mean_error);

    float duration = 0;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    for (int i = 0; i < 128; i++) {
        ape::apeGemmFP32(ape::APE_TRANS_N, ape::APE_TRANS_N, m, n, k, &alpha_eval, data_eval_a, m, data_eval_b, k, &beta_eval,
                         data_eval_c, m, algo);
    }
    cudaEventRecord(st, 0);
    for (int i = 0; i < 128; i++) {
        ape::apeGemmFP32(ape::APE_TRANS_N, ape::APE_TRANS_N, m, n, k, &alpha_eval, data_eval_a, m, data_eval_b, k, &beta_eval,
                         data_eval_c, m, algo);
    }
    cudaEventRecord(ed, 0);
    cudaEventSynchronize(st);
    cudaEventSynchronize(ed);
    cudaEventElapsedTime(&duration, st, ed);
    double perf = double(m) * double(n) * double(k) * 2.0f * 128.0f / duration / 1e9;

    std::cout << "[TEST] " << getApeAlgoName(algo) << " (" << m << " " << n << " " << k << ") max_error: " << max_error
              << " mean_error: " << mean_error << " perf(TFLOPS): " << perf << std::endl;
    cudaFree(data_eval_a);
    cudaFree(data_eval_b);
    cudaFree(data_eval_c);
    cudaFree(data_res_a);
    cudaFree(data_res_b);
    cudaFree(data_res_c);
}

} // namespace ape
