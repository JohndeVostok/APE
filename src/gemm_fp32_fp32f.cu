
#include "cuda_utils.h"
#include "kernel.h"

namespace ape {
void gemm_fp32_fp32f(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc) {
    half *buf, *buf_a, *buf_b;
    cudaSafeCall(cudaMalloc((void **)&buf, sizeof(half) * (m * k + k * n) * 2));
    buf_a = buf;
    buf_b = buf + m * k * 2;

    split_fp32_to_fp16(buf_a, A, m * k);
    split_fp32_to_fp16(buf_b, B, k * n);

    float alpha0 = *alpha, alpha1 = *alpha / 4096.0f, beta0 = *beta, beta1 = 1;
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha0,
                                buf_a, CUDA_R_16F, lda, buf_b, CUDA_R_16F, ldb, &beta0, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha1,
                                buf_a + m * k, CUDA_R_16F, lda, buf_b, CUDA_R_16F, ldb, &beta1, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha1,
                                buf_a, CUDA_R_16F, lda, buf_b + k * n, CUDA_R_16F, ldb, &beta1, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    cudaSafeCall(cudaFree(buf));
}

} // namespace ape