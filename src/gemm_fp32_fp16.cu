
#include "cuda_utils.h"
#include "kernel.h"

namespace ape {
void gemm_fp32_fp16(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                      const float *B, int ldb, const float *beta, float *C, int ldc) {
    
    half *half_A = (half*)ape_buffer, *half_B = half_A + m * k * 2;
    //cudaSafeCall(cudaMalloc((void**) &half_A, sizeof(half) * m * k * 2));
    //cudaSafeCall(cudaMalloc((void**) &half_B, sizeof(half) * k * n * 2));

    split_fp32_to_fp16(half_A, A, m*k);
    split_fp32_to_fp16(half_B, B, k*n);

    float alpha0 = *alpha, alpha1 = *alpha / 4096.0f, beta0 = *beta, beta1 = 1;
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha0, half_A, CUDA_R_16F, 
        lda, half_B, CUDA_R_16F, ldb, &beta0, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha1, half_A + m*k, CUDA_R_16F, 
        lda, half_B, CUDA_R_16F, ldb, &beta1, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha1, half_A, CUDA_R_16F, 
        lda, half_B + k*n, CUDA_R_16F, ldb, &beta1, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    
    //cudaSafeCall(cudaFree(half_A));
    //cudaSafeCall(cudaFree(half_B));
}

} // namespace ape
