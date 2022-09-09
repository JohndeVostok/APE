#include "cuda_utils.h"
#include "kernel.h"

namespace ape
{
void gemm_fp32_bf16(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                      const float *B, int ldb, const float *beta, float *C, int ldc) {
    __nv_bfloat16 *bf16_A = (__nv_bfloat16*)ape_buffer, *bf16_B = bf16_A + m * k * 3;
    //cudaSafeCall(cudaMalloc((void**) &bf16_A, sizeof(__nv_bfloat16) * m * k * 3));
    //cudaSafeCall(cudaMalloc((void**) &bf16_B, sizeof(__nv_bfloat16) * k * n * 3));

    convert_fp32_to_bf16(bf16_A, A, m*k);
    convert_fp32_to_bf16(bf16_B, B, k*n);

    float beta0 = *beta, beta1 = 1;
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, alpha, bf16_A, CUDA_R_16BF, 
        lda, bf16_B, CUDA_R_16BF, ldb, &beta0, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, alpha, bf16_A + m*k, CUDA_R_16BF, 
        lda, bf16_B, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, alpha, bf16_A, CUDA_R_16BF, 
        lda, bf16_B + k*n, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, alpha, bf16_A + m*k, CUDA_R_16BF, 
        lda, bf16_B + k*n, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, alpha, bf16_A + m*k*2, CUDA_R_16BF, 
        lda, bf16_B, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, alpha, bf16_A, CUDA_R_16BF, 
        lda, bf16_B + k*n*2, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    
    //cudaSafeCall(cudaFree(bf16_A));
    //cudaSafeCall(cudaFree(bf16_B));
}
} // namespace ape
