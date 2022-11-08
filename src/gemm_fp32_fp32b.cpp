#include "common.h"
#include "kernel.h"

namespace ape {
void gemm_fp32_fp32b(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc) {
    assert((m * k + k * n) * 6 <= APEHandler::getBufSize());
    __nv_bfloat16 *buf = (__nv_bfloat16 *)APEHandler::getBuf();
    __nv_bfloat16 *buf_a, *buf_b;
    buf_a = buf;
    buf_b = buf + m * k * 3;

    split_fp32_to_bf16(buf_a, A, m * k);
    split_fp32_to_bf16(buf_b, B, k * n);

    float beta0 = *beta, beta1 = 1;
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                alpha, buf_a, CUDA_R_16BF, lda, buf_b, CUDA_R_16BF, ldb, &beta0, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                alpha, buf_a + m * k, CUDA_R_16BF, lda, buf_b, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                alpha, buf_a, CUDA_R_16BF, lda, buf_b + k * n, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                alpha, buf_a + m * k, CUDA_R_16BF, lda, buf_b + k * n, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F,
                                ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                alpha, buf_a + m * k * 2, CUDA_R_16BF, lda, buf_b, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                alpha, buf_a, CUDA_R_16BF, lda, buf_b + k * n * 2, CUDA_R_16BF, ldb, &beta1, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
} // namespace ape
