#include "common.h"
#include "kernel.h"

namespace ape {
// TODO: the interface of apeGemmINT16 seems contradictory with current cublasGemmEx
void gemm_int16_emu(ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda,
                    const int16_t *B, int ldb, const int32_t *beta, int32_t *C, int ldc) {
    assert((m * k + k * n) * 2 <= APEHandler::getBufSize());
    int8_t *buf = (int8_t *) APEHandler::getBuf();
    int8_t *buf_a, *buf_b;
    buf_a = buf;
    buf_b = buf + m * k * 2;

    split_int16_to_int8(buf_a, A, m * k);
    split_int16_to_int8(buf_b, B, k * n);

    int alpha0 = *alpha * 256 * 256, alpha1 = *alpha * 256, alpha2 = *alpha;
    int beta0 = *beta, beta1 = 1;
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                &alpha0, buf_a, CUDA_R_8I, lda, buf_b, CUDA_R_8I, ldb, &beta0, C, CUDA_R_32I, ldc,
                                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                &alpha1, buf_a + m * k, CUDA_R_8I, lda, buf_b, CUDA_R_8I, ldb, &beta1, C, CUDA_R_32I, ldc,
                                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                &alpha1, buf_a, CUDA_R_8I, lda, buf_b + k * n, CUDA_R_8I, ldb, &beta1, C, CUDA_R_32I, ldc,
                                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                &alpha2, buf_a + m * k, CUDA_R_8I, lda, buf_b + k * n, CUDA_R_8I, ldb, &beta1, C, CUDA_R_32I,
                                ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
}
} // namespace ape