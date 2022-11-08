#include "common.h"
#include "kernel.h"

namespace ape {
void gemm_fp32_auto(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                    const float *B, int ldb, const float *beta, float *C, int ldc) {
    if (count_overflow_fp32f(A, m, k) > 0 || count_overflow_fp32f(B, k, n) > 0) {
        gemm_fp32_fp32b(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        gemm_fp32_fp32f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

void gemm_fp32_auto_strict(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C, int ldc) {
    if (count_overflow_fp32f_strict(A, m, k) > 0 || count_overflow_fp32f_strict(B, k, n) > 0) {
        gemm_fp32_fp32b(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        gemm_fp32_fp32f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

} // namespace ape
