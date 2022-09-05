#include <iostream>

#include "ape.h"
#include "common.h"
#include "kernel.h"

namespace ape {

void apeGemmFP64(ApeTrans transa, ApeTrans transb, int m, int n, int k, const double *alpha, const double *A, int lda,
                 const double *B, int ldb, const double *beta, double *C, int ldc, ApeAlgo algo) {
    gemm_fp64_cublas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void apeGemmFP32(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                 const float *B, int ldb, const float *beta, float *C, int ldc, ApeAlgo algo) {
    switch (algo) {
    case APE_ALGO_AUTO:
        ape_error("Not impl.");
        break;
    case APE_ALGO_CUBLAS:
        gemm_fp32_cublas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    case APE_ALGO_FP32F:
        ape_error("Not impl.");
        break;
    case APE_ALGO_FP32B:
        ape_error("Not impl.");
        break;
    default:
        ape_error("Invalid algo.");
    }
}

void apeGemmINT16(ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda,
                  const int16_t *B, int ldb, const int16_t *beta, int16_t *C, int ldc, ApeAlgo algo) {
    switch (algo) {
    case APE_ALGO_AUTO:
        ape_error("Not impl.");
        break;
    case APE_ALGO_INT16:
        ape_error("Not impl.");
        break;
    default:
        ape_error("Invalid algo.");
    }
}

} // namespace ape