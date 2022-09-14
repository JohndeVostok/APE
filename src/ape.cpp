#include <iostream>

#include "ape.h"
#include "common.h"
#include "kernel.h"

namespace ape {

void apeInit(const size_t buf_size) {
    APEHandler::initCublas();
    if (buf_size > 0) {
        APEHandler::initBuffer(buf_size);
    }
}

void apeGemmFP32(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                 const float *B, int ldb, const float *beta, float *C, int ldc, const ApeAlgo algo) {
    switch (algo) {
    case APE_ALGO_AUTO:
        gemm_fp32_auto(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    case APE_ALGO_AUTO_STRICT:
        gemm_fp32_auto_strict(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    case APE_ALGO_CUBLAS:
        gemm_fp32_cublas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    case APE_ALGO_FP32F:
        gemm_fp32_fp32f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    case APE_ALGO_FP32B:
        gemm_fp32_fp32b(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    case APE_ALGO_FP32T:
        gemm_fp32_fp32t(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    default:
        assert(false);
    }
}

void apeGemmFP64(ApeTrans transa, ApeTrans transb, int m, int n, int k, const double *alpha, const double *A, int lda,
                 const double *B, int ldb, const double *beta, double *C, int ldc, ApeAlgo algo) {
    switch (algo) {
    case APE_ALGO_AUTO:
        gemm_fp64_cublas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    case APE_ALGO_CUBLAS:
        gemm_fp64_cublas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    default:
        assert(false);
    }
}

void apeGemmINT16(ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda,
                  const int16_t *B, int ldb, const int32_t *beta, int32_t *C, int ldc, ApeAlgo algo) {
    switch (algo) {
    case APE_ALGO_AUTO:
        // TODO: check layout
        assert(count_overflow_int16emu(A, m, k) == 0);
        assert(count_overflow_int16emu(B, k, n) == 0);
        gemm_int16_emu(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    case APE_ALGO_INT16:
        gemm_int16_emu(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    default:
        assert(false);
    }
}

} // namespace ape