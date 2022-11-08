#pragma once

#include <cstdint>
#include <cstdlib>

namespace ape {

enum ApeTrans {
    APE_TRANS_N = 0,
    APE_TRANS_T,
};

enum ApeAlgo {
    APE_ALGO_AUTO = 1,
    APE_ALGO_CUBLAS,
    APE_ALGO_FP32F,
    APE_ALGO_FP32B,
    APE_ALGO_FP32T,
    APE_ALGO_INT16,
};

void apeInit(const size_t buf_size = 0);

void apeGemmFP32(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                 const float *B, int ldb, const float *beta, float *C, int ldc, const ApeAlgo algo = APE_ALGO_AUTO);

void apeGemmFP64(ApeTrans transa, ApeTrans transb, int m, int n, int k, const double *alpha, const double *A, int lda,
                 const double *B, int ldb, const double *beta, double *C, int ldc, ApeAlgo algo = APE_ALGO_AUTO);

void apeGemmINT16(ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda,
                  const int16_t *B, int ldb, const int32_t *beta, int32_t *C, int ldc, ApeAlgo algo = APE_ALGO_AUTO);

} // namespace ape