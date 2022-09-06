#pragma once

#include <cstdint>

#include "common.h"

namespace ape {

void apeGemmFP32(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                 const float *B, int ldb, const float *beta, float *C, int ldc, ApeAlgo algo = APE_ALGO_AUTO);

void apeGemmFP64(ApeTrans transa, ApeTrans transb, int m, int n, int k, const double *alpha, const double *A, int lda,
                 const double *B, int ldb, const double *beta, double *C, int ldc, ApeAlgo algo = APE_ALGO_AUTO);

void apeGemmINT16(ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda,
                  const int16_t *B, int ldb, const int16_t *beta, int16_t *C, int ldc, ApeAlgo algo = APE_ALGO_AUTO);

} // namespace ape