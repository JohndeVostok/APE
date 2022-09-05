#pragma once
#include "common.h"

namespace ape {
void gemm_fp32_auto(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                    const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_cublas(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                      const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_fp32f(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_fp32b(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_fp32t(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp64_cublas(ApeTrans transa, ApeTrans transb, int m, int n, int k, const double *alpha, const double *A, int lda,
                      const double *B, int ldb, const double *beta, double *C, int ldc);

void convert_fp64_to_fp32(float *dst, double *src, size_t size);
void convert_fp32_to_fp64(double *dst, float *src, size_t size);
void compare_fp32_to_fp64(const float *src, const double *dst, size_t size, double &max_error, double &mean_error);

} // namespace ape