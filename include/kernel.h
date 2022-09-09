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
void gemm_fp32_fp16(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_bf16(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                      const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp64_cublas(ApeTrans transa, ApeTrans transb, int m, int n, int k, const double *alpha, const double *A, int lda,
                      const double *B, int ldb, const double *beta, double *C, int ldc);

void gemm_int16_int8(ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda,
                      const int16_t *B, int ldb, const int16_t *beta, int16_t *C, int ldc);

void convert_fp64_to_fp32(float *dst, double *src, size_t size);
void convert_fp32_to_fp64(double *dst, float *src, size_t size);
void convert_int32_to_int16(int16_t* dst, const int32_t* src, size_t size);
void convert_int16_to_int32(int32_t* dst, const int16_t* src, size_t size);
void compare_fp32_to_fp64(const float *src, const double *dst, size_t size, double &max_error, double &mean_error);

void split_fp32_to_fp16(half *dst, const float *src, size_t size);
void merge_fp16_to_fp32(float *dst, const half *src, size_t size);
void split_fp32_to_bf16(__nv_bfloat16 *dst, const float *src, uint32_t size);
void merge_bf16_to_fp32(float *dst, const __nv_bfloat16 *src, uint32_t size);
void split_int16_to_int8(int8_t* dst, const int16_t* src, size_t size);
void merge_int8_to_int16(int16_t* dst, const int8_t* src, size_t size);
} // namespace ape