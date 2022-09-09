#include "cuda_utils.h"
#include "kernel.h"

namespace ape
{
//TODO: the interface of apeGemmINT16 seems contradictory with current cublasGemmEx
void gemm_int16_int8(ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda,
                      const int16_t *B, int ldb, const int16_t *beta, int16_t *C, int ldc) {
    int8_t *int8_A = (int8_t*)ape_buffer, *int8_B = int8_A + m * k * 2;
    int32_t *int32_C = (int32_t*)(int8_B + k * n * 2);
    //cudaSafeCall(cudaMalloc((void**) &int8_A, sizeof(int8_t) * m * k * 2));
    //cudaSafeCall(cudaMalloc((void**) &int8_B, sizeof(int8_t) * k * n * 2));
    //cudaSafeCall(cudaMalloc((void**) &int32_C, sizeof(int32_t) * m * n));

    convert_int16_to_int8(int8_A, A, m*k);
    convert_int16_to_int8(int8_B, B, k*n);
    convert_int16_to_int32(int32_C, C, m*n);

    int alpha0 = *alpha * 256 * 256, alpha1 = *alpha * 256, alpha2 = *alpha;
    int beta0 = *beta, beta1 = 1;
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha0, int8_A, CUDA_R_8I, 
        lda, int8_B, CUDA_R_8I, ldb, &beta0, int32_C, CUDA_R_32I, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha1, int8_A + m*k, CUDA_R_8I, 
        lda, int8_B, CUDA_R_8I, ldb, &beta1, int32_C, CUDA_R_32I, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha1, int8_A, CUDA_R_8I, 
        lda, int8_B + k*n, CUDA_R_8I, ldb, &beta1, int32_C, CUDA_R_32I, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, &alpha2, int8_A + m*k, CUDA_R_8I, 
        lda, int8_B + k*n, CUDA_R_8I, ldb, &beta1, int32_C, CUDA_R_32I, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));

    convert_int32_to_int16(C, int32_C, m*n);

    //cudaSafeCall(cudaFree(int8_A));
    //cudaSafeCall(cudaFree(int8_B));
    //cudaSafeCall(cudaFree(int32_C));
}
} // namespace ape