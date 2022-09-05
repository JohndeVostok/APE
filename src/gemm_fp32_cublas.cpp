
#include "cuda_utils.h"
#include "kernel.h"

namespace ape {
void gemm_fp32_cublas(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                      const float *B, int ldb, const float *beta, float *C, int ldc) {
    cublasSafeCall(cublasSgemm(ape_cublas_handle, cublasOperation_t(transa), cublasOperation_t(transb), m, n, k, alpha, A, lda,
                               B, ldb, beta, C, ldc));
}

} // namespace ape
