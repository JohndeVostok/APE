#include "common.h"

namespace ape {

cublasHandle_t ape_cublas_handle = nullptr;

void apeInit() {
    if (ape_cublas_handle == nullptr) {
        cublasCreate(&ape_cublas_handle);
        cublasSetMathMode(ape_cublas_handle, CUBLAS_DEFAULT_MATH);
    }
}

} // namespace ape