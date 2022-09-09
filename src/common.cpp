#include "common.h"
#include "cuda_utils.h"

namespace ape {

cublasHandle_t ape_cublas_handle = nullptr;
cudaDeviceProp ape_gpu_prop;

//TODO: initialize for buffer
void apeInit() {
    if (ape_cublas_handle == nullptr) {
        cublasSafeCall(cublasCreate(&ape_cublas_handle));
        cublasSafeCall(cublasSetMathMode(ape_cublas_handle, CUBLAS_DEFAULT_MATH));
        cudaSafeCall(cudaGetDeviceProperties(&ape_gpu_prop, 0));
    }
}

} // namespace ape