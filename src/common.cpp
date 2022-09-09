#include "common.h"
#include "cuda_utils.h"

namespace ape {

cublasHandle_t ape_cublas_handle = nullptr;
void* ape_buffer = nullptr;

//TODO: initialize for buffer
void apeInit(size_t buffer_size = 0) {
    if (ape_cublas_handle == nullptr) {
        cublasSafeCall(cublasCreate(&ape_cublas_handle));
        cublasSafeCall(cublasSetMathMode(ape_cublas_handle, CUBLAS_DEFAULT_MATH));
    }

    if (ape_buffer == nullptr && buffer_size != 0) {
        cudaSafeCall(cudaMalloc((void**) &ape_buffer, buffer_size));
    }
}

void apeFinal() {if (ape_buffer) cudaSafeCall(cudaFree(ape_buffer));};

} // namespace ape