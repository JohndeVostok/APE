#include "common.h"
#include "kernel.h"

namespace ape {

__global__ void kernel_convert_fp64_to_fp32(float *dst, const double *src, size_t size) {
    size_t base = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    size_t step = 2 * blockDim.x * gridDim.x;
    for (size_t i = base; i < size; i += step) {
        double2 base = (double2 &)src[i];
        float2 buf;
        buf.x = float(base.x);
        buf.y = float(base.y);
        (float2 &)dst[i] = (float2 &)buf;
    }
}

void convert_fp64_to_fp32(float *dst, const double *src, size_t size) {
    dim3 grid_size(NUM_SM, 1);
    dim3 block_size(MAX_THREAD, 1);
    kernel_convert_fp64_to_fp32<<<grid_size, block_size>>>(dst, src, size);
    cudaCheckError();
}

} // namespace ape