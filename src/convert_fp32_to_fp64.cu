#include "cuda_utils.h"
#include "kernel.h"

namespace ape {

__global__ void kernel_convert_fp32_to_fp64(double *dst, float *src, size_t size) {
    uint32_t base = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = 2 * blockDim.x * gridDim.x;
    for (uint32_t i = base; i < size; i += step) {
        float2 base = (float2 &)src[i];
        double2 buf;
        buf.x = double(base.x);
        buf.y = double(base.y);
        (double2 &)dst[i] = (double2 &)buf;
    }
}

void convert_fp32_to_fp64(double *dst, float *src, size_t size) {
    dim3 grid_size(NUM_SM, 1);
    dim3 block_size(MAX_THREAD, 1);
    kernel_convert_fp32_to_fp64<<<grid_size, block_size>>>(dst, src, size);
    cudaCheckError();
}

} // namespace ape