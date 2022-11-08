#include "common.h"
#include "kernel.h"

namespace ape {
__global__ void kernel_split_fp32_to_tf32(float *dst, const float *src, size_t size) {
    uint32_t base = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = 2 * blockDim.x * gridDim.x;
    for (uint32_t i = base; i < size; i += step) {
        float2 base = (float2 &)src[i];
        float2 buf[2];
        buf[0].x = base.x;
        buf[0].y = base.y;
        buf[1].x = (base.x - float(buf[0].x)) * 4096.0f;
        buf[1].y = (base.y - float(buf[0].y)) * 4096.0f;
        (float2 &)dst[i] = buf[0];
        (float2 &)dst[size + i] = buf[1];
    }
    return;
}

void split_fp32_to_tf32(float *dst, const float *src, size_t size) {
    dim3 grid_size(NUM_SM, 1);
    dim3 block_size(MAX_THREAD, 1);
    kernel_split_fp32_to_tf32<<<grid_size, block_size>>>(dst, src, size);
    cudaCheckError();
}
} // namespace ape