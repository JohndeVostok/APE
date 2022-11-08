#include "common.h"
#include "kernel.h"

namespace ape {
__global__ void kernel_merge_fp16_to_fp32(float *dst, const half *src, size_t size) {
    uint32_t base = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = 2 * blockDim.x * gridDim.x;
    for (uint32_t i = base; i < size; i += step) {
        half2 tmp[2];
        tmp[0] = (half2 &)src[i];
        tmp[1] = (half2 &)src[size + i];
        float2 buf;
        buf.x = float(tmp[0].x) + float(tmp[1].x) / 4096.0f;
        buf.y = float(tmp[0].y) + float(tmp[1].y) / 4096.0f;
        (float2 &)dst[i] = buf;
    }
    return;
}

void merge_fp16_to_fp32(float *dst, const half *src, size_t size) {
    dim3 grid_size(NUM_SM, 1);
    dim3 block_size(MAX_THREAD, 1);
    kernel_merge_fp16_to_fp32<<<grid_size, block_size>>>(dst, src, size);
    cudaCheckError();
}
} // namespace ape
