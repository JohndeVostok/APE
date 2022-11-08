#include "common.h"
#include "kernel.h"

namespace ape {

__global__ void kernel_merge_bf16_to_fp32(float *dst, const __nv_bfloat16 *src, uint32_t size) {
    uint32_t base = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = 2 * blockDim.x * gridDim.x;
    for (uint32_t i = base; i < size; i += step) {
        __nv_bfloat162 a0, a1, a2;
        a0 = (__nv_bfloat162 &)src[i];
        a1 = (__nv_bfloat162 &)src[i + size];
        a2 = (__nv_bfloat162 &)src[i + size * 2];
        float2 b;
        b.x = float(a0.x) + float(a1.x) + float(a2.x);
        b.y = float(a0.y) + float(a1.y) + float(a2.y);
        (float2 &)dst[i] = b;
    }
    return;
}

void merge_bf16_to_fp32(float *dst, const __nv_bfloat16 *src, uint32_t size) {
    dim3 grid_size(NUM_SM, 1);
    dim3 block_size(MAX_THREAD, 1);
    kernel_merge_bf16_to_fp32<<<grid_size, block_size>>>(dst, src, size);
    cudaCheckError();
}

} // namespace ape
