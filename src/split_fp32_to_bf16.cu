#include "cuda_utils.h"
#include "kernel.h"

namespace ape {

__global__ void kernel_split_fp32_to_bf16(__nv_bfloat16 *dst, const float *src,
                                uint32_t size) {
    uint32_t base = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = 2 * blockDim.x * gridDim.x;
    for (uint32_t i = base; i < size; i += step) {
        float2 base = (float2 &)src[i];
        __nv_bfloat162 buf[3];
        buf[0].x = __float2bfloat16(base.x);
        buf[0].y = __float2bfloat16(base.y);
        buf[1].x = __float2bfloat16(base.x - float(buf[0].x));
        buf[1].y = __float2bfloat16(base.y - float(buf[0].y));
        buf[2].x = __float2bfloat16(base.x - float(buf[0].x) - float(buf[1].x));
        buf[2].y = __float2bfloat16(base.y - float(buf[0].y) - float(buf[1].y));

        (__nv_bfloat162 &)dst[i] = buf[0];
        (__nv_bfloat162 &)dst[size + i] = buf[1];
        (__nv_bfloat162 &)dst[size * 2 + i] = buf[2];
    }
    return;
}

void split_fp32_to_bf16(__nv_bfloat16 *dst, const float *src,
                                uint32_t size) {
    dim3 grid_size(NUM_SM, 1);
    dim3 block_size(MAX_THREAD, 1);
    kernel_split_fp32_to_bf16<<<grid_size, block_size>>>(dst, src, size);
    cudaCheckError();
}

} // namespace ape

