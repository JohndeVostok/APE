#include "common.h"
#include "kernel.h"

namespace ape {
__global__ void kernel_convert_int16_to_int32(int32_t *dst, const int16_t *src, size_t size) {
    uint32_t base = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = 4 * blockDim.x * gridDim.x;
    for (uint32_t i = base; i < size; i += step) {
        short4 base = (short4 &)src[i];
        int4 buf;
        buf.x = int(base.x);
        buf.y = int(base.y);
        buf.z = int(base.z);
        buf.w = int(base.w);
        (int4 &)dst[i] = buf;
    }
    return;
}

void convert_int16_to_int32(int32_t *dst, const int16_t *src, size_t size) {
    dim3 grid(NUM_SM, 1, 1);
    dim3 block(MAX_THREAD, 1, 1);
    kernel_convert_int16_to_int32<<<grid, block>>>(dst, src, size);
    cudaCheckError();
}
} // namespace ape