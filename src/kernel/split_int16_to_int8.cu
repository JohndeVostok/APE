#include "common.h"
#include "kernel.h"

namespace ape {
__global__ void kernel_split_int16_to_int8(int8_t *dst, const int16_t *src, size_t size) {
    uint32_t base = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = 4 * blockDim.x * gridDim.x;
    for (uint32_t i = base; i < size; i += step) {
        short4 base = (short4 &)src[i];
        char4 buf[2];
        buf[0].x = char(base.x);
        buf[0].y = char(base.y);
        buf[0].z = char(base.z);
        buf[0].w = char(base.w);
        buf[1].x = char((base.x - int16_t(buf[0].x)) / 256);
        buf[1].y = char((base.y - int16_t(buf[0].y)) / 256);
        buf[1].z = char((base.z - int16_t(buf[0].z)) / 256);
        buf[1].w = char((base.w - int16_t(buf[0].w)) / 256);

        (char4 &)dst[i] = buf[1];
        (char4 &)dst[size + i] = buf[0];
    }
    return;
}

void split_int16_to_int8(int8_t *dst, const int16_t *src, size_t size) {
    dim3 grid(NUM_SM, 1, 1);
    dim3 block(MAX_THREAD, 1, 1);

    kernel_split_int16_to_int8<<<grid, block>>>(dst, src, size);
    cudaCheckError();
}
} // namespace ape