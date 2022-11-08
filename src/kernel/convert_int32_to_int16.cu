#include "common.h"
#include "kernel.h"

namespace ape {
// TODO: change names of other converts into merge or split, because this convert has different meanings from others
__global__ void kernel_convert_int32_to_int16(int16_t *dst, const int32_t *src, size_t size) {
    uint32_t base = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = 2 * blockDim.x * gridDim.x;
    for (uint32_t i = base; i < size; i += step) {
        int2 base = (int2 &)src[i];
        short2 buf;
        buf.x = short(base.x);
        buf.y = short(base.y);
        (short2 &)dst[i] = buf;
    }
    return;
}

void convert_int32_to_int16(int16_t *dst, const int32_t *src, size_t size) {
    dim3 grid(NUM_SM, 1, 1);
    dim3 block(MAX_THREAD, 1, 1);
    kernel_convert_int32_to_int16<<<grid, block>>>(dst, src, size);
    cudaCheckError();
}
} // namespace ape