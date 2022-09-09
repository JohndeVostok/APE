#include "cuda_utils.h"
#include "kernel.h"

namespace ape {
__global__ void kernel_merge_int8_to_int16(int16_t* dst, const int8_t* src, size_t size) {
    uint32_t base = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t step = 4 * blockDim.x * gridDim.x;
    for (uint32_t i = base; i < size; i += step) {
        char4 tmp[2];

        tmp[0] = (char4 &)src[size + i];
        tmp[1] = (char4 &)src[size];

        short4 buf;
        buf.x = tmp[0].x + tmp[1].x * 256;
        buf.y = tmp[0].y + tmp[1].y * 256;
        buf.z = tmp[0].z + tmp[1].z * 256;
        buf.w = tmp[0].w + tmp[1].w * 256;

        (short4 &)dst[i] = buf;
    }
    return;
}

void merge_int8_to_int16(int16_t* dst, const int8_t* src, size_t size) {
    dim3 grid(NUM_SM, 1, 1);
    dim3 block(MAX_THREAD, 1, 1);

    kernel_merge_int8_to_int16<<<grid, block>>>(dst, src, size);
    cudaCheckError();
}
} //namespace ape