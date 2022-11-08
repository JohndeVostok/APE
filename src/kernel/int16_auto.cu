#include "kernel.h"
#include "thrust/device_vector.h"
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace ape {

struct OpINT16 {
    __host__ __device__ int operator()(int16_t x) { return (x > INT16C_MAX); }
};

int count_overflow_int16emu(const int16_t *src, size_t row, size_t col) {
    thrust::device_ptr<int16_t> d_src(const_cast<int16_t *>(src));
    return thrust::transform_reduce(d_src, d_src + row * col, OpINT16(), 0, thrust::plus<int>());
}
} // namespace ape
