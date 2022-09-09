#include "test_fp32.h"

int main() {
    //TODO: should we merge all fp32 tests into one ?
    ape::apeInit(8192 * 8192 * sizeof(float) * 4);
    test::test_error(128, 128, 128, ape::APE_ALGO_FP32F);
    test::test_error(256, 256, 256, ape::APE_ALGO_FP32F);
    test::test_error(512, 512, 512, ape::APE_ALGO_FP32F);
    test::test_error(1024, 1024, 1024, ape::APE_ALGO_FP32F);
    test::test_error(2048, 2048, 2048, ape::APE_ALGO_FP32F);
    test::test_error(4096, 4096, 4096, ape::APE_ALGO_FP32F);
    test::test_error(8192, 8192, 8192, ape::APE_ALGO_FP32F);
    ape::apeFinal();
}