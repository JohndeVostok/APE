#include "ape.h"
#include "test/test_gemm_fp32.h"

int main() {
    ape::apeInit();
    ape::test::test_gemm_fp32(128, 128, 128, ape::APE_ALGO_AUTO);
    ape::test::test_gemm_fp32(256, 256, 256, ape::APE_ALGO_AUTO);
    ape::test::test_gemm_fp32(512, 512, 512, ape::APE_ALGO_AUTO);
    ape::test::test_gemm_fp32(1024, 1024, 1024, ape::APE_ALGO_AUTO);
    ape::test::test_gemm_fp32(2048, 2048, 2048, ape::APE_ALGO_AUTO);
    ape::test::test_gemm_fp32(4096, 4096, 4096, ape::APE_ALGO_AUTO);
    ape::test::test_gemm_fp32(8192, 8192, 8192, ape::APE_ALGO_AUTO);
}