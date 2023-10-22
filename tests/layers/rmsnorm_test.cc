#include <gtest/gtest.h>
#include <math.h>
#include <layers/rmsnorm.h>


TEST(RMSNormTest_Forward, BasicAssertions) {
    printf("start\n");
    std::vector<int> dims = {2};
    float data[2] = {1,2};
    Tensor input(dims, data);
    Tensor output;
    float weights[2] = {1,1};
    RMSNormParameter param(dims[0], 1e-5, weights);
    RMSNorm rmsnorm(param);
    printf("input.data() = %p\n", input.data());
    printf("output.data() = %p\n", output.data());
    return;
    rmsnorm.forward(input, output);
    printf("output.data() = %p\n", output.data());
    EXPECT_LE(fabs(*(output.data()) - 0.6324555320336759f), 1e-5f);
    EXPECT_LE(fabs(*(output.data()+1) - 1.2649110640673518f), 1e-5f);
    printf("done\n");
}