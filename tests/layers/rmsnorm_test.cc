#include <vector>
#include <gtest/gtest.h>
#include <math.h>
#include <layers/rmsnorm.h>


TEST(RMSNormTest_Forward, BasicAssertions) {
    std::vector<int> dims = {2};
    float data[2] = {1,2};
    Tensor input(dims, data);
    Tensor output;
    float weights[2] = {1, 1};
    Tensor weights_tensor(dims, weights);
    RMSNorm rmsnorm(weights_tensor);
    rmsnorm.forward(input, output);
    EXPECT_LE(fabs(*(output.data()) - 0.6324555320336759f), 1e-5f);
    EXPECT_LE(fabs(*(output.data()+1) - 1.2649110640673518f), 1e-5f);
}