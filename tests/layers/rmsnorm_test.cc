#include <gtest/gtest.h>
#include <math.h>
#include <layers/rmsnorm.h>


TEST(RMSNormTest_Forward, BasicAssertions) {
    std::vector<int> dims = {2};
    float data[2] = {1,2};
    Tensor input(dims, data);
    Tensor output(dims);
    float weights[2] = {1,1};
    RMSNormParameter param(dims[0], 1e-5, weights);
    RMSNorm rmsnorm(&param);
    rmsnorm.forward(input, output);
    EXPECT_LE(fabs(*(output.data()) - 0.4472135954999579f), 1e-5f) << output.data() << " != " << 0.4472135954999579f;
    EXPECT_LE(fabs(*(output.data()+1) - 0.8944271909999159f), 1e-5f) << output.data() + 1 << " != " << 0.8944271909999159f;
}