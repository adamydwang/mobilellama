#include <stdio.h>
#include <vector>
#include <gtest/gtest.h>
#include <data/tensor.h>


TEST(TensorTest_Dims, BasicAssertions) {
    std::vector<int> dims = {1,2,3};
    Tensor tensor(dims);
    EXPECT_EQ(tensor.size(), 6);
}

TEST(TensorTest_Data, BasicAssertions) {
    std::vector<int> dims = {2};
    float data[2] = {1,2};
    Tensor tensor(dims, data);
    EXPECT_EQ(*tensor.data(), *data) << tensor.data() << " != " << data;
    EXPECT_EQ(*(tensor.data()+1), *(data+1)) << tensor.data() + 1 << " != " << data + 1;
}

TEST(TensorTest_Assignment, BasicAssertions) {
    std::vector<int> dims = {2};
    float data[2] = {1,2};
    Tensor tensor1(dims, data);
    Tensor tensor2 = tensor1;
    EXPECT_EQ(tensor1.data(), tensor2.data());
}