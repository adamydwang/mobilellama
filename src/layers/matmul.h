#pragma once

#include <vector>
#include <layers/base_layer.h>
#include <data/tensor.h>


class Matmul: public BaseLayer {
public:
    Matmul(int m, int n): m(m), n(n) {}
    void forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
private:
    int m;
    int n;
};