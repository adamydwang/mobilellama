#pragma once

#include <vector>
#include <memory.h>
#include <data/tensor.h>
#include <layers/base_layer.h>


class RMSNorm : public BaseLayer {
public:
    RMSNorm(Tensor& weights, float eps = 1e-5f): weights(weights), eps(eps) {}
    ~RMSNorm() {}
    void forward(Tensor& input, Tensor& output);
private:
    Tensor weights;
    float eps;
};