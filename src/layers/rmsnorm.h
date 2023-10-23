#pragma once

#include <vector>
#include <memory.h>
#include <layers/base_layer.h>
#include <data/tensor.h>


class RMSNorm: public BaseLayer {
public:
    RMSNorm(float _eps = 1e-5f): eps(_eps) {}
    ~RMSNorm() {
    }

    void forward(Tensor& input, Tensor& output);
private:
    float eps;
};