#pragma once

#include <vector>
#include <layers/base_layer.h>


class Softmax: public BaseLayer {
public:
    Softmax() {}
    void forward(Tensor& input, Tensor& output);
    void load_weights(FILE*& fp) {
        return;
    }
};