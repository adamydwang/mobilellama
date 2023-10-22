#pragma once

#include <vector>
#include <layers/base_layer.h>
#include <data/tensor.h>


class Matmul: public BaseLayer {
public:
    Matmul(): cache(nullptr) {}
    ~Matmul() {
        if (cache != nullptr) {
            delete cache;
        }
    } 
    void forward(Tensor& input0, Tensor& input1, Tensor& output);
    void load_weights(FILE*& fp) {
        return;
    }
private:
    Tensor* cache;
};