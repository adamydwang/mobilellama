#pragma once

#include <vector>
#include <stdio.h>
#include <layers/base_layer.h>
#include <data/tensor.h>


class RMSNormParameter: public BaseLayerParameter {
public:
    RMSNormParameter(int dim=0, float eps=0, float* weights=nullptr): dim(dim), eps(eps), weights(weights) {}
    ~RMSNormParameter() {
        delete[] weights;
    }  
    int load(FILE*& fp) {
        int res = fread(&dim, sizeof(int), 1, fp);
        res += fread(&eps, sizeof(float), 1, fp);
        weights = new float[dim];
        res += fread(weights, sizeof(float), dim, fp);
        return res;
    }
public:
    int dim;
    float eps;
    float* weights;
};


class RMSNorm: public BaseLayer {
public:
    RMSNorm(RMSNormParameter* params=nullptr) {
        params = params;
    }

    ~RMSNorm() {
        if (params != nullptr) {
            delete params;
        }
    }

    void forward(Tensor& input, Tensor& output);
    void load_weights(FILE*& fp) {
        params = new RMSNormParameter();
        params->load(fp);
    }
private:
    RMSNormParameter* params;
};