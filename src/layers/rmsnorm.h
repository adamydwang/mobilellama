#pragma once

#include <vector>
#include <memory.h>
#include <layers/base_layer.h>
#include <data/tensor.h>


class RMSNormParameter: public BaseLayerParameter {
public:
    RMSNormParameter(int _dim=0, float _eps=0, float* _weights=nullptr): dim(_dim), eps(_eps) {
        ref_count = new int(1);
        if (_weights != nullptr) {
            weights = new float[dim];
            memcpy(weights, _weights, dim * sizeof(float));
        } else {
            weights = nullptr;
        }
    }
    ~RMSNormParameter() {
        (*ref_count)--;
        if (*ref_count <= 0) {
            printf("rms delete ref_count: %p\n", ref_count);
            delete ref_count;
            if (weights != nullptr) {
                printf("rms delete weights: %p\n", weights);
                delete[] weights;
                weights = nullptr;
            }   
        }
    }  

    RMSNormParameter& operator=(const RMSNormParameter& other) {
        (*other.ref_count)++;
        (*ref_count)--;
        if (*ref_count <= 0) {
            printf("delete ref_count: %p\n", ref_count);
            delete ref_count;
            if (weights != nullptr) {
                printf("delete weights: %p\n", weights);
                delete[] weights;
                weights = nullptr;
            }   
        }
        this->dim = other.dim;
        this->eps = other.eps;
        this->weights = other.weights;
        this->ref_count = other.ref_count;
        return *this;
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
    int* ref_count;
};


class RMSNorm: public BaseLayer {
public:
    RMSNorm() {}
    RMSNorm(RMSNormParameter& params): params(params) {
    }

    ~RMSNorm() {
    }

    void forward(Tensor& input, Tensor& output);
    void load_weights(FILE*& fp) {
        params.load(fp);
    }
private:
    RMSNormParameter params;
};