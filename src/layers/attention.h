#pragma once
#include <vector>
#include <layers/attention.h>
#include <data/tensor.h>
#include <layers/base_layer.h>
#include <layers/matmul.h>
#include <layers/softmax.h>


class Attention: public BaseLayer {
public:
    Attention(
            int _n_heads,
            int _n_kv_heads,
            int _dim,
            int _dim_kv,
            float* _wq,
            float* _wk,
            float* _wv,
            Tensor* _cache_q,
            Tensor* _cache_k,
            Tensor* _cache_v,
            Tensor* _cache_out
    ):
            n_heads(_n_heads),
            n_kv_heads(_n_kv_heads),
            dim(_dim),
            dim_kv(_dim_kv),
            wq(_wq),
            wk(_wk),
            wv(_wv),
            cache_q(_cache_q),
            cache_k(_cache_k),
            cache_v(_cache_v),
            cache_out(_cache_out){}

    ~Attention() {}

    void forward(Tensor& input, Tensor& output);

    int load_weights(void* cur_ptr) {
        return (dim + dim_kv * 2) * sizeof(float);
    }
private:
    int max_len;
    int n_heads;
    int n_kv_heads;
    int dim;
    int dim_kv;
    float* wq;
    float* wk;
    float* wv;
    Tensor* cache_out;
    Tensor* cache_q;
    Tensor* cache_k;
    Tensor* cache_v;
};