#pragma once
#include <vector>
#include <layers/attention.h>
#include <data/tensor.h>
#include <layers/base_layer.h>


class Attention: public BaseLayer {
public:
    Attention(
            int _n_heads,
            int _n_kv_heads,
            int _dim,  // token embedding dim
            int _max_len, // max sequence length
            Tensor& _wq, // [dim, n_heads * head_size]
            Tensor& _wk, // [dim, n_kv_heads * head_size]
            Tensor& _wv, // [dim, n_kv_heads * head_size]
            Tensor& _wo, // [n_heads * head_size, dim]
            Tensor& _cache_q, // [n_heads * head_size]
            Tensor& _cache_attn // [n_heads, max_len]
    ):
            n_heads(_n_heads),
            n_kv_heads(_n_kv_heads),
            dim(_dim),
            max_len(_max_len),
            wq(_wq),
            wk(_wk),
            wv(_wv),
            wo(_wo),
            cache_q(_cache_q),
            cache_attn(_cache_attn) {
        this->head_size = this->dim / this->n_heads;
        this->kv_dim = this->dim * this->n_kv_heads / this->n_heads;
        this->kv_mul = this->n_heads / this->n_kv_heads;
        std::vector<int> dims = {this->max_len, this->n_kv_heads, this->head_size};
        this->cache_k = Tensor(dims, this->kv_dim * this->max_len);
        this->cache_v = Tensor(dims, this->kv_dim * this->max_len);
    }

    ~Attention() {}

    void forward(Tensor& input, int pos, Tensor& output);

    int load_weights(void* cur_ptr) {
        return 0;
    }
private:
    int max_len;
    int n_heads;
    int head_size;
    int n_kv_heads;
    int kv_mul;
    int dim;
    int kv_dim;
    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor wo;
    Tensor cache_k; //[max_len, n_kv_heads * head_size]
    Tensor cache_v; //[max_len, n_kv_heads * head_size]
    Tensor cache_q; //[n_heads * head_size]
    Tensor cache_attn; //[n_kv_heads * head_size]
};