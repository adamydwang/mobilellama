#pragma once

#include <data/tensor.h>
#include <layers/base_layer.h>
#include <layers/attention.h>
#include <layers/mlp.h>
#include <layers/rmsnorm.h>


class Transformer: public BaseLayer {
public:
    Transformer(
            // attention
            int _n_heads,
            int _n_kv_heads,
            int _dim,  // token embedding dim
            int _max_len, // max sequence length
            Tensor& _wq, // [dim, n_heads * head_size]
            Tensor& _wk, // [dim, n_kv_heads * head_size]
            Tensor& _wv, // [dim, n_kv_heads * head_size]
            Tensor& _wo, // [n_heads * head_size, dim]
            Tensor& _cache_q, // [n_heads * head_size]
            Tensor& _cache_attn, // [n_heads, max_len]
            // mlp
            Tensor& _w_gate,
            Tensor& _w_up,
            Tensor& _w_down,
            Tensor& _cache_gate,
            Tensor& _cache_up,
            // other
            Tensor& _w_rms_attn,
            Tensor& _w_rms_ffn,
            Tensor& _cache_i // [n_heads * head_size]
    ): attention(_n_heads, _n_kv_heads, _dim, _max_len, _wq, _wk, _wv, _wo, _cache_q, _cache_attn),
    mlp(_w_gate, _w_up, _w_down, _cache_gate, _cache_up),
    rms_attn(_w_rms_attn),
    rms_ffn(_w_rms_ffn),
    cache_i(_cache_i), dim(_dim) {}

    void forward(Tensor& input, int pos, Tensor& output);

private:
    Attention attention;
    MLP mlp;
    RMSNorm rms_attn;
    RMSNorm rms_ffn;
    Tensor cache_i; // [n_heads * head_size]
    int dim;
};