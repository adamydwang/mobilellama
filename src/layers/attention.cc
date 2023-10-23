#include <layers/attention.h>
#include <layers/matmul.h>


void Attention::forward(Tensor& input, Tensor& output) {
    // input: [n, d] , n=num_tokens, d=vocab_size
    // wq: [d_q, d], d_q = d
    // wk = wv = [d_kv, d]
    // wq * wk * kv = [d_q, d]

    int n = input.dims[0];
    int d = input.dims[1];
    int d_kv = d / this->n_heads;
    int d_q = d;
    int d_k = d;
    int d_v = d;
    float* data_in = input.data();
    float* data_out = output.data();
    float* data_q = wq;
    float* data_k = wk;
    float* data_v = wv;
}