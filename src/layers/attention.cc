#include <vector>
#include <layers/attention.h>
#include <ops/matmul.h>
#include <ops/rope.h>
#include <ops/softmax.h>


void Attention::forward(Tensor& input, int pos, Tensor& output) {
    std::vector<int> dims = {1, this->dim};
    input.reshape(dims);
    dims = {this->n_kv_heads, this->head_size};
    Tensor kcache = Tensor(dims, this->cache_k.data() + this->kv_dim * pos); // current kcache: [n_kv_heads, head_size]
    Tensor vcache = Tensor(dims, this->cache_v.data() + this->kv_dim * pos); // current vcache: [n_kv_heads, head_size]
    matmul(input, this->wq, this->cache_q); // [1, dim] x [dim, n_heads * head_size] = [1, n_heads * head_size]
    matmul(input, this->wk, kcache); // [1, dim] x [dim, n_kv_heads * head_size] = [1, n_kv_heads * head_size]
    matmul(input, this->wv, vcache); // [1, dim] x [dim, n_kv_heads * head_size] = [1, n_kv_heads * head_size]
    rope(input, kcache, this->dim, this->kv_dim, head_size, pos);

    // multihead attention
    for (int h = 0; h < this->n_heads; ++h) {
        float* q = this->cache_q.data() + h * this->head_size;
        float* att = this->cache_attn.data() + h * this->max_len;
        for (int i = 0; i <= pos; ++i) {
            float* k = this->cache_k.data() + i * this->kv_dim + h * this->head_size;
            float sum = 0;
            for (int j = 0; j < this->head_size; ++j) {
                sum += q[j] * k[j];
            }
            att[i] = sum;
        }
        dims = {pos + 1};
        Tensor att_tensor = Tensor(dims, att);
        softmax(att_tensor, att_tensor);
        float* input_head = input.data() + h * this->head_size;
        memset(input_head, 0, this->head_size * sizeof(float));
        for (int i = 0; i <= pos; ++i) {
            float* v = this->cache_v.data() + i * this->kv_dim + h * this->head_size;
            float att_val = att[i];
            for (int j = 0; j < this->head_size; ++j) {
                input_head[j] += att_val * v[j];
            }
        }
    }
    matmul(input, this->wo, input); // [1, n_heads * head_size] x [n_heads * head_size, dim] = [1, dim]
}