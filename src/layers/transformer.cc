#include <vector>
#include <data/tensor.h>
#include <layers/transformer.h>
#include <ops/matmul.h>

inline void add(Tensor& input1, Tensor& input2, Tensor& output) {
    float* data1 = input1.data();
    float* data2 = input2.data();
    float* data3 = output.data();
    input2.reshape(input1.dims);
    output.reshape(input1.dims);
    int size = input1.size();
    for (int i = 0; i < size; i++) {
        data3[i] = data1[i] + data2[i];
    }
}

void Transformer::forward(int token, int pos, Tensor& output) {
    std::vector<int> dims = {this->dim};
    Tensor embedding = Tensor(dims, this->embeddings.data() + token * this->dim);
    this->input_tensor.copy_from(embedding);
    this->rms_attn.forward(this->input_tensor, this->cache_i);
    this->attention.forward(this->cache_i, pos, this->cache_i);
    add(this->cache_i, this->input_tensor, this->cache_i);
    this->rms_ffn.forward(this->cache_i, this->cache_i);
    this->mlp.forward(this->cache_i, output);
    add(this->cache_i, output, output);
    this->rms_final.forward(output, output);
    // logits
    // [1, dim] x [dim, vocab_size] = [1, vocab_size]
    matmul(output, this->w_classifier, output);
    output.reshape(dims);
}