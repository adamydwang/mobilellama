#include <vector>
#include <math.h>
#include <layers/mlp.h>
#include <ops/matmul.h>

void MLP::forward(Tensor& input, Tensor& output) {
    std::vector<int> dims = {input.size(), 1};
    input.reshape(dims);
    // [hidden, dim] x [dim, 1] = [hidden, 1
    matmul(this->w_gate, input, this->cache_gate);
    matmul(this->w_up, input, this->cache_up);
    float* data = this->cache_gate.data();
    float* data_up = this->cache_up.data();
    int size = this->cache_gate.size();
    for (int i = 0; i < size; ++i) {
        data[i] *= (1.0f / (1.0f + expf(-data[i])));
        data[i] *= data_up[i];
    }
    // [dim, hidden] x [hidden, 1] = [dim, 1]
    matmul(this->w_down, this->cache_gate, output); 
}