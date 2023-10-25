#include <math.h>
#include <layers/rmsnorm.h>

void RMSNorm::forward(Tensor& input, Tensor& output) {
    float* data_in = input.data();
    float* data_out = input.data();
    float* w = this->weights.data();
    float sum = 0;
    int size = input.size();
    for (int i = 0; i < size; i++) {
        sum += data_in[i] * data_in[i];
    }
    sum /= size;
    sum += eps;
    sum = 1.0f / sqrt(sum);
    for (int i = 0; i < size; i++) {
        data_out[i] = data_in[i] * sum * w[i];
    }
    output = input;
}