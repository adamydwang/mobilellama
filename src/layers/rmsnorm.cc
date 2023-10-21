#include <math.h>
#include <vector>
#include <layers/rmsnorm.h>

void RMSNorm::forward(Tensor& input, Tensor& output) {
    float* data_in = input.data();
    float* data_out = output.data();
    float sum = 0;
    for (int i = 0; i < this->params->dim; i++) {
        sum += data_in[i] * data_in[i];
    }
    sum /= this->params->dim;
    sum += this->params->eps;
    sum = 1.0f / sqrt(sum);
    for (int i = 0; i < this->params->dim; i++) {
        data_out[i] = data_in[i] * sum * this->params->weights[i];
    }
}