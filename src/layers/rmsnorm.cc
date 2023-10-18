#include <math.h>
#include <vector>
#include "rmsnorm.h"

void RMSNorm::forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    float* input = inputs[0]->data_fp;
    float* output = outputs[0]->data_fp;
    float sum = 0;
    for (int i = 0; i < this->dim; i++) {
        sum += input[i] * input[i];
    }
    sum /= this->dim;
    sum += this->eps;
    sum = 1.0f / sqrt(sum);
    for (int i = 0; i < this->dim; i++) {
        output[i] = input[i] * sum;
    }
}