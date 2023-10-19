#include <vector>
#include <src/layers/softmax.h>
#include <src/data/tensor.h>


void Softmax::forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    float* input = inputs[0]->get_float_data();
    float* output = outputs[0]->get_float_data();
    float max = input[0];
    for (int i = 1; i < this->size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < this->size; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < this->size; i++) {
        output[i] /= sum;
    }
}
