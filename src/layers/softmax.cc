#include <vector>
#include <layers/softmax.h>
#include <data/tensor.h>


void Softmax::forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    float* input = inputs[0]->data();
    float* output = outputs[0]->data();
    int size = inputs[0]->size();
    float max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}
