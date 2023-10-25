#include <vector>
#include <ops/softmax.h>
#include <data/tensor.h>


void softmax(Tensor& input, Tensor& output) {
    float* data_in = input.data();
    float* data_out = input.data();
    int size = input.size();
    float maxv = data_in[0];
    for (int i = 1; i < size; i++) {
        if (data_in[i] > maxv) {
            maxv = data_in[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < size; i++) {
        data_out[i] = exp(data_in[i] - maxv);
        sum += data_out[i];
    }
    for (int i = 0; i < size; i++) {
        data_out[i] /= sum;
    }
    output = input;
}
