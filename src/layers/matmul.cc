#include <vector>
#include "matmul.h"
#include <src/data/tensor.h>


void Matmul::forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // input1: m * k
    float* input1 = inputs[0]->data_fp;
    // input2: k * n
    float* input2 = inputs[1]->data_fp;
    // output: m * n
    float* output = outputs[0]->data_fp;
    int m = inputs[0]->dims[0];
    int k = inputs[0]->dims[1];
    int n = inputs[1]->dims[1];
    #pragma omp parallel for private(i)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += input1[i * k + l] * input2[l * n + j];
            }
            output[i * n + j] = sum;
        }
    }
}