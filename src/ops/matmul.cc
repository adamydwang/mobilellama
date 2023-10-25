#include <vector>
#include <ops/matmul.h>
#include <vector>
#include <data/tensor.h>


void matmul(Tensor& input0, Tensor& input1, Tensor& output) {
    // input0: m * k
    // input1: k *n 
    // output: m * n
    std::vector<int> dims = {input0.dims[0], input1.dims[1]};
    output.reshape(dims);
    float* data_in0 = input0.data();
    float* data_in1 = input1.data();
    float* data_out = output.data();
    int m = input0.dims[0];
    int k = input0.dims[1];
    int n = input1.dims[1];
    #pragma omp parallel for private(i)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += data_in0[i * k + l] * data_in1[l * k + j];
            }
            data_out[i * n + j] = sum;
        }
    }
}