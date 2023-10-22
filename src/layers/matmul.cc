#include <vector>
#include <layers/matmul.h>
#include <data/tensor.h>


void Matmul::forward(Tensor& input0, Tensor& input1, Tensor& output) {
    if (cache == nullptr || cache->dims[0] != input0.dims[0] || cache->dims[1] != input1.dims[1]) {
        std::vector<int> output_dims = {input0.dims[0], input1.dims[1]};
        cache = new Tensor(output_dims);
    }   
    // input1: m * k
    float* data_in0 = input0.data();
    // input2: k * n
    float* data_in1 = input1.data();
    // output: m * n
    float* data_out = cache->data();
    int m = input0.dims[0];
    int k = input0.dims[1];
    int n = input1.dims[1];
    #pragma omp parallel for private(i)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += data_in0[i * k + l] * data_in1[l * n + j];
            }
            data_out[i * n + j] = sum;
        }
    }
    output = *cache;
}