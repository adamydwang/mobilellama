#include <math.h>
#include <data/tensor.h>
#include <ops/rope.h>


void rope(Tensor& query, Tensor& key, int dim, int kv_dim, int head_size, int pos) {
    float* data_q = query.data();
    float* data_k = key.data();
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float cosv = cosf(val);
        float sinv = sinf(val);
        // query
        float tmp0 = data_q[i];
        float tmp1 = data_q[i+1];
        data_q[i] = tmp0 * cosv - tmp1 * sinv;
        data_q[i+1] = tmp0 * sinv + tmp1 * cosv;
        // key
        if (i < kv_dim) {
            tmp0 = data_k[i];
            tmp1 = data_k[i+1];
            data_k[i] = tmp0 * cosv - tmp1 * sinv;
            data_k[i+1] = tmp0 * sinv + tmp1 * cosv;
        }
    }
}