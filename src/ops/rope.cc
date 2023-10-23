#include <math.h>


void rope(float* query, float* key, int dim, int kv_dim, int head_size, int pos) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float cosv = cosf(val);
        float sinv = sinf(val);
        // query
        float tmp0 = query[i];
        float tmp1 = query[i+1];
        query[i] = tmp0 * cosv - tmp1 * sinv;
        query[i+1] = tmp0 * sinv + tmp1 * cosv;
        // key
        if (i < kv_dim) {
            tmp0 = key[i];
            tmp1 = key[i+1];
            key[i] = tmp0 * cosv - tmp1 * sinv;
            key[i+1] = tmp0 * sinv + tmp1 * cosv;
        }
    }
}