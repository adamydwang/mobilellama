#include <string>
#include <stdio.h>


class ModelWeights {
public:
    float* token_embedding_weights;
    //rmsnorm
    float* rms_attn_weights;
    float* rms_ffn_weights;
    //
};