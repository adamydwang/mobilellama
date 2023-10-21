#pragma once

struct TransformerConfig {
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int hidden_size;
    int vocab_size;
    int max_seq_len;
    TransformerConfig(int num_layers, int num_heads, int num_kv_heads, int hidden_size, int vocab_size, int max_seq_len) {
        this->num_layers = num_layers;
        this->num_heads = num_heads;
        this->num_kv_heads = num_kv_heads;
        this->hidden_size = hidden_size;
        this->vocab_size = vocab_size;
        this->max_seq_len = max_seq_len;
    }
};

struct TransformerWeights {
    float* token_embedding_table;
    float* rms_attn_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;

    float* w1;
    float* w2;
    float* w3;

    float* rms_final_weight;
    float* wcls;
    bool load_weights(float* data, const TransformerConfig& config) {
    }
};


class Transformer {
public:
    Transformer(int num_layers, int num_heads, int hidden_size, int vocab_size, int max_seq_len, float dropout, float* weights);
};