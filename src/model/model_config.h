#pragma once

#include <stdio.h>
#include <string>

class ModelConfig {
public:
    ModelConfig(int version = 0, int dim=0, int hidden_dim=0, int n_layers=0, int n_heads=0, int n_kv_heads=0, int vocab_size=0, int max_seq_len=0) :
        version(version), dim(dim), hidden_dim(hidden_dim), n_layers(n_layers), n_heads(n_heads), n_kv_heads(n_kv_heads), vocab_size(vocab_size), max_seq_len(max_seq_len) {}

    void print() {
        printf("ModelConfig: version=%d, dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, max_seq_len=%d\n",
            version, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len);
    }

    void load(FILE*& fp) {
        fread(&version, sizeof(int), 1, fp);
        fread(&dim, sizeof(int), 1, fp);
        fread(&hidden_dim, sizeof(int), 1, fp);
        fread(&n_layers, sizeof(int), 1, fp);
        fread(&n_heads, sizeof(int), 1, fp);
        fread(&n_kv_heads, sizeof(int), 1, fp);
        fread(&vocab_size, sizeof(int), 1, fp);
        fread(&max_seq_len, sizeof(int), 1, fp);
    }
    
public:
    int version;
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
};