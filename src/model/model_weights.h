#pragma once
#include <vector>
#include <model/model_config.h>
#include <data/memory_manager.h>


extern MemoryManager memory_manager;


class TransformerWeights {
public:
    TransformerWeights(
        ModelConfig* config):
            config(config), wq(nullptr), wk(nullptr), wv(nullptr), wo(nullptr),
            w_gate(nullptr), w_up(nullptr), w_down(nullptr),
            w_rms_attn(nullptr), w_rms_ffn(nullptr){}

    ~TransformerWeights() {}

    void load(FILE*& fp) {
        int head_size = config->dim / config->n_heads;
        w_rms_attn = memory_manager.allocate(config->dim);
        fread(w_rms_attn, sizeof(float), config->dim, fp);
        wq = memory_manager.allocate(config->dim * config->n_heads * head_size);
        fread(wq, sizeof(float), config->dim * config->n_heads * head_size, fp);
        wk = memory_manager.allocate(config->dim * config->n_kv_heads * head_size);
        fread(wk, sizeof(float), config->dim * config->n_kv_heads * head_size, fp);
        wv = memory_manager.allocate(config->dim * config->n_kv_heads * head_size);
        fread(wv, sizeof(float), config->dim * config->n_kv_heads * head_size, fp);
        wo = memory_manager.allocate(config->dim * config->n_heads * head_size);
        fread(wo, sizeof(float), config->dim * config->n_heads * head_size, fp);
        w_rms_ffn = memory_manager.allocate(config->dim);
        fread(w_rms_ffn, sizeof(float), config->dim, fp);
        w_gate = memory_manager.allocate(config->dim * config->dim);
        fread(w_gate, sizeof(float), config->dim * config->hidden_dim, fp);
        w_down = memory_manager.allocate(config->dim * config->hidden_dim);
        fread(w_down, sizeof(float), config->dim * config->hidden_dim, fp);
        w_up = memory_manager.allocate(config->dim * config->hidden_dim);
        fread(w_up, sizeof(float), config->dim * config->hidden_dim, fp);
    }
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w_gate;
    float* w_up;
    float* w_down;
    float* w_rms_attn;
    float* w_rms_ffn;
private:
    ModelConfig* config;
};


class ModelWeights {
public:
    ModelWeights(ModelConfig* config):
        config(config), token_embedding_weights(nullptr), w_rms_final(nullptr), transformer_weights(config->n_layers, nullptr) {}
    ~ModelWeights() {
        for (auto ptr : transformer_weights) {
            delete ptr;
        }
    }
    void load(FILE*& fp) {
        token_embedding_weights = memory_manager.allocate(config->vocab_size * config->dim);
        fread(token_embedding_weights, sizeof(float), config->vocab_size * config->dim, fp);
        for (int i = 0; i < config->n_layers; i++) {
            transformer_weights[i] = new TransformerWeights(config);
            transformer_weights[i]->load(fp);
        }
        w_rms_final = memory_manager.allocate(config->dim);
        fread(w_rms_final, sizeof(float), config->dim, fp);
        w_classifier = token_embedding_weights;
    }
    float* token_embedding_weights;
    std::vector<TransformerWeights*> transformer_weights;
    float* w_rms_final;
    float* w_classifier;
private:
    ModelConfig* config;
};
