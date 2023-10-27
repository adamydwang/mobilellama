#pragma once

#include <string>
#include <vector>
#include <data/tensor.h>
#include <model/model_config.h>
#include <model/model_weights.h>
#include <layers/transformer.h>
#include <layers/rmsnorm.h>
#include <ops/matmul.h>
#include <layers/sampler.h>
#include <spdlog/spdlog.h>


class Model {
public:
    Model(const std::string& model_path) {
        FILE* fp = fopen(model_path.c_str(), "rb");
        if (fp == NULL) {
            spdlog::error("Model file not found: {}", model_path);
            throw std::runtime_error("Model file not found");
        }
        this->config = new ModelConfig();
        this->config->load(fp);
        this->weights = new ModelWeights(this->config);
        this->weights->load(fp);
        fclose(fp);
    }

    int build() {
        std::vector<int> dims;
        int head_size = this->config->dim / this->config->n_heads;
        for (int i = 0; i < this->config->n_layers; i++) {
            
            dims = {this->config->dim, this->config->n_heads * head_size};
            Tensor wq(dims, this->weights->transformer_weights[i]->wq);
            dims = {this->config->dim, this->config->n_kv_heads * head_size};
            Tensor wk(dims, this->weights->transformer_weights[i]->wk);
            Tensor wv(dims, this->weights->transformer_weights[i]->wv);
            dims = {this->config->n_heads * head_size, this->config->dim};
            Tensor wo(dims, this->weights->transformer_weights[i]->wo);
            dims = {this->config->dim};
            Tensor cache_q(dims);
            dims = {this->config->n_heads, this->config->max_seq_len};
            Tensor cache_attn(dims);
            dims = {this->config->hidden_dim, this->config->dim};
            Tensor w_gate(dims, this->weights->transformer_weights[i]->w_gate);
            Tensor w_up(dims, this->weights->transformer_weights[i]->w_up);
            dims = {this->config->dim, this->config->hidden_dim};
            Tensor w_down(dims, this->weights->transformer_weights[i]->w_down);
            dims = {this->config->hidden_dim};
            Tensor cache_gate(dims);
            Tensor cache_up(dims);
            dims = {this->config->dim};
            Tensor w_rms_attn(dims, this->weights->transformer_weights[i]->w_rms_attn);
            Tensor w_rms_ffn(dims, this->weights->transformer_weights[i]->w_rms_ffn);
            
            dims = {this->config->dim};
            Tensor cache_i(dims);
            Transformer* transformer = new Transformer(
                this->config->n_heads, this->config->n_kv_heads, this->config->dim, this->config->max_seq_len,
                wq, wk, wv, wo, cache_q, cache_attn,
                w_gate, w_up, w_down, cache_gate, cache_up,
                w_rms_attn, w_rms_ffn, cache_i
            );
            this->transformers.push_back(transformer);
        }
        dims = {this->config->dim};
        input_tensor = Tensor(dims);
        dims = {this->config->vocab_size, this->config->dim};
        embeddings = Tensor(dims, this->weights->token_embedding_weights);
        dims = {this->config->dim};
        Tensor w_rms_final(dims, this->weights->w_rms_final);
        this->rms_final = new RMSNorm(w_rms_final);
        dims = {this->config->vocab_size, this->config->dim};
        this->w_classifier = Tensor(dims, this->weights->w_classifier);
        this->sampler = new Sampler(this->config->vocab_size);
        dims = {1, this->config->vocab_size};
        this->output_tensor = Tensor(dims);
        return 0;
    }

    int forward(int token, int pos, float temperature=1.0, float top_p=0.85) {
        std::vector<int> dims = {this->config->dim};
        Tensor embedding = Tensor(dims, this->embeddings.data() + token * this->config->dim);
        this->input_tensor.copy_from(embedding);
        for (int i = 0; i < this->config->n_layers; i++) {
            this->transformers[i]->forward(this->input_tensor, pos, this->input_tensor);
        }
        this->rms_final->forward(this->input_tensor, this->input_tensor);
        // logits
        // [vocab_size, dim] x [dim, 1] = [vocab_size, 1]
        dims = {this->config->dim, 1};
        this->input_tensor.reshape(dims);
        matmul(this->w_classifier, this->input_tensor, this->output_tensor);
        dims = {this->config->vocab_size};
        output_tensor.reshape(dims);
        return this->sampler->sample(output_tensor, temperature, top_p);
    }

    ~Model() {
        delete this->config;
        delete this->weights;
        delete this->sampler;
        delete this->rms_final;
        for (auto ptr : this->transformers) {
            delete ptr;
        } 
    }
    ModelConfig* config;
private:
    ModelWeights* weights;
    std::vector<Transformer*> transformers;
    Sampler* sampler;
    Tensor output_tensor;
    Tensor input_tensor;
    RMSNorm* rms_final;
    Tensor w_classifier; // [dim, vocab_size]
    Tensor embeddings; // [vocab_size, dim]
};