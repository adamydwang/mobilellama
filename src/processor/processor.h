#pragma once

#include <string>
#include <vector>
#include <data/memory_manager.h>
#include <data/tensor.h>
#include <model/model_config.h>
#include <model/model_weights.h>
#include <model/model.h>
#include <model/tokenizer.h>
#include <spdlog/spdlog.h>


class LlmProcessor {
public:
    LlmProcessor(const std::string& model_path, const std::string& tokenizer_path);

    void generate(const std::string& text, std::string& output, float temperature=1.0, float top_p = 0.85, int max_len = 100);

    ~LlmProcessor() {
        delete this->model;
        delete this->tokenizer;
    }
private:
    Model* model;
    Tokenizer* tokenizer;
};