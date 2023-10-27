#include <processor/processor.h>

LlmProcessor::LlmProcessor(const std::string& model_path, const std::string& tokenizer_path) {
    FILE* fp = fopen(model_path.c_str(), "rb");
    if (fp == NULL) {
        spdlog::error("Model file not found: {}", model_path);
        throw std::runtime_error("Model file not found");
    }
    this->model = new Model(model_path);
    this->model->build();
    fclose(fp);
    this->tokenizer = new Tokenizer(tokenizer_path);
}

void LlmProcessor::generate(const std::string& text, std::string& output, float temperature, float top_p, int max_len) {
        max_len = std::min(max_len, this->model->config->max_seq_len);
        std::vector<int> ids;
        this->tokenizer->encode(text, ids);
        if (ids.size() > this->model->config->max_seq_len) {
            return;
        }
        int pos = 0;
        int token = 0;
        for (auto id : ids) {
            token = this->model->forward(id, pos++);
        }
        std::vector<int> output_tokens;
        output_tokens.push_back(token);
        for (int i  = ids.size(); i < max_len; i++) {
            token = this->model->forward(token, pos++, temperature, top_p);
            if (token == this->tokenizer->eos_id) {
                break;
            }
            output_tokens.push_back(token);
        }
        this->tokenizer->decode(output_tokens, output);
    }