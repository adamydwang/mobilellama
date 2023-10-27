#pragma once
#include <spdlog/spdlog.h>
#include <sentencepiece_processor.h>

class Tokenizer {
public:
    Tokenizer(const std::string& model_path) {
        const auto status = this->processor.Load(model_path);
        if (!status.ok()) {
            spdlog::error("Tokenizer load error: {}", status.ToString());
            throw std::runtime_error("Tokenizer load error");
        }
        this->bos_id = this->processor.bos_id();
        this->eos_id = this->processor.eos_id();
        this->pad_id = this->processor.pad_id();
    }

    ~Tokenizer();
    void encode(const std::string& text, std::vector<int>& ids) {
        this->processor.Encode(text, &ids);
    }

    void decode(const std::vector<int>& ids, std::string& text) {
        this->processor.Decode(ids, &text);
    }

    int bos_id;
    int eos_id;
    int pad_id;
private:
    sentencepiece::SentencePieceProcessor processor;
};
