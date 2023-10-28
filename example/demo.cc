#include <string>
#include <iostream>
#include <processor/processor.h>


int main(int argc, char** argv) {
    if (argc < 3) {
        spdlog::error("Usage: {} <model_path> <tokenizer_path>", argv[0]);
        return 1;
    }
    float temperature = 0.0f;
    float top_p = 0.0f;
    int max_len = 32;
    std::string model_path = argv[1];
    std::string tokenizer_path = argv[2];
    LlmProcessor *processor = new LlmProcessor(model_path, tokenizer_path);
    while (true) {
        std::cout << "Input: ";
        std::string input;
        std::getline(std::cin, input);
        std::string output;
        processor->generate(input, output, temperature, top_p, max_len);
        std::cout << "Output: " << output << std::endl;
    }
    return 0;
}