#pragma once
#include <vector>
#include <random>
#include <data/tensor.h>
#include <layers/base_layer.h>


struct ProbIndex {
    int index;
    float prob;
};

class Sampler: public BaseLayer {
public:
    Sampler(int vocab_size): vocab_size(vocab_size), prob_index(vocab_size), distribution(0.0f, 1.0f) {}

    int sample(Tensor& logits, float tempature, float top_p);
private:
    std::vector<ProbIndex> prob_index;
    int vocab_size;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution;
    int sample_argmax(Tensor& logits);
    int sample_topp(Tensor& logits, float temperature, float top_p);
};