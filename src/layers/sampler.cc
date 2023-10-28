#include <layers/sampler.h>
#include <algorithm>
#include <math.h>
#include <random>
#include <data/tensor.h>
#include <ops/softmax.h>


int Sampler::sample_argmax(Tensor& logits) {
    float* probs = logits.data();
    float max_prob = probs[0];
    int max_index = 0;
    for (int i = 0; i < this->vocab_size; ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_index = i;
        }
    }
    printf("max_index=%d, max_prob = %f\n", max_index, max_prob);
    return max_index;
}


int Sampler::sample_topp(Tensor& logits, float temperature, float top_p) {
    float* probs = logits.data();
    for (int i = 0; i < this->vocab_size; ++i) {
        probs[i] /= temperature;
    }
    softmax(logits, logits);
    float random_value = this->distribution(this->generator);
    int length = 0;
    float threshold = (1.0f - top_p) / (this->vocab_size - 1);
    for (int i = 0; i < this->vocab_size; ++i) {
        if (probs[i] >= threshold) {
            this->prob_index[length].index = i;
            this->prob_index[length].prob = probs[i];
            length++;
        }
    }
    std::sort(prob_index.begin(), prob_index.begin() + length, [](const ProbIndex& a, const ProbIndex& b) {
        return a.prob > b.prob;
    });

    float cumulated_prob = 0;
    int last_index = length - 1;
    for (int i = 0; i < length; ++i) {
        cumulated_prob += this->prob_index[i].prob;
        if (cumulated_prob > top_p) {
            last_index = i;
            break;
        }
    }

    threshold = random_value * cumulated_prob;
    cumulated_prob = 0;
    for (int i = 0; i <= last_index; ++i) {
        cumulated_prob += this->prob_index[i].prob;
        if (cumulated_prob > threshold) {
            return this->prob_index[i].index;
        }
    }
    return this->prob_index[last_index].index;
}


int Sampler::sample(Tensor& logits, float temperature, float top_p) {
    printf("temperature=%f, top_p=%f\n", temperature, top_p);
    if (top_p < 0.0f || top_p > 1.0f) {
        printf("top_p should be in [0.0, 1.0]\n");
        top_p = 1.0f;
    }
    if (top_p < 1e-5f) {
        printf("sample_argmax\n");
        return this->sample_argmax(logits);
    } else {
        printf("sample_topp\n");
        return this->sample_topp(logits, temperature, top_p);
    }
}