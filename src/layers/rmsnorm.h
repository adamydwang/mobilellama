#include <vector>
#include <stdio.h>
#include <src/layers/base_layer.h>
#include <src/data/tensor.h>


class RMSNormWeights: public BaseLayerWeights {
public:
    RMSNormWeights(): dim(0), eps(0), weights(nullptr) {}
    int load(FILE* fp) {
        int res = fread(&dim, sizeof(int), 1, fp);
        res += fread(&eps, sizeof(float), 1, fp);
        weights = new float[dim];
        res += fread(weights, sizeof(float), dim, fp);
        return res;
    }
private:
    int dim;
    float eps;
    float* weights;
};


class RMSNorm: public BaseLayer {
public:
    RMSNorm() {}
    void forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    void load_weights(FILE*& fp) {
        weights = new RMSNormWeights();
        weights->load(fp);
    }
private:
    RMSNormWeights* weights;
};