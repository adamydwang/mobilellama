#include <vector>
#include <src/layers/attention.h>
#include <src/data/tensor.h>
#include <src/layers/base_layer.h>


class AttentionArgs {
public:
    AttentionArgs(int n_heads, int n)
};

class Attention: public BaseLayer {
public:
    Attention(int dim): dim(dim) {}
    void forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
private:

};