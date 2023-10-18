#include <vector>
#include "base_layer.h"
#include <src/data/tensor.h>


class RMSNorm: public BaseLayer {
public:
    RMSNorm(int dim, float eps): dim(dim), eps(eps) {}
    void forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
private:
    int dim;
    float eps;
};