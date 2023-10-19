#include <vector>
#include <stdio.h>
#include <src/data/tensor.h>


class BaseLayerWeights {
public:
    virtual int load(FILE*& fp) = 0;
};


};
class BaseLayer {
public:
    virtual void forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) = 0;
    virtual void load_weights(FILE*& fp) = 0;
};