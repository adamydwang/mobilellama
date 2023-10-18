#include <vector>
#include "src/data/tensor.h"


class BaseLayer {
public:
    virtual void forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) = 0;
};