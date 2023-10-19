#include <vector>
#include <src/layers/base_layer.h>


class Softmax: public BaseLayer {
public:
    Softmax() {}
    void forward(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
};