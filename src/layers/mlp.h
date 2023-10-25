#include <layers/base_layer.h>
#include <data/tensor.h>


class MLP: public BaseLayer {
public:
    MLP(
        Tensor& _w_gate,
        Tensor& _w_up,
        Tensor& _w_down,
        Tensor& _cache_gate,
        Tensor& _cache_up
    ): w_gate(_w_gate), w_up(_w_up), w_down(_w_down),
    cache_gate(_cache_gate), cache_up(_cache_up) {

    }

    void forward(Tensor& input, Tensor& output);

private:
    Tensor w_gate; // [hidden_dim, dim]
    Tensor w_up;  // [hidden_dim, dim]
    Tensor w_down; // [dim, hidden_dim]
    Tensor cache_gate; // [hidden_dim]
    Tensor cache_up; // [hidden_dim]
};