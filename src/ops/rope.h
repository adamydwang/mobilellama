#pragma once

#include <data/tensor.h>


void rope(Tensor& query, Tensor& key, int dim, int kv_dim, int head_size, int pos);