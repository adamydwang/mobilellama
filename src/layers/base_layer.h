#pragma once

#include <vector>
#include <stdio.h>
#include <data/tensor.h>


class BaseLayer {
public:
    BaseLayer() {}
    virtual ~BaseLayer() {}
    virtual int weights_size() {
        return 0;
    }
};