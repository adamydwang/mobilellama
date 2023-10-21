#pragma once

#include <vector>
#include <stdio.h>
#include <data/tensor.h>


class BaseLayerParameter {
public:
    BaseLayerParameter() {}
    virtual ~BaseLayerParameter() {}
    virtual int load(FILE*& fp) = 0;
};


class BaseLayer {
public:
    BaseLayer() {}
    virtual ~BaseLayer() {}
    virtual void load_weights(FILE*& fp) = 0;
};