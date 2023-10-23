#pragma once

#include <string.h>
#include <vector>
#include <memory>
#include <spdlog/spdlog.h>
#include <stdio.h>


class Tensor {
public:
    Tensor() {
        this->dims = {0};
        this->data_ptr = nullptr;
    }

    Tensor(std::vector<int>& _dims) {
        this->dims = _dims;
        float* data = new float[this->size()];
        this->data_ptr = std::shared_ptr<float[]>(data, std::default_delete<float[]>());
    }

    Tensor(std::vector<int>& _dims, float* _data) {
        this->dims = _dims;
        float* data = new float[this->size()];
        memcpy(data, _data, this->size() * sizeof(float));
        this->data_ptr = std::shared_ptr<float[]>(data, std::default_delete<float[]>());
    }
    
    ~Tensor() {}

    float* data() {
        return this->data_ptr.get();
    }

    int size() {
        int size = 1;
        for (int i = 0; i < dims.size(); i++) {
            size *= dims[i];
        }
        return size;
    }

    std::vector<int> dims;
private:
    std::shared_ptr<float[]> data_ptr;
};