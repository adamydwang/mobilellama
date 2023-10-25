#pragma once

#include <string.h>
#include <vector>
#include <spdlog/spdlog.h>
#include <stdio.h>
#include <data/memory_manager.h>

extern MemoryManager memory_manager;

class Tensor {
public:
    Tensor() {
        this->dims = {0};
        this->data_ptr = nullptr;
    }

    Tensor(std::vector<int>& dims, int space_size = 0) {
        this->dims = dims;
        if (space_size == 0) {
            space_size = this->size();
        }
        this->space_size = space_size;
        this->data_ptr = memory_manager.allocate(this->space_size);
    }
    Tensor(std::vector<int>& dims, float* data, int space_size = 0) {
        this->dims = dims;
        if (space_size == 0) {
            space_size = this->size();
        }
        this->space_size = space_size;
        this->data_ptr = data;
    }
    
    ~Tensor() {}

    float* data() {
        return this->data_ptr;
    }

    void reshape(std::vector<int>& dims) {
        this->dims = dims;
    }

    void copy_from(Tensor& other) {
        if (this->space_size != other.space_size) {
            spdlog::error("Tensor copy_from error: space_size not equal");
            return;
        }
        memcpy(this->data_ptr, other.data_ptr, this->space_size * sizeof(float));
    }
    
    int size() {
        int size = 1;
        for (int i = 0; i < dims.size(); i++) {
            size *= dims[i];
        }
        return size;
    }

    std::vector<int> dims;
    int space_size;
private:
    float* data_ptr;
};