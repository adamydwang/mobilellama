#pragma once

#include <string.h>
#include <vector>
#include <memory>
#include <spdlog/spdlog.h>


class Tensor {
public:
    Tensor() {
        this->data_ptr = nullptr;
        ref_count = new int(1);
    }

    Tensor(std::vector<int>& dims) {
        this->dims = dims;
        this->data_ptr = new float[this->size()];
        ref_count = new int(1);
    }

    Tensor(std::vector<int>& dims, float* data) {
        this->dims = dims;
        this->data_ptr = new float[this->size()];
        memcpy(this->data_ptr, data, this->size() * sizeof(float));
        ref_count = new int(1);
    }
    
    ~Tensor() {
        (*this->ref_count)--;
        if (*this->ref_count <= 0) {
            if (this->data_ptr != nullptr) {
                delete[] this->data_ptr;
            }
            delete this->ref_count;
        }
    }

    Tensor& operator=(const Tensor& other) {
        (*other.ref_count)++;
        (*this->ref_count)--;
        if (*this->ref_count <= 0) {
            if (this->data_ptr != nullptr) {
                delete[] this->data_ptr;
            }
            delete this->ref_count;
        }
        this->dims = other.dims;
        this->data_ptr = other.data_ptr;
        this->ref_count = other.ref_count;
        return *this;
    }

    float* data() {
        return this->data_ptr;
    }

    int size() {
        int size = 1;
        for (int i = 0; i < dims.size(); i++) {
            size *= dims[i];
        }
        return size;
    }

    float* data_ptr;
    std::vector<int> dims;
private:
    int *ref_count;
};