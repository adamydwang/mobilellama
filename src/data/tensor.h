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
        ref_count = new int(1);
        printf("new ref_count: %p\n", ref_count);
    }

    Tensor(std::vector<int>& _dims) {
        this->dims = _dims;
        this->data_ptr = new float[this->size()];
        printf("new data_ptr: %p\n", this->data_ptr);
        ref_count = new int(1);
        printf("new ref_count: %p\n", ref_count);
    }

    Tensor(std::vector<int>& _dims, float* _data) {
        this->dims = _dims;
        this->data_ptr = new float[this->size()];
        printf("new data_ptr: %p\n", this->data_ptr);
        memcpy(this->data_ptr, _data, this->size() * sizeof(float));
        ref_count = new int(1);
        printf("new ref_count: %p\n", ref_count);
    }
    
    ~Tensor() {
        (*this->ref_count)--;
        if (*this->ref_count <= 0) {
            if (this->data_ptr != nullptr) {
                printf("delete data_ptr: %p\n", this->data_ptr);
                delete[] this->data_ptr;
            }
            printf("delete ref_count: %p\n", this->ref_count);
            delete this->ref_count;
        }
    }

    Tensor& operator=(const Tensor& other) {
        (*other.ref_count)++;
        (*this->ref_count)--;
        if (*this->ref_count <= 0) {
            if (this->data_ptr != nullptr) {
                printf("delete data_ptr: %p\n", this->data_ptr);
                delete[] this->data_ptr;
            }
            printf("delete ref_count: %p\n", this->ref_count);
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

    std::vector<int> dims;
private:
    int *ref_count;
    float* data_ptr;
};