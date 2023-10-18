#include <vector>


class Tensor {
public:
    Tensor(std::vector<int>& dims) {
        this->data_int = nullptr;
        this->dims = dims;
        int size = 1;
        for (int i = 0; i < dims.size(); i++) {
            size *= dims[i];
        }
        this->data_fp = new float[size];
    }
    Tensor(std::vector<int>& dims, float* data) {
        this->data_int = nullptr;
        this->dims = dims;
        this->data_fp = data;
    }

    Tensor(int data) {
        dims = {1};
        data_int = new int[1];
        data_int[0] = data;
        data_fp = nullptr;
    }

    Tensor(float data) {
        dims = {1};
        data_fp = new float[1];
        data_fp[0] = data;
        data_int = nullptr;
    }

    ~Tensor() {
        if (data_fp != nullptr) {
            delete[] data_fp;
        }
        if (data_int != nullptr) {
            delete[] data_int;
        }
    }

    int size() {
        int size = 1;
        for (int i = 0; i < dims.size(); i++) {
            size *= dims[i];
        }
        return size;
    }

    float* data_fp;
    int* data_int;
    std::vector<int> dims;
};
