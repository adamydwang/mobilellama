#include <vector>


enum DataType {
    DTYPE_FLOAT,
    DTYPE_INT,
    DTYPE_BOOL
};


class Tensor {
public:
    Tensor(std::vector<int>& dims, DataType dtype=DTYPE_FLOAT) {
        this->dims = dims;
        this->dtype = dtype;
        int size = 1;
        for (int i = 0; i < dims.size(); i++) {
            size *= dims[i];
        }
        this->data = new float[size * this->get_bytes()];
    }

    Tensor(std::vector<int>& dims, float* data) {
        this->dims = dims;
        this->dtype = DTYPE_FLOAT;
        this->data = data;
    }

    Tensor(int data) {
        this->dims = {1};
        this->dtype = DTYPE_INT;
        this->data = new int[1];
        (int*)this->data[0] = data;
    }

    Tensor(float data) {
        this->dims = {1};
        this->dtype = DTYPE_FLOAT;
        this->data = new float[1];
        (float*)this->data[0] = data;
    }

    ~Tensor() {
        if (data != nullptr) {
            delete[] data;
        }
    }

    float* get_float_data() {
        return (float*)this->data;
    }

    int get_int() {
        return (int*)this->data[0];
    }

    int size() {
        int size = 1;
        for (int i = 0; i < dims.size(); i++) {
            size *= dims[i];
        }
        return size;
    }

    int bytes() {
        return this->size() * this->get_bytes();
    }

    void* data;
    std::vector<int> dims;
    DataType dtype;
private:
    int get_bytes() {
        switch (this->dtype) {
            case DTYPE_FLOAT:
                return 4;
            case DTYPE_INT:
                return 4;
            case DTYPE_BOOL:
                return 1;
        }
        return 1;
    }
};
