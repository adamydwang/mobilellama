#pragma once
#include <vector>


class MemoryManager {
public:
    MemoryManager() {
    }

    ~MemoryManager() {
        for (auto ptr : memory_pool) {
            delete[] ptr;
        }
    }

    float* allocate(int size) {
        float* ptr = new float[size];
        memory_pool.push_back(ptr);
        return ptr;
    }

private:
    std::vector<float*> memory_pool;
};


static MemoryManager memory_manager;