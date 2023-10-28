# MobileLLaMA
## A Lightweight C++ Implementation of LLaMA for Mobile Devices

# 1. Milestones

- [ ] Naive version: pure c++ implementation of Matrix manipulation
  - [x] C++ Inference Engine
  - [ ] Model Transform tool: PyTorch model to MobileLLaMA model 
- [ ] OpenBLAS version: speedup matrix manipulation by OpenBLAS


# 2. Build and Run

## How to build?

```
$ git clone --recurse-submodules https://github.com/adamydwang/mobilellama.git
$ cd mobilellama
$ cd deps && bash build.sh   #build thirdparty dependencies
$ mkdir ../build && cd ../build && cmake .. && make
```

***Reminds:*** executables and libraries are output to directories: *lib* and *bin*

## How to run demo

***!!!caution: model transformer tool has not been ready, so can not run yet***

```
$ cd bin
$ ./demo ${model_path} ${tokenizer_path}
```

