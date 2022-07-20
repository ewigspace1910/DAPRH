#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>

#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace std;
using namespace nvinfer1;

// void readPGMFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
// {
//     std::ifstream infile(fileName, std::ifstream::binary);
//     assert(infile.is_open() && "Attempting to read from a file that is not open.");
//     std::string magic, h, w, max;
//     infile >> magic >> h >> w >> max;
//     infile.seekg(1, infile.cur);
//     infile.read(reinterpret_cast<char*>(buffer), inH * inW);
// }

namespace sample_onnx {

class Engine {
public:
    Engine(const string &engine_path, bool verbose=false);
    Engine(const char *onnx_model, size_t onnx_size, const vector<int>& dynamic_batch_opts, bool verbose, size_t workspace_size=(1ULL << 30));

    ~Engine();

    void save(const string &path);
    void infer(vector<void *> &buffers, int batch);

private:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;

    int mNumber{0};

    void _load(const string &path);
    void _prepare();

};
}