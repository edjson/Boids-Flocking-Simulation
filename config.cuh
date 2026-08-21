#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

constexpr int COLS = 120;
constexpr int ROWS = 34;
constexpr int W    = COLS;
constexpr int H    = ROWS * 2;

#define CUDA_CHECK(call)                                                \
    do{                                                                 \
        cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess){                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",            \
                         cudaGetErrorString(err_), __FILE__, __LINE__); \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

struct Sim{
    int N;
    float2 *pos[2];
    float2 *vel[2];
    int    *cellId;
    int    *fishId;
    int    *cellStart;
    int    *cellEnd;
    float3 *color;
    float2 *posSort, *velSort;
    float3 *colSort;
    int     cur;
    int     simW;
    int     simH;
    int     gridW;
    int     gridH;
    int     nCells;
    float   cellSize;
};
