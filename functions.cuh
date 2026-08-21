#pragma once
#include "config.cuh"
#include "kernels.cuh"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

static void render(const uchar3* fb){
    std::string out;
    out.reserve(W * H * 24);
    for(int r = 0; r < ROWS; r++){
        int lastF = -1;
        int lastB = -1;
        for(int x = 0; x < COLS; x++){
            uchar3 t = fb[(2*r)   * W + x];
            uchar3 b = fb[(2*r+1) * W + x];
            char buf[32];

            int ft = (t.x << 16) | (t.y << 8) | t.z;
            int bt = (b.x << 16) | (b.y << 8) | b.z;
            if(ft != lastF){out.append(buf, snprintf(buf, sizeof buf,
                            "\x1b[38;2;%d;%d;%dm", t.x, t.y, t.z)); lastF = ft;
            }
            if(bt != lastB){out.append(buf, snprintf(buf, sizeof buf,
                            "\x1b[48;2;%d;%d;%dm", b.x, b.y, b.z)); lastB = bt;
            }
            out += "\xE2\x96\x80";
        }
        out += "\x1b[0m\x1b[K\n";
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
}

static int sideForDensity(int n, float fishPerCell)
{
    double area = (double)n * (double)(VIEW_R * VIEW_R) / (double)fishPerCell;
    return (int)std::sqrt(area) + 1;
}

static void simInit(Sim& s, int N, int simW = W, int simH = H)
{
    s.N        = N;
    s.cur      = 0;
    s.simW     = simW;
    s.simH     = simH;
    s.cellSize = VIEW_R;
    s.gridW    = (int)(s.simW / s.cellSize) + 1;
    s.gridH    = (int)(s.simH / s.cellSize) + 1;
    s.nCells   = s.gridW * s.gridH;

    size_t fbytes = (size_t)N * sizeof(float2);
    size_t ibytes = (size_t)N * sizeof(int);
    size_t cbytes = (size_t)N * sizeof(float3);

    for (int k = 0; k < 2; k++) {
        CUDA_CHECK(cudaMalloc(&s.pos[k], fbytes));
        CUDA_CHECK(cudaMalloc(&s.vel[k], fbytes));
    }
    CUDA_CHECK(cudaMalloc(&s.cellId,    ibytes));
    CUDA_CHECK(cudaMalloc(&s.fishId,    ibytes));
    CUDA_CHECK(cudaMalloc(&s.cellStart, (size_t)s.nCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.cellEnd,   (size_t)s.nCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.color,     cbytes));
    CUDA_CHECK(cudaMalloc(&s.posSort,   fbytes));
    CUDA_CHECK(cudaMalloc(&s.velSort,   fbytes));
    CUDA_CHECK(cudaMalloc(&s.colSort,   cbytes));

    std::vector<float2> hp(N), hv(N);
    std::vector<float3> hc(N);
    const float3 ROYGBIV[7] = {
        make_float3(1.0f, 0.0f, 0.0f),   // red
        make_float3(1.0f, 0.5f, 0.0f),   // orange
        make_float3(1.0f, 1.0f, 0.0f),   // yellow
        make_float3(0.0f, 1.0f, 0.0f),   // green
        make_float3(0.0f, 0.3f, 1.0f),   // blue
        make_float3(0.3f, 0.0f, 1.0f),   // indigo
        make_float3(0.6f, 0.0f, 1.0f),   // violet
    };

    for (int i = 0; i < N; i++) {
        float rx = (rand() / (float)RAND_MAX) * (s.simW - 1.0f);
        float ry = (rand() / (float)RAND_MAX) * (s.simH - 1.0f);
        hp[i] = make_float2(rx, ry);
        float a = (rand() / (float)RAND_MAX) * 6.2832f;
        hv[i] = make_float2(cosf(a) * SPEED, sinf(a) * SPEED);
        hc[i] = ROYGBIV[i % 7];
    }
    CUDA_CHECK(cudaMemcpy(s.pos[0], hp.data(), fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.vel[0], hv.data(), fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.color,  hc.data(), cbytes, cudaMemcpyHostToDevice));
}

static void simFree(Sim& s)
{
    for (int k = 0; k < 2; k++) { cudaFree(s.pos[k]); cudaFree(s.vel[k]); }
    cudaFree(s.cellId);
    cudaFree(s.fishId);
    cudaFree(s.cellStart);
    cudaFree(s.cellEnd);
    cudaFree(s.color);
    cudaFree(s.posSort);
    cudaFree(s.velSort);
    cudaFree(s.colSort);
}

static void simStep(Sim& s, float dt)
{
    const int TPB  = 256;
    int fishBlocks = (s.N + TPB - 1) / TPB;

    k_assign<<<fishBlocks, TPB>>>(s.pos[s.cur], s.cellId, s.fishId,
                                  s.N, s.gridW, s.gridH, s.cellSize);
    thrust::sort_by_key(thrust::device, s.cellId, s.cellId + s.N, s.fishId);
    CUDA_CHECK(cudaMemset(s.cellStart, -1, (size_t)s.nCells * sizeof(int)));
    k_bounds<<<fishBlocks, TPB>>>(s.cellId, s.cellStart, s.cellEnd, s.N);

    // Permute pos/vel/color into cell-sorted order so the neighbour gather
    // in k_boids reads contiguously instead of chasing fishId indirection.
    k_reorder<<<fishBlocks, TPB>>>(s.pos[s.cur], s.vel[s.cur], s.color,
                                   s.fishId, s.posSort, s.velSort,
                                   s.colSort, s.N);
    k_boids<<<fishBlocks, TPB>>>(s.posSort, s.velSort,
                                 s.pos[1 - s.cur], s.vel[1 - s.cur],
                                 s.cellStart, s.cellEnd,
                                 s.N, dt, s.simW, s.simH,
                                 s.gridW, s.gridH, s.cellSize);
    std::swap(s.color, s.colSort);
    CUDA_CHECK(cudaGetLastError());

    s.cur = 1 - s.cur;
}

static void benchSweep(const int* sizes, int nSizes, bool fixedDomain,
                       int warmup, int iters, float fishPerCell)
{
    std::printf("\n  %9s  %11s  %9s  %10s  %10s  %11s\n",
                "fish", "domain", "grid", "fish/cell", "ms/step", "steps/sec");
    std::printf("  %9s  %11s  %9s  %10s  %10s  %11s\n",
                "---------", "-----------", "---------",
                "----------", "----------", "-----------");

    for (int si = 0; si < nSizes; si++) {
        int n = sizes[si];
        Sim s{};

        if (fixedDomain) {
            simInit(s, n);
        } else {
            int side = sideForDensity(n, fishPerCell);
            simInit(s, n, side, side);
        }

        for (int k = 0; k < warmup; k++) simStep(s, 1.0f / 60.0f);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) simStep(s, 1.0f / 60.0f);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                    / iters;
        double perCell = (double)n / (double)s.nCells;

        char dom[32], grd[32];
        std::snprintf(dom, sizeof dom, "%dx%d", s.simW,  s.simH);
        std::snprintf(grd, sizeof grd, "%dx%d", s.gridW, s.gridH);
        std::printf("  %9d  %11s  %9s  %10.1f  %10.3f  %11.0f\n",
                    n, dom, grd, perCell, ms, 1000.0 / ms);

        simFree(s);
    }
    std::printf("\n");
}

static void benchScaling()
{
    const int sizes[] = {10000, 50000, 200000, 500000,
                         1000000, 2000000, 4000000};
    const float FPC = 10.0f;

    std::printf("\n  CONSTANT DENSITY (~%.0f fish/cell), world grows as sqrt(N).\n"
                "  Each fish sees a fixed number of candidates regardless of N,\n"
                "  so this is the case the spatial grid was built for: expect\n"
                "  ms/step to rise roughly LINEARLY.\n", FPC);

    benchSweep(sizes, (int)(sizeof(sizes) / sizeof(sizes[0])),
               false, 500, 100, FPC);
}

static void benchDensity()
{
    const int sizes[] = {1000, 5000, 20000, 50000, 100000};

    std::printf("\n  FIXED %dx%d WORLD -- only 45 cells, at every N. Each fish you\n"
                "  add lands in somebody's 3x3 neighbourhood, so per-fish work\n"
                "  grows with N and the total goes QUADRATIC. Same kernels, same\n"
                "  grid; the difference is entirely the configuration.\n", W, H);

    benchSweep(sizes, (int)(sizeof(sizes) / sizeof(sizes[0])),
               true, 50, 20, 0.0f);
}
