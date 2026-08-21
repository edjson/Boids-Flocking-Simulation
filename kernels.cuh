#pragma once
#include "config.cuh"

#define SEP_R   5.0f
#define VIEW_R  14.0f
#define SEP_W   45.0f
#define ALI_W   8.0f
#define COH_W   3.0f
#define SPEED   30.0f

__global__ void k_assign(const float2* pos, int* cellId, int* fishId,
                         int N, int gridW, int gridH, float cellSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cx = (int)(pos[i].x / cellSize);
    int cy = (int)(pos[i].y / cellSize);
    cx = min(max(cx, 0), gridW - 1);
    cy = min(max(cy, 0), gridH - 1);

    cellId[i] = cy * gridW + cx;
    fishId[i] = i;
}

__global__ void k_bounds(const int* cellId, int* cellStart, int* cellEnd, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int c = cellId[i];
    if (i == 0     || cellId[i - 1] != c) cellStart[c] = i;
    if (i == N - 1 || cellId[i + 1] != c) cellEnd[c]   = i + 1;
}

__global__ void k_boids(const float2* posIn, const float2* velIn,
                        float2* posOut, float2* velOut,
                        const int* cellStart, const int* cellEnd,
                        int N, float dt, int simW, int simH,
                        int gridW, int gridH, float cellSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float2 p = posIn[i], v = velIn[i];
    float2 sep = {0, 0}, ali = {0, 0}, coh = {0, 0};
    int    n   = 0;

    int cx = min(max((int)(p.x / cellSize), 0), gridW - 1);
    int cy = min(max((int)(p.y / cellSize), 0), gridH - 1);

    // 3x3 cell scan. cellSize >= VIEW_R guarantees no in-view fish is missed.
    for (int oy = -1; oy <= 1; oy++) {
        int ny = cy + oy;
        if (ny < 0 || ny >= gridH) continue;

        for (int ox = -1; ox <= 1; ox++) {
            int nx = cx + ox;
            if (nx < 0 || nx >= gridW) continue;

            int c = ny * gridW + nx;
            int s = cellStart[c];
            if (s < 0) continue;                  // empty cell

            for (int j = s, e = cellEnd[c]; j < e; j++) {
                if (j == i) continue;

                float2 q  = posIn[j];
                float  dx = q.x - p.x, dy = q.y - p.y;
                float  d2 = dx * dx + dy * dy;
                if (d2 > VIEW_R * VIEW_R) continue;

                if (d2 < SEP_R * SEP_R) {         // push away, 1/d weighted
                    float inv = rsqrtf(fmaxf(d2, 1e-4f));
                    sep.x -= dx * inv;
                    sep.y -= dy * inv;
                }
                ali.x += velIn[j].x;  ali.y += velIn[j].y;
                coh.x += q.x;         coh.y += q.y;
                n++;
            }
        }
    }

    float ax = sep.x * SEP_W, ay = sep.y * SEP_W;
    if (n > 0) {                                  // match heading, seek centroid
        float inv = 1.0f / n;
        ax += (ali.x * inv - v.x) * ALI_W + (coh.x * inv - p.x) * COH_W;
        ay += (ali.y * inv - v.y) * ALI_W + (coh.y * inv - p.y) * COH_W;
    }

    v.x += ax * dt;
    v.y += ay * dt;

    float sp = sqrtf(v.x * v.x + v.y * v.y);      // renormalize to fixed speed
    if (sp > 1e-4f) {
        v.x = v.x / sp * SPEED;
        v.y = v.y / sp * SPEED;
    } else {                                      // stalled: golden-angle restart
        float a = i * 2.39996f;
        v.x = cosf(a) * SPEED;
        v.y = sinf(a) * SPEED;
    }

    p.x += v.x * dt;
    p.y += v.y * dt;

    float maxX = simW - 1.0f, maxY = simH - 1.0f; // reflect off walls
    if (p.x < 0.0f) { p.x = -p.x;               v.x = -v.x; }
    if (p.y < 0.0f) { p.y = -p.y;               v.y = -v.y; }
    if (p.x > maxX) { p.x = 2.0f * maxX - p.x;  v.x = -v.x; }
    if (p.y > maxY) { p.y = 2.0f * maxY - p.y;  v.y = -v.y; }
    p.x = fminf(fmaxf(p.x, 0.0f), maxX);
    p.y = fminf(fmaxf(p.y, 0.0f), maxY);

    posOut[i] = p;
    velOut[i] = v;
}

__global__ void k_fade(float3* acc, int n, float decay)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float3 c = acc[idx];
    acc[idx] = make_float3(c.x * decay, c.y * decay, c.z * decay);
}

__global__ void k_splat(float3* acc, const float2* pos, const float2* vel,
                        const float3* color, int N, int simW, int simH){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int x = (int)pos[i].x;
    int y = (int)pos[i].y;
    if (x < 0 || y < 0 || x >= simW || y >= simH) return;
    int idx = y * simW + x;

    float3 c = color[i];
    float  t = 0.5f + 0.5f * vel[i].x / SPEED;
    float  s = 0.55f + 0.75f * t;

    atomicAdd(&acc[idx].x, c.x * s);
    atomicAdd(&acc[idx].y, c.y * s);
    atomicAdd(&acc[idx].z, c.z * s);
}

__global__ void k_resolve(const float3* acc, uchar3* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float3 c = acc[idx];

    float r = 1.0f - __expf(-fmaxf(c.x, 0.0f));
    float g = 1.0f - __expf(-fmaxf(c.y, 0.0f));
    float b = 1.0f - __expf(-fmaxf(c.z, 0.0f));

    const float ig = 1.0f / 2.2f;
    r = __powf(r, ig);  g = __powf(g, ig);  b = __powf(b, ig);

    out[idx] = make_uchar3(
        (unsigned char)(255.0f * fminf(fmaxf(r, 0.0f), 1.0f)),
        (unsigned char)(255.0f * fminf(fmaxf(g, 0.0f), 1.0f)),
        (unsigned char)(255.0f * fminf(fmaxf(b, 0.0f), 1.0f)));
}

__global__ void k_reorder(const float2* pos, const float2* vel,
                          const float3* col, const int* fishId,
                          float2* posOut, float2* velOut, float3* colOut,
                          int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    int j = fishId[k];
    posOut[k] = pos[j];
    velOut[k] = vel[j];
    colOut[k] = col[j];
}
