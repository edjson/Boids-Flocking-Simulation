#include "functions.cuh"

#include <thread>
#include <csignal>

static volatile sig_atomic_t g_stop    = 0;
static volatile sig_atomic_t g_resized = 1;

static void on_sigint(int){
    g_stop = 1;
}

static void on_winch(int){
    g_resized = 1;
}

static uchar3 fb[W * H];

int main(int argc, char** argv){
    int nFish = 500;
    int mode  = 0;

    for (int i = 1; i < argc; i++){
        if(!std::strcmp(argv[i], "--bench")){
            mode = 1;
        }
        else if(!std::strcmp(argv[i], "--bench-dense")){
            mode = 2;
        }
        else if(!std::strcmp(argv[i], "--fish") && i + 1 < argc){
            nFish = std::atoi(argv[i++]);
        }
        else{
            std::printf("usage: %s [--bench | --bench-dense] [--fish N]\n",
                        argv[0]);
            return 1;
        }
    }

    if (nFish < 1){
        nFish = 1;
    }

    if (mode == 1){
        benchScaling();
        return 0;
    }

    if (mode == 2){
        benchDensity();
        return 0;
    }

    std::signal(SIGINT, on_sigint);
    std::signal(SIGWINCH, on_winch);

    Sim s{};
    simInit(s, nFish);

    float3* d_acc;
    uchar3* d_fb;
    CUDA_CHECK(cudaMalloc(&d_acc, (size_t)W * H * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_fb, sizeof(fb)));
    CUDA_CHECK(cudaMemset(d_acc, 0, (size_t)W * H * sizeof(float3)));

    const int TPB  = 256;
    int nPix       = W * H;
    int pixBlocks  = (nPix  + TPB - 1) / TPB;
    int fishBlocks = (nFish + TPB - 1) / TPB;

    std::fputs("\x1b[?1049h\x1b[?7l\x1b[?25l\x1b[2J", stdout);

    const float dt    = 1.0f / 60.0f;
    const float decay = 0.72f;
    long frameNo      = 0;

    while (!g_stop){
        if(g_resized){
            g_resized = 0;
            std::fputs("\x1b[2J", stdout);
        }
        k_fade<<<pixBlocks, TPB>>>(d_acc, nPix, decay);
        simStep(s, dt);
        k_splat<<<fishBlocks, TPB>>>(d_acc, s.pos[s.cur], s.vel[s.cur],
                                     s.color, s.N, W, H);
        k_resolve<<<pixBlocks, TPB>>>(d_acc, d_fb, nPix);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(fb, d_fb, sizeof(fb), cudaMemcpyDeviceToHost));

        std::fputs("\x1b[H", stdout);
        render(fb);

        std::printf("  fish %d   world %dx%d   grid %dx%d   frame %ld\x1b[K",
                    s.N, s.simW, s.simH, s.gridW, s.gridH, frameNo);
        std::fflush(stdout);

        frameNo++;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    std::fputs("\x1b[0m\x1b[?25h\x1b[?7h\x1b[?1049l", stdout);
    std::fflush(stdout);
    std::printf("ran %ld frames with %d fish\n", frameNo, s.N);

    simFree(s);
    cudaFree(d_acc);
    cudaFree(d_fb);
    return 0;
}