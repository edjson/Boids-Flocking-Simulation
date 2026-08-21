# CUDA Boids

Boids flocking simulation rendered as truecolor half-blocks in a terminal.

## Build

```
nvcc -O3 -std=c++17 -arch=sm_120 main.cu -o boids
```

`sm_120` is Blackwell architecture. For other cards:
`nvidia-smi --query-gpu=compute_cap --format=csv`

## Run

```
./boids                  # interactive, 500 fish
./boids --fish 140       # interactive, N fish
./boids --bench          # constant-density scaling sweep
./boids --bench-dense    # fixed-domain density sweep
```

## How it works

Each step hashes a half-block into a uniform grid, sorts them by cell id, then has
every fish scan its 3x3 cell neighbourhood for separation, alignment, and
cohesion. Cell size is tied to the view radius, smaller cells would let an
in-view fish sit two cells away and be silently missed. Positions are
double-buffered so no thread reads a value another has already written.

## The reorder pass

Sorting only the index array leaves positions in spawn order, making the
neighbour gather `pos[fishId[k]]`,  a random access, with every warp lane
reading a different part of the world. That works while the data fits in
cache and falls off a cliff once it doesn't.

`k_reorder` permutes positions, velocities, and colours into cell-sorted order
so the gather reads contiguously. Cost of running with and without it:

| fish | no reorder (ms) | with reorder (ms) | speedup |
|---:|---:|---:|---:|
| 10,000 | 0.404 | 0.240 | 1.7x |
| 50,000 | 0.727 | 1.019 | 0.7x |
| 200,000 | 2.779 | 1.050 | 2.6x |
| 500,000 | 6.271 | 1.184 | 5.3x |
| 1,000,000 | 11.732 | 1.456 | 8.0x |
| 2,000,000 | 43.274 | 2.334 | 18.5x |
| 4,000,000 | 260.814 | 4.882 | **53.4x** |

Cost per million fish flattens to ~1.2 ms and holds from 2M to 4M. The 50k row
is ~40% slower with the pass enabled an extra kernel launch with no bandwidth
savings to pay for it at that size. Run-to-run variance is ~10%.

At 4M fish the reordered version moves ~5.8 GB per step in 4.9 ms, about
1.2 TB/s, roughly 2.7x the card's 448 GB/s VRAM bandwidth. 

The loop does ~6 FLOPs per candidate against 16 bytes loaded, so this is a
memory problem, not an arithmetic one. Every optimization here targets data
movement.

## Files

| file | contents |
|---|---|
| `config.cuh` | dimensions, `CUDA_CHECK`, the `Sim` struct |
| `kernels.cuh` | device code |
| `functions.cuh` | init/step/free, renderer, benchmarks |
| `main.cu` | arg parsing, frame loop, terminal setup |

Measured on RTX 5060 Ti (16 GB), CUDA 13.3, WSL2.
