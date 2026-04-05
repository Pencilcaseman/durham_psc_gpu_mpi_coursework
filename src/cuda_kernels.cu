#include "kernels.hpp"
#include "check_cuda.hpp"
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════════
// Part A1 – Vector operations
// ═══════════════════════════════════════════════════════════════════════════════

// --- AXPY: y = alpha*x + y ---------------------------------------------------
__global__ void k_axpy(int n, float alpha,
                       const float* __restrict__ x,
                             float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + y[i];
}

void launch_axpy(int n, float alpha, const float* x, float* y, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    k_axpy<<<grid, block, 0, stream>>>(n, alpha, x, y);
    CUDA_CHECK_LAST("k_axpy");
}

// --- ADD: z = x + y ----------------------------------------------------------
// TODO (students): implement this kernel for Part A1
__global__ void k_add(int n,
                      const float* __restrict__ x,
                      const float* __restrict__ y,
                            float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

void launch_add(int n, const float* x, const float* y, float* z, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    k_add<<<grid, block, 0, stream>>>(n, x, y, z);
    CUDA_CHECK_LAST("k_add");
}

// --- COPY: y = x -------------------------------------------------------------
// TODO (students): implement this kernel for Part A1
__global__ void k_copy(int n,
                       const float* __restrict__ x,
                             float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

void launch_copy(int n, const float* x, float* y, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    k_copy<<<grid, block, 0, stream>>>(n, x, y);
    CUDA_CHECK_LAST("k_copy");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part A2 – Parallel reduction (tree reduction, two-stage)
// ═══════════════════════════════════════════════════════════════════════════════

// Stage 1: each block reduces its chunk into a single partial sum
// TODO (students): you may add warp-level primitives (__shfl_down_sync) here
__global__ void k_reduce_block_sum(const float* __restrict__ x,
                                         float* __restrict__ partials,
                                   int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] : 0.f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

float gpu_reduce_sum(const float* d_x, int n, cudaStream_t stream) {
    const int block = 256;
    int grid = (n + block - 1) / block;

    float* d_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partials, grid * sizeof(float)));
    k_reduce_block_sum<<<grid, block, 0, stream>>>(d_x, d_partials, n);
    CUDA_CHECK_LAST("k_reduce_block_sum");

    // Stage 2+: reduce partial sums until a single value remains
    while (grid > 1) {
        int next = (grid + block - 1) / block;
        float* d_next = nullptr;
        CUDA_CHECK(cudaMalloc(&d_next, next * sizeof(float)));
        k_reduce_block_sum<<<next, block, 0, stream>>>(d_partials, d_next, grid);
        CUDA_CHECK_LAST("k_reduce_block_sum(iter)");
        CUDA_CHECK(cudaFree(d_partials));
        d_partials = d_next;
        grid = next;
    }

    float h = 0.f;
    CUDA_CHECK(cudaMemcpyAsync(&h, d_partials, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_partials));
    return h;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part B1 – Naïve GEMM
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void k_gemm_naive(int M, int N, int K,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                                   float* __restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

void launch_gemm_naive(int M, int N, int K,
                       const float* A, const float* B, float* C,
                       cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    k_gemm_naive<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAST("k_gemm_naive");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part B2 – Tiled GEMM with shared memory
// ═══════════════════════════════════════════════════════════════════════════════

// TODO (students): tune TILE size (try 16 and 32), experiment with
//                  #pragma unroll and __ldg() for extra performance
template <int TILE>
__global__ void k_gemm_tiled(int M, int N, int K,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                                    float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row*K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row*N + col] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

void launch_gemm_tiled16(int M, int N, int K,
                         const float* A, const float* B, float* C,
                         cudaStream_t stream) {
    constexpr int T = 16;
    dim3 block(T, T);
    dim3 grid((N + T-1)/T, (M + T-1)/T);
    k_gemm_tiled<T><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAST("k_gemm_tiled<16>");
}
