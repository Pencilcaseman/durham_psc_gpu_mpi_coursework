#include <coursework/kernels.hpp>
#include <coursework/check_cuda.hpp>
#include <coursework/util.hpp>

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ═══════════════════════════════════════════════════════════════════════════════
// Part A1 – Vector operations
// ═══════════════════════════════════════════════════════════════════════════════

// --- AXPY: y = alpha*x + y ---------------------------------------------------
__global__ void
k_axpy(int n, float alpha, const float *__restrict__ x, float *__restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}

void launch_axpy(
    int n,
    float alpha,
    const float *x,
    float *y,
    cudaStream_t stream) {

    int block = 256;
    int grid = (n + block - 1) / block;

    k_axpy<<<grid, block, 0, stream>>>(n, alpha, x, y);
    CUDA_CHECK_LAST("k_axpy");
}

// --- ADD: z = x + y ----------------------------------------------------------
// TODO (students): implement this kernel for Part A1
__global__ void k_add(
    int n,
    const float *__restrict__ x,
    const float *__restrict__ y,
    float *__restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

void launch_add(
    int n,
    const float *x,
    const float *y,
    float *z,
    cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    k_add<<<grid, block, 0, stream>>>(n, x, y, z);
    CUDA_CHECK_LAST("k_add");
}

// --- COPY: y = x -------------------------------------------------------------
// TODO (students): implement this kernel for Part A1
__global__ void
k_copy(int n, const float *__restrict__ x, float *__restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}

void launch_copy(int n, const float *x, float *y, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    k_copy<<<grid, block, 0, stream>>>(n, x, y);
    CUDA_CHECK_LAST("k_copy");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part A2 – Parallel reduction (tree reduction, two-stage)
// ═══════════════════════════════════════════════════════════════════════════════

__inline__ __device__ float warp_reduce_sum(float sum) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    return sum;
}

__global__ void reduce_sum_inner(const float *__restrict__ x, int n, float *__restrict__ scratch) {
    // Requires blockDim.x <= 1024 -- see below

    float sum = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Leaves us with blockDim.x sums
    for (int i = idx; i < n; i += stride) {
        sum += __ldg(&x[i]);
    }

    // Must be ceil(blockDim.x / warpSize) elements long
    // blockDim.x == 1024 => 32 elements
    __shared__ float shared_sums[32];

    int warp_idx = threadIdx.x / warpSize;
    int lane_idx = threadIdx.x % warpSize;

    // Reduces blockDim.x sums into blockDim.x / warpSize sums.
    // lane 0 in each warp holds the sum
    sum = warp_reduce_sum(sum);

    if (lane_idx == 0) {
        shared_sums[warp_idx] = sum;
    }

    __syncthreads();

    // Currently have e.g. 32 numbers to sum.
    // Only first warp in a block needs to do this

    if (warp_idx == 0) {
        sum = lane_idx < blockDim.x / warpSize ? shared_sums[lane_idx] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane_idx == 0) {
            scratch[blockIdx.x] = sum;
        }
    }

    // scratch[0..gridDim.x - 1] now contains partial sums of each block.
}

template<int UNROLL = 4>
__global__ void reduce_sum_inner_float4(const float4 *__restrict__ x, int n, float *__restrict__ scratch) {
    // Requires blockDim.x <= 1024 -- see below

    float sum = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Unrolled grid-stride loop: each iteration loads UNROLL float4s
    int i = idx;
    for (; i + stride * (UNROLL - 1) < n; i += stride * UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            float4 tmp = __ldg(&x[i + stride * u]);
            sum += tmp.x + tmp.y + tmp.z + tmp.w;
        }
    }

    // Tail
    for (; i < n; i += stride) {
        float4 tmp = __ldg(&x[i]);
        sum += tmp.x + tmp.y + tmp.z + tmp.w;
    }

    // Must be ceil(blockDim.x / warpSize) elements long
    // blockDim.x == 1024 => 32 elements
    __shared__ float shared_sums[32];

    int warp_idx = threadIdx.x / warpSize;
    int lane_idx = threadIdx.x % warpSize;

    // Reduces blockDim.x sums into blockDim.x / warpSize sums.
    // lane 0 in each warp holds the sum
    sum = warp_reduce_sum(sum);

    if (lane_idx == 0) {
        shared_sums[warp_idx] = sum;
    }

    __syncthreads();

    // Currently have e.g. 32 numbers to sum.
    // Only first warp in a block needs to do this

    if (warp_idx == 0) {
        sum = lane_idx < blockDim.x / warpSize ? shared_sums[lane_idx] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane_idx == 0) {
            scratch[blockIdx.x] = sum;
        }
    }

    // scratch[0..gridDim.x - 1] now contains partial sums of each block.
}

// The scratch buffer must be 1024 elements long and will be modified during the
// execution of this function. If multiple instances of this function are called
// in parallel, each must use a separate, non-overlapping scratch buffer to
// guarantee correct results.
float gpu_reduce_sum(
        const float *__restrict__ device_x,
        int n,
        float *__restrict__ scratch,
        cudaStream_t stream
    ) {

    const int tail_size = n % 4;
    const int float4_size = n - tail_size;
    const int float4_elements = float4_size / 4;

    const int block_dim = 1024;
    const int grid_dim  = std::min((float4_elements + block_dim - 1) / block_dim, 256);

    float tail[4] = {0, 0, 0, 0};

    CUDA_CHECK(cudaMemcpyAsync(
        &tail,
        device_x + float4_size,
        tail_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    ));

    float res = 0;

    if (n > 4) {
        reduce_sum_inner_float4<<<grid_dim, block_dim, 0, stream>>>(
            reinterpret_cast<const float4*>(device_x),
            float4_elements,
            scratch
        );

        CUDA_CHECK_LAST("reduce_sum_inner");

        reduce_sum_inner<<<1, block_dim, 0, stream>>>(scratch, grid_dim, scratch);
        CUDA_CHECK_LAST("reduce_sum_inner");

        util::cuda_memcpy_checked(&res, scratch, 1, cudaMemcpyDeviceToHost, stream);
    }

    res += tail[0];
    res += tail[1];
    res += tail[2];
    res += tail[3];

    return res;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part B1 – Naïve GEMM
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void k_gemm_naive(
    int M,
    int N,
    int K,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void launch_gemm_naive(
    int M,
    int N,
    int K,
    const float *A,
    const float *B,
    float *C,
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
template<int TILE>
__global__ void k_gemm_tiled(
    int M,
    int N,
    int K,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row   = blockIdx.y * TILE + threadIdx.y;
    int col   = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_gemm_tiled16(
    int M,
    int N,
    int K,
    const float *A,
    const float *B,
    float *C,
    cudaStream_t stream) {
    constexpr int T = 32;
    dim3 block(T, T);
    dim3 grid((N + T - 1) / T, (M + T - 1) / T);
    k_gemm_tiled<T><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAST("k_gemm_tiled<32>");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part B2 – Optimised GEMM
// ═══════════════════════════════════════════════════════════════════════════════

constexpr static int WMMA_SIZE = 16;

// Optimised GEMM using WMMA Tensor Cores with:
//   - Thread coarsening: each warp computes a 2x2 grid of 16x16 WMMA tiles (32x32)
//   - Double buffering: overlaps global memory loads with Tensor Core computation
//   - Vectorised loads: float4 (128-bit) coalesced global memory access
//   - Shared memory padding: eliminates bank conflicts for WMMA loads
template<int TILE_M = 128, int TILE_N = 128, int TILE_K = 32, int PAD = 8>
__global__ void gemm_optimised_kernel(
    int m,
    int n,
    int k,
    const float *__restrict__ mat_a,
    const float *__restrict__ mat_b,
    float *__restrict__ mat_c) {
    // 16 warps in a 4x4 grid, each computing 2x2 WMMA tiles = 32x32 output
    // 16 warps * 32 threads/warp = 512 threads per block
    // Block covers TILE_M x TILE_N = 128 x 128 output elements

    constexpr int LD_A = TILE_K + PAD;
    constexpr int LD_B = TILE_N + PAD;
    constexpr int WARPS_M = TILE_M / (WMMA_SIZE * 2); // 4
    constexpr int WARPS_N = TILE_N / (WMMA_SIZE * 2); // 4

    __shared__ __half tile_a[2][TILE_M][LD_A];
    __shared__ __half tile_b[2][TILE_K][LD_B];

    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    const int warp_id  = tid / warpSize;
    const int warp_row = warp_id / WARPS_N; // 0..3
    const int warp_col = warp_id % WARPS_N; // 0..3

    // Each warp accumulates a 2x2 grid of 16x16 fragments
    wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE,
                   __half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE,
                   __half, wmma::row_major> frag_b[2];
    wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE,
                   float> frag_acc[2][2];

    #pragma unroll
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(frag_acc[i][j], 0.0f);

    // Number of float4s to load per tile
    constexpr int FLOAT4S_A = (TILE_M * TILE_K) / 4; // 128*32/4 = 1024
    constexpr int FLOAT4S_B = (TILE_K * TILE_N) / 4; // 32*128/4 = 1024

    const int num_k_tiles = (k + TILE_K - 1) / TILE_K;

    // === Lambda-style cooperative load using float4 ===
    // Loads a TILE_M x TILE_K block of A and TILE_K x TILE_N block of B
    // into shared memory buffer `buf`, converting float32 -> half
    auto load_tile = [&](int buf, int k_offset) {
        // Load tile_a: 128 rows x 32 cols = 4096 floats = 1024 float4s
        for (int i = tid; i < FLOAT4S_A; i += blockDim.x) {
            int r = i / (TILE_K / 4);
            int c4 = i % (TILE_K / 4);
            int global_r = block_row + r;
            int global_c = k_offset + c4 * 4;

            float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
            if (global_r < m && global_c + 3 < k) {
                val = __ldg(reinterpret_cast<const float4*>(&mat_a[global_r * k + global_c]));
            } else if (global_r < m) {
                // Scalar fallback for boundary
                const float *base = &mat_a[global_r * k + global_c];
                if (global_c     < k) val.x = __ldg(base);
                if (global_c + 1 < k) val.y = __ldg(base + 1);
                if (global_c + 2 < k) val.z = __ldg(base + 2);
                if (global_c + 3 < k) val.w = __ldg(base + 3);
            }
            int sc = c4 * 4;
            tile_a[buf][r][sc    ] = __float2half(val.x);
            tile_a[buf][r][sc + 1] = __float2half(val.y);
            tile_a[buf][r][sc + 2] = __float2half(val.z);
            tile_a[buf][r][sc + 3] = __float2half(val.w);
        }

        // Load tile_b: 32 rows x 128 cols = 4096 floats = 1024 float4s
        for (int i = tid; i < FLOAT4S_B; i += blockDim.x) {
            int r = i / (TILE_N / 4);
            int c4 = i % (TILE_N / 4);
            int global_r = k_offset + r;
            int global_c = block_col + c4 * 4;

            float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
            if (global_r < k && global_c + 3 < n) {
                val = __ldg(reinterpret_cast<const float4*>(&mat_b[global_r * n + global_c]));
            } else if (global_r < k) {
                const float *base = &mat_b[global_r * n + global_c];
                if (global_c     < n) val.x = __ldg(base);
                if (global_c + 1 < n) val.y = __ldg(base + 1);
                if (global_c + 2 < n) val.z = __ldg(base + 2);
                if (global_c + 3 < n) val.w = __ldg(base + 3);
            }
            int sc = c4 * 4;
            tile_b[buf][r][sc    ] = __float2half(val.x);
            tile_b[buf][r][sc + 1] = __float2half(val.y);
            tile_b[buf][r][sc + 2] = __float2half(val.z);
            tile_b[buf][r][sc + 3] = __float2half(val.w);
        }
    };

    // Compute on a given shared memory buffer
    auto compute_tile = [&](int buf) {
        const int smem_row = warp_row * 32;
        const int smem_col = warp_col * 32;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += WMMA_SIZE) {
            wmma::load_matrix_sync(frag_a[0], &tile_a[buf][smem_row     ][kk], LD_A);
            wmma::load_matrix_sync(frag_a[1], &tile_a[buf][smem_row + 16][kk], LD_A);
            wmma::load_matrix_sync(frag_b[0], &tile_b[buf][kk][smem_col     ], LD_B);
            wmma::load_matrix_sync(frag_b[1], &tile_b[buf][kk][smem_col + 16], LD_B);

            wmma::mma_sync(frag_acc[0][0], frag_a[0], frag_b[0], frag_acc[0][0]);
            wmma::mma_sync(frag_acc[0][1], frag_a[0], frag_b[1], frag_acc[0][1]);
            wmma::mma_sync(frag_acc[1][0], frag_a[1], frag_b[0], frag_acc[1][0]);
            wmma::mma_sync(frag_acc[1][1], frag_a[1], frag_b[1], frag_acc[1][1]);
        }
    };

    // === Double-buffered main loop ===

    // Prologue: load first K-tile into buffer 0
    load_tile(0, 0);
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
        int cur = t & 1;
        int nxt = 1 - cur;

        // Load next K-tile into the other buffer (if not last)
        if (t + 1 < num_k_tiles) {
            load_tile(nxt, (t + 1) * TILE_K);
        }

        // Compute WMMA on current buffer
        compute_tile(cur);

        __syncthreads();
    }

    // === Store results back to global C ===
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int c_row = block_row + warp_row * 32 + i * WMMA_SIZE;
            int c_col = block_col + warp_col * 32 + j * WMMA_SIZE;
            if (c_row < m && c_col < n) {
                wmma::store_matrix_sync(
                    &mat_c[c_row * n + c_col],
                    frag_acc[i][j], n, wmma::mem_row_major);
            }
        }
    }
}

void launch_gemm_optimised(
    int M,
    int N,
    int K,
    const float *A,
    const float *B,
    float *C,
    cudaStream_t stream) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    constexpr int PAD    = 8;
    // 4x4 warp grid, each warp covers 32x32 (2x2 coarsened WMMA tiles)
    constexpr int NUM_WARPS = (TILE_M / 32) * (TILE_N / 32); // 16
    constexpr int threads   = NUM_WARPS * 32;           // 512

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_optimised_kernel<TILE_M, TILE_N, TILE_K, PAD>
        <<<grid, threads, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAST("gemm_optimised_kernel");
}
