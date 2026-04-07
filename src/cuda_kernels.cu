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
    float sum = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        sum += __ldg(&x[i]);
    }

    __shared__ float shared_sums[32];

    int warp_idx = threadIdx.x / warpSize;
    int lane_idx = threadIdx.x % warpSize;

    sum = warp_reduce_sum(sum);

    if (lane_idx == 0) {
        shared_sums[warp_idx] = sum;
    }

    __syncthreads();

    if (warp_idx == 0) {
        sum = lane_idx < blockDim.x / warpSize ? shared_sums[lane_idx] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane_idx == 0) {
            scratch[blockIdx.x] = sum;
        }
    }
}

template<int UNROLL = 4>
__global__ void reduce_sum_inner_float4(const float4 *__restrict__ x, int n, float *__restrict__ scratch) {
    float sum = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int i = idx;
    for (; i + stride * (UNROLL - 1) < n; i += stride * UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            float4 tmp = __ldg(&x[i + stride * u]);
            sum += tmp.x + tmp.y + tmp.z + tmp.w;
        }
    }

    for (; i < n; i += stride) {
        float4 tmp = __ldg(&x[i]);
        sum += tmp.x + tmp.y + tmp.z + tmp.w;
    }

    __shared__ float shared_sums[32];

    int warp_idx = threadIdx.x / warpSize;
    int lane_idx = threadIdx.x % warpSize;

    sum = warp_reduce_sum(sum);

    if (lane_idx == 0) {
        shared_sums[warp_idx] = sum;
    }

    __syncthreads();

    if (warp_idx == 0) {
        sum = lane_idx < blockDim.x / warpSize ? shared_sums[lane_idx] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane_idx == 0) {
            scratch[blockIdx.x] = sum;
        }
    }
}

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
// Part B2 – Optimised GEMM (Tensor Core WMMA, double-buffered)
// ═══════════════════════════════════════════════════════════════════════════════

constexpr static int WMMA_SIZE = 16;

// ---------- FP32 -> FP16 conversion kernel ----------
__global__ void convert_f32_to_f16(const float *__restrict__ in,
                                   __half *__restrict__ out, int count) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 7 < count) {
        float4 a = __ldg(reinterpret_cast<const float4 *>(in + i));
        float4 b = __ldg(reinterpret_cast<const float4 *>(in + i + 4));
        __half2 h0 = __floats2half2_rn(a.x, a.y);
        __half2 h1 = __floats2half2_rn(a.z, a.w);
        __half2 h2 = __floats2half2_rn(b.x, b.y);
        __half2 h3 = __floats2half2_rn(b.z, b.w);
        *reinterpret_cast<__half2 *>(out + i)     = h0;
        *reinterpret_cast<__half2 *>(out + i + 2) = h1;
        *reinterpret_cast<__half2 *>(out + i + 4) = h2;
        *reinterpret_cast<__half2 *>(out + i + 6) = h3;
    } else {
        for (int j = 0; j < 8 && i + j < count; ++j)
            out[i + j] = __float2half(in[i + j]);
    }
}

void launch_convert_f32_to_f16(const float *in, __half *out, int count,
                               cudaStream_t stream) {
    int threads = 256;
    int blocks = (count + threads * 8 - 1) / (threads * 8);
    convert_f32_to_f16<<<blocks, threads, 0, stream>>>(in, out, count);
}

// ---------- Optimised GEMM ----------
//
// PAD=2 rationale:
//   LD_A = 32 + 2 = 34 halves.  34/2 = 17 four-byte banks.  gcd(17, 32) = 1  ✓
//   LD_B = 128 + 2 = 130 halves. 130/2 = 65 four-byte banks. gcd(65, 32) = 1  ✓
//   Total smem = 2*(128*34 + 32*130)*2 = 34,048 bytes ≈ 33.25 KB
//
// __launch_bounds__(512, 1):
//   512 threads/CTA is fixed.  minBlocks=1 because 2 CTAs would need ~66 KB
//   smem which exceeds Turing's 64 KB max.  With minBlocks=1 the compiler gets
//   128 regs/thread budget (65536 / 512) — generous, no spills.
//
template <int TILE_M = 128, int TILE_N = 128, int TILE_K = 32, int PAD = 2>
__global__ void __launch_bounds__(512, 1)
gemm_optimised_kernel(int m, int n, int k,
                      const __half *__restrict__ mat_a,
                      const __half *__restrict__ mat_b,
                      float *__restrict__ mat_c) {

    constexpr int LD_A = TILE_K + PAD;   // 34
    constexpr int LD_B = TILE_N + PAD;   // 130

    constexpr int WARPS_N = TILE_N / (WMMA_SIZE * 2);  // 4

    __shared__ __half tile_a[2][TILE_M][LD_A];
    __shared__ __half tile_b[2][TILE_K][LD_B];

    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    const int warp_id  = tid / warpSize;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, float>
        frag_acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(frag_acc[i][j], 0.0f);

    const int num_k_tiles = (k + TILE_K - 1) / TILE_K;

    // Register prefetch buffers (1 float4 = 8 halves per thread, per matrix)
    float4 prefetch_a, prefetch_b;

    auto global_load_a = [&](int k_offset) {
        int r  = tid / (TILE_K / 8);
        int c8 = tid % (TILE_K / 8);
        int gr = block_row + r;
        int gc = k_offset + c8 * 8;
        if (gr < m && gc + 7 < k) {
            prefetch_a = __ldg(reinterpret_cast<const float4 *>(
                &mat_a[gr * k + gc]));
        } else {
            __half tmp[8] = {};
            if (gr < m)
                for (int j = 0; j < 8 && gc + j < k; ++j)
                    tmp[j] = __ldg(&mat_a[gr * k + gc + j]);
            prefetch_a = *reinterpret_cast<float4 *>(tmp);
        }
    };

    auto global_load_b = [&](int k_offset) {
        int r  = tid / (TILE_N / 8);
        int c8 = tid % (TILE_N / 8);
        int gr = k_offset + r;
        int gc = block_col + c8 * 8;
        if (gr < k && gc + 7 < n) {
            prefetch_b = __ldg(reinterpret_cast<const float4 *>(
                &mat_b[gr * n + gc]));
        } else {
            __half tmp[8] = {};
            if (gr < k)
                for (int j = 0; j < 8 && gc + j < n; ++j)
                    tmp[j] = __ldg(&mat_b[gr * n + gc + j]);
            prefetch_b = *reinterpret_cast<float4 *>(tmp);
        }
    };

    auto store_a_to_smem = [&](int buf) {
        int r  = tid / (TILE_K / 8);
        int c8 = tid % (TILE_K / 8);
        *reinterpret_cast<float4 *>(&tile_a[buf][r][c8 * 8]) = prefetch_a;
    };

    auto store_b_to_smem = [&](int buf) {
        int r  = tid / (TILE_N / 8);
        int c8 = tid % (TILE_N / 8);
        *reinterpret_cast<float4 *>(&tile_b[buf][r][c8 * 8]) = prefetch_b;
    };

    auto compute_tile = [&](int buf) {
        wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE,
                       __half, wmma::row_major> frag_a[2];
        wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE,
                       __half, wmma::row_major> frag_b[2];

        const int smem_row = warp_row * 32;
        const int smem_col = warp_col * 32;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += WMMA_SIZE) {
            wmma::load_matrix_sync(frag_a[0],
                &tile_a[buf][smem_row     ][kk], LD_A);
            wmma::load_matrix_sync(frag_a[1],
                &tile_a[buf][smem_row + 16][kk], LD_A);
            wmma::load_matrix_sync(frag_b[0],
                &tile_b[buf][kk][smem_col     ], LD_B);
            wmma::load_matrix_sync(frag_b[1],
                &tile_b[buf][kk][smem_col + 16], LD_B);

            wmma::mma_sync(frag_acc[0][0], frag_a[0], frag_b[0], frag_acc[0][0]);
            wmma::mma_sync(frag_acc[0][1], frag_a[0], frag_b[1], frag_acc[0][1]);
            wmma::mma_sync(frag_acc[1][0], frag_a[1], frag_b[0], frag_acc[1][0]);
            wmma::mma_sync(frag_acc[1][1], frag_a[1], frag_b[1], frag_acc[1][1]);
        }
    };

    // === Prologue ===
    global_load_a(0);
    global_load_b(0);
    store_a_to_smem(0);
    store_b_to_smem(0);
    __syncthreads();

    // === Main loop ===
    for (int t = 0; t < num_k_tiles - 1; ++t) {
        int cur = t & 1;
        int nxt = 1 - cur;

        global_load_a((t + 1) * TILE_K);
        global_load_b((t + 1) * TILE_K);

        compute_tile(cur);

        __syncthreads();

        store_a_to_smem(nxt);
        store_b_to_smem(nxt);

        __syncthreads();
    }

    // === Epilogue ===
    if (num_k_tiles > 0)
        compute_tile((num_k_tiles - 1) & 1);

    // === Store C ===
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int c_row = block_row + warp_row * 32 + i * WMMA_SIZE;
            int c_col = block_col + warp_col * 32 + j * WMMA_SIZE;
            if (c_row < m && c_col < n)
                wmma::store_matrix_sync(&mat_c[c_row * n + c_col],
                                        frag_acc[i][j], n,
                                        wmma::mem_row_major);
        }
    }
}

// ---------- Launch ----------
void launch_gemm_optimised(int M, int N, int K,
                           const float *A, const float *B, float *C,
                           cudaStream_t stream) {
    __half *A_h, *B_h;
    cudaMalloc(&A_h, (size_t)M * K * sizeof(__half));
    cudaMalloc(&B_h, (size_t)K * N * sizeof(__half));
    launch_convert_f32_to_f16(A, A_h, M * K, stream);
    launch_convert_f32_to_f16(B, B_h, K * N, stream);

    constexpr int TILE_M = 128, TILE_N = 128, TILE_K = 32, PAD = 2;
    constexpr int threads = ((TILE_M / 32) * (TILE_N / 32)) * 32;  // 512

    // Opt into max smem carveout on Turing (up to 64 KB from the 96 KB L1/smem)
    cudaFuncSetAttribute(
        gemm_optimised_kernel<TILE_M, TILE_N, TILE_K, PAD>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_optimised_kernel<TILE_M, TILE_N, TILE_K, PAD>
        <<<grid, threads, 0, stream>>>(M, N, K, A_h, B_h, C);

    cudaStreamSynchronize(stream);

    cudaFree(A_h);
    cudaFree(B_h);
}
