/*
 * COMP3741 – Coursework: self-contained correctness tests
 * Run with:  mpirun -np <P> ./mpi_cuda_tests
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>

#include "mpi_utils.hpp"
#include "check_cuda.hpp"
#include "kernels.hpp"
#include "cpu_reference.hpp"

static int g_pass = 0, g_fail = 0;

static void report(const char* name, bool ok) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    ok ? ++g_pass : ++g_fail;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    auto info = mpi_info();

    // All ranks use GPU 0 (single-GPU assumption)
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { std::cerr << "No GPU found\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    CUDA_CHECK(cudaSetDevice(0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int N = 1 << 18;   // 256 K elements – quick to run
    auto d = dist_1d(N, info.rank, info.size);

    // ── Test 1: AXPY ─────────────────────────────────────────────────────────
    {
        std::vector<float> x(d.N_local, 1.f), y(d.N_local, 2.f), yref(d.N_local, 2.f);
        const float alpha = 3.f;
        cpu_axpy((int)d.N_local, alpha, x.data(), yref.data());  // ref: y = 3*1+2 = 5

        float *dx = nullptr, *dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dy, y.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        launch_axpy((int)d.N_local, alpha, dx, dy, stream);
        CUDA_CHECK(cudaMemcpyAsync(y.data(), dy, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float err = max_abs_diff((int)d.N_local, y.data(), yref.data());
        if (info.rank == 0) report("AXPY correctness", err < 1e-5f);
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }

    // ── Test 2: Global reduction ──────────────────────────────────────────────
    {
        // All elements == 1  →  global sum == N
        std::vector<float> x(d.N_local, 1.f);
        float* dx = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float local  = gpu_reduce_sum(dx, (int)d.N_local, stream);
        float global = 0.f;
        MPI_Allreduce(&local, &global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if (info.rank == 0)
            report("Reduction global sum", std::fabs(global - (float)N) < 0.5f);
        CUDA_CHECK(cudaFree(dx));
    }

    // ── Test 3: Tiled GEMM 32×32 identity-like check ─────────────────────────
    {
        // Use a single rank's local portion only for this test
        const int m = 64, n = 64, k = 64;
        if (d.N_local >= m || info.size == 1) {
            std::vector<float> A(m*k, 0.f), B(k*n, 0.f), C(m*n, 0.f), Cref(m*n, 0.f);
            for (int i = 0; i < m && i < k; ++i) A[i*k+i] = 1.f;   // identity-like
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j) B[i*n+j] = (float)(i+j);

            cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

            float *dA=nullptr,*dB=nullptr,*dC=nullptr;
            CUDA_CHECK(cudaMalloc(&dA, m*k*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, k*n*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, m*n*sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), m*k*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), k*n*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(dC, 0, m*n*sizeof(float), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            launch_gemm_tiled16(m, n, k, dA, dB, dC, stream);
            CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float err = max_abs_diff(m*n, C.data(), Cref.data());
            if (info.rank == 0) report("Tiled GEMM correctness", err < 1e-3f);
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }
    }

    if (info.rank == 0)
        std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed.\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return g_fail > 0 ? 1 : 0;
}
