/*
 * COMP3741 – Parallel Scientific Computing: GPU & MPI Programming
 * Coursework starter – main.cpp
 *
 * Single-GPU design: ALL MPI ranks share the ONE available GPU (device 0).
 * Each rank allocates its own device memory and issues its own CUDA kernels
 * independently.  No CUDA peer-access or multi-GPU logic is required.
 *
 * Usage:
 *   mpirun -np <P> ./mpi_cuda_coursework --mode axpy|reduce|gemm [options]
 *
 * Common options:
 *   --N       <int>    global vector length  (default 10 000 000)
 *   --M       <int>    global matrix rows    (default 1024)
 *   --Nmat    <int>    matrix columns        (default 1024)
 *   --K       <int>    inner dimension       (default 1024)
 *   --alpha   <float>  AXPY scalar           (default 2.0)
 *   --kernel  naive|tiled  GEMM kernel       (default tiled)
 *   --csv     <file>   output CSV path       (default results.csv)
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>

#include <mpi.h>
#include <cuda_runtime.h>

#include <coursework/benchmarks.hpp>
#include <coursework/check_cuda.hpp>
#include <coursework/cli.hpp>
#include <coursework/cpu_reference.hpp>
#include <coursework/kernels.hpp>
#include <coursework/mpi_distribution.hpp>
#include <coursework/mpi_utils.hpp>
#include <coursework/timer.hpp>
#include <coursework/util.hpp>

// ── GPU selection
// ───────────────────────────────────────────────────────────── With a single
// GPU all ranks share device 0. If multiple GPUs happen to be available we
// still assign device 0 to every rank to stay within the "at most one GPU"
// assumption on Colab / NCC.
static void select_gpu() {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        std::fprintf(stderr, "No CUDA device found. Aborting.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(0)); // always use GPU 0
}

// ── Random fill ──────────────────────────────────────────────────────────────
static void fill_random(std::vector<float> &v, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x: v) {
        x = dist(gen);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    auto info = mpi_info();
    Args args = Args::parse(argc, argv);

    const std::string mode = args.get("mode", "");
    if (mode.empty()) {
        if (info.rank == 0) {
            std::cout
                << "Usage: --mode axpy|add|copy|reduce|gemm  [see source for full options]\n";
        }

        MPI_Finalize();
        return 0;
    }

    select_gpu();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const std::string csv = args.get("csv", "results.csv");

    // ══════════════════════════════════════════════════════════════════════════
    // Part A1 – AXPY
    // ══════════════════════════════════════════════════════════════════════════
    if (mode == "axpy") {
        long long N = args.get_ll("N", 10'000'000LL);
        float alpha = (float)args.get_double("alpha", 2.0);
        auto d      = dist_1d(N, info.rank, info.size);

        std::vector<float> x(d.N_local);
        std::vector<float> y(d.N_local);
        std::vector<float> yref(d.N_local);

        fill_random(x, 1000 + info.rank);
        fill_random(y, 2000 + info.rank);
        yref = y;

        float *dx = util::cuda_malloc_checked<float>(d.N_local);
        float *dy = util::cuda_malloc_checked<float>(d.N_local);

        util::cuda_memcpy_checked(
            dx,
            x.data(),
            d.N_local,
            cudaMemcpyHostToDevice,
            stream);

        util::cuda_memcpy_checked(
            dy,
            y.data(),
            d.N_local,
            cudaMemcpyHostToDevice,
            stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ── Warm-up (not timed)
        // ───────────────────────────────────────────────
        launch_axpy((int)d.N_local, alpha, dx, dy, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Re-upload y so timing run starts from a clean state
        util::cuda_memcpy_checked(
            dy,
            y.data(),
            d.N_local,
            cudaMemcpyHostToDevice,
            stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer gt;
        gt.start(stream);
        launch_axpy((int)d.N_local, alpha, dx, dy, stream);
        float ms = gt.stop(stream);

        util::cuda_memcpy_checked(
            y.data(),
            dy,
            d.N_local,
            cudaMemcpyDeviceToHost,
            stream
        );

        CUDA_CHECK(cudaStreamSynchronize(stream));

        cpu_axpy((int)d.N_local, alpha, x.data(), yref.data());
        float err = max_abs_diff((int)d.N_local, y.data(), yref.data());

        // Effective bandwidth: read x, read y, write y = 3 arrays
        double bytes = 3.0 * (double)d.N_local * sizeof(float);
        double gbs   = (bytes / 1e9) / (ms / 1e3);

        if (info.rank == 0) {
            std::cout << "[AXPY] rank=" << info.rank << " N_local=" << d.N_local
                      << " ms=" << ms << " GB/s=" << gbs << " max_err=" << err
                      << "\n";
        }

        append_csv(
            csv,
            info.rank,
            "axpy",
            "axpy",
            d.N_local,
            0,
            0,
            0,
            ms,
            0.0,
            gbs);

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
    }

    else if (mode == "add") {
        long long N = args.get_ll("N", 10'000'000LL);
        auto d      = dist_1d(N, info.rank, info.size);

        std::vector<float> x(d.N_local);
        std::vector<float> y(d.N_local);
        std::vector<float> z(d.N_local);
        std::vector<float> zref(d.N_local);

        fill_random(x, 1000 + info.rank);
        fill_random(y, 2000 + info.rank);

        float *dx = util::cuda_malloc_checked<float>(d.N_local);
        float *dy = util::cuda_malloc_checked<float>(d.N_local);
        float *dz = util::cuda_malloc_checked<float>(d.N_local);

        util::cuda_memcpy_checked(
            dx,
            x.data(),
            d.N_local,
            cudaMemcpyHostToDevice,
            stream);

        util::cuda_memcpy_checked(
            dy,
            y.data(),
            d.N_local,
            cudaMemcpyHostToDevice,
            stream);

        util::cuda_memcpy_checked(
            dz,
            z.data(),
            d.N_local,
            cudaMemcpyHostToDevice,
            stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ── Warm-up (not timed)
        // ───────────────────────────────────────────────
        launch_add((int)d.N_local, dx, dy, dz, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Re-upload z so timing run starts from a clean state
        util::cuda_memcpy_checked(
            dz,
            z.data(),
            d.N_local,
            cudaMemcpyHostToDevice,
            stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer gt;
        gt.start(stream);

        launch_add((int)d.N_local, dx, dy, dz, stream);

        float ms = gt.stop(stream);

        util::cuda_memcpy_checked(
            z.data(),
            dz,
            d.N_local,
            cudaMemcpyDeviceToHost,
            stream
        );

        CUDA_CHECK(cudaStreamSynchronize(stream));

        cpu_add((int)d.N_local, x.data(), y.data(), zref.data());
        float err = max_abs_diff((int)d.N_local, z.data(), zref.data());

        // Effective bandwidth: read x, read y, write z = 3 arrays
        double bytes = 3.0 * (double)d.N_local * sizeof(float);
        double gbs   = (bytes / 1e9) / (ms / 1e3);

        if (info.rank == 0) {
            std::cout << "[ADD] rank=" << info.rank << " N_local=" << d.N_local
                      << " ms=" << ms << " GB/s=" << gbs << " max_err=" << err
                      << "\n";
        }

        append_csv(
            csv,
            info.rank,
            "add",
            "add",
            d.N_local,
            0,
            0,
            0,
            ms,
            0.0,
            gbs);

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
        CUDA_CHECK(cudaFree(dz));
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Part A1 – COPY
    // ══════════════════════════════════════════════════════════════════════════
    else if (mode == "copy") {
        long long N = args.get_ll("N", 10'000'000LL);
        auto d      = dist_1d(N, info.rank, info.size);

        std::vector<float> x(d.N_local);
        std::vector<float> y(d.N_local, 0.f);

        fill_random(x, 1000 + info.rank);

        float *dx = util::cuda_malloc_checked<float>(d.N_local);
        float *dy = util::cuda_malloc_checked<float>(d.N_local);

        util::cuda_memcpy_checked(
            dx, x.data(), d.N_local, cudaMemcpyHostToDevice, stream);
        util::cuda_memcpy_checked(
            dy, y.data(), d.N_local, cudaMemcpyHostToDevice, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Warm-up
        launch_copy((int)d.N_local, dx, dy, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Re-upload y so timing run starts from a clean state
        util::cuda_memcpy_checked(
            dy, y.data(), d.N_local, cudaMemcpyHostToDevice, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer gt;
        gt.start(stream);
        launch_copy((int)d.N_local, dx, dy, stream);
        float ms = gt.stop(stream);

        util::cuda_memcpy_checked(
            y.data(), dy, d.N_local, cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float err = max_abs_diff((int)d.N_local, y.data(), x.data());

        // Effective bandwidth: read x, write y = 2 arrays
        double bytes = 2.0 * (double)d.N_local * sizeof(float);
        double gbs   = (bytes / 1e9) / (ms / 1e3);

        if (info.rank == 0) {
            std::cout << "[COPY] rank=" << info.rank << " N_local=" << d.N_local
                      << " ms=" << ms << " GB/s=" << gbs << " max_err=" << err
                      << "\n";
        }

        append_csv(
            csv, info.rank, "copy", "copy", d.N_local, 0, 0, 0, ms, 0.0, gbs);

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Part A2 – Parallel reduction
    // ══════════════════════════════════════════════════════════════════════════
    else if (mode == "reduce")
    {
        long long N = args.get_ll("N", 10'000'000LL);
        auto d      = dist_1d(N, info.rank, info.size);

        // Scratch for reduction
        float *scratch = util::cuda_malloc_checked<float>(1024);

        std::vector<float> x(d.N_local);
        fill_random(x, 3000 + info.rank);

        float *dx = util::cuda_malloc_checked<float>(d.N_local);
        util::cuda_memcpy_checked(
            dx, x.data(), d.N_local, cudaMemcpyHostToDevice, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Warm-up (not timed)
        gpu_reduce_sum(dx, (int)d.N_local, scratch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Re-upload data since reduce allocates/frees temp buffers
        util::cuda_memcpy_checked(
            dx, x.data(), d.N_local, cudaMemcpyHostToDevice, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer gt;
        gt.start(stream);
        float local_gpu = gpu_reduce_sum(dx, (int)d.N_local, scratch, stream);
        float ms        = gt.stop(stream);

        // Combine partial sums from all ranks
        float global_gpu = 0.f;
        MPI_Allreduce(
            &local_gpu,
            &global_gpu,
            1,
            MPI_FLOAT,
            MPI_SUM,
            MPI_COMM_WORLD);

        // CPU reference
        float local_ref  = cpu_reduce_sum((int)d.N_local, x.data());
        float global_ref = 0.f;
        MPI_Allreduce(
            &local_ref,
            &global_ref,
            1,
            MPI_FLOAT,
            MPI_SUM,
            MPI_COMM_WORLD);

        // Tolerance: single-precision epsilon * sqrt(N) is a conservative bound
        float tol = 1e-3f * std::sqrt((float)N);
        float err = std::fabs(global_gpu - global_ref);

        // Effective bandwidth: read N_local floats (1 array, read only)
        double bytes = (double)d.N_local * sizeof(float);
        double gbs   = (bytes / 1e9) / (ms / 1e3);

        if (info.rank == 0) {
            std::cout << "[REDUCE] N=" << N << " gpu=" << global_gpu
                      << " ref=" << global_ref << " err=" << err
                      << " tol=" << tol << " ms=" << ms
                      << " GB/s=" << gbs
                      << (err < tol ? "  PASS" : "  FAIL") << "\n";
        }

        append_csv(csv, info.rank, "reduce", "tree", N, 0, 0, 0, ms, 0.0, gbs);

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(scratch));
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Part B – GEMM
    // ══════════════════════════════════════════════════════════════════════════
    else if (mode == "gemm")
    {
        int M              = args.get_int("M", 1024);
        int Nmat           = args.get_int("N", 1024); // matrix N (not vector N)
        int K              = args.get_int("K", 1024);
        std::string kernel = args.get("kernel", "tiled");

        auto d = dist_rows(M, Nmat, K, info.rank, info.size);

        // Rank 0 generates the full A; all ranks receive their local rows
        std::vector<float> A_full, B;
        if (info.rank == 0) {
            A_full.resize((size_t)M * K);
            fill_random(A_full, 111);
        }
        B.resize((size_t)K * Nmat);
        if (info.rank == 0) {
            fill_random(B, 222);
        }

        // Scatter rows of A
        std::vector<int> countsA, displsA;
        build_counts_displs_rows(M, K, info.size, countsA, displsA);

        std::vector<float> A_local((size_t)d.M_local * K);
        MPI_Scatterv(
            info.rank == 0 ? A_full.data() : nullptr,
            countsA.data(),
            displsA.data(),
            MPI_FLOAT,
            A_local.data(),
            (int)A_local.size(),
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD);

        // Broadcast B to all ranks
        MPI_Bcast(B.data(), (int)B.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Allocate GPU memory
        GemmScratch gemm_scratch(d.M_local, Nmat, K);
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, A_local.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, (size_t)d.M_local * Nmat * sizeof(float)));

        CUDA_CHECK(cudaMemcpyAsync(
            dA,
            A_local.data(),
            A_local.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            dB,
            B.data(),
            B.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaMemsetAsync(
            dC,
            0,
            (size_t)d.M_local * Nmat * sizeof(float),
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ── Timed kernel
        // ────────────────────────────────────────────────────── Measure MPI
        // barrier (communication/synchronization) cost separately
        CpuTimer comm_timer;
        comm_timer.start();
        MPI_Barrier(MPI_COMM_WORLD);
        double comm_ms = comm_timer.stop() * 1e3;

        GpuTimer gt;
        gt.start(stream);
        if (kernel == "naive") {
            launch_gemm_naive(d.M_local, Nmat, K, dA, dB, dC, stream);
        } else if (kernel == "optimised") {
            launch_gemm_optimised(d.M_local, Nmat, K, dA, dB, dC, gemm_scratch, stream);
        } else {
            launch_gemm_tiled16(d.M_local, Nmat, K, dA, dB, dC, stream);
        }
        float comp_ms = gt.stop(stream);

        std::vector<float> C_local((size_t)d.M_local * Nmat);
        CUDA_CHECK(cudaMemcpyAsync(
            C_local.data(),
            dC,
            (size_t)d.M_local * Nmat * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Validation (only for small matrices to keep CI fast)
        float err = -1.f;
        if ((long long)M * Nmat <= 1024LL * 1024LL) {
            std::vector<float> C_ref((size_t)d.M_local * Nmat);
            cpu_gemm(
                d.M_local,
                Nmat,
                K,
                A_local.data(),
                B.data(),
                C_ref.data());
            err =
                max_abs_diff((int)C_local.size(), C_local.data(), C_ref.data());
        }

        double flops  = 2.0 * (double)d.M_local * Nmat * K;
        double gflops = (flops / 1e9) / (comp_ms / 1e3);

        if (info.rank == 0) {
            std::cout << "[GEMM] kernel=" << kernel << " M=" << M
                      << " N=" << Nmat << " K=" << K << " comp_ms=" << comp_ms
                      << " comm_ms=" << comm_ms << " GFLOP/s=" << gflops
                      << (err >= 0 ? " max_err=" + std::to_string(err) : "")
                      << "\n";
        }

        append_csv(
            csv,
            info.rank,
            "gemm",
            kernel,
            0,
            M,
            Nmat,
            K,
            comp_ms,
            gflops,
            comm_ms);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    else {
        if (info.rank == 0) {
            std::cerr << "Unknown mode '" << mode
                      << "'. Use axpy|reduce|gemm.\n";
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return 0;
}
