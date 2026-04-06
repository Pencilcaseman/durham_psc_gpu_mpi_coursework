#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace util {
    void exit_with(const char *msg) {
        std::fprintf( stderr, msg);
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }

    template<typename T>
    T *cuda_malloc_checked(size_t n) {
        T *res = nullptr;
        CUDA_CHECK(cudaMalloc(&res, n * sizeof(T)));

        if (res == nullptr) {
            exit_with("Error: Failed to allocate CUDA memory\n");
        }

        return res;
    }

    template<typename T>
    void cuda_memcpy_checked(T *dst, const T *src, size_t n, cudaMemcpyKind direction, cudaStream_t stream) {
        if (dst == nullptr) {
            exit_with("Memcpy destination is nullptr\n");
        }

        if (src == nullptr) {
            exit_with("Memcpy source is nullptr\n");
        }

        CUDA_CHECK(cudaMemcpyAsync(
            dst,
            src,
            n * sizeof(T),
            direction,
            stream));
    }
}
