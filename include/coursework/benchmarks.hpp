#pragma once
#include <string>

void append_csv(
    const std::string &path,
    int rank,
    int num_ranks,
    const std::string &mode,
    const std::string &kernel,
    long long N,
    long long global_N,
    int M,
    int Nmat,
    int K,
    double ms_gpu,
    double gflops,
    double gbs);
