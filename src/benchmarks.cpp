#include <coursework/benchmarks.hpp>

#include <cstdio>

void append_csv(
    const std::string &path,
    int rank,
    const std::string &mode,
    const std::string &kernel,
    long long N,
    int M,
    int Nmat,
    int K,
    double ms_gpu,
    double gflops,
    double gbs) {
    // Write header only when creating the file for the first time
    bool write_header = false;
    {
        FILE *test = std::fopen(path.c_str(), "r");
        if (!test) {
            write_header = true;
        } else {
            std::fclose(test);
        }
    }

    FILE *f = std::fopen(path.c_str(), "a");
    if (!f) {
        return;
    }
    if (write_header) {
        std::fprintf(f, "rank,mode,kernel,N,M,Nmat,K,ms_gpu,GFLOPs,GBs\n");
    }
    std::fprintf(
        f,
        "%d,%s,%s,%lld,%d,%d,%d,%.6f,%.3f,%.3f\n",
        rank,
        mode.c_str(),
        kernel.c_str(),
        N,
        M,
        Nmat,
        K,
        ms_gpu,
        gflops,
        gbs);
    std::fclose(f);
}
