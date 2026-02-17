#include <omp.h>
#include <iostream>
#include <vector>

constexpr int CACHE_LINE = 64;

int main(int argc, char** argv) {
    long iters = (argc > 1) ? std::stol(argv[1]) : 200000000;
    int T = omp_get_max_threads();

    std::vector<long long> plain(T, 0);
    std::vector<std::vector<long long>> padded(T, std::vector<long long>(CACHE_LINE / sizeof(long long), 0));

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (long i = 0; i < iters; ++i) plain[tid]++;
    }
    double tPlain = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (long i = 0; i < iters; ++i) padded[tid][0]++;
    }
    double tPadded = omp_get_wtime() - t0;

    std::cout << "Unpadded time=" << tPlain << " sec\n";
    std::cout << "Padded time=" << tPadded << " sec\n";
    return 0;
}
