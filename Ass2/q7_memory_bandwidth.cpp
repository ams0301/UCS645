#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    long N = (argc > 1) ? std::stol(argv[1]) : 100000000;
    double alpha = 2.5;
    std::vector<double> A(N, 0.0), B(N, 1.0), C(N, 2.0);

    double t0 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; ++i) A[i] = B[i] + alpha * C[i];
    double tp = omp_get_wtime() - t0;

    double bytes = 3.0 * N * sizeof(double);
    double bandwidthGBs = bytes / tp / 1e9;

    std::cout << "Time=" << tp << " sec\n";
    std::cout << "Effective bandwidth=" << bandwidthGBs << " GB/s\n";
    return 0;
}
