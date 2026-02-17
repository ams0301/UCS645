#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    long N = (argc > 1) ? std::stol(argv[1]) : 50000000;
    std::vector<double> A(N, 1.0);

    double sum1 = 0.0, sum2 = 0.0;

    double t0 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; ++i) {
        #pragma omp critical
        sum1 += A[i];
    }
    double tCritical = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum2)
    for (long i = 0; i < N; ++i) sum2 += A[i];
    double tReduction = omp_get_wtime() - t0;

    std::cout << "critical: " << tCritical << " sec, sum=" << sum1 << "\n";
    std::cout << "reduction: " << tReduction << " sec, sum=" << sum2 << "\n";
    return 0;
}
