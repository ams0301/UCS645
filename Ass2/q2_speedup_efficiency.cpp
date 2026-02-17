#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    long N = (argc > 1) ? std::stol(argv[1]) : 100000000;
    std::vector<double> A(N, 1.0), B(N, 2.0), C(N, 0.0);

    double start = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; ++i) C[i] = A[i] + B[i];
    double tp = omp_get_wtime() - start;

    std::cout << "Tp=" << tp << " sec, Throughput=" << (N / tp) << " elements/sec\n";
    std::cout << "Run with OMP_NUM_THREADS=1 once to get T1, then compute:\n";
    std::cout << "Speedup S(p)=T1/Tp, Efficiency E(p)=S(p)/p\n";
    return 0;
}
