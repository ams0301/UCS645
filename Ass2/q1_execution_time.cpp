#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    long N = (argc > 1) ? std::stol(argv[1]) : 100000000;
    std::vector<double> A(N, 1.0), B(N, 2.0), C(N, 0.0);

    double start = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; ++i) C[i] = A[i] + B[i];
    double end = omp_get_wtime();

    std::cout << "N=" << N << " Threads=" << omp_get_max_threads()
              << " Time=" << (end - start) << " sec\n";
    std::cout << "Sample checksum=" << C[0] + C[N - 1] << "\n";
    return 0;
}
