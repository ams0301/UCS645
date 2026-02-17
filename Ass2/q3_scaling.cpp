#include <omp.h>
#include <iostream>
#include <vector>

static double run(long N) {
    std::vector<double> A(N, 1.0), B(N, 2.0), C(N, 0.0);
    double t0 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; ++i) C[i] = A[i] + B[i];
    return omp_get_wtime() - t0;
}

int main(int argc, char** argv) {
    int p = omp_get_max_threads();
    long strongN = (argc > 1) ? std::stol(argv[1]) : 100000000;
    long weakN = (argc > 2) ? std::stol(argv[2]) : 10000000L * p;

    double ts = run(strongN);
    double tw = run(weakN);

    std::cout << "Threads=" << p << "\n";
    std::cout << "Strong scaling (fixed N=" << strongN << "): " << ts << " sec\n";
    std::cout << "Weak scaling (N proportional to p, N=" << weakN << "): " << tw << " sec\n";
    return 0;
}
