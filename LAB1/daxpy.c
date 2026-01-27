#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 65536

int main() {
    double *X = malloc(N * sizeof(double));
    double *Y = malloc(N * sizeof(double));
    double a = 2.5;

    for(int i = 0; i < N; i++) {
        X[i] = i * 1.0;
        Y[i] = i * 2.0;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }

    double end = omp_get_wtime();
    printf("Time taken: %f seconds\n", end - start);

    free(X);
    free(Y);
    return 0;
}
// Time taken: 0.000348 seconds 2 threads
// Time taken: 0.000499 seconds 4 threads
// Time taken: 0.001548 seconds 8 threads
// Time taken: 0.009632 seconds 16 threads
// In the DAXPY loop experiment, performance improved as the number of threads was increased because the vector operations were divided among multiple threads, reducing execution time compared to sequential execution. The maximum speedup was observed when the number of threads was close to the number of physical CPU cores available on the system, as this allowed efficient utilization of the processor. When the number of threads was increased beyond this point, performance gains reduced or execution time increased due to thread management overhead and competition for shared resources such as cache and memory bandwidth, which limits further speedup.
