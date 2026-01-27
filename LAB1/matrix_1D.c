#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 500   // use 500 or 1000 depending on system

int main() {
    int i, j, k;
    double **A, **B, **C;

    A = (double **)malloc(N * sizeof(double *));
    B = (double **)malloc(N * sizeof(double *));
    C = (double **)malloc(N * sizeof(double *));

    for (i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }

    // Initialize matrices
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }

    double start = omp_get_wtime();

    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double end = omp_get_wtime();

    printf("1D Parallel Time: %f seconds\n", end - start);

    return 0;
}
//1D Parallel Time: 0.074493 seconds  2 threads
//1D Parallel Time: 0.042550 seconds  4 threads
//1D Parallel Time: 0.039021 seconds  8 threads