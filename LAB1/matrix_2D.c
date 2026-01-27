#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 500

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

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }

    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2) private(k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double end = omp_get_wtime();

    printf("2D Parallel Time: %f seconds\n", end - start);

    return 0;
}
// 2D Parallel Time: 0.068277 seconds  2 threads
// 2D Parallel Time: 0.044401 seconds  4 threads
// 2D Parallel Time: 0.035204 seconds  8 threads
// In this experiment, matrix multiplication was parallelized using OpenMP to study the effect of different ways of distributing work among threads. In the one dimensional threading approach, only the outer loop was parallelized, so each thread was responsible for computing one or more rows of the result matrix. This reduced execution time compared to sequential execution but limited the amount of parallelism. In the two dimensional threading approach, nested loops were parallelized so that the computation of matrix elements was shared more evenly among threads. This created more parallel tasks and improved load balancing, leading to better processor utilization and faster execution for large matrices.
