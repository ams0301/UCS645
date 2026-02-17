#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 8;
    if (argc > 1) N = atoi(argv[1]);

    if (N % size != 0) {
        if (rank == 0) printf("N must be divisible by process count. Given N=%d, p=%d\n", N, size);
        MPI_Finalize();
        return 0;
    }

    int rows_per_proc = N / size;

    int* matrix = NULL;
    int* vector = (int*)malloc((size_t)N * sizeof(int));
    int* result = NULL;

    int* local_matrix = (int*)malloc((size_t)rows_per_proc * N * sizeof(int));
    int* local_result = (int*)malloc((size_t)rows_per_proc * sizeof(int));

    if (rank == 0) {
        matrix = (int*)malloc((size_t)N * N * sizeof(int));
        result = (int*)malloc((size_t)N * sizeof(int));

        for (int i = 0; i < N; ++i) {
            vector[i] = i;
            for (int j = 0; j < N; ++j) matrix[i * N + j] = i + j;
        }
    }

    MPI_Scatter(matrix, rows_per_proc * N, MPI_INT,
                local_matrix, rows_per_proc * N, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Bcast(vector, N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; ++i) {
        int sum = 0;
        for (int j = 0; j < N; ++j) sum += local_matrix[i * N + j] * vector[j];
        local_result[i] = sum;
    }

    MPI_Gather(local_result, rows_per_proc, MPI_INT,
               result, rows_per_proc, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Result vector:\n");
        for (int i = 0; i < N; ++i) printf("%d ", result[i]);
        printf("\n");
    }

    free(local_matrix);
    free(local_result);
    free(vector);
    if (rank == 0) {
        free(matrix);
        free(result);
    }

    MPI_Finalize();
    return 0;
}
