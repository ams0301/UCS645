#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 200000000; // workload per process
    if (argc > 1) N = atoll(argv[1]);

    MPI_Barrier(MPI_COMM_WORLD);

    double comm_start = MPI_Wtime();
    long long token = rank;
    if (size > 1) {
        if (rank == 0) {
            MPI_Send(&token, 1, MPI_LONG_LONG, 1, 7, MPI_COMM_WORLD);
            MPI_Recv(&token, 1, MPI_LONG_LONG, size - 1, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(&token, 1, MPI_LONG_LONG, rank - 1, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&token, 1, MPI_LONG_LONG, (rank + 1) % size, 7, MPI_COMM_WORLD);
        }
    }
    double comm_time = MPI_Wtime() - comm_start;

    double comp_start = MPI_Wtime();
    volatile double sum = 0.0;
    for (long long i = 1; i <= N; ++i) {
        sum += 1.0 / (double)(i + rank);
    }
    double comp_time = MPI_Wtime() - comp_start;

    double total = comp_time + comm_time;
    double max_total = 0.0;
    MPI_Reduce(&total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double global_sum = 0.0;
    double local_sum = (double)sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Processes=%d N=%lld\n", size, N);
        printf("Tp (max process total time): %.6f sec\n", max_total);
        printf("Communication time (rank0): %.6f sec\n", comm_time);
        printf("Computation time (rank0): %.6f sec\n", comp_time);
        printf("Checksum (ignore value, for anti-opt): %.6f\n", global_sum);
        printf("Run with -np 1,2,4,8 and compute:\n");
        printf("Speedup Sp=T1/Tp, Efficiency Ep=Sp/p\n");
    }
    MPI_Finalize();
    return 0;
}
