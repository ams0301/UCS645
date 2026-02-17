# Assignment 4 (MPI) - Question-wise Solutions

## Files
- `q1_hello_mpi.c` - Hello World MPI
- `q2_send_recv.c` - Point-to-point communication
- `q3_broadcast.c` - MPI_Bcast
- `q4_reduce_sum.c` - MPI_Reduce (sum)
- `q5_matrix_vector_mpi.c` - Parallel matrix-vector multiplication
- `q6_perf_metrics.c` - Timing, communication/computation split, speedup/efficiency guidance
- `Makefile`

## Build
```bash
make
```

## Run examples
```bash
mpirun -np 4 ./q1_hello_mpi
mpirun -np 2 ./q2_send_recv
mpirun -np 4 ./q3_broadcast
mpirun -np 4 ./q4_reduce_sum
mpirun -np 4 ./q5_matrix_vector_mpi 8

mpirun -np 1 ./q6_perf_metrics
mpirun -np 2 ./q6_perf_metrics
mpirun -np 4 ./q6_perf_metrics
mpirun -np 8 ./q6_perf_metrics
```

For q6, compute:
- `Sp = T1 / Tp`
- `Ep = Sp / p`
