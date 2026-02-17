# Assignment 3 (Makefile + Correlation)

Files are split per question:
- `q1_sequential.cpp` → baseline sequential
- `q2_parallel_openmp.cpp` → OpenMP parallel
- `q3_optimized.cpp` → optimized (SIMD + blocking + OpenMP)
- `main.cpp` → CLI runner (`ny nx` from command line)
- `Makefile`

## Build
```bash
make
```

## Run
```bash
./q1_seq 2000 1000
OMP_NUM_THREADS=8 ./q2_parallel 2000 1000
OMP_NUM_THREADS=8 ./q3_optimized 2000 1000
```

## Perf stats
```bash
./perf_compare.sh 2000 1000
```
