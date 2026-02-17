# Assignment 2 (OpenMP Performance Metrics)

## Build
```bash
make
```

## Run examples
```bash
OMP_NUM_THREADS=1 ./q1_execution_time
OMP_NUM_THREADS=8 ./q1_execution_time

OMP_NUM_THREADS=8 ./q2_speedup_efficiency
OMP_NUM_THREADS=8 ./q3_scaling

OMP_SCHEDULE="static" OMP_NUM_THREADS=8 ./q4_scheduling_imbalance
OMP_SCHEDULE="dynamic,10" OMP_NUM_THREADS=8 ./q4_scheduling_imbalance
OMP_SCHEDULE="guided" OMP_NUM_THREADS=8 ./q4_scheduling_imbalance

OMP_NUM_THREADS=8 ./q5_sync_overhead
OMP_NUM_THREADS=8 ./q6_false_sharing
OMP_NUM_THREADS=8 ./q7_memory_bandwidth
```

## perf (optional)
```bash
perf stat -e cycles,instructions,cache-misses ./q7_memory_bandwidth
```
