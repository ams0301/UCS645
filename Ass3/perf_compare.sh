#!/usr/bin/env bash
set -e
NY=${1:-2000}
NX=${2:-1000}

echo "Building..."
make

echo "\nSequential"
OMP_NUM_THREADS=1 /usr/bin/time -f "real=%E" ./q1_seq "$NY" "$NX"

echo "\nParallel (all threads)"
OMP_NUM_THREADS=$(nproc) /usr/bin/time -f "real=%E" ./q2_parallel "$NY" "$NX"

echo "\nOptimized (all threads)"
OMP_NUM_THREADS=$(nproc) /usr/bin/time -f "real=%E" ./q3_optimized "$NY" "$NX"

echo "\nPerf stats (optimized)"
perf stat -e cycles,instructions,cache-misses ./q3_optimized "$NY" "$NX"
