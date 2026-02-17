#include <omp.h>
#include <cmath>
#include <iostream>
#include <vector>

static inline double work(int i) {
    double x = 0.0;
    int reps = (i % 1000) + 50;
    for (int k = 0; k < reps; ++k) x += std::sin(i * 0.001 + k);
    return x;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::stoi(argv[1]) : 200000;
    int T = omp_get_max_threads();
    std::vector<double> thread_time(T, 0.0);
    volatile double sink = 0.0;

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double s = omp_get_wtime();
        #pragma omp for schedule(runtime)
        for (int i = 0; i < N; ++i) sink += work(i);
        double e = omp_get_wtime();
        thread_time[tid] = e - s;
    }
    double total = omp_get_wtime() - t0;

    double tmax = 0.0, tsum = 0.0;
    for (double t : thread_time) { tmax = std::max(tmax, t); tsum += t; }
    double tavg = tsum / T;
    double imbalance = (tmax - tavg) / tavg;

    std::cout << "Total=" << total << " sec, Tmax=" << tmax << " Tavg=" << tavg
              << " Imbalance=" << imbalance << "\n";
    std::cout << "Try: OMP_SCHEDULE=\"static\" OR \"dynamic,10\" OR \"guided\"\n";
    return (sink == -1.0);
}
