#include "correlate.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<double> norm((size_t)ny * nx);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        const float* row = data + (size_t)y * nx;
        double mean = 0.0;
        #pragma omp simd reduction(+:mean)
        for (int x = 0; x < nx; ++x) mean += row[x];
        mean /= nx;

        double sq = 0.0;
        #pragma omp simd reduction(+:sq)
        for (int x = 0; x < nx; ++x) {
            double v = row[x] - mean;
            norm[(size_t)y * nx + x] = v;
            sq += v * v;
        }

        double inv = (sq > 0.0) ? 1.0 / std::sqrt(sq) : 0.0;
        #pragma omp simd
        for (int x = 0; x < nx; ++x) norm[(size_t)y * nx + x] *= inv;
    }

    const int block = 16;
    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < ny; ii += block) {
        for (int jj = 0; jj <= ii; jj += block) {
            int iMax = std::min(ii + block, ny);
            int jMax = std::min(jj + block, ny);
            for (int i = ii; i < iMax; ++i) {
                for (int j = jj; j < jMax && j <= i; ++j) {
                    const double* a = &norm[(size_t)i * nx];
                    const double* b = &norm[(size_t)j * nx];
                    double dot = 0.0;
                    #pragma omp simd reduction(+:dot)
                    for (int x = 0; x < nx; ++x) dot += a[x] * b[x];
                    result[i + (size_t)j * ny] = (float)dot;
                }
            }
        }
    }
}
