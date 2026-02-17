#include "correlate.h"
#include <cmath>
#include <vector>

void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<double> norm((size_t)ny * nx);

    for (int y = 0; y < ny; ++y) {
        double mean = 0.0;
        for (int x = 0; x < nx; ++x) mean += data[x + y * nx];
        mean /= nx;

        double sq = 0.0;
        for (int x = 0; x < nx; ++x) {
            double v = data[x + y * nx] - mean;
            norm[x + y * nx] = v;
            sq += v * v;
        }
        double inv = (sq > 0.0) ? 1.0 / std::sqrt(sq) : 0.0;
        for (int x = 0; x < nx; ++x) norm[x + y * nx] *= inv;
    }

    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double dot = 0.0;
            for (int x = 0; x < nx; ++x) dot += norm[x + i * nx] * norm[x + j * nx];
            result[i + j * ny] = (float)dot;
        }
    }
}
