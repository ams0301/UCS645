#include "correlate.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <ny(rows)> <nx(cols)>\n";
        return 1;
    }

    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);

    std::vector<float> data((size_t)ny * nx), result((size_t)ny * ny, 0.0f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : data) v = dist(rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    correlate(ny, nx, data.data(), result.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "ny=" << ny << " nx=" << nx << " time=" << sec << " sec\n";
    std::cout << "sample: corr(0,0)=" << std::fixed << std::setprecision(6) << result[0] << "\n";
    return 0;
}
