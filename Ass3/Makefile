CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -fopenmp -march=native

all: q1_seq q2_parallel q3_optimized

q1_seq: main.cpp q1_sequential.cpp correlate.h
	$(CXX) $(CXXFLAGS) main.cpp q1_sequential.cpp -o q1_seq

q2_parallel: main.cpp q2_parallel_openmp.cpp correlate.h
	$(CXX) $(CXXFLAGS) main.cpp q2_parallel_openmp.cpp -o q2_parallel

q3_optimized: main.cpp q3_optimized.cpp correlate.h
	$(CXX) $(CXXFLAGS) main.cpp q3_optimized.cpp -o q3_optimized

clean:
	rm -f q1_seq q2_parallel q3_optimized

.PHONY: all clean
