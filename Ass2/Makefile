CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -fopenmp

TARGETS = q1_execution_time q2_speedup_efficiency q3_scaling q4_scheduling_imbalance q5_sync_overhead q6_false_sharing q7_memory_bandwidth

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
