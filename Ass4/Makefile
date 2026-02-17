MPICC = mpicc
CFLAGS = -O3 -Wall

TARGETS = q1_hello_mpi q2_send_recv q3_broadcast q4_reduce_sum q5_matrix_vector_mpi q6_perf_metrics

all: $(TARGETS)

q1_hello_mpi: q1_hello_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@
q2_send_recv: q2_send_recv.c
	$(MPICC) $(CFLAGS) $< -o $@
q3_broadcast: q3_broadcast.c
	$(MPICC) $(CFLAGS) $< -o $@
q4_reduce_sum: q4_reduce_sum.c
	$(MPICC) $(CFLAGS) $< -o $@
q5_matrix_vector_mpi: q5_matrix_vector_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@
q6_perf_metrics: q6_perf_metrics.c
	$(MPICC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
