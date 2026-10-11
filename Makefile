include deps/flags.mk

all: bench_kernels

bench_kernels: bench_kernels.cpp kernels.h deps/gemm.h deps/util.h
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $< -o $@ $(OMP_LDFLAGS)

run: bench_kernels
	mkdir -p results
	./bench_kernels results | tee results/kernels.txt

clean:
	rm -f bench_kernels

.PHONY: all run clean
