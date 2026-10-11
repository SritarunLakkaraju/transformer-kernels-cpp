# Softmax, LayerNorm and attention kernels in C++

The building blocks of a transformer layer, each as a readable scalar reference and a NEON version,
with numerical stability demonstrated rather than assumed.

## Results

| Kernel | Result |
|---|---|
| Vectorised `exp` (4 floats) | Max relative error 8.4e-08 over 8 million inputs |
| Softmax, NEON vs stable scalar | 3.3x to 3.5x faster; within 1e-7 of float64 |
| LayerNorm, NEON vs stable scalar | 3.9x to 8.3x faster |
| Attention block, L = 1024, 1 thread | 507 ms (naive loops) -> 21.4 ms |
| Attention block, L = 4096, 11 threads | 53 ms |
| Attention score memory, L = 4096 | 64 MB -> 4 MB with tiling |

Measured on an Apple M3 Pro (5 performance + 6 efficiency cores), Apple clang 21, `-O3 -mcpu=native`, no `-ffast-math`. Raw output is in `results/`.

### Stability

Softmax of `[1000, 999, 995, 0, -1000, 90]`: the textbook formula returns NaN; subtracting the
maximum first gives `0.727475 0.267623 0.004902 ...`, matching float64.

LayerNorm on `offset + noise`, n = 768, maximum error against float64:

| Offset | One-pass variance | Two-pass scalar | Two-pass NEON |
|---:|---:|---:|---:|
| 0 | 1.7e-07 | 5.3e-07 | 1.5e-07 |
| 100 | 1.9e-02 | 3.6e-05 | 1.7e-05 |
| 1000 | 4.2e-01 | 4.7e-04 | 4.9e-05 |
| 10000 | NaN | 5.6e-03 | 5.1e-04 |

The one-pass formula `E[x^2] - E[x]^2` suffers catastrophic cancellation. The NEON version is the
most accurate of the three because its 16 partial sums amount to pairwise summation.

### Attention tiling

The tiled version never materialises the L x L score matrix. Queries are processed 256 rows at a
time: scores via GEMM, softmax in place, then the product with V. Scratch memory grows linearly
with sequence length instead of quadratically. On this machine tiling is a memory optimisation,
not a speed one: single-threaded time is the same with and without it.

## What is in the attention block

```
h = LayerNorm(x)
Q = h Wq,  K = h Wk,  V = h Wv
per head:  P = softmax(Q K^T / sqrt(d)),  O = P V
y = x + concat(O) Wo
```

## Requirements

- macOS on Apple Silicon (the SIMD code is ARM NEON) with Apple clang
- Homebrew `libomp`

```sh
brew install libomp
```

## Build and run

```sh
make          # build bench_kernels
make run      # accuracy, stability demonstrations and timings -> results/
./bench_kernels "" 128     # optional second argument: attention tile size in query rows
```

## Files

| File | Contents |
|---|---|
| `kernels.h` | `exp_ps`, three softmax and three LayerNorm variants, `attention_naive`, `attention_fast` |
| `bench_kernels.cpp` | Benchmarks and stability demonstrations |
| `deps/gemm.h` | The matrix multiply from [fast-gemm-cpp](https://github.com/SritarunLakkaraju/fast-gemm-cpp) |

## Limitations

- No attention mask, no projection biases; inference only, one sequence at a time.
- Tiling bounds the score memory but not the O(L^2 d) arithmetic.
- The fast paths are ARM NEON only.

## Part of a series

Five projects that build on each other:

1. [fast-gemm-cpp](https://github.com/SritarunLakkaraju/fast-gemm-cpp) - matrix multiply, the kernel everything else uses
2. [conv2d-kernels-cpp](https://github.com/SritarunLakkaraju/conv2d-kernels-cpp) - convolution on top of that GEMM
3. [tiny-inference-engine-cpp](https://github.com/SritarunLakkaraju/tiny-inference-engine-cpp) - a small inference engine on top of both
4. [int8-quantized-kernels-cpp](https://github.com/SritarunLakkaraju/int8-quantized-kernels-cpp) - the same ideas in 8-bit integers
5. [transformer-kernels-cpp](https://github.com/SritarunLakkaraju/transformer-kernels-cpp) - softmax, LayerNorm and attention

Each repository is self-contained: the headers it needs from the earlier projects are copied into `deps/`.
