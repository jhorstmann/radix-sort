# Radix sort implementation using AVX-512

This needs a nightly version of rustc in order to use avx512 intrinsics.

Run benchmarks using

```
RUSTFLAGS=-Ctarget-cpu=native cargo bench
```