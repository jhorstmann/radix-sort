# Radix sort implementation using AVX-512

This needs at least version 1.89 of rustc in order to use avx512 intrinsics on the stable channel.

Run benchmarks using

```
RUSTFLAGS=-Ctarget-cpu=native cargo bench
```