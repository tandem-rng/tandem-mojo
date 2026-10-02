# tandem-mojo

Prototype of [Tandem8x32](https://github.com/tandem-rng/spec) in Mojo: the specification's
building blocks, a CPU row fill over eight SIMD lanes, and a GPU row fill with one thread per
chunk, in one file, `tandem.mojo`. It produces the stream the specification defines, bit for
bit, for 32-bit words. Scalar draws, other widths, and the float mappings are not written yet.

```sh
pixi install          # Mojo 1.1 and MAX 26.6 from the Modular conda channel
pixi run test         # vectors and the reference u32 dump, CPU
pixi run bench        # CPU fill, 2^24 words
pixi run test-gpu     # on a host with a supported GPU
pixi run bench-gpu
```

The GPU fill needs the `max` package. On an NVIDIA driver older than 580 set
`MODULAR_NVPTX_COMPILER_PATH` to a CUDA 12.8 `ptxas`. On macOS, Mojo compiles GPU kernels
through the Metal toolchain, which needs a full Xcode install, not the Command Line Tools.

## Speed

Apple M4, one thread, `pixi run bench`, 2^24 words, minimum of seven after a warm-up: `fill_u32` 17.0 GiB/s.

NVIDIA A100 40 GB (PCIe), `pixi run bench-gpu`, 2^28 words into device memory, minimum of 21
after a 0.5 s warm-up, GPU idle: `fill_u32_gpu` 1188 GiB/s. The kernel stores
each block from registers; it has no shared-memory tile yet.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
