# Speed

`pixi run bench` produces the CPU figures, `pixi run bench-gpu` the GPU figures and
`pixi run bench-exponential` the comparison with the standard library.

## CPU

One thread, `pixi run bench`, minimum of seven runs of 2^24 elements, in GiB/s.

| Apple M4 | GiB/s |
|---|---|
| `fill_u32` | 17.9 |
| `fill_u64` | 17.8 |
| `fill_f32` | 15.6 |
| `fill_f64` | 15.6 |
| `fill_below_u32`, bound 1000 | 11.5 |
| `fill_below_u64`, bound 1000 | 10.5 |
| `fill_normal_f32` | 5.5 |
| `fill_normal_f64` | 6.54 |
| `next_f64` chain, ns per draw | 2.68 |

The CPU fill converts floats in the same pass that stores the row. The 32-bit low word of each
product is a plain vector multiply, and only the high word is a widening one: taking both from
one 64-bit product made LLVM emit two widening multiplies. The bounded and normal fills do
extra arithmetic per draw, so they run below the plain rate. The `f32` normals are limited by
the vector pipes, not by memory. The `f64` normals spend most of their time in the fallbacks
of the 0.43 % of draws that miss the ziggurat's inner rectangles.

## GPU

GPU fills into device memory, `pixi run bench-gpu`, 1 GiB per fill, minimum of 21 after a
half-second warm-up, GPU idle, in GiB/s.

| NVIDIA A100 40 GB PCIe | GiB/s |
|---|---|
| `fill_u32_gpu` | 1186 |
| `fill_u64_gpu` | 1208 |
| `fill_f32_gpu` | 1160 |
| `fill_f64_gpu` | 1188 |

The GPU kernel stores each block from registers and has no shared-memory tile.

## Other generators

Exponential fills against `-log(1 - random_float64())`, the only exponential in the Mojo
standard library, `pixi run bench-exponential`, 2^22 elements, in GiB/s.

| Apple M4 | `f64` | `f32` |
|---|---|---|
| `fill_exponential_*` | 5.34 | 5.95 |
| `-log(1 - random_float64())` | 0.31 | 0.15 |
