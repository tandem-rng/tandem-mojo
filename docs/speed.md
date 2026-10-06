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
| `next_f64` chain | 2.78 |

The CPU fill converts floats in the same pass that stores the row. The 32-bit low word of each
product is a plain vector multiply, and only the high word is a widening one: taking both from
one 64-bit product made LLVM emit two widening multiplies. The bounded and normal fills do
extra arithmetic per draw, so they run below the plain rate. The `f32` normals are limited by
the vector pipes, not by memory. The `f64` normals spend most of their time in the fallbacks
of the 0.43 % of draws that miss the ziggurat's inner rectangles.

## GPU

GPU fills into device memory, `pixi run bench-gpu`, 1 GiB per fill, minimum of 21 after a
half-second warm-up, in GiB/s. The GPU had no other process, and a second run agreed within 1 %.
The cuRAND column is Philox4x32-10 of cuRAND 10.3.9 in the same run, by the same method, through
its host API in the `DeviceContext`'s context. cuRAND has no 64-bit integer output for Philox, so
the `u64` row's figure is `curandGenerate` writing the same bytes as 32-bit words, marked
"nearest".

| NVIDIA A100 40 GB PCIe | GiB/s | cuRAND Philox4x32-10 | cuRAND call |
|---|---|---|---|
| `fill_u32_gpu` | 1189 | 1309 | `curandGenerate` |
| `fill_u64_gpu` | 1201 | 1302 | `curandGenerate`, nearest |
| `fill_f32_gpu` | 1171 | 1277 | `curandGenerateUniform` |
| `fill_f64_gpu` | 1188 | 797 | `curandGenerateUniformDouble` |

The GPU kernel stores each block from registers and has no shared-memory tile, so the 32-bit
fills run below cuRAND's. tandem-cuda's tile kernel reaches 1377 to 1388 GiB/s on the same card.

## Other generators

Exponential fills against `-log(1 - random_float64())`, the only exponential in the Mojo
standard library, `pixi run bench-exponential`, 2^22 elements, in GiB/s.

| Apple M4 | `f64` | `f32` |
|---|---|---|
| `fill_exponential_*` | 5.34 | 5.95 |
| `-log(1 - random_float64())` | 0.31 | 0.15 |
