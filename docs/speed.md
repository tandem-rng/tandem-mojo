# Speed

`pixi run bench` and `pixi run bench-exponential` produce the CPU figures, `pixi run bench-gpu`
the GPU figures.

## CPU

One thread, minimum of seven runs of 2^24 elements (2^22 and five runs for the exponentials),
in GiB/s of output, every row from one session. A `next_f64` draw counts 8 bytes. The baseline
is the Mojo standard library: Philox4x32-10 of `std.random.philox` for the plain fills, its
`NormalRandom` for the `f32` normals, and `std.random` for the rest: `rand` for bounded
integers, `randn` for the `f64` normals, `-log(1 - random_float64())` for the exponentials and
`random_float64` for the chain.

| Apple M4 Pro | Tandem | Mojo standard library |
|---|---|---|
| `fill_u32` | 17.8 | 1.42 |
| `fill_u64` | 18.0 | 1.42 |
| `fill_f32` | 15.6 | 1.38 |
| `fill_f64` | 15.4 | 1.37 |
| `fill_below_u32`, bound 1000 | 11.5 | 0.27 |
| `fill_below_u64`, bound 1000 | 10.4 | 0.53 |
| `fill_normal_f32` | 5.16 | 0.33 |
| `fill_normal_f64` | 6.59 | 0.15 |
| `fill_exponential_f32` | 6.12 | 0.16 |
| `fill_exponential_f64` | 5.50 | 0.32 |
| `next_f64` chain | 8.71 | 0.58 |

The CPU fill converts floats in the same pass that stores the row. The 32-bit low word of each
product is a plain vector multiply, and only the high word is a widening one: taking both from
one 64-bit product made LLVM emit two widening multiplies. The bounded and normal fills do
extra arithmetic per draw, so they run below the plain rate. The `f32` normals are limited by
the vector pipes, not by memory. The `f64` normals spend most of their time in the fallbacks
of the 0.43 % of draws that miss the ziggurat's inner rectangles.

A scalar draw reads one word of the cached row through a pointer. Indexing the 32-wide row
vector by a runtime index spilled the whole vector on every draw, which held the chain at
2.8 GiB/s. F's rounds unroll on the GPU only: on the CPU the unrolled copy inlines into the row
seek of each scalar draw and cost the chain a third.

## GPU

GPU fills into device memory, `pixi run bench-gpu`, 1 GiB per fill, minimum of 21 after a
half-second warm-up, in GiB/s. The GPU had no other process, and a second run agreed within 1 %.
The cuRAND column is Philox4x32-10 of cuRAND 10.3.9 in the same run, by the same method, through
its host API in the `DeviceContext`'s context. cuRAND has no 64-bit integer output for Philox, so
the `u64` row's figure is `curandGenerate` writing the same bytes as 32-bit words, marked
"nearest".

| NVIDIA A100 40 GB PCIe | GiB/s | cuRAND Philox4x32-10 | cuRAND call |
|---|---|---|---|
| `fill_u32_gpu` | 1379 | 1304 | `curandGenerate` |
| `fill_u64_gpu` | 1379 | 1306 | `curandGenerate`, nearest |
| `fill_f32_gpu` | 1378 | 1287 | `curandGenerateUniform` |
| `fill_f64_gpu` | 1378 | 801 | `curandGenerateUniformDouble` |

For K a multiple of 8, the default, the kernel stages eight steps of 32 groups in 32 KiB of
shared memory and stores them as 512 contiguous bytes per warp, as tandem-cuda's tile kernel
does, at the card's memory bandwidth. Other K take one thread per chunk with direct stores. The
fills ran at 1171 to 1201 GiB/s before the tile and before F's round constants were unrolled:
indexing them with the round number put them in local memory.
