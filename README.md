<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .mojo"></p>

# tandem-mojo

Mojo implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator. It produces the stream the specification defines, bit for bit.
It is fast on CPUs and GPUs alike. The whole port is one file, `tandem.mojo`.

## Install

```sh
pixi install          # Mojo 1.1 and MAX 26.6 from the Modular conda channel
```

The GPU fills need the `max` package. On an NVIDIA driver older than 580, set
`MODULAR_NVPTX_COMPILER_PATH` to a CUDA 12.8 `ptxas`. On macOS, the Metal toolchain needs a
full Xcode install.

## Use

```mojo
from std.memory.alloc import unsafe_alloc
from tandem import Tandem

def main() raises:
    var rng = Tandem(42)                       # 128-bit seed, default K
    var x = rng.next_f64()
    var words = unsafe_alloc[UInt32](1 << 20)
    rng.fill_u32(words, 1 << 20)
    var i = rng.below_u32(10)                  # uniform in 0..10, Lemire
    var z = rng.normal_f64()                   # Box-Muller from two f64 draws
    var e = rng.exponential_f64()              # -log(1 - u), one f64 draw
    var worker = rng.split(7)                  # by index, from the key alone
    var kids = rng.fork(4)                     # from the current block, parent moves on
```

```mojo
from max.gpu.host import DeviceContext
from tandem import fill_f64_gpu, seed

var ctx = DeviceContext()
var dev = ctx.enqueue_create_buffer[DType.float64](1 << 24)
fill_f64_gpu(ctx, seed(42), 0, (1 << 24) // 16, 32, dev.unsafe_ptr())   # rows 0 to 2^20
```

## What it provides

- `Tandem`: a generator with a 128-bit key, 64-bit bit position and chunk length `K`.
  `Tandem(seed)` and `Tandem.from_key` raise when `K` is not a power of two in 1 to 65536.
- Scalar draws of every specification type: `Bool`, 8 to 128-bit integers, `Float32`,
  `Float64`, binary16 as `UInt16`, `char` as `UInt32`, complex `Float16`, `Float32`, `Float64`.
- `at_*` for random access, `split`, `fork`, `sub`, `from_key`.
- CPU fills for every type: `fill_u8` to `fill_u128`, `fill_i8` to `fill_i128`, `fill_f32`,
  `fill_f64`, `fill_f16_bits`, `fill_char`, `fill_bool`, `fill_c16_bits`, `fill_c32`,
  `fill_c64`. Each equals the scalar draws it replaces.
- `below_u32`, `below_u64`, `fill_below_u32`, `fill_below_u64`. A fill cut into chunks equals
  the whole fill.
- `normal_f64`, `normal_f32`, `normal2_f64`, `normal2_f32` and `fill_normal_*`. They are byte
  identical to tandem-c.
- `exponential_f64`, `exponential_f32`, `fill_exponential_f64`, `fill_exponential_f32`. They
  are byte identical to tandem-c.
- GPU fills `fill_u32_gpu`, `fill_u64_gpu`, `fill_f32_gpu`, `fill_f64_gpu`. They take a key
  and a row range, write whole rows and agree with the CPU fill from bit position
  `1024 * first_row`.
- Parallel use: element `i` of a fill is draw `i`, so any decomposition reproduces a serial run.
  See [Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative).

## Tests

```sh
pixi run test         # CPU tests
pixi run test-gpu     # on a host with a supported GPU
```

- The specification vectors in `tests/vectors_data.mojo`, generated from the spec's `vectors.json`.
- Long fills, scalar draws and random access against the dumps in `tests/data`, copied from
  `tandem-c/tests/data`.
- Fills against the scalar draws at offsets and lengths that cut rows and chunks.
- Bounded integers, normals and exponentials against the tandem-c cross fixtures and hashes.
- GPU fills against the CPU fills. CI does not run them.

## Speed

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
| `fill_normal_f64` | 4.99 |
| `next_f64` chain, ns per draw | 2.68 |

GPU fills into device memory, `pixi run bench-gpu`, 1 GiB per fill, minimum of 21, in GiB/s.

| NVIDIA A100 40 GB PCIe | GiB/s |
|---|---|
| `fill_u32_gpu` | 1186 |
| `fill_u64_gpu` | 1208 |
| `fill_f32_gpu` | 1160 |
| `fill_f64_gpu` | 1188 |

Exponential fills against `-log(1 - random_float64())`, the only exponential in the Mojo
standard library, `pixi run bench-exponential`, 2^22 elements, in GiB/s.

| Apple M4 | `f64` | `f32` |
|---|---|---|
| `fill_exponential_*` | 5.34 | 5.95 |
| `-log(1 - random_float64())` | 0.31 | 0.15 |

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
