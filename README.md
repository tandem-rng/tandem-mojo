<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .mojo"></p>

# tandem-mojo

Mojo implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. It produces the stream
the specification defines, bit for bit. The whole port is one file, `tandem.mojo`.

- A `Tandem` is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus a
  cache of the current 1024-bit row.
- Scalar draws of every type in the specification: `Bool`, 8 to 128-bit unsigned and signed
  integers, `Float32`, `Float64`, binary16 as `UInt16` bit patterns, `char` as a `UInt32` code
  point, complex `Float16`, `Float32` and `Float64` as pairs. Random access without advancing
  (`at_*`). Split by index, fork at the current block, sub by purpose, seed whitening, and a raw-key
  constructor.
- CPU fills for every one of those types over eight SIMD lanes: `fill_u8` to `fill_u128`,
  `fill_i8` to `fill_i128`, `fill_f32`, `fill_f64`, `fill_f16_bits`, `fill_char`, `fill_bool`,
  `fill_c16_bits`, `fill_c32`, `fill_c64`. Every fill equals the scalar draws it replaces.
- GPU fills into device memory for `u32`, `u64`, `f32` and `f64`: one thread per chunk, each
  thread stores its blocks from registers.
- Bounded integers (`below_u32`, `below_u64`) and standard normals (`normal_f64`, `normal_f32`,
  and the pairs `normal2_f64`, `normal2_f32`) with fills. They are not in the specification.
  They follow the shared device core in `tandem-cuda`, so every port returns the same integers.
  A bound of 0 returns 0 after one draw. `fill_below_*` takes draw `i` of the plain fill for
  element `i` and consumes exactly one draw per element. A rejected draw retries on
  `sub(purpose).split(g)` of the key, with `g` the global draw index, the aligned start position
  over the draw width plus `i`, so a fill cut into chunks equals the whole fill. The bounded
  APIs are typed by draw width, `below_u32` and `below_u64`, so the draw width never depends on
  a result type. They exist on the CPU only. A normal step uses two uniforms and returns the cos
  half then the sin half. A scalar normal is the cos half, and a normal fill is the flattened
  pairs, so an odd count consumes both uniforms of its last pair. An empty bounded or normal
  fill moves nothing. The normals run on SIMD lanes with the algorithm and coefficients of
  `tandem-c` and an explicit fused multiply-add for every multiply-add, so they are byte
  identical to `tandem-c` (same SHA-256 from `tools/dump_normals.mojo` and
  `tools/dump_normals.c`, checked on arm64). They agree with libm to about 1e-15 in `f64`. The
  `f32` normal runs in `f32`. The bounded fills do the multiply-high and the compare on SIMD
  lanes (u32) or on the integer pipes beside the row generator (u64), and only a rejection takes
  a scalar fixup.
- Exponential draws `exponential_f64` and `exponential_f32` with fills `fill_exponential_f64`
  and `fill_exponential_f32`, `-log(1 - u)` of one uniform per element. Element `i` of a fill is
  the scalar draw `i`, and an empty fill moves nothing. They reuse the polynomial logarithm of the
  normals with an explicit fused multiply-add for every multiply-add, in `f64` from `f64` draws
  and in `f32` from `f32` draws, so the bytes equal `tandem-c`'s (FNV-1a `47f8f98297d94ee2` over
  1e6 values of each width from five positions).

## Use

```sh
pixi install          # Mojo 1.1 and MAX 26.6 from the Modular conda channel
pixi run test         # CPU tests
pixi run bench        # CPU fills
pixi run test-gpu     # on a host with a supported GPU
pixi run bench-gpu
```

```mojo
from std.memory.alloc import unsafe_alloc
from tandem import Tandem

def main() raises:
    var rng = Tandem(42)                       # 128-bit seed, default K
    var x = rng.next_f64()
    var words = unsafe_alloc[UInt32](1 << 20)
    rng.fill_u32(words, 1 << 20)
    var c = rng.next_c64()                     # (re, im), two f64 draws
    var i = rng.below_u32(10)                  # uniform in 0..10, Lemire
    var z = rng.normal_f64()                   # Box-Muller from two f64 draws
    var e = rng.exponential_f64()              # -log(1 - u), one f64 draw
    var worker = rng.split(7)                  # by index, from the key alone
    var kids = rng.fork(4)                     # from the current block, parent moves on
    var raw = Tandem.from_key(rng.key, rng.position(), rng.k)
```

`Tandem(seed)` and `Tandem.from_key` raise when `K` is not a power of two in 1 to 65536.
Every draw aligns the position to the width of its type first, and every fill returns the
generator where the same number of scalar draws would leave it. Fills take a pointer and a count.
Complex fills take the number of complex values and write interleaved `(re, im)` components.

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

## GPU

```mojo
from max.gpu.host import DeviceContext
from tandem import fill_f64_gpu, seed

var ctx = DeviceContext()
var dev = ctx.enqueue_create_buffer[DType.float64](1 << 24)
fill_f64_gpu(ctx, seed(42), 0, (1 << 24) // 16, 32, dev.unsafe_ptr())   # rows 0 to 2^20
```

`fill_u32_gpu`, `fill_u64_gpu`, `fill_f32_gpu` and `fill_f64_gpu` write whole rows,
`[first_row, first_row + nrows)`, of 32, 16, 32 and 16 values, so a GPU fill and a CPU fill from
bit position `1024 * first_row` agree. The GPU fills take a key and a row range, not a `Tandem`,
and do not move a position.

The GPU fill needs the `max` package. On an NVIDIA driver older than 580 set
`MODULAR_NVPTX_COMPILER_PATH` to a CUDA 12.8 `ptxas`. On macOS, Mojo compiles GPU kernels
through the Metal toolchain, which needs a full Xcode install, not the Command Line Tools.

## Tests

```sh
pixi run test
```

- `tests/test_vectors.mojo` checks every vector of the specification.
  `tests/vectors_data.mojo` is generated from the spec repository's `vectors.json` by
  `tools/gen_vectors.py`, and CI fails when it is out of date.
- `tests/test_dumps.mojo` compares long fills, scalar draws and random access with the
  reference stream dumps in `tests/data`, complex fills included. The dumps are copies of
  `tandem-c/tests/data`, and CI fails when they differ.
- `tests/test_tandem.mojo` covers scalar draws of every width, alignment, random access,
  `set_position`, chunk lengths, split, sub and fork.
- `tests/test_fills.mojo` compares every fill with the scalar draws of its type, at chunk
  lengths, offsets and lengths that cut rows and chunks, and checks the position afterwards.
- `tests/test_derived.mojo` compares bounded integers and normals with the cross-check values
  of `tandem-c`, which it generates from the `tandem-cuda` core, and the bounded and normal
  fills with the fill fixtures of `tandem-cuda` that `tandem-c` carries
  (`tools/gen_derived.py` converts both), the bounded ones from the starts 0, 1 and 12345 bits.
  It checks the bounded fills against their definition, at every length that cuts a SIMD block,
  and cut into chunks against the whole fill with rejections. It checks the normal fills against
  the flattened pairs, the series against libm over 2^18 pairs and the edges of the range, the
  hash of 1e6 pairs from five positions against the value of `tandem-c`, the moments of the
  normals, and that an empty fill moves nothing.
  It checks the exponentials against `tandem-c`'s `cross_exponential.h` bit for bit, the hash of
  1e6 values of each width from five positions, fills cut across the L1 block against the whole
  fill and the scalar draws, that an empty fill moves nothing, and the moments to the fourth
  order and the KS distance of 1e7 draws of each width against Exp(1).
- `tests/test_gpu.mojo` (`pixi run test-gpu`, on a GPU host) compares the GPU fills with the
  CPU fills over chunk lengths and row ranges, and with the dump. CI does not run it.

## Speed

One thread, `pixi run bench`, minimum of seven runs of 2^24 elements after a warm-up, in GiB/s.

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

GPU fills into device memory, `pixi run bench-gpu`, 1 GiB per fill, minimum of 21 after a
half-second warm-up, GPU idle, in GiB/s.

| NVIDIA A100 40 GB PCIe | GiB/s |
|---|---|
| `fill_u32_gpu` | 1186 |
| `fill_u64_gpu` | 1208 |
| `fill_f32_gpu` | 1160 |
| `fill_f64_gpu` | 1188 |

The exponential fills against the only exponential the Mojo standard library offers, `-log(1 - u)`
over `random_float64` (it has no exponential sampler), `pixi run bench-exponential`, one thread,
2^22 elements, minimum of five, in GiB/s.

| Apple M4 | `f64` | `f32` |
|---|---|---|
| `fill_exponential_*` | 5.34 | 5.95 |
| `-log(1 - random_float64())` | 0.31 | 0.15 |

The CPU fill converts floats in the same pass that stores the row. The 32-bit low word of each
product is a plain vector multiply, and only the high word is a widening one: taking both from
one 64-bit product made LLVM emit two widening multiplies. The bounded and normal fills do
extra arithmetic per draw, so they run below the plain rate: the normals are limited by the
vector pipes, not by memory. The GPU kernel stores each block from registers and has no
shared-memory tile.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
