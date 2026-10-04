# API

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

## Design

- A `Tandem` is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus a
  cache of the current 1024-bit row.
- Scalar draws include signed integers. Seed whitening and a raw-key constructor are
  available. Random access is `at_*`.
- GPU fills into device memory for `u32`, `u64`, `f32` and `f64`: one thread per chunk, each
  thread stores its blocks from registers.
- Bounded integers and standard normals are not in the specification. They follow the shared
  device core in `tandem-cuda`, so every port returns the same integers. A bound of 0 returns 0
  after one draw. `fill_below_*` takes draw `i` of the plain fill for element `i` and consumes
  exactly one draw per element. A rejected draw retries on `sub(purpose).split(g)` of the
  key, with `g` the global draw index, the aligned start position over the draw width plus `i`,
  so a fill cut into chunks equals the whole fill. The bounded APIs are typed by draw width,
  `below_u32` and `below_u64`, so the draw width never depends on a result type. They exist on
  the CPU only.
- A normal step uses two uniforms and returns the cos half then the sin half. A scalar normal
  is the cos half, and a normal fill is the flattened pairs, so an odd count consumes both
  uniforms of its last pair. An empty bounded or normal fill moves nothing.
- The normals run on SIMD lanes with the algorithm and coefficients of `tandem-c` and an
  explicit fused multiply-add for every multiply-add, so they are byte identical to `tandem-c`
  (same SHA-256 from `tools/dump_normals.mojo` and `tools/dump_normals.c`, checked on arm64).
  They agree with libm to about 1e-15 in `f64`. The `f32` normal runs in `f32`.
- The bounded fills do the multiply-high and the compare on SIMD lanes (u32) or on the integer
  pipes beside the row generator (u64), and only a rejection takes a scalar fixup.
- Exponentials are `-log(1 - u)` of one uniform per element. Element `i` of a fill is the
  scalar draw `i`, and an empty fill moves nothing. They reuse the polynomial logarithm of the
  normals with an explicit fused multiply-add for every multiply-add, in `f64` from `f64` draws
  and in `f32` from `f32` draws, so the bytes equal `tandem-c`'s (FNV-1a `47f8f98297d94ee2`
  over 1e6 values of each width from five positions).

## Positions and fills

`Tandem(seed)` and `Tandem.from_key` raise when `K` is not a power of two in 1 to 65536.
Every draw aligns the position to the width of its type first, and every fill returns the
generator where the same number of scalar draws would leave it. Fills take a pointer and a
count. Complex fills take the number of complex values and write interleaved `(re, im)`
components.

```mojo
var c = rng.next_c64()                     # (re, im), two f64 draws
var raw = Tandem.from_key(rng.key, rng.position(), rng.k)
```

`fill_u32_gpu`, `fill_u64_gpu`, `fill_f32_gpu` and `fill_f64_gpu` write whole rows,
`[first_row, first_row + nrows)`, of 32, 16, 32 and 16 values, so a GPU fill and a CPU fill
from bit position `1024 * first_row` agree. The GPU fills take a key and a row range, not a
`Tandem`, and do not move a position.
