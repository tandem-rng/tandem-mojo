# API

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
    var z = rng.normal_f64()                   # ziggurat from one u64 draw
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

## Reference

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
- `normal_f64`, `normal_f32`, `normal2_f32` and `fill_normal_*`. `f64` normals are the ziggurat
  of Appendix A, `f32` normals Box-Muller pairs. They are byte identical to tandem-c. An empty
  `f64` fill aligns the position to 64 bits.
- `exponential_f64`, `exponential_f32`, `fill_exponential_f64`, `fill_exponential_f32`. They
  are byte identical to tandem-c.
- GPU fills `fill_u32_gpu`, `fill_u64_gpu`, `fill_f32_gpu`, `fill_f64_gpu`. They take a key
  and a row range, write whole rows and agree with the CPU fill from bit position
  `1024 * first_row`.

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

## Parallel use

Element `i` of a fill is draw `i`, so any decomposition reproduces a serial run.
See [Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative).
