# tandem-mojo notes

Detail moved out of the README. Sections follow the README headings.

## What it provides

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

## Use

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

## Tests

- `tests/test_vectors.mojo` checks every vector of the specification.
  `tests/vectors_data.mojo` is generated from the spec repository's `vectors.json` by
  `tools/gen_vectors.py`.
- `tests/test_dumps.mojo` compares long fills, scalar draws and random access with the
  reference stream dumps in `tests/data`, complex fills included. The dumps are copies of
  `tandem-c/tests/data`.
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
  normals, and that an empty fill moves nothing. It checks the exponentials against
  `tandem-c`'s `cross_exponential.h` bit for bit, the hash of 1e6 values of each width from
  five positions, fills cut across the L1 block against the whole fill and the scalar draws,
  that an empty fill moves nothing, and the moments to the fourth order and the KS distance of
  1e7 draws of each width against Exp(1).
- `tests/test_gpu.mojo` compares the GPU fills with the CPU fills over chunk lengths and row
  ranges, and with the dump.

## Speed

GPU fills: 1 GiB per fill, minimum of 21 after a half-second warm-up, GPU idle.

The CPU fill converts floats in the same pass that stores the row. The 32-bit low word of each
product is a plain vector multiply, and only the high word is a widening one: taking both from
one 64-bit product made LLVM emit two widening multiplies. The bounded and normal fills do
extra arithmetic per draw, so they run below the plain rate: the normals are limited by the
vector pipes, not by memory. The GPU kernel stores each block from registers and has no
shared-memory tile.
