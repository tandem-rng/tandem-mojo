# Design

## Fills

- A `Tandem` is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus a
  cache of the current 1024-bit row.
- Scalar draws include signed integers. Seed whitening and a raw-key constructor are
  available. Random access is `at_*`.
- GPU fills into device memory for `u32`, `u64`, `f32` and `f64`: one thread per chunk, each
  thread stores its blocks from registers.

## Bounded integers

- Bounded integers and standard normals are not in the specification. They follow the shared
  device core in `tandem-cuda`, so every port returns the same integers. A bound of 0 returns 0
  after one draw. `fill_below_*` takes draw `i` of the plain fill for element `i` and consumes
  exactly one draw per element. A rejected draw retries on `sub(purpose).split(g)` of the
  key, with `g` the global draw index, the aligned start position over the draw width plus `i`,
  so a fill cut into chunks equals the whole fill. The bounded APIs are typed by draw width,
  `below_u32` and `below_u64`, so the draw width never depends on a result type. They exist on
  the CPU only.
- The bounded fills do the multiply-high and the compare on SIMD lanes (u32) or on the integer
  pipes beside the row generator (u64), and only a rejection takes a scalar fixup.

## Normals

- A normal step uses two uniforms and returns the cos half then the sin half. A scalar normal
  is the cos half, and a normal fill is the flattened pairs, so an odd count consumes both
  uniforms of its last pair. An empty bounded or normal fill moves nothing.
- The normals run on SIMD lanes with the algorithm and coefficients of `tandem-c` and an
  explicit fused multiply-add for every multiply-add, so they are byte identical to `tandem-c`
  (same SHA-256 from `tools/dump_normals.mojo` and `tools/dump_normals.c`, checked on arm64).
  They agree with libm to about 1e-15 in `f64`. The `f32` normal runs in `f32`.

## Exponentials

- Exponentials are `-log(1 - u)` of one uniform per element. Element `i` of a fill is the
  scalar draw `i`, and an empty fill moves nothing. They reuse the polynomial logarithm of the
  normals with an explicit fused multiply-add for every multiply-add, in `f64` from `f64` draws
  and in `f32` from `f32` draws, so the bytes equal `tandem-c`'s (FNV-1a `47f8f98297d94ee2`
  over 1e6 values of each width from five positions).
