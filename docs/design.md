# Design

## Fills

- A `Tandem` is its transport form (128-bit key, 64-bit bit position, chunk length `K`) plus a
  cache of the current 1024-bit row.
- Scalar draws include signed integers. Seed whitening and a raw-key constructor are
  available. Random access is `at_*`.
- GPU fills into device memory for `u32`, `u64`, `f32` and `f64`: one thread per chunk, 32 groups
  per block staging eight steps in shared memory, so a warp stores 512 contiguous bytes. For K not
  a multiple of 8 each thread stores its blocks from registers.

## Bounded integers

- Bounded integers and standard normals follow Appendix A of the specification, which is not
  normative, so every port returns the same integers. A bound of 0 returns 0
  after one draw. `fill_below_*` takes draw `i` of the plain fill for element `i` and consumes
  exactly one draw per element. A rejected draw retries on `sub(purpose).split(g)` of the
  key, with `g` the global draw index, the aligned start position over the draw width plus `i`,
  so a fill cut into chunks equals the whole fill. The bounded APIs are typed by draw width,
  `below_u32` and `below_u64`, so the draw width never depends on a result type. They exist on
  the CPU only.
- The bounded fills do the multiply-high and the compare on SIMD lanes (u32) or on the integer
  pipes beside the row generator (u64), and only a rejection takes a scalar fixup.

## Normals

- `f64` normals are the 1024-layer ziggurat of Appendix A, one `u64` draw per element. Element
  `i` of a fill comes from draw `i` of the `u64` fill, and the scalar draw equals element 0.
  99.57 % of the draws land in an inner rectangle and cost a table lookup and a multiply. A
  draw that misses continues on its own fallback generator, `sub(0x4e524d3634).split(g)` of the
  key at position 0, with `g` the global draw index. So a fill cut anywhere equals the whole
  fill, and a fill consumes exactly `n` draws. An empty `f64` fill aligns the position to 64.
- `zig_tables.mojo` is generated from the spec's `tables/normal_f64_zig1024.json` by
  `tools/gen_zig_tables.py`, which checks the file's SHA-256, and CI checks it is current. It
  holds each `Float64` as its bits. The tables are read through a pointer to a global constant,
  because indexing the constant itself copies the whole table per lookup.
- A fill writes every candidate in one pass over 512 draws and lists the misses. The misses
  queue across passes, and their fallbacks are seeded eight at a time on SIMD lanes, as
  `tandem-c` does. Each fallback then computes one block per two draws.
- An `f32` normal step uses two uniforms and returns the cos half then the sin half. A scalar
  `f32` normal is the cos half, and an `f32` fill is the flattened pairs, so an odd count
  consumes both uniforms of its last pair. An empty bounded or `f32` normal fill moves nothing.
- The logarithm and the `f32` normals run on SIMD lanes with the algorithm and coefficients of
  `tandem-c` and an explicit fused multiply-add for every multiply-add. So both kinds of normal
  are byte identical to `tandem-c`: `tools/dump_normals.mojo` and `tools/dump_normals.c` give
  the same SHA-256 on arm64, and the hash tests pass on Linux x86_64. The `f64` logarithm agrees
  with libm to 2e-15. The `f32` normal runs in `f32`.

## Exponentials

- Exponentials are `-log(1 - u)` of one uniform per element. Element `i` of a fill is the
  scalar draw `i`, and an empty fill moves nothing. They reuse the polynomial logarithm of the
  normals with an explicit fused multiply-add for every multiply-add, in `f64` from `f64` draws
  and in `f32` from `f32` draws, so the bytes equal `tandem-c`'s (FNV-1a `47f8f98297d94ee2`
  over 1e6 values of each width from five positions).
