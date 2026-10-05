# Tests

```sh
pixi run test         # CPU tests
pixi run test-gpu     # on a host with a supported GPU
```

## Suite

- The specification vectors in `tests/vectors_data.mojo`, generated from the spec's `vectors.json`.
- Long fills, scalar draws and random access against the dumps in `tests/data`, copied from
  `tandem-c/tests/data`.
- Fills against the scalar draws at offsets and lengths that cut rows and chunks.
- Bounded integers, normals and exponentials against the tandem-c cross fixtures and hashes.
- GPU fills against the CPU fills.

### Files

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
  of `tandem-c`, and the bounded and normal fills with the fill fixtures of `tandem-cuda` that
  `tandem-c` carries (`tools/gen_derived.py` converts both), the bounded ones from the starts
  0, 1 and 12345 bits. The `f64` normal rows start unaligned and hold a wedge accept, a wedge
  reject and a tail, for fills and scalar draws, bit for bit. It checks the bounded fills
  against their definition, at every length that cuts a SIMD block, and cut into chunks against
  the whole fill with rejections. It checks the `f64` normal fills against the scalar draws and
  cut into chunks against the whole fill, the `f32` normal fills against the flattened pairs,
  the logarithm and the `f32` series against libm over 2^18 values and the edges of the range,
  and the moments of the normals. It compares FNV-1a hashes with `tandem-c`'s: 1e6 `f64`
  normals from five positions, 2e5 `f64` normals at two positions of the spec's Python
  reference with their end positions, and 2e6 - 1 `f32` normals from five positions. It checks
  that an empty bounded or `f32` fill moves nothing and an empty `f64` fill aligns to 64. It checks the exponentials against
  `tandem-c`'s `cross_exponential.h` bit for bit, the hash of 1e6 values of each width from
  five positions, fills cut across the L1 block against the whole fill and the scalar draws,
  that an empty fill moves nothing, and the moments to the fourth order and the KS distance of
  1e7 draws of each width against Exp(1).
- `tests/test_gpu.mojo` compares the GPU fills with the CPU fills over chunk lengths and row
  ranges, and with the dump.

## Fixtures

`tests/vectors_data.mojo` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`. The dumps in `tests/data` are copies of `tandem-c/tests/data`.
`tools/gen_derived.py` converts the cross-check values of `tandem-c` and the fill fixtures of
`tandem-cuda` that `tandem-c` carries, from `tandem-c` 684e273. `tools/gen_zig_tables.py` writes
`zig_tables.mojo` from the spec's `tables/normal_f64_zig1024.json`. CI checks all of them.
`mojo run -I . tools/dump_normals.mojo out.bin` writes the bytes of `tandem-c`'s
`tools/dump_normals.c`, SHA-256
`700ec4d2f4d6b82aaa56c6eff18a4e5919585fdbd093988773383d580ea610d1`.

## CI

- CI runs `pixi run test` on Ubuntu. CI does not run the GPU tests.
- One job checks that the vector data is current, and one that the derived data and dumps are.
