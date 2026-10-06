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
- The conformance files of the spec: bounded integers, normals, exponentials and weighted choice,
  with the stream and dump hashes.
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
- `tests/test_conformance.mojo` reads the byte copies of the spec's `conformance/*.json` in
  `tests/conformance` through `tests/conformance.mojo`, and demonstrates every item of the spec's
  `conformance/CHECKLIST.md`. Every case of the bounded, normal, exponential and weighted
  choice files runs as a fill, as scalar draws and cut at elements 1, 7, 20, 21 and n - 1, bit
  for bit with the end position, including the `f32` normals. It builds the choice tables and
  compares `S`, `cut` and `alias`, and checks the stream and dump hashes of `hashes.json` by
  SHA-256 and FNV-1a from the key. It checks the global draw index of the fallback, the width of
  the bounded draws, empty fills, the pair rule and odd `n` of the `f32` normals, rejected choice
  weights and the choice law, random access, a complex draw across a block, and the position
  bounds.
- `tests/test_derived.mojo` checks the bounded fills against their definition at every length
  that cuts a SIMD block, long `f64` normal, bounded and exponential fills cut into chunks, the
  `f32` normals against the exact formula, the logarithm and the `f32` series against libm,
  and the moments of the normals and the moments and KS distance of 1e7 exponentials.
- `tests/test_gpu.mojo` compares the GPU fills with the CPU fills over chunk lengths and row
  ranges, and with the dump.

## Fixtures

`tests/vectors_data.mojo` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`. The dumps in `tests/data` are copies of `tandem-c/tests/data`.
`tests/conformance/*.json` are byte copies of tandem-spec f420545 `conformance/*.json`.
`tools/gen_zig_tables.py` writes
`zig_tables.mojo` from the spec's `tables/normal_f64_zig1024.json`. CI checks all of them.
`mojo run -I . tools/dump_normals.mojo out.bin` writes the bytes of `tandem-c`'s
`tools/dump_normals.c`, SHA-256
`700ec4d2f4d6b82aaa56c6eff18a4e5919585fdbd093988773383d580ea610d1`.

## CI

- CI runs `pixi run test` on Ubuntu. CI does not run the GPU tests.
- One job checks that the vector data is current, and one that the dumps and the conformance files are.
