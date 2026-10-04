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

## Fixtures

`tests/vectors_data.mojo` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`. The dumps in `tests/data` are copies of `tandem-c/tests/data`.
`tools/gen_derived.py` converts the cross-check values of `tandem-c` and the fill fixtures of
`tandem-cuda` that `tandem-c` carries.

## CI

- CI runs `pixi run test` on Ubuntu. CI does not run the GPU tests.
- One job checks that the vector data is current, and one that the derived data and dumps are.
