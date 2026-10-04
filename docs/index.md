# tandem-mojo documentation

- [API](api.md): the generator, draws, fills, bounded integers, normals, exponentials and GPU fills.
- [Tests](tests.md): what the suite checks.
- [Speed](speed.md): Apple M4 and A100 figures.

## Install

```sh
pixi install          # Mojo 1.1 and MAX 26.6 from the Modular conda channel
```

The whole port is one file, `tandem.mojo`. The GPU fills need the `max` package. On an NVIDIA
driver older than 580, set `MODULAR_NVPTX_COMPILER_PATH` to a CUDA 12.8 `ptxas`. On macOS, the
Metal toolchain needs a full Xcode install.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
