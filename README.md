<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .mojo"></p>

# tandem-mojo

[![CI](https://github.com/tandem-rng/tandem-mojo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-mojo/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tandem--rng.github.io-7fb3ee.svg)](https://tandem-rng.github.io/tandem-mojo/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Mojo implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator, in one file, `tandem.mojo`. It produces the stream the
specification defines, bit for bit, with SIMD fills on CPUs and MAX fills on GPUs.

## Install

Install with pixi.

```sh
pixi install
pixi run test
```

## Quick start

```mojo
from std.memory.alloc import unsafe_alloc
from tandem import Tandem

def main() raises:
    var rng = Tandem(42)                       # 128-bit seed, default K
    var words = unsafe_alloc[UInt32](1 << 20)
    rng.fill_u32(words, 1 << 20)
    var worker = rng.split(7)                  # by index, from the key alone
    var z = worker.normal_f64()                # ziggurat, byte identical to tandem-c
```

## Requirements

Needs Mojo 1.1 and MAX 26.6 from the Modular conda channel.

## Documentation

See [API](docs/api.md) for every draw and the GPU fills, and [tests](docs/tests.md) and
[speed](docs/speed.md) for the rest.

Portions of the code were generated with the assistance of LLMs.

[Docs](https://tandem-rng.github.io/tandem-mojo/) · [Specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) · [Website](https://tandem-rng.github.io/) · [Issues](https://github.com/tandem-rng/tandem-mojo/issues) · [License](LICENSE)
