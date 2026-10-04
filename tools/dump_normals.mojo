# Writes the normal fills of tandem-c's tools/dump_normals.c as raw bytes to the file named by
# the first argument, for comparing ports by SHA-256:
#   mojo run -I . tools/dump_normals.mojo out.bin && shasum -a 256 out.bin
from std.memory.alloc import unsafe_alloc
from std.sys import argv

from tandem import Tandem


def main() raises:
    comptime PAIRS = 1000000
    var starts: List[UInt64] = [0, 1, 77, 12345, 1 << 30]
    var d = unsafe_alloc[Float64](2 * PAIRS)
    var f = unsafe_alloc[Float32](2 * PAIRS)
    with open(argv()[1], "w") as out:
        for s in starts:
            var g = Tandem(UInt128(2026) | (UInt128(7) << 64))
            g.set_position(s)
            g.fill_normal_f64(d, 2 * PAIRS - 1)
            out.write_bytes(Span(unsafe_ptr=d.unsafe_bitcast[UInt8](), length=(2 * PAIRS - 1) * 8))
            g.fill_normal_f32(f, 2 * PAIRS - 1)
            out.write_bytes(Span(unsafe_ptr=f.unsafe_bitcast[UInt8](), length=(2 * PAIRS - 1) * 4))
