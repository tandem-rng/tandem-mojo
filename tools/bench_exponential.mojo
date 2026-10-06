# Exponential fills against the standard library's only route to an exponential, -log(1 - u)
# over its uniform: 2^22 elements, 0.2 s warm-up, minimum of five, one thread.
from std.math import log
from std.memory.alloc import unsafe_alloc
from std.random import random_float64
from std.time import perf_counter_ns

from tandem import Tandem, seed


def gibs(bytes: Int, seconds: Float64) -> Float64:
    return Float64(bytes) / seconds / 1073741824.0


def run[which: Int](mut g: Tandem, d: Pointer[Float64, MutUntrackedOrigin], f: Pointer[Float32, MutUntrackedOrigin], n: Int) raises:
    g.set_position(0)
    comptime if which == 0:
        g.fill_exponential_f64(d, n)
    elif which == 1:
        for i in range(n):
            d.unsafe_offset(i).unsafe_store(-log(1.0 - random_float64()))
    elif which == 2:
        g.fill_exponential_f32(f, n)
    else:
        for i in range(n):
            f.unsafe_offset(i).unsafe_store(-log(1.0 - random_float64().cast[DType.float32]()))


def time_best[which: Int](name: String, bytes: Int, mut g: Tandem, d: Pointer[Float64, MutUntrackedOrigin], f: Pointer[Float32, MutUntrackedOrigin], n: Int) raises:
    var warm = perf_counter_ns()
    while perf_counter_ns() - warm < 200_000_000:
        run[which](g, d, f, n)
    var best = Float64.MAX
    for _ in range(5):
        var t0 = perf_counter_ns()
        run[which](g, d, f, n)
        best = min(best, Float64(perf_counter_ns() - t0) * 1e-9)
    print(name, "2^22 elements", gibs(bytes, best), "GiB/s")


def main() raises:
    var n = 1 << 22
    var d = unsafe_alloc[Float64](n)
    var f = unsafe_alloc[Float32](n)
    var g = Tandem.from_key(seed(42), 0)
    time_best[0]("tandem fill_exponential_f64", 8 * n, g, d, f, n)
    time_best[1]("std.random -log(1 - u) f64", 8 * n, g, d, f, n)
    time_best[2]("tandem fill_exponential_f32", 4 * n, g, d, f, n)
    time_best[3]("std.random -log(1 - u) f32", 4 * n, g, d, f, n)
    d.unsafe_free()
    f.unsafe_free()
