# Throughput of the CPU fill. tools/bench_gpu.mojo times the GPU fill.
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns

from tandem import fill_u32, seed


def gibs(bytes: Int, seconds: Float64) -> Float64:
    return Float64(bytes) / seconds / 1073741824.0


def main() raises:
    var key = seed(42)
    var n = 1 << 24
    var buf = unsafe_alloc[UInt32](n)
    var warm = perf_counter_ns()
    while perf_counter_ns() - warm < 500_000_000:
        _ = fill_u32(key, 0, 32, buf, n)
    var best = Float64.MAX
    for _ in range(7):
        var t0 = perf_counter_ns()
        _ = fill_u32(key, 0, 32, buf, n)
        best = min(best, Float64(perf_counter_ns() - t0) * 1e-9)
    print("cpu fill_u32  2^24 words ", gibs(4 * n, best), "GiB/s")
    buf.unsafe_free()
