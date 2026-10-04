# Throughput of the CPU fills, one thread: 2^24 elements, 0.5 s warm-up, minimum of seven.
# Also the cost of a scalar draw in a chain. tools/bench_gpu.mojo times the GPU fills.
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.time import perf_counter_ns

from tandem import KIND_F32, KIND_F64, KIND_INT, Tandem, seed


def gibs(bytes: Int, seconds: Float64) -> Float64:
    return Float64(bytes) / seconds / 1073741824.0


def bench[kind: Int, W: Int, T: DType](name: String, key: SIMD[DType.uint32, 4]) raises:
    var n = 1 << 24
    var buf = unsafe_alloc[Scalar[T]](n)
    var g = Tandem.from_key(key, 0)
    var warm = perf_counter_ns()
    while perf_counter_ns() - warm < 500_000_000:
        g.set_position(0)
        g.fill[kind, W, T](buf, n)
    var best = Float64.MAX
    for _ in range(7):
        g.set_position(0)
        var t0 = perf_counter_ns()
        g.fill[kind, W, T](buf, n)
        best = min(best, Float64(perf_counter_ns() - t0) * 1e-9)
    print(name, "2^24 elements", gibs(n * size_of[Scalar[T]](), best), "GiB/s")
    buf.unsafe_free()


struct Buffers:
    var u32: Pointer[UInt32, MutUntrackedOrigin]
    var u64: Pointer[UInt64, MutUntrackedOrigin]
    var f32: Pointer[Float32, MutUntrackedOrigin]
    var f64: Pointer[Float64, MutUntrackedOrigin]

    def __init__(out self, n: Int):
        self.u32 = unsafe_alloc[UInt32](n)
        self.u64 = unsafe_alloc[UInt64](n)
        self.f32 = unsafe_alloc[Float32](n)
        self.f64 = unsafe_alloc[Float64](n)


def run[which: Int](mut g: Tandem, b: Buffers, n: Int):
    comptime if which == 0:
        g.fill_below_u32(b.u32, n, 1000)
    elif which == 1:
        g.fill_below_u64(b.u64, n, 1000)
    elif which == 2:
        g.fill_normal_f32(b.f32, n)
    else:
        g.fill_normal_f64(b.f64, n)


def time_best[which: Int](name: String, bytes: Int, mut g: Tandem, b: Buffers, n: Int):
    var warm = perf_counter_ns()
    while perf_counter_ns() - warm < 500_000_000:
        g.set_position(0)
        run[which](g, b, n)
    var best = Float64.MAX
    for _ in range(7):
        g.set_position(0)
        var t0 = perf_counter_ns()
        run[which](g, b, n)
        best = min(best, Float64(perf_counter_ns() - t0) * 1e-9)
    print(name, "2^24 elements", gibs(bytes, best), "GiB/s")


def main() raises:
    var key = seed(42)
    bench[KIND_INT, 32, DType.uint32]("cpu fill_u32", key)
    bench[KIND_INT, 64, DType.uint64]("cpu fill_u64", key)
    bench[KIND_F32, 32, DType.float32]("cpu fill_f32", key)
    bench[KIND_F64, 64, DType.float64]("cpu fill_f64", key)

    var n = 1 << 24
    var bufs = Buffers(n)
    var g = Tandem.from_key(key, 0)
    time_best[0]("cpu fill_below_u32", 4 * n, g, bufs, n)
    time_best[1]("cpu fill_below_u64", 8 * n, g, bufs, n)
    time_best[2]("cpu fill_normal_f32", 4 * n, g, bufs, n)
    time_best[3]("cpu fill_normal_f64", 8 * n, g, bufs, n)

    var sink = Float64(0)
    var best = Float64.MAX
    for _ in range(8):
        g.set_position(0)
        var t0 = perf_counter_ns()
        for _ in range(n):
            sink += g.next_f64()
        best = min(best, Float64(perf_counter_ns() - t0) / Float64(n))
    print("cpu next_f64 chain", best, "ns per draw", sink > 0)
