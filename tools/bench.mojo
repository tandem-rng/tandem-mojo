# Throughput of the CPU fills, one thread: 2^24 elements, 0.5 s warm-up, minimum of seven, each
# next to a baseline from the Mojo standard library. Also the cost of a scalar draw in a chain.
# tools/bench_gpu.mojo times the GPU fills.
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.random import rand, randn, random_float64
from std.random.philox import NormalRandom, Random
from std.time import perf_counter_ns

from tandem import Tandem, seed

comptime N = 1 << 24


def gibs(bytes: Int, seconds: Float64) -> Float64:
    return Float64(bytes) / seconds / 1073741824.0


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


def tandem[which: Int](mut g: Tandem, b: Buffers) raises:
    g.set_position(0)
    comptime if which == 0:
        g.fill_u32(b.u32, N)
    elif which == 1:
        g.fill_u64(b.u64, N)
    elif which == 2:
        g.fill_f32(b.f32, N)
    elif which == 3:
        g.fill_f64(b.f64, N)
    elif which == 4:
        g.fill_below_u32(b.u32, N, 1000)
    elif which == 5:
        g.fill_below_u64(b.u64, N, 1000)
    elif which == 6:
        g.fill_normal_f32(b.f32, N)
    else:
        g.fill_normal_f64(b.f64, N)


def baseline[which: Int](b: Buffers):
    """Philox4x32-10 of std.random.philox for the plain fills, std.random for the rest."""
    var r = Random(seed=42)
    comptime if which == 0 or which == 1:
        for i in range(0, N if which == 0 else 2 * N, 4):
            b.u32.unsafe_offset(i).unsafe_store(r.step())
    elif which == 2:
        for i in range(0, N, 4):
            b.f32.unsafe_offset(i).unsafe_store(r.step_uniform())
    elif which == 3:
        for i in range(0, N, 2):
            var x = bitcast[DType.uint64, 2](r.step())
            b.f64.unsafe_offset(i).unsafe_store((x >> 11).cast[DType.float64]() * 1.1102230246251565e-16)
    elif which == 4:
        rand[DType.uint32](b.u32, N, min=0, max=999)
    elif which == 5:
        rand[DType.uint64](b.u64, N, min=0, max=999)
    elif which == 6:
        var nr = NormalRandom(seed=42)
        for i in range(0, N, 8):
            b.f32.unsafe_offset(i).unsafe_store(nr.step_normal())
    else:
        randn[DType.float64](b.f64, N)


def best[which: Int, ours: Bool](mut g: Tandem, b: Buffers) raises -> Float64:
    var warm = perf_counter_ns()
    while perf_counter_ns() - warm < 500_000_000:
        comptime if ours:
            tandem[which](g, b)
        else:
            baseline[which](b)
    var t = Float64.MAX
    for _ in range(7):
        var t0 = perf_counter_ns()
        comptime if ours:
            tandem[which](g, b)
        else:
            baseline[which](b)
        t = min(t, Float64(perf_counter_ns() - t0) * 1e-9)
    return t


def row[which: Int](name: String, bytes: Int, mut g: Tandem, b: Buffers) raises:
    var ours = gibs(bytes * N, best[which, True](g, b))
    var theirs = gibs(bytes * N, best[which, False](g, b))
    print("cpu", name, "2^24 elements", ours, "GiB/s, baseline", theirs, "GiB/s")


def chain[ours: Bool](n: Int) raises -> Float64:
    """The best of eight timed chains of n scalar f64 draws on a fresh generator."""
    var r = Tandem.from_key(seed(42), 0)
    var sink = Float64(0)
    var t = Float64.MAX
    for _ in range(8):
        r.set_position(0)
        var t0 = perf_counter_ns()
        for _ in range(n):
            comptime if ours:
                sink += r.next_f64()
            else:
                sink += random_float64()
        t = min(t, Float64(perf_counter_ns() - t0) * 1e-9)
    if sink < 0:
        print(sink)
    return t


def main() raises:
    var b = Buffers(N)
    var g = Tandem.from_key(seed(42), 0)
    row[0]("fill_u32", 4, g, b)
    row[1]("fill_u64", 8, g, b)
    row[2]("fill_f32", 4, g, b)
    row[3]("fill_f64", 8, g, b)
    row[4]("fill_below_u32", 4, g, b)
    row[5]("fill_below_u64", 8, g, b)
    row[6]("fill_normal_f32", 4, g, b)
    row[7]("fill_normal_f64", 8, g, b)

    # Scalar chains: Tandem's next_f64 against std.random.random_float64.
    var ours = chain[True](N)
    var theirs = chain[False](N)
    print("cpu next_f64 chain", gibs(8 * N, ours), "GiB/s, baseline", gibs(8 * N, theirs), "GiB/s")
