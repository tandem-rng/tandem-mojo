# Throughput of the GPU fill: 2^28 words into device memory, 0.5 s warm-up, minimum of 21.
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from tandem import fill_u32_gpu, seed


def main() raises:
    var key = seed(42)
    var ctx = DeviceContext()
    var m = 1 << 28
    var dev = ctx.enqueue_create_buffer[DType.uint32](m)
    var rows = UInt64(m // 32)
    var warm = perf_counter_ns()
    while perf_counter_ns() - warm < 500_000_000:
        fill_u32_gpu(ctx, key, 0, rows, 32, dev.unsafe_ptr())
        ctx.synchronize()
    var best = Float64.MAX
    for _ in range(21):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        fill_u32_gpu(ctx, key, 0, rows, 32, dev.unsafe_ptr())
        ctx.synchronize()
        best = min(best, Float64(perf_counter_ns() - t0) * 1e-9)
    print("gpu fill_u32  2^28 words ", Float64(4 * m) / best / 1073741824.0, "GiB/s on", ctx.name())
