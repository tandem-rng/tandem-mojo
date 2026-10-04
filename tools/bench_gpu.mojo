# Throughput of the GPU fills: 1 GiB into device memory, 0.5 s warm-up, minimum of 21.
from std.sys import size_of
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from tandem import KIND_F32, KIND_F64, KIND_INT, fill_gpu, seed


def bench[kind: Int, T: DType](ctx: DeviceContext, name: String, key: SIMD[DType.uint32, 4]) raises:
    var m = (1 << 30) // size_of[Scalar[T]]()
    var dev = ctx.enqueue_create_buffer[T](m)
    var rows = UInt64(m * size_of[Scalar[T]]() // 128)
    var warm = perf_counter_ns()
    while perf_counter_ns() - warm < 500_000_000:
        fill_gpu[kind, T](ctx, key, 0, rows, 32, dev.unsafe_ptr())
        ctx.synchronize()
    var best = Float64.MAX
    for _ in range(21):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        fill_gpu[kind, T](ctx, key, 0, rows, 32, dev.unsafe_ptr())
        ctx.synchronize()
        best = min(best, Float64(perf_counter_ns() - t0) * 1e-9)
    print(name, "1 GiB", Float64(m * size_of[Scalar[T]]()) / best / 1073741824.0, "GiB/s on", ctx.name())


def main() raises:
    var key = seed(42)
    var ctx = DeviceContext()
    bench[KIND_INT, DType.uint32](ctx, "gpu fill_u32", key)
    bench[KIND_INT, DType.uint64](ctx, "gpu fill_u64", key)
    bench[KIND_F32, DType.float32](ctx, "gpu fill_f32", key)
    bench[KIND_F64, DType.float64](ctx, "gpu fill_f64", key)
