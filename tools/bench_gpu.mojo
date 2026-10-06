# Throughput of the GPU fills: 1 GiB into device memory, 0.5 s warm-up, minimum of 21. Then
# cuRAND Philox4x32-10 for each output type, by the same method, through its host API in the
# context of the DeviceContext. cuRAND has no 64-bit integer output for Philox, so the u64 row
# takes curandGenerate into the same bytes. libcurand and libcudart come from the pixi environment.
from std.ffi import OwnedDLHandle, c_int
from std.os import getenv
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


def curand_fill(cr: OwnedDLHandle, rt: OwnedDLHandle, gen: Int, dst: Int, n: Int, call: String) raises:
    var status: c_int
    if call == "curandGenerate":
        status = cr.call["curandGenerate", c_int](gen, dst, n)
    elif call == "curandGenerateUniform":
        status = cr.call["curandGenerateUniform", c_int](gen, dst, n)
    else:
        status = cr.call["curandGenerateUniformDouble", c_int](gen, dst, n)
    if status != 0 or rt.call["cudaDeviceSynchronize", c_int]() != 0:
        raise Error("cuRAND failed")


def bench_curand[T: DType](ctx: DeviceContext, name: String, call: String) raises:
    var prefix = getenv("CONDA_PREFIX")
    var cr = OwnedDLHandle(prefix + "/lib/libcurand.so.10")
    var rt = OwnedDLHandle(prefix + "/lib/libcudart.so.12")
    var gen: Int = 0
    _ = cr.call["curandCreateGenerator", c_int](Pointer(to=gen), c_int(161))
    _ = cr.call["curandSetPseudoRandomGeneratorSeed", c_int](gen, UInt64(42))
    var m = (1 << 30) // size_of[Scalar[T]]()
    var dev = ctx.enqueue_create_buffer[T](m)
    ctx.synchronize()
    var dst = Int(dev.unsafe_ptr())
    # curandGenerate writes 32-bit words, so a u64 buffer takes twice its element count.
    var n = (1 << 30) // 4 if call == "curandGenerate" else m
    var warm = perf_counter_ns()
    while perf_counter_ns() - warm < 500_000_000:
        curand_fill(cr, rt, gen, dst, n, call)
    var best = Float64.MAX
    for _ in range(21):
        var t0 = perf_counter_ns()
        curand_fill(cr, rt, gen, dst, n, call)
        best = min(best, Float64(perf_counter_ns() - t0) * 1e-9)
    _ = cr.call["curandDestroyGenerator", c_int](gen)
    _ = dev^
    print(name, "1 GiB", Float64(1 << 30) / best / 1073741824.0, "GiB/s with", call)


def main() raises:
    var key = seed(42)
    var ctx = DeviceContext()
    bench[KIND_INT, DType.uint32](ctx, "gpu fill_u32", key)
    bench[KIND_INT, DType.uint64](ctx, "gpu fill_u64", key)
    bench[KIND_F32, DType.float32](ctx, "gpu fill_f32", key)
    bench[KIND_F64, DType.float64](ctx, "gpu fill_f64", key)
    bench_curand[DType.uint32](ctx, "cuRAND u32", "curandGenerate")
    bench_curand[DType.uint64](ctx, "cuRAND u64", "curandGenerate")
    bench_curand[DType.float32](ctx, "cuRAND f32", "curandGenerateUniform")
    bench_curand[DType.float64](ctx, "cuRAND f64", "curandGenerateUniformDouble")
