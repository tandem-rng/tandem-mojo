# The GPU fills against the reference u32 stream dump and against the CPU fills. Run on a
# GPU host: mojo run -I . tests/test_gpu.mojo
from std.testing import assert_equal, assert_true
from std.memory.alloc import unsafe_alloc
from std.sys import bit_width_of
from max.gpu.host import DeviceContext

from tandem import KIND_F32, KIND_F64, KIND_INT, Tandem, fill_gpu, fill_u32_gpu

comptime KEY = SIMD[DType.uint32, 4](1, 2, 3, 4)


def test_u32_dump(ctx: DeviceContext) raises:
    var want: List[UInt8]
    with open("tests/data/k1234_K32_u32.bin", "r") as file:
        want = file.read_bytes()
    var words = want.unsafe_ptr().unsafe_bitcast[UInt32]()
    var n = 65536
    for first_row in [0, 1, 31, 32, 33, 100]:
        var nrows = n // 32 - first_row
        var dev = ctx.enqueue_create_buffer[DType.uint32](nrows * 32)
        var host = ctx.enqueue_create_host_buffer[DType.uint32](nrows * 32)
        fill_u32_gpu(ctx, KEY, UInt64(first_row), UInt64(nrows), 32, dev.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=host, src_buf=dev)
        ctx.synchronize()
        for i in range(nrows * 32):
            if host[i] != words.unsafe_offset(first_row * 32 + i).unsafe_load():
                assert_true(False, String("gpu fill from row ", first_row, " differs at word ", i))


def check[kind: Int, T: DType](ctx: DeviceContext, label: String) raises:
    """Rows from the GPU equal the same bits from the CPU fill, at chunk lengths and row offsets that cut groups."""
    comptime per_row = 1024 // bit_width_of[T]()
    for K in [32, 8, 1]:
        for first_row in [0, 1, 7, 8, 31, 33, 100]:
            for nrows in [1, 7, 64, 300]:
                var n = nrows * per_row
                var dev = ctx.enqueue_create_buffer[T](n)
                var host = ctx.enqueue_create_host_buffer[T](n)
                fill_gpu[kind, T](ctx, KEY, UInt64(first_row), UInt64(nrows), UInt32(K), dev.unsafe_ptr())
                ctx.enqueue_copy(dst_buf=host, src_buf=dev)
                ctx.synchronize()
                var cpu = Tandem.from_key(KEY, UInt64(first_row) * 1024, UInt32(K))
                var want = unsafe_alloc[Scalar[T]](n)
                comptime if kind == KIND_INT and T == DType.uint32:
                    cpu.fill_u32(want.unsafe_bitcast[UInt32](), n)
                elif kind == KIND_INT:
                    cpu.fill_u64(want.unsafe_bitcast[UInt64](), n)
                elif kind == KIND_F32:
                    cpu.fill_f32(want.unsafe_bitcast[Float32](), n)
                else:
                    cpu.fill_f64(want.unsafe_bitcast[Float64](), n)
                for i in range(n):
                    if host[i] != want.unsafe_offset(i).unsafe_load():
                        assert_true(False, String(label, " K=", K, " first_row=", first_row, " nrows=", nrows, " differs at ", i))
                want.unsafe_free()


def main() raises:
    var ctx = DeviceContext()
    test_u32_dump(ctx)
    check[KIND_INT, DType.uint32](ctx, "u32")
    check[KIND_INT, DType.uint64](ctx, "u64")
    check[KIND_F32, DType.float32](ctx, "f32")
    check[KIND_F64, DType.float64](ctx, "f64")
    print("mojo gpu: ok on", ctx.name())
