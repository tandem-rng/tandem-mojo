# The GPU fill against the reference u32 stream dump. Run on a GPU host:
# mojo run -I . tests/test_gpu.mojo
from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext

from tandem import fill_u32_gpu


def main() raises:
    var want: List[UInt8]
    with open("tests/data/k1234_K32_u32.bin", "r") as file:
        want = file.read_bytes()
    var words = want.unsafe_ptr().unsafe_bitcast[UInt32]()
    var n = 65536
    var ctx = DeviceContext()
    for first_row in [0, 1, 31, 32, 33, 100]:
        var nrows = n // 32 - first_row
        var dev = ctx.enqueue_create_buffer[DType.uint32](nrows * 32)
        var host = ctx.enqueue_create_host_buffer[DType.uint32](nrows * 32)
        fill_u32_gpu(ctx, SIMD[DType.uint32, 4](1, 2, 3, 4), UInt64(first_row), UInt64(nrows), 32, dev.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=host, src_buf=dev)
        ctx.synchronize()
        for i in range(nrows * 32):
            if host[i] != words.unsafe_offset(first_row * 32 + i).unsafe_load():
                assert_true(False, String("gpu fill from row ", first_row, " differs at word ", i))
    print("mojo gpu: ok on", ctx.name())
