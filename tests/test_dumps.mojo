# Long streams against the reference dumps of the Julia implementation, copied byte for byte
# from tandem-c/tests/data. Each dump is compared with a fill, with scalar draws, and at
# sampled indices with random access.
# Run: mojo run -I . -I tests tests/test_dumps.mojo
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.testing import assert_equal, assert_true

from tandem import (
    KIND_BOOL,
    KIND_CHAR,
    KIND_F16,
    KIND_F32,
    KIND_F64,
    KIND_INT,
    Tandem,
    fill_u32,
    seed,
)
from test_fills import draw

comptime KEY = SIMD[DType.uint32, 4](1, 2, 3, 4)


def slurp(name: String) raises -> List[UInt8]:
    with open("tests/data/" + name, "r") as file:
        return file.read_bytes()


def check[kind: Int, W: Int, T: DType](name: String, key: SIMD[DType.uint32, 4], k: UInt32) raises:
    var want = slurp(name)
    var n = len(want) // size_of[Scalar[T]]()
    var words = want.unsafe_ptr().unsafe_bitcast[Scalar[T]]()
    var a = Tandem.from_key(key, 0, k)
    var b = a.copy()
    var got = unsafe_alloc[Scalar[T]](n)
    a.fill[kind, W, T](got, n)
    for i in range(n):
        if got.unsafe_offset(i).unsafe_load() != words.unsafe_offset(i).unsafe_load():
            assert_true(False, String(name, ": fill differs at element ", i))
        if draw[kind, W, T](b) != words.unsafe_offset(i).unsafe_load():
            assert_true(False, String(name, ": scalar draw differs at element ", i))
    assert_true(a == b, String(name, ": fill and draws end at different positions"))
    got.unsafe_free()


def test_integer_dumps() raises:
    check[KIND_INT, 32, DType.uint32]("k1234_K32_u32.bin", KEY, 32)
    check[KIND_INT, 32, DType.uint32]("k1234_K8_u32.bin", KEY, 8)
    check[KIND_INT, 64, DType.uint64]("k1234_K32_u64.bin", KEY, 32)
    check[KIND_INT, 8, DType.uint8]("seed42_K32_u8.bin", seed(42), 32)


def test_mapped_dumps() raises:
    check[KIND_F64, 64, DType.float64]("seed42_K32_f64.bin", seed(42), 32)
    check[KIND_F32, 32, DType.float32]("seed42_K32_f32.bin", seed(42), 32)
    check[KIND_F16, 16, DType.uint16]("seed42_K32_f16bits.bin", seed(42), 32)
    check[KIND_CHAR, 64, DType.uint32]("seed42_K32_char.bin", seed(42), 32)
    check[KIND_BOOL, 1, DType.uint8]("seed42_K32_bool.bin", seed(42), 32)


def test_wide_and_complex_dumps() raises:
    var key = seed(42)
    var raw = slurp("seed42_K32_u128.bin")
    var n = len(raw) // 16
    var wide = raw.unsafe_ptr().unsafe_bitcast[UInt128]()
    var a = Tandem.from_key(key, 0)
    var b = a.copy()
    var got = unsafe_alloc[UInt128](n)
    a.fill_u128(got, n)
    for i in range(n):
        assert_equal(got.unsafe_offset(i).unsafe_load(), wide.unsafe_offset(i).unsafe_load())
        assert_equal(b.next_u128(), wide.unsafe_offset(i).unsafe_load())
    assert_true(a == b)
    got.unsafe_free()

    var c64 = slurp("seed42_K32_c64.bin")
    n = len(c64) // 16
    var z64 = c64.unsafe_ptr().unsafe_bitcast[Float64]()
    a = Tandem.from_key(key, 0)
    b = a.copy()
    var out64 = unsafe_alloc[Float64](2 * n)
    a.fill_c64(out64, n)
    for i in range(n):
        var z = b.next_c64()
        assert_equal(z[0], z64.unsafe_offset(2 * i).unsafe_load())
        assert_equal(z[1], z64.unsafe_offset(2 * i + 1).unsafe_load())
        assert_equal(out64.unsafe_offset(2 * i).unsafe_load(), z[0])
        assert_equal(out64.unsafe_offset(2 * i + 1).unsafe_load(), z[1])
    assert_true(a == b)
    out64.unsafe_free()

    var c32 = slurp("seed42_K32_c32.bin")
    n = len(c32) // 8
    var z32 = c32.unsafe_ptr().unsafe_bitcast[Float32]()
    a = Tandem.from_key(key, 0)
    b = a.copy()
    var out32 = unsafe_alloc[Float32](2 * n)
    a.fill_c32(out32, n)
    for i in range(n):
        var z = b.next_c32()
        assert_equal(z[0], z32.unsafe_offset(2 * i).unsafe_load())
        assert_equal(z[1], z32.unsafe_offset(2 * i + 1).unsafe_load())
        assert_equal(out32.unsafe_offset(2 * i).unsafe_load(), z[0])
        assert_equal(out32.unsafe_offset(2 * i + 1).unsafe_load(), z[1])
    assert_true(a == b)
    out32.unsafe_free()


def test_random_access_against_dumps() raises:
    var u32 = slurp("k1234_K32_u32.bin")
    var w32 = u32.unsafe_ptr().unsafe_bitcast[UInt32]()
    var u64 = slurp("k1234_K32_u64.bin")
    var w64 = u64.unsafe_ptr().unsafe_bitcast[UInt64]()
    var u8 = slurp("seed42_K32_u8.bin")
    var f64 = slurp("seed42_K32_f64.bin")
    var wf64 = f64.unsafe_ptr().unsafe_bitcast[Float64]()
    var f32 = slurp("seed42_K32_f32.bin")
    var wf32 = f32.unsafe_ptr().unsafe_bitcast[Float32]()
    var g = Tandem.from_key(KEY, 0)
    var s = Tandem(42)
    for i in range(0, 65536, 97):
        assert_equal(g.at_u32(UInt64(i)), w32.unsafe_offset(i).unsafe_load())
    for i in range(0, 2048, 97):
        assert_equal(g.at_u64(UInt64(i)), w64.unsafe_offset(i).unsafe_load())
    for i in range(0, len(u8), 97):
        assert_equal(s.at_u8(UInt64(i)), u8[i])
    for i in range(0, 4096, 97):
        assert_equal(s.at_f64(UInt64(i)), wf64.unsafe_offset(i).unsafe_load())
    for i in range(0, 4096, 97):
        assert_equal(s.at_f32(UInt64(i)), wf32.unsafe_offset(i).unsafe_load())


def test_u32_dump_from_offsets() raises:
    """The stream of key (1, 2, 3, 4), K = 32, as 65536 words, from several start words."""
    var want = slurp("k1234_K32_u32.bin")
    assert_equal(len(want), 262144)
    var words = want.unsafe_ptr().unsafe_bitcast[UInt32]()
    var n = 65536
    for start in [0, 1, 7, 31, 32, 33, 1000, 65535]:
        var buf = unsafe_alloc[UInt32](n)
        var end = fill_u32(KEY, UInt64(start) * 32, 32, buf, n - start)
        assert_equal(end, UInt64(n) * 32)
        var g = Tandem.from_key(KEY, UInt64(start) * 32)
        var buf2 = unsafe_alloc[UInt32](n)
        g.fill_u32(buf2, n - start)
        assert_equal(g.position(), UInt64(n) * 32)
        for i in range(n - start):
            if buf.unsafe_offset(i).unsafe_load() != words.unsafe_offset(start + i).unsafe_load():
                assert_true(False, String("fill from word ", start, " differs at element ", i))
            if buf2.unsafe_offset(i).unsafe_load() != words.unsafe_offset(start + i).unsafe_load():
                assert_true(False, String("Tandem fill from word ", start, " differs at element ", i))
        buf.unsafe_free()
        buf2.unsafe_free()


def main() raises:
    test_integer_dumps()
    test_mapped_dumps()
    test_wide_and_complex_dumps()
    test_random_access_against_dumps()
    test_u32_dump_from_offsets()
    print("mojo dumps: ok")
