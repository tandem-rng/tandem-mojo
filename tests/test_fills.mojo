# Every fill equals the scalar draws of its type, from offsets and lengths that cut rows and
# chunks, and leaves the generator where the draws leave it.
# Run: mojo run -I . tests/test_fills.mojo
from std.testing import assert_equal, assert_true
from std.memory.alloc import unsafe_alloc

from tandem import (
    KIND_BOOL,
    KIND_CHAR,
    KIND_F16,
    KIND_F32,
    KIND_F64,
    KIND_INT,
    Tandem,
    to_char,
    to_f16_bits,
    to_f32,
    to_f64,
)

comptime U4 = SIMD[DType.uint32, 4]
comptime KEY = U4(1, 2, 3, 4)

def lengths() -> List[Int]:
    return [0, 1, 2, 31, 32, 33, 127, 128, 129, 500, 1031]


def starts() -> List[UInt64]:
    """The last two pass 2^63, where a 1-bit element index leaves the range of Int."""
    return [0, 1, 7, 33, 100, 1000, 1023, 1024, 5000, (1 << 63) + 5, (1 << 64) - (1 << 17)]


def chunks() -> List[Int]:
    return [32, 8, 1, 1024]


def draw[kind: Int, W: Int, T: DType](mut g: Tandem) -> Scalar[T]:
    """The scalar draw whose value a fill of this kind stores."""
    var raw = g.next[W]()
    comptime if kind == KIND_INT:
        return raw.cast[T]()
    elif kind == KIND_F32:
        return rebind[Scalar[T]](to_f32(UInt32(raw)))
    elif kind == KIND_F64:
        return rebind[Scalar[T]](to_f64(raw))
    elif kind == KIND_F16:
        return rebind[Scalar[T]](to_f16_bits(UInt16(raw)))
    elif kind == KIND_CHAR:
        return rebind[Scalar[T]](to_char(raw))
    else:
        return rebind[Scalar[T]](UInt8(raw))


def check[kind: Int, W: Int, T: DType](label: String) raises:
    for k in chunks():
        for start in starts():
            for n in lengths():
                var a = Tandem.from_key(KEY, start, UInt32(k))
                var b = a.copy()
                var buf = unsafe_alloc[Scalar[T]](n + 1)
                var guard = Scalar[T](0) - 1 if T.is_integral() else Scalar[T](7)
                buf.unsafe_offset(n).unsafe_store(guard)
                a.fill[kind, W, T](buf, n)
                for i in range(n):
                    if buf.unsafe_offset(i).unsafe_load() != draw[kind, W, T](b):
                        assert_true(False, String(label, " K=", k, " start=", start, " n=", n, " differs at ", i))
                assert_equal(buf.unsafe_offset(n).unsafe_load(), guard)
                if n > 0:
                    assert_true(a == b, String(label, " position K=", k, " start=", start, " n=", n))
                    assert_equal(a.next_u64(), b.next_u64())
                buf.unsafe_free()


def test_integer_fills() raises:
    check[KIND_INT, 8, DType.uint8]("u8")
    check[KIND_INT, 16, DType.uint16]("u16")
    check[KIND_INT, 32, DType.uint32]("u32")
    check[KIND_INT, 64, DType.uint64]("u64")
    check[KIND_INT, 8, DType.int8]("i8")
    check[KIND_INT, 16, DType.int16]("i16")
    check[KIND_INT, 32, DType.int32]("i32")
    check[KIND_INT, 64, DType.int64]("i64")


def test_mapped_fills() raises:
    check[KIND_F32, 32, DType.float32]("f32")
    check[KIND_F64, 64, DType.float64]("f64")
    check[KIND_F16, 16, DType.uint16]("f16 bits")
    check[KIND_CHAR, 64, DType.uint32]("char")
    check[KIND_BOOL, 1, DType.uint8]("bool")


def test_wide_and_complex_fills() raises:
    """A 128-bit fill is the 128-bit draws and a complex fill is the real fill of twice the length."""
    for start in starts():
        for n in [0, 1, 7, 8, 9, 70]:
            var a = Tandem.from_key(KEY, start)
            var b = a.copy()
            var wide = unsafe_alloc[UInt128](n + 1)
            a.fill_u128(wide, n)
            for i in range(n):
                assert_equal(wide.unsafe_offset(i).unsafe_load(), b.next_u128())
            if n > 0:
                assert_true(a == b, String("u128 position start=", start, " n=", n))
            var signed = unsafe_alloc[Int128](n + 1)
            a = Tandem.from_key(KEY, start)
            b = a.copy()
            a.fill_i128(signed, n)
            for i in range(n):
                assert_equal(signed.unsafe_offset(i).unsafe_load(), b.next_i128())
            wide.unsafe_free()
            signed.unsafe_free()

            var c64 = unsafe_alloc[Float64](2 * n + 1)
            a = Tandem.from_key(KEY, start)
            b = a.copy()
            a.fill_c64(c64, n)
            for i in range(n):
                var z = b.next_c64()
                assert_equal(c64.unsafe_offset(2 * i).unsafe_load(), z[0])
                assert_equal(c64.unsafe_offset(2 * i + 1).unsafe_load(), z[1])
            c64.unsafe_free()

            var c32 = unsafe_alloc[Float32](2 * n + 1)
            a = Tandem.from_key(KEY, start)
            b = a.copy()
            a.fill_c32(c32, n)
            for i in range(n):
                var z = b.next_c32()
                assert_equal(c32.unsafe_offset(2 * i).unsafe_load(), z[0])
                assert_equal(c32.unsafe_offset(2 * i + 1).unsafe_load(), z[1])
            c32.unsafe_free()

            var c16 = unsafe_alloc[UInt16](2 * n + 1)
            a = Tandem.from_key(KEY, start)
            b = a.copy()
            a.fill_c16_bits(c16, n)
            for i in range(n):
                var z = b.next_c16_bits()
                assert_equal(c16.unsafe_offset(2 * i).unsafe_load(), z[0])
                assert_equal(c16.unsafe_offset(2 * i + 1).unsafe_load(), z[1])
            if n > 0:
                assert_true(a == b, String("c16 position start=", start, " n=", n))
            c16.unsafe_free()


def test_empty_fill_aligns() raises:
    """An empty fill moves the position to the component width and writes nothing."""
    var g = Tandem.from_key(KEY, 3)
    g.fill_u64(unsafe_alloc[UInt64](1), 0)
    assert_equal(g.position(), UInt64(64))
    g = Tandem.from_key(KEY, 3)
    g.fill_u128(unsafe_alloc[UInt128](1), 0)
    assert_equal(g.position(), UInt64(128))
    g = Tandem.from_key(KEY, 3)
    g.fill_bool(unsafe_alloc[Bool](1), 0)
    assert_equal(g.position(), UInt64(3))


def test_fills_chain() raises:
    """Fills continue where the previous one stopped, across widths, on one cache."""
    var a = Tandem(42)
    var b = Tandem(42)
    var x = unsafe_alloc[UInt32](100)
    var y = unsafe_alloc[Float64](100)
    var z = unsafe_alloc[UInt8](100)
    a.fill_u32(x, 100)
    a.fill_f64(y, 100)
    a.fill_u8(z, 100)
    for i in range(100):
        assert_equal(x.unsafe_offset(i).unsafe_load(), b.next_u32())
    for i in range(100):
        assert_equal(y.unsafe_offset(i).unsafe_load(), b.next_f64())
    for i in range(100):
        assert_equal(z.unsafe_offset(i).unsafe_load(), b.next_u8())
    assert_true(a == b)
    x.unsafe_free()
    y.unsafe_free()
    z.unsafe_free()


def main() raises:
    test_integer_fills()
    test_mapped_fills()
    test_wide_and_complex_fills()
    test_empty_fill_aligns()
    test_fills_chain()
    print("mojo fills: ok")
