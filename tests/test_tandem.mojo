# Scalar draws, alignment, random access and derived generators of Tandem.
# Run: mojo run -I . tests/test_tandem.mojo
from std.testing import assert_equal, assert_true
from std.memory.alloc import unsafe_alloc

from tandem import Tandem, fill_u32

comptime U4 = SIMD[DType.uint32, 4]
comptime KEY = U4(1, 2, 3, 4)


def fresh() raises -> Tandem:
    return Tandem.from_key(KEY, 0)


def test_draw_widths() raises:
    """Every width reads little-endian from the first block 0a5bcb90 6dfe98bc 9612198a ac115fd8."""
    var g = fresh()
    assert_equal(g.next_u8(), UInt8(0x90))
    assert_equal(g.next_u8(), UInt8(0xCB))
    assert_equal(g.next_u16(), UInt16(0x0A5B))
    assert_equal(g.position(), UInt64(32))
    assert_equal(g.next_u64(), UInt64(0xAC115FD89612198A))
    assert_equal(g.position(), UInt64(128))
    g = fresh()
    assert_equal(g.next_u64(), UInt64(0x6DFE98BC0A5BCB90))
    g = fresh()
    assert_equal(g.next_u128(), UInt128(0xAC115FD89612198A6DFE98BC0A5BCB90))
    assert_equal(g.position(), UInt64(128))
    g = fresh()
    assert_equal(g.next_i8(), Int8(-112))
    g = fresh()
    assert_equal(g.next_i16(), Int16(-13424))
    g = fresh()
    assert_equal(g.next_i32(), Int32(0x0A5BCB90))
    g = fresh()
    assert_equal(g.next_i64(), Int64(0x6DFE98BC0A5BCB90))
    g = fresh()
    assert_equal(g.next_i128(), Int128(0x2C115FD89612198A6DFE98BC0A5BCB90) - (Int128(1) << 127))


def test_float_and_char_mappings() raises:
    var g = fresh()
    assert_equal(g.next_f32(), Float32(0.040463149547576904))
    g = fresh()
    assert_equal(g.next_f16_bits(), UInt16(0x3A5C))
    g = fresh()
    assert_equal(g.next_char(), UInt32(479864))
    g = fresh()
    var c = g.next_c64()
    var h = fresh()
    assert_equal(c, SIMD[DType.float64, 2](h.next_f64(), h.next_f64()))
    assert_equal(g.position(), UInt64(128))
    g = fresh()
    var c32 = g.next_c32()
    assert_equal(c32, SIMD[DType.float32, 2](fresh().at_f32(0), fresh().at_f32(1)))


def test_alignment() raises:
    """A draw aligns the position to its width first, and a pair may span blocks."""
    var g = fresh()
    _ = g.next_bool()
    assert_equal(g.position(), UInt64(1))
    _ = g.next_u32()
    assert_equal(g.position(), UInt64(64))
    _ = g.next_bool()
    _ = g.next_u64()
    assert_equal(g.position(), UInt64(192))
    g.set_position(120)
    var c = g.next_c64()
    assert_equal(g.position(), UInt64(256))
    var h = Tandem.from_key(KEY, 128)
    assert_equal(c[0], h.next_f64())


def test_set_position_and_access() raises:
    var g = fresh()
    for _ in range(5000):
        _ = g.next_u64()
    for p in [0, 64, 1000, 1024 * 33 + 64, 4096, 17]:
        var f = Tandem.from_key(KEY, UInt64(p))
        g.set_position(UInt64(p))
        assert_equal(g.next_u64(), f.next_u64())
        assert_equal(g.next_u32(), f.next_u32())
    var a = Tandem.from_key(KEY, 3)
    var before = a.copy()
    assert_true(a == before)
    var b = Tandem.from_key(KEY, 3)
    for i in range(70):
        assert_equal(a.at_u8(UInt64(i)), b.next_u8())
    b = Tandem.from_key(KEY, 3)
    for i in range(70):
        assert_equal(a.at_u16(UInt64(i)), b.next_u16())
    b = Tandem.from_key(KEY, 3)
    for i in range(70):
        assert_equal(a.at_u32(UInt64(i)), b.next_u32())
    b = Tandem.from_key(KEY, 3)
    for i in range(70):
        assert_equal(a.at_u64(UInt64(i)), b.next_u64())


def test_chunk_lengths() raises:
    """K changes the stream's row layout, so K = 8 draws are block B(c, j) of that layout."""
    var g = Tandem.from_key(KEY, 0, 8)
    var buf = unsafe_alloc[UInt32](8 * 32 + 4)
    _ = fill_u32(KEY, 0, 8, buf, 8 * 32 + 4)
    for i in range(8 * 32 + 4):
        assert_equal(g.next_u32(), buf.unsafe_offset(i).unsafe_load())
    buf.unsafe_free()
    var raised = False
    try:
        _ = Tandem.from_key(KEY, 0, 24)
    except:
        raised = True
    assert_true(raised)


def test_derived_generators() raises:
    var g = Tandem.from_key(KEY, 77)
    assert_true(g.split(0).key != g.split(1).key)
    assert_equal(g.split(5).k, g.k)
    assert_equal(g.sub(7).position(), UInt64(0))
    assert_equal(g.split(1).position(), UInt64(0))
    assert_equal(g.position(), UInt64(77))
    var parent = fresh()
    var kids = parent.fork(3)
    assert_equal(len(kids), 3)


def test_fork_advances_the_parent() raises:
    var g = Tandem.from_key(KEY, 130)
    var one = g.fork()
    assert_equal(g.position(), UInt64(256))
    var again = g.fork()
    assert_equal(g.position(), UInt64(384))
    assert_true(one.key != again.key)
    var empty = g.fork(0)
    assert_equal(len(empty), 0)
    assert_equal(g.position(), UInt64(512))
    var h = Tandem.from_key(KEY, 130)
    var batch = h.fork(2)
    var h2 = Tandem.from_key(KEY, 130)
    assert_true(batch[0] == h2.fork())
    assert_true(batch[1].key != batch[0].key)


def main() raises:
    test_draw_widths()
    test_float_and_char_mappings()
    test_alignment()
    test_set_position_and_access()
    test_chunk_lengths()
    test_derived_generators()
    test_fork_advances_the_parent()
    print("mojo tandem: ok")
