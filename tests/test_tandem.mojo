# The specification's vectors and the reference u32 stream dump, on the CPU fill.
# Run: mojo run -I . tests/test_tandem.mojo
from std.testing import assert_equal, assert_true
from std.memory.alloc import unsafe_alloc

from tandem import Tandem, block, f_keyed, fill_u32, seed, split, sub, t, AUX_STREAM, DOMAIN_STREAM

comptime U4 = SIMD[DType.uint32, 4]
comptime KEY = U4(1, 2, 3, 4)


def test_step() raises:
    var o = U4(1, 2, 3, 4)
    var h = U4(5, 6, 7, 8)
    t(o, h)
    assert_equal(o, U4(0x00000017, 0x00150007, 0x00000001, 0x00050008))
    assert_equal(h, U4(0x9E377CA9, 0x0000E006, 0x02000007, 0x00001820))
    var z = U4(0)
    var zh = U4(0)
    t(z, zh)
    assert_equal(z, U4(0))
    assert_equal(zh, U4(0x9E3779B9, 0, 0, 0))


def test_seeding_function() raises:
    var oh = f_keyed(KEY, 0, DOMAIN_STREAM, AUX_STREAM)
    assert_equal(oh[0], U4(0x472BEF12, 0xC0977C66, 0xD330AC3A, 0xB11A020D))
    assert_equal(oh[1], U4(0xBFA0B6BA, 0xECDBC48E, 0xF1989116, 0xC2374D96))
    oh = f_keyed(KEY, 1, DOMAIN_STREAM, AUX_STREAM)
    assert_equal(oh[0], U4(0xB777C10C, 0x2F6B5A5D, 0x67A9CE03, 0x7DE06A50))
    assert_equal(oh[1], U4(0xD0C2DC4F, 0xE1E50E0F, 0x89EFC72C, 0x6E82B062))


def test_stream_words() raises:
    assert_equal(block(KEY, 0, 0), U4(0x0A5BCB90, 0x6DFE98BC, 0x9612198A, 0xAC115FD8))
    assert_equal(block(KEY, 1, 0), U4(0x33F598D7, 0xB5C280CA, 0x7A8E7B99, 0x8C362290))
    assert_equal(block(KEY, 0, 1), U4(0x9DA7BAC0, 0x4ACA79EB, 0xBEB1F65A, 0xE2F0D5A1))
    var buf = unsafe_alloc[UInt32](40)
    var end = fill_u32(KEY, 0, 32, buf, 40)
    assert_equal(end, UInt64(40 * 32))
    assert_equal(buf.unsafe_load[width=4](), U4(0x0A5BCB90, 0x6DFE98BC, 0x9612198A, 0xAC115FD8))
    assert_equal(buf.unsafe_offset(4).unsafe_load[width=4](), U4(0x33F598D7, 0xB5C280CA, 0x7A8E7B99, 0x8C362290))
    assert_equal(buf.unsafe_offset(32).unsafe_load[width=4](), U4(0x9DA7BAC0, 0x4ACA79EB, 0xBEB1F65A, 0xE2F0D5A1))
    buf.unsafe_free()


def test_derived_keys() raises:
    assert_equal(split(KEY, 0), U4(0xE256E9A1, 0x5020F806, 0x3BD3F7DC, 0x5328763D))
    assert_equal(split(KEY, 1), U4(0xA9EA3F0B, 0x47F97AF0, 0xD1844C53, 0x97E9CEE2))
    assert_equal(sub(KEY, 7), U4(0x8048398F, 0x1678E814, 0xD8823983, 0xC4B1045C))
    assert_equal(seed(42), U4(0x421D21EB, 0x32D31777, 0x62E7564B, 0xDF2BDF82))
    var buf = unsafe_alloc[UInt32](1)
    _ = fill_u32(seed(42), 0, 32, buf, 1)
    assert_equal(buf.unsafe_load(), UInt32(0x05E80CEC))
    buf.unsafe_free()


def test_u32_dump() raises:
    """The reference stream of key (1, 2, 3, 4), K = 32, as 65536 words, from several offsets."""
    var want: List[UInt8]
    with open("tests/data/k1234_K32_u32.bin", "r") as file:
        want = file.read_bytes()
    assert_equal(len(want), 262144)
    var words = want.unsafe_ptr().unsafe_bitcast[UInt32]()
    var n = 65536
    for start in [0, 1, 7, 31, 32, 33, 1000, 65535]:
        var buf = unsafe_alloc[UInt32](n)
        var end = fill_u32(KEY, UInt64(start) * 32, 32, buf, n - start)
        assert_equal(end, UInt64(n) * 32)
        for i in range(n - start):
            if buf.unsafe_offset(i).unsafe_load() != words.unsafe_offset(start + i).unsafe_load():
                assert_true(False, String("fill from word ", start, " differs at element ", i))
        buf.unsafe_free()


def fresh() raises -> Tandem:
    return Tandem.from_key(KEY, 0)


def test_spec_draws() raises:
    var g = fresh()
    assert_equal(g.next_f64(), 0.4296660861094629)
    assert_equal(fresh().at_f64(16), 0.2921520424112306)
    assert_equal(fresh().at_f32(2), Float32(0.58621365))
    var bits = fresh()
    var want = [False, False, False, False, True, False, False, True]
    for i in range(8):
        assert_equal(bits.next_bool(), want[i])
    var z = Tandem(42)
    assert_equal(z.key, U4(0x421D21EB, 0x32D31777, 0x62E7564B, 0xDF2BDF82))
    assert_equal(Tandem(42).at_f64(0), 0.9829130398628935)
    assert_equal(Tandem(42).at_f64(2), 0.47759300283385586)
    assert_equal(Tandem(42).at_f64(16), 0.9692135305890753)
    assert_equal(z.next_u32(), UInt32(0x05E80CEC))


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


def test_children() raises:
    var g = Tandem.from_key(KEY, 77)
    assert_equal(g.split(0).key, U4(0xE256E9A1, 0x5020F806, 0x3BD3F7DC, 0x5328763D))
    assert_equal(g.split(1).key, U4(0xA9EA3F0B, 0x47F97AF0, 0xD1844C53, 0x97E9CEE2))
    assert_equal(g.sub(7).key, U4(0x8048398F, 0x1678E814, 0xD8823983, 0xC4B1045C))
    assert_equal(g.split(1).position(), UInt64(0))
    assert_equal(g.position(), UInt64(77))
    var parent = fresh()
    var kids = parent.fork(3)
    assert_equal(kids[0].key, U4(0x67FD37C5, 0x9DC0B8C6, 0x4E3BD55E, 0xAF3F2216))
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
    test_step()
    test_seeding_function()
    test_stream_words()
    test_derived_keys()
    test_u32_dump()
    test_spec_draws()
    test_draw_widths()
    test_float_and_char_mappings()
    test_alignment()
    test_set_position_and_access()
    test_chunk_lengths()
    test_children()
    test_fork_advances_the_parent()
    print("mojo: ok")
