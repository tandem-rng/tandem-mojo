# The specification's vectors and the reference u32 stream dump, on the CPU fill.
# Run: mojo run -I . tests/test_tandem.mojo
from std.testing import assert_equal, assert_true
from std.memory.alloc import unsafe_alloc

from tandem import block, f_keyed, fill_u32, seed, split, sub, t, AUX_STREAM, DOMAIN_STREAM

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


def main() raises:
    test_step()
    test_seeding_function()
    test_stream_words()
    test_derived_keys()
    test_u32_dump()
    print("mojo: ok")
