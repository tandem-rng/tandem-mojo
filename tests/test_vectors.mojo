# Every vector of the specification, from tests/vectors_data.mojo, which tools/gen_vectors.py
# generates from the spec repository's vectors.json.
# Run: mojo run -I . -I tests tests/test_vectors.mojo
from std.memory.alloc import unsafe_alloc
from std.testing import assert_equal, assert_true

from tandem import AUX_STREAM, DOMAIN_STREAM, Tandem, block, f_keyed, fill_u32, seed, t
from vectors_data import *


def test_step() raises:
    var o = t_o()
    var h = t_h()
    var o_out = t_o_out()
    var h_out = t_h_out()
    for i in range(len(o)):
        var oo = o[i]
        var hh = h[i]
        t(oo, hh)
        assert_equal(oo, o_out[i])
        assert_equal(hh, h_out[i])


def test_seeding_function() raises:
    var counter = f_counter()
    var want_o = f_o()
    var want_h = f_h()
    for i in range(len(counter)):
        var oh = f_keyed(KEY, counter[i], DOMAIN_STREAM, AUX_STREAM)
        assert_equal(oh[0], want_o[i])
        assert_equal(oh[1], want_h[i])


def test_stream_words() raises:
    """Each listed block by B(c, j), by the fill, and by scalar draws."""
    var first = stream_first_word()
    var want = stream_words()
    var buf = unsafe_alloc[UInt32](64)
    _ = fill_u32(KEY, 0, K, buf, 64)
    for i in range(len(first)):
        var lane = (first[i] // 4) % 8
        var row = first[i] // 32
        assert_equal(block(KEY, UInt64(lane), UInt32(row)), want[i])
        assert_equal(buf.unsafe_offset(first[i]).unsafe_load[width=4](), want[i])
        var g = Tandem.from_key(KEY, UInt64(first[i]) * 32, K)
        for w in range(4):
            assert_equal(g.next_u32(), want[i][w])
    buf.unsafe_free()


def test_float_and_bool_draws() raises:
    var g = Tandem.from_key(KEY, 0, K)
    var i64 = float64_index()
    var v64 = float64_value()
    for i in range(len(i64)):
        assert_equal(g.at_f64(UInt64(i64[i])), v64[i])
    var i32 = float32_index()
    var v32 = float32_value()
    for i in range(len(i32)):
        assert_equal(g.at_f32(UInt64(i32[i])), v32[i])
    var ib = bool_index()
    var vb = bool_value()
    var bits = unsafe_alloc[Bool](256)
    g.fill_bool(bits, 256)
    for i in range(len(ib)):
        assert_equal(bits.unsafe_bitcast[UInt8]().unsafe_offset(ib[i]).unsafe_load() != 0, vb[i])
    bits.unsafe_free()


def test_derived_keys() raises:
    var g = Tandem.from_key(KEY, 0, K)
    assert_equal(g.split(0).key, SPLIT0)
    assert_equal(g.split(1).key, SPLIT1)
    assert_equal(g.sub(7).key, PURPOSE7)
    assert_equal(g.fork().key, FORK0)


def test_seed_whitening() raises:
    assert_equal(seed(SEED), SEED_KEY)
    var g = Tandem(SEED)
    assert_equal(g.key, SEED_KEY)
    var i64 = seed_f64_index()
    var v64 = seed_f64_value()
    for i in range(len(i64)):
        assert_equal(g.at_f64(UInt64(i64[i])), v64[i])
    var iu = seed_u32_index()
    var vu = seed_u32_value()
    for i in range(len(iu)):
        assert_equal(g.at_u32(UInt64(iu[i])), vu[i])
        var buf = unsafe_alloc[UInt32](1)
        _ = fill_u32(SEED_KEY, UInt64(iu[i]) * 32, K, buf, 1)
        assert_equal(buf.unsafe_load(), vu[i])
        buf.unsafe_free()


def main() raises:
    test_step()
    test_seeding_function()
    test_stream_words()
    test_float_and_bool_draws()
    test_derived_keys()
    test_seed_whitening()
    print("mojo vectors: ok")
