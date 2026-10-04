# Bounded integers and normals agree with the shared device core, and fills agree with scalar
# draws. The fixed values come from tandem-c's cross-check headers, which it generates from
# tandem-cuda's core.hpp. Run: mojo run -I tests -I . tests/test_derived.mojo
from std.math import cos, log, sqrt
from std.memory.alloc import unsafe_alloc
from std.testing import assert_equal, assert_true

from tandem import PURPOSE_BELOW32, PURPOSE_BELOW64, Tandem
from derived_data import (
    NORMAL_F32_END,
    NORMAL_F64_END,
    below_u32_bounds,
    below_u32_end,
    below_u32_want,
    below_u64_bounds,
    below_u64_end,
    below_u64_want,
    fill_below_u32_bounds,
    fill_below_u32_end,
    fill_below_u32_want,
    fill_below_u64_bounds,
    fill_below_u64_end,
    fill_below_u64_want,
    normal_f32,
    normal_f64,
)


def start() raises -> Tandem:
    """The fixtures start after one bit draw, which leaves the position unaligned."""
    var g = Tandem(42)
    _ = g.next_bool()
    return g^


def test_below_matches_the_device_core() raises:
    var n32 = below_u32_bounds()
    var want32 = below_u32_want()
    var end32 = below_u32_end()
    for c in range(len(n32)):
        var g = start()
        for i in range(64):
            assert_equal(g.below_u32(n32[c]), want32[64 * c + i], String("below_u32 bound ", n32[c], " element ", i))
        assert_equal(g.position(), end32[c])
    var n64 = below_u64_bounds()
    var want64 = below_u64_want()
    var end64 = below_u64_end()
    for c in range(len(n64)):
        var g = start()
        for i in range(64):
            assert_equal(g.below_u64(n64[c]), want64[64 * c + i], String("below_u64 bound ", n64[c], " element ", i))
        assert_equal(g.position(), end64[c])


def test_fill_below_matches_the_device_core() raises:
    """The large bounds reject often in 64 elements, so the fallback generator runs."""
    var n32 = fill_below_u32_bounds()
    var want32 = fill_below_u32_want()
    var end32 = fill_below_u32_end()
    var differs = 0
    for c in range(len(n32)):
        var g = start()
        var scalar = g.copy()
        var out = unsafe_alloc[UInt32](64)
        g.fill_below_u32(out, 64, n32[c])
        for i in range(64):
            assert_equal(out.unsafe_offset(i).unsafe_load(), want32[64 * c + i], String("fill_below_u32 bound ", n32[c], " element ", i))
            if out.unsafe_offset(i).unsafe_load() != scalar.below_u32(n32[c]):
                differs += 1
        assert_equal(g.position(), end32[c])
        out.unsafe_free()
    assert_true(differs > 0)
    var n64 = fill_below_u64_bounds()
    var want64 = fill_below_u64_want()
    var end64 = fill_below_u64_end()
    for c in range(len(n64)):
        var g = start()
        var out = unsafe_alloc[UInt64](64)
        g.fill_below_u64(out, 64, n64[c])
        for i in range(64):
            assert_equal(out.unsafe_offset(i).unsafe_load(), want64[64 * c + i], String("fill_below_u64 bound ", n64[c], " element ", i))
        assert_equal(g.position(), end64[c])
        out.unsafe_free()


def test_bound_zero_returns_zero_after_one_draw() raises:
    var a = Tandem(3)
    var b = Tandem(3)
    assert_equal(a.below_u32(0), UInt32(0))
    _ = b.next_u32()
    assert_true(a == b)
    assert_equal(a.below_u64(0), UInt64(0))
    _ = b.next_u64()
    assert_true(a == b)


def below_by_definition[bits: Int](rng: Tandem, raw: List[UInt64], n: UInt64) raises -> List[UInt64]:
    """Element i takes draw i of the raw fill. A rejected draw retries on sub(purpose).split(i) of the key at position 0."""
    var base = Tandem.from_key(rng.key, 0, rng.k)
    var sub = base.sub(PURPOSE_BELOW32 if bits == 32 else PURPOSE_BELOW64)
    var mask = (UInt128(1) << UInt128(bits)) - 1
    var reject = (UInt128(1) << UInt128(bits)) % UInt128(max(n, 1))
    var out = List[UInt64]()
    for i in range(len(raw)):
        var m = UInt128(raw[i]) * UInt128(n)
        if (m & mask) < reject:
            var r = sub.split(UInt64(i))
            comptime if bits == 32:
                m = UInt128(r.next_u32()) * UInt128(n)
                while (m & mask) < reject:
                    m = UInt128(r.next_u32()) * UInt128(n)
            else:
                m = UInt128(r.next_u64()) * UInt128(n)
                while (m & mask) < reject:
                    m = UInt128(r.next_u64()) * UInt128(n)
        out.append(UInt64(m >> UInt128(bits)))
    return out^


def test_bounded_fills_follow_the_definition() raises:
    """Bounds near 2^32 and 2^64 reject a quarter of the draws, so the retry runs."""
    for count in [0, 1, 31, 32, 33, 1000, 5000]:
        for n in [0, 1, 3, 1000, 0xC0000000]:
            var rng = start()
            var raw = start()
            var got = unsafe_alloc[UInt32](count + 1)
            rng.fill_below_u32(got, count, UInt32(n))
            var words = unsafe_alloc[UInt32](count + 1)
            raw.fill_u32(words, count)
            var draws = List[UInt64]()
            for i in range(count):
                draws.append(UInt64(words.unsafe_offset(i).unsafe_load()))
            var want = below_by_definition[32](rng, draws, UInt64(n))
            for i in range(count):
                assert_equal(UInt64(got.unsafe_offset(i).unsafe_load()), want[i], String("u32 n=", n, " count=", count, " element ", i))
            assert_true(rng == raw, "u32 consumes exactly count draws")
            got.unsafe_free()
            words.unsafe_free()
        for n in [0, 1, 3, 1_000_000_000_000, 0xC000000000000000]:
            var rng = start()
            var raw = start()
            var got = unsafe_alloc[UInt64](count + 1)
            rng.fill_below_u64(got, count, UInt64(n))
            var words = unsafe_alloc[UInt64](count + 1)
            raw.fill_u64(words, count)
            var draws = List[UInt64]()
            for i in range(count):
                draws.append(words.unsafe_offset(i).unsafe_load())
            var want = below_by_definition[64](rng, draws, UInt64(n))
            for i in range(count):
                assert_equal(got.unsafe_offset(i).unsafe_load(), want[i], String("u64 n=", n, " count=", count, " element ", i))
            assert_true(rng == raw, "u64 consumes exactly count draws")
            got.unsafe_free()
            words.unsafe_free()


def test_bounded_fill_without_rejection_is_the_scalar_draws() raises:
    """A power of two never rejects."""
    var a = start()
    var b = start()
    var got = unsafe_alloc[UInt32](777)
    a.fill_below_u32(got, 777, 1 << 20)
    for i in range(777):
        assert_equal(got.unsafe_offset(i).unsafe_load(), b.below_u32(1 << 20))
    assert_true(a == b)
    got.unsafe_free()


def test_normal_matches_the_device_core() raises:
    """The libraries differ in the last bits of log and cos so f64 matches to 1e-12 relative
    and f32 to 8 ulps plus 1e-6, which covers the zeros of cos. The positions are exact."""
    var want = normal_f64()
    var g = start()
    for i in range(len(want)):
        var z = g.normal_f64()
        if abs(z - want[i]) > 1e-12 * abs(want[i]):
            assert_true(False, String("normal_f64 element ", i, ": ", z, " against ", want[i]))
    assert_equal(g.position(), NORMAL_F64_END)
    var want32 = normal_f32()
    g = start()
    for i in range(len(want32)):
        var z = g.normal_f32()
        if abs(z - want32[i]) > 8.0 * 1.1920929e-07 * abs(want32[i]) + 1e-6:
            assert_true(False, String("normal_f32 element ", i, ": ", z, " against ", want32[i]))
    assert_equal(g.position(), NORMAL_F32_END)


def test_normal_f32_is_box_muller_of_two_f32_draws() raises:
    """The oracle is the exact formula in f64 on the same two draws."""
    var a = start()
    var b = start()
    for i in range(1000):
        var got = Float64(a.normal_f32())
        var u = Float64(b.next_f32())
        var v = Float64(b.next_f32())
        var want = sqrt(-2.0 * log(1.0 - u)) * cos(6.283185307179586 * v)
        if abs(got - want) > 16.0 * 1.1920929e-07 * max(abs(want), 1.0):
            assert_true(False, String("element ", i, ": ", got, " against ", want))
    assert_true(a == b)


def test_normal_fills_are_scalar_draws() raises:
    """Lengths cross the block of the normal fill and a row."""
    for count in [0, 1, 127, 128, 129, 300, 1000]:
        var a = Tandem(7)
        var b = Tandem(7)
        var got = unsafe_alloc[Float64](count + 1)
        a.fill_normal_f64(got, count)
        for i in range(count):
            assert_equal(got.unsafe_offset(i).unsafe_load(), b.normal_f64())
        assert_true(a == b, String("normal_f64 position at ", count))
        got.unsafe_free()

        a = Tandem(7)
        b = Tandem(7)
        var got32 = unsafe_alloc[Float32](count + 1)
        a.fill_normal_f32(got32, count)
        for i in range(count):
            assert_equal(got32.unsafe_offset(i).unsafe_load(), b.normal_f32())
        assert_true(a == b, String("normal_f32 position at ", count))
        got32.unsafe_free()


def test_normals_have_unit_moments() raises:
    """Mean 0 and variance 1 to within 5 standard errors of 2^20 draws."""
    var n = 1 << 20
    var z = unsafe_alloc[Float64](n)
    var y = unsafe_alloc[Float32](n)
    var g = Tandem(1)
    g.fill_normal_f64(z, n)
    g = Tandem(1)
    g.fill_normal_f32(y, n)
    for which in range(2):
        var sum = Float64(0)
        for i in range(n):
            sum += z.unsafe_offset(i).unsafe_load() if which == 0 else Float64(y.unsafe_offset(i).unsafe_load())
        var mean = sum / Float64(n)
        var sq = Float64(0)
        for i in range(n):
            var x = z.unsafe_offset(i).unsafe_load() if which == 0 else Float64(y.unsafe_offset(i).unsafe_load())
            sq += (x - mean) * (x - mean)
        var variance = sq / Float64(n)
        assert_true(abs(mean) < 5.0 / sqrt(Float64(n)), String("mean ", mean, " kind ", which))
        assert_true(abs(variance - 1.0) < 5.0 * sqrt(2.0 / Float64(n)), String("variance ", variance, " kind ", which))
    z.unsafe_free()
    y.unsafe_free()


def main() raises:
    test_below_matches_the_device_core()
    test_fill_below_matches_the_device_core()
    test_bound_zero_returns_zero_after_one_draw()
    test_bounded_fills_follow_the_definition()
    test_bounded_fill_without_rejection_is_the_scalar_draws()
    test_normal_matches_the_device_core()
    test_normal_f32_is_box_muller_of_two_f32_draws()
    test_normal_fills_are_scalar_draws()
    test_normals_have_unit_moments()
    print("mojo derived: ok")
