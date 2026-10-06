# Bounded integers, normals and exponentials against their definitions, and long fills against
# the scalar draws and cut into pieces. The fixtures of tandem-c and tandem-cuda are in
# test_conformance.mojo.
# Run: mojo run -I tests -I . tests/test_derived.mojo
from std.ffi import external_call
from std.math import cos, exp, log, sin, sqrt
from std.memory.alloc import unsafe_alloc
from std.builtin.sort import sort
from std.testing import assert_equal, assert_true

from tandem import PURPOSE_BELOW32, PURPOSE_BELOW64, Tandem, neg2_log_f64, normal2_f32
from tandem import seed


def start() raises -> Tandem:
    """The fixtures start after one bit draw, which leaves the position unaligned."""
    var g = Tandem(42)
    _ = g.next_bool()
    return g^


def below_by_definition[bits: Int](rng: Tandem, raw: List[UInt64], n: UInt64, first: Int) raises -> List[UInt64]:
    """Element i takes draw i of the raw fill. A rejected draw retries on sub(purpose).split(g) of
    the key at position 0, with the global draw index g = first + i."""
    var base = Tandem.from_key(rng.key, 0, rng.k)
    var sub = base.sub(PURPOSE_BELOW32 if bits == 32 else PURPOSE_BELOW64)
    var mask = (UInt128(1) << UInt128(bits)) - 1
    var reject = (UInt128(1) << UInt128(bits)) % UInt128(max(n, 1))
    var out = List[UInt64]()
    for i in range(len(raw)):
        var m = UInt128(raw[i]) * UInt128(n)
        if (m & mask) < reject:
            var r = sub.split(UInt64(first + i))
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
    """Bounds near 2^32 and 2^64 reject a quarter of the draws, so the retry runs. The lengths
    cut the SIMD block, the draw block and a row from an unaligned start."""
    for count in [0, 1, 7, 15, 16, 17, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 2049, 5000]:
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
            var want = below_by_definition[32](rng, draws, UInt64(n), 1)
            for i in range(count):
                assert_equal(UInt64(got.unsafe_offset(i).unsafe_load()), want[i], String("u32 n=", n, " count=", count, " element ", i))
            if count > 0:
                assert_true(rng == raw, "u32 consumes exactly count draws")
            else:
                assert_true(rng == start(), "u32 empty fill moves nothing")
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
            var want = below_by_definition[64](rng, draws, UInt64(n), 1)
            for i in range(count):
                assert_equal(got.unsafe_offset(i).unsafe_load(), want[i], String("u64 n=", n, " count=", count, " element ", i))
            if count > 0:
                assert_true(rng == raw, "u64 consumes exactly count draws")
            else:
                assert_true(rng == start(), "u64 empty fill moves nothing")
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


def test_normal_f32_is_box_muller_of_two_f32_draws() raises:
    """The oracle is the exact formula in f64 on the same two draws, for both halves."""
    var a = start()
    var b = start()
    for i in range(1000):
        var got = a.normal2_f32()
        var u = Float64(b.next_f32())
        var v = Float64(b.next_f32())
        var r = sqrt(-2.0 * log(1.0 - u))
        var want = SIMD[DType.float64, 2](r * cos(6.283185307179586 * v), r * sin(6.283185307179586 * v))
        for h in range(2):
            if abs(Float64(got[h]) - want[h]) > 16.0 * 1.1920929e-07 * max(abs(want[h]), 1.0):
                assert_true(False, String("pair ", i, " half ", h, ": ", got[h], " against ", want[h]))
    assert_true(a == b)


def test_normal_fills_are_the_scalar_draws() raises:
    """Lengths cross the block of the normal fill and a row. An f64 fill equals the scalar f64
    draws, misses included. An odd f32 count keeps the cos half of its last pair and consumes
    both draws."""
    for count in [1, 2, 3, 127, 128, 129, 257, 300, 513, 1000, 3000]:
        var a = start()
        var b = start()
        var got = unsafe_alloc[Float64](count)
        a.fill_normal_f64(got, count)
        for i in range(count):
            assert_equal(got.unsafe_offset(i).unsafe_load(), b.normal_f64(), String("normal_f64 at ", count, " element ", i))
        assert_true(a == b, String("normal_f64 position at ", count))
        got.unsafe_free()

        a = start()
        b = start()
        var got32 = unsafe_alloc[Float32](count + 1)
        a.fill_normal_f32(got32, count)
        for i in range(count // 2 + count % 2):
            var z = b.normal2_f32()
            assert_equal(got32.unsafe_offset(2 * i).unsafe_load(), z[0])
            if 2 * i + 1 < count:
                assert_equal(got32.unsafe_offset(2 * i + 1).unsafe_load(), z[1])
        assert_true(a == b, String("normal_f32 position at ", count))
        got32.unsafe_free()


def test_empty_fills() raises:
    """A bounded fill advances exactly its draws and an f32 normal fill its pairs, so an empty one
    does not even align the position. An empty f64 normal fill aligns it to 64 bits, as
    Appendix A requires."""
    for p in [1, 5, 33, 65, 1001]:
        var g = Tandem.from_key(seed(1), UInt64(p))
        g.fill_below_u32(unsafe_alloc[UInt32](1), 0, 10)
        g.fill_below_u64(unsafe_alloc[UInt64](1), 0, 10)
        g.fill_normal_f32(unsafe_alloc[Float32](1), 0)
        assert_equal(g.position(), UInt64(p))
        g.fill_normal_f64(unsafe_alloc[Float64](1), 0)
        assert_equal(g.position(), (UInt64(p) + 63) & ~UInt64(63))


def test_cut_normal_f64_fills_equal_the_whole_fill() raises:
    """The fallback is keyed by the global draw index, so the misses of fills cut at any element
    equal those of the whole fill. 3000 draws hold about 13 misses, and the start is unaligned."""
    var cuts = [0, 1, 15, 16, 513, 1025, 2000, 3000]
    var whole = unsafe_alloc[Float64](3000)
    var cut = unsafe_alloc[Float64](3000)
    var a = Tandem.from_key(seed(42), 12345)
    a.fill_normal_f64(whole, 3000)
    var b = Tandem.from_key(seed(42), 12345)
    for c in range(len(cuts) - 1):
        b.fill_normal_f64(cut.unsafe_offset(cuts[c]), cuts[c + 1] - cuts[c])
    for i in range(3000):
        assert_equal(cut.unsafe_offset(i).unsafe_load(), whole.unsafe_offset(i).unsafe_load(), String("element ", i))
    assert_true(a == b)
    whole.unsafe_free()
    cut.unsafe_free()


def libm_normal2(a: Float64, b: Float64) -> SIMD[DType.float64, 2]:
    var r = sqrt(-2.0 * external_call["log", Float64](1.0 - a))
    var angle = 6.283185307179586 * b
    return SIMD[DType.float64, 2](r * external_call["cos", Float64](angle), r * external_call["sin", Float64](angle))


def test_series_against_libm() raises:
    """The SIMD series against libm on 2^18 uniform pairs, including the ends of the range
    where u = 1 - a is tiny or 1 and where cos or sin crosses zero: the f64 logarithm, and the
    f32 Box-Muller pair."""
    var g = Tandem(11)
    var edges = [0.0, 1.1102230246251565e-16, 0.25, 0.5, 0.75, 0.9999999999999999, 0.125, 0.375]
    var worst = Float64(0)
    var worst32 = Float64(0)
    for i in range(1 << 18):
        var a = g.next_f64()
        var b = g.next_f64()
        if i < 64:
            a = edges[i % 8]
            b = edges[(i // 8) % 8]
        var x = 1.0 - a
        var want = -2.0 * external_call["log", Float64](x)
        worst = max(worst, abs(neg2_log_f64[1](SIMD[DType.float64, 1](x))[0] - want) / max(abs(want), 1e-300))
        var af = min(Float32(a), Float32(0.99999994))
        var bf = Float32(b)
        var z32 = normal2_f32[1](SIMD[DType.float32, 1](af), SIMD[DType.float32, 1](bf))
        # The contract rounds u = 1 - a to f32, so the oracle starts from that u.
        var w32 = libm_normal2(1.0 - Float64(Float32(1.0) - af), Float64(bf))
        worst32 = max(worst32, max(abs(Float64(z32[0][0]) - w32[0]) / max(abs(w32[0]), 1.0), abs(Float64(z32[1][0]) - w32[1]) / max(abs(w32[1]), 1.0)))
    assert_true(worst < 2e-15, String("f64 worst relative error ", worst))
    assert_true(worst32 < 16.0 * 1.1920929e-07, String("f32 worst error ", worst32))


def test_cut_bounded_fills_equal_the_whole_fill() raises:
    """Fills cut at arbitrary element boundaries and run one after the other equal the whole fill,
    at a start that is not on a draw boundary and with rejections in every piece."""
    var cuts = [0, 1, 17, 18, 1025, 2000, 3000]
    var whole32 = unsafe_alloc[UInt32](3000)
    var cut32 = unsafe_alloc[UInt32](3000)
    var a = Tandem.from_key(seed(42), 12345)
    a.fill_below_u32(whole32, 3000, 0xC0000001)
    var b = Tandem.from_key(seed(42), 12345)
    var rejected = 0
    var plain = Tandem.from_key(seed(42), 12345)
    for _ in range(3000):
        rejected += Int(UInt32(UInt64(plain.next_u32()) * 0xC0000001 & 0xFFFFFFFF) < 0x3FFFFFFF)
    for c in range(len(cuts) - 1):
        b.fill_below_u32(cut32.unsafe_offset(cuts[c]), cuts[c + 1] - cuts[c], 0xC0000001)
    assert_true(rejected > 100)
    for i in range(3000):
        assert_equal(cut32.unsafe_offset(i).unsafe_load(), whole32.unsafe_offset(i).unsafe_load(), String("u32 element ", i))
    assert_true(a == b)
    whole32.unsafe_free()
    cut32.unsafe_free()
    var whole64 = unsafe_alloc[UInt64](3000)
    var cut64 = unsafe_alloc[UInt64](3000)
    var c0 = Tandem.from_key(seed(42), 12345)
    c0.fill_below_u64(whole64, 3000, 0xC000000000000001)
    var d = Tandem.from_key(seed(42), 12345)
    for c in range(len(cuts) - 1):
        d.fill_below_u64(cut64.unsafe_offset(cuts[c]), cuts[c + 1] - cuts[c], 0xC000000000000001)
    for i in range(3000):
        assert_equal(cut64.unsafe_offset(i).unsafe_load(), whole64.unsafe_offset(i).unsafe_load(), String("u64 element ", i))
    assert_true(c0 == d)
    whole64.unsafe_free()
    cut64.unsafe_free()


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


def test_cut_exponential_fills_equal_the_whole_fill() raises:
    """Fills cut across the L1 block and at positions off a draw boundary equal the whole fill
    and the scalar draws, and leave the generator where they leave it."""
    var cuts = [0, 1, 17, 18, 1025, 2000, 3000]
    var whole = unsafe_alloc[Float64](3000)
    var cut = unsafe_alloc[Float64](3000)
    var a = Tandem.from_key(seed(42), 12345)
    var b = a.copy()
    var c = a.copy()
    a.fill_exponential_f64(whole, 3000)
    for k in range(len(cuts) - 1):
        b.fill_exponential_f64(cut.unsafe_offset(cuts[k]), cuts[k + 1] - cuts[k])
    for i in range(3000):
        assert_equal(cut.unsafe_offset(i).unsafe_load(), whole.unsafe_offset(i).unsafe_load(), String("f64 element ", i))
        assert_equal(c.exponential_f64(), whole.unsafe_offset(i).unsafe_load(), String("f64 draw ", i))
    assert_true(a == b)
    assert_true(a == c)
    whole.unsafe_free()
    cut.unsafe_free()
    var whole32 = unsafe_alloc[Float32](3000)
    var cut32 = unsafe_alloc[Float32](3000)
    var d = Tandem.from_key(seed(42), 12345)
    var e = d.copy()
    var f = d.copy()
    d.fill_exponential_f32(whole32, 3000)
    for k in range(len(cuts) - 1):
        e.fill_exponential_f32(cut32.unsafe_offset(cuts[k]), cuts[k + 1] - cuts[k])
    for i in range(3000):
        assert_equal(cut32.unsafe_offset(i).unsafe_load(), whole32.unsafe_offset(i).unsafe_load(), String("f32 element ", i))
        assert_equal(f.exponential_f32(), whole32.unsafe_offset(i).unsafe_load(), String("f32 draw ", i))
    assert_true(d == e)
    assert_true(d == f)
    whole32.unsafe_free()
    cut32.unsafe_free()


def check_exp1[T: DType](mut x: List[Scalar[T]]) raises:
    """Raw moments 1 to 4 within 5 standard errors of k!, and the Kolmogorov distance within the
    0.1% critical value."""
    var n = len(x)
    var fact = [1.0, 2.0, 6.0, 24.0]
    var fact2 = [2.0, 24.0, 720.0, 40320.0]
    var sums = [Float64(0), Float64(0), Float64(0), Float64(0)]
    for i in range(n):
        var v = Float64(x[i])
        var q = v
        for k in range(4):
            sums[k] += q
            q *= v
    for k in range(4):
        var se = sqrt((fact2[k] - fact[k] * fact[k]) / Float64(n))
        assert_true(abs(sums[k] / Float64(n) - fact[k]) < 5.0 * se, String("moment ", k + 1, ": ", sums[k] / Float64(n)))
    sort(x)
    var dmax = Float64(0)
    for i in range(n):
        var cdf = 1.0 - exp(-Float64(x[i]))
        dmax = max(dmax, max(abs(Float64(i + 1) / Float64(n) - cdf), abs(Float64(i) / Float64(n) - cdf)))
    assert_true(dmax < 1.9495 / sqrt(Float64(n)), String("KS distance ", dmax))


def test_exponentials_are_exp1() raises:
    """1e7 draws of each width: moments to the fourth order and the KS distance to 1 - exp(-x)."""
    var n = 10_000_000
    var d = unsafe_alloc[Float64](n)
    var g = Tandem(7)
    g.fill_exponential_f64(d, n)
    var xd = List[Float64](length=n, fill=0.0)
    for i in range(n):
        xd[i] = d.unsafe_offset(i).unsafe_load()
    d.unsafe_free()
    check_exp1[DType.float64](xd)
    var f = unsafe_alloc[Float32](n)
    g = Tandem(7)
    g.fill_exponential_f32(f, n)
    var xf = List[Float32](length=n, fill=0.0)
    for i in range(n):
        xf[i] = f.unsafe_offset(i).unsafe_load()
    f.unsafe_free()
    check_exp1[DType.float32](xf)


def main() raises:
    test_bounded_fills_follow_the_definition()
    test_bounded_fill_without_rejection_is_the_scalar_draws()
    test_normal_f32_is_box_muller_of_two_f32_draws()
    test_normal_fills_are_the_scalar_draws()
    test_empty_fills()
    test_cut_normal_f64_fills_equal_the_whole_fill()
    test_series_against_libm()
    test_cut_bounded_fills_equal_the_whole_fill()
    test_normals_have_unit_moments()
    test_cut_exponential_fills_equal_the_whole_fill()
    test_exponentials_are_exp1()
    print("mojo derived: ok")
