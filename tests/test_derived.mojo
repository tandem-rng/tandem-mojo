# Bounded integers, normals and exponentials agree with tandem-c's cross fixtures and the
# tandem-cuda fixtures it carries, and fills agree with scalar draws.
# Run: mojo run -I tests -I . tests/test_derived.mojo
from std.ffi import external_call
from std.math import cos, exp, log, sin, sqrt
from std.memory.alloc import unsafe_alloc
from std.builtin.sort import sort
from std.testing import assert_equal, assert_true

from tandem import PURPOSE_BELOW32, PURPOSE_BELOW64, Tandem, neg2_log_f64, normal2_f32
from tandem import seed
from derived_data import (
    NORMAL_F32_END,
    below_u32_bounds,
    below_u32_end,
    below_u32_want,
    below_u64_bounds,
    below_u64_end,
    below_u64_want,
    fill_below_u32_bounds,
    fill_below_u32_end,
    fill_below_u32_starts,
    fill_below_u32_want,
    fill_below_u64_bounds,
    fill_below_u64_end,
    fill_below_u64_starts,
    fill_below_u64_want,
    cuda_below_u32_range,
    cuda_below_u32_rejected,
    cuda_below_u32_want,
    cuda_below_u64_range,
    cuda_below_u64_rejected,
    cuda_below_u64_want,
    cuda_normal_f32_n,
    cuda_normal_f32_pos,
    cuda_normal_f32_want,
    cuda_normal_f64_n,
    cuda_normal_f64_pos,
    cuda_normal_f64_want,
    exponential_f32_end,
    exponential_f32_starts,
    exponential_f32_want,
    exponential_f64_end,
    exponential_f64_starts,
    exponential_f64_want,
    normal_f32,
    normal_f64_end,
    normal_f64_starts,
    normal_f64_want,
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
    """The starts 0, 1 and 12345 bits give the global draw indices 0, 1 and 386 (193 for u64), so the
    fallback key differs from the element index. The large bounds reject often in 64 elements."""
    var s32 = fill_below_u32_starts()
    var n32 = fill_below_u32_bounds()
    var want32 = fill_below_u32_want()
    var end32 = fill_below_u32_end()
    var differs = 0
    for c in range(len(n32)):
        var g = Tandem.from_key(seed(42), s32[c])
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
    var s64 = fill_below_u64_starts()
    var n64 = fill_below_u64_bounds()
    var want64 = fill_below_u64_want()
    var end64 = fill_below_u64_end()
    for c in range(len(n64)):
        var g = Tandem.from_key(seed(42), s64[c])
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


def test_normal_matches_tandem_c() raises:
    """The values of tandem-c's cross_normal.h bit for bit, and its end positions. The f64 rows
    start unaligned and put a wedge accept, a wedge reject and a tail at element 20. Fills and
    scalar draws both match them."""
    var starts = normal_f64_starts()
    var want = normal_f64_want()
    var end = normal_f64_end()
    var out = unsafe_alloc[Float64](64)
    for c in range(len(starts)):
        var g = Tandem(42)
        g.set_position(starts[c])
        g.fill_normal_f64(out, 64)
        assert_equal(g.position(), end[c])
        var s = Tandem(42)
        s.set_position(starts[c])
        for i in range(64):
            assert_equal(out.unsafe_offset(i).unsafe_load(), want[64 * c + i], String("normal_f64 fill at ", starts[c], " element ", i))
            assert_equal(s.normal_f64(), want[64 * c + i], String("normal_f64 at ", starts[c], " element ", i))
        assert_equal(s.position(), end[c])
    out.unsafe_free()
    var want32 = normal_f32()
    var g = start()
    for i in range(len(want32) // 2):
        var z = g.normal2_f32()
        for h in range(2):
            assert_equal(z[h], want32[2 * i + h], String("normal2_f32 pair ", i, " half ", h))
    assert_equal(g.position(), NORMAL_F32_END)


def test_scalar_normal_f32_is_the_cos_half() raises:
    var a = Tandem(3)
    var b = Tandem(3)
    assert_equal(a.normal_f32(), b.normal2_f32()[0])
    assert_true(a == b)


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


def test_fills_match_the_cuda_fixtures() raises:
    """The fixtures of tandem-cuda: the key of seed 42, K = 32. The bounded fills hold 64 elements
    from position 0, and the ranges near 2^31 and 2^63 reject, so the fallback generator runs.
    The normal fills hold 33 elements from several positions, which cuts rows and pairs. The f64
    normals match bit for bit and the f32 normals, which use __sincosf on the device, to 8 ulps."""
    var r32 = cuda_below_u32_range()
    var rej32 = cuda_below_u32_rejected()
    var w32 = cuda_below_u32_want()
    var rejecting = 0
    for c in range(len(r32)):
        var g = Tandem.from_key(seed(42), 0)
        var out = unsafe_alloc[UInt32](64)
        g.fill_below_u32(out, 64, r32[c])
        for i in range(64):
            assert_equal(out.unsafe_offset(i).unsafe_load(), w32[64 * c + i], String("below_u32 range ", r32[c], " element ", i))
        out.unsafe_free()
        rejecting += rej32[c]
    var r64 = cuda_below_u64_range()
    var rej64 = cuda_below_u64_rejected()
    var w64 = cuda_below_u64_want()
    for c in range(len(r64)):
        var g = Tandem.from_key(seed(42), 0)
        var out = unsafe_alloc[UInt64](64)
        g.fill_below_u64(out, 64, r64[c])
        for i in range(64):
            assert_equal(out.unsafe_offset(i).unsafe_load(), w64[64 * c + i], String("below_u64 range ", r64[c], " element ", i))
        out.unsafe_free()
        rejecting += rej64[c]
    assert_true(rejecting > 0)

    var pos = cuda_normal_f64_pos()
    var n = cuda_normal_f64_n()
    var want = cuda_normal_f64_want()
    var base = 0
    for c in range(len(pos)):
        var g = Tandem.from_key(seed(42), pos[c])
        var out = unsafe_alloc[Float64](n[c])
        g.fill_normal_f64(out, n[c])
        for i in range(n[c]):
            assert_equal(out.unsafe_offset(i).unsafe_load(), want[base + i], String("normal f64 pos ", pos[c], " element ", i))
        base += n[c]
        out.unsafe_free()
    var pos32 = cuda_normal_f32_pos()
    var n32 = cuda_normal_f32_n()
    var want32 = cuda_normal_f32_want()
    base = 0
    for c in range(len(pos32)):
        var g = Tandem.from_key(seed(42), pos32[c])
        var out = unsafe_alloc[Float32](n32[c])
        g.fill_normal_f32(out, n32[c])
        for i in range(n32[c]):
            var w = want32[base + i]
            assert_true(abs(out.unsafe_offset(i).unsafe_load() - w) <= 8.0 * 1.1920929e-07 * abs(w) + 1e-6, String("normal f32 pos ", pos32[c], " element ", i))
        base += n32[c]
        out.unsafe_free()


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


def fnv[origin: Origin[mut=True]](h0: UInt64, bytes: Pointer[UInt8, origin], n: Int) -> UInt64:
    var h = h0
    for i in range(n):
        h = (h ^ UInt64(bytes.unsafe_offset(i).unsafe_load())) * 0x100000001B3
    return h


def test_normal_fills_have_the_bytes_of_tandem_c() raises:
    """FNV-1a hashes equal to those of tandem-c's tests/test_normal_bits.c: 1e6 f64 normals from
    five positions, whose SHA-256 dump this port matches too, 2e5 f64 normals at two positions
    of the spec's Python reference with their end positions, and 2e6 - 1 f32 normals from the
    five positions."""
    comptime N = 1000000
    comptime BASIS = UInt64(0xCBF29CE484222325)
    var starts: List[UInt64] = [0, 1, 77, 12345, 1 << 30]
    var d = unsafe_alloc[Float64](N)
    var h = BASIS
    for s in starts:
        var g = Tandem(UInt128(2026) | (UInt128(7) << 64))
        g.set_position(s)
        g.fill_normal_f64(d, N)
        h = fnv(h, d.unsafe_bitcast[UInt8](), N * 8)
    assert_equal(h, UInt64(0xA61CFA844C85F7C1))
    var ref_starts: List[UInt64] = [0, 2373]
    var ref_hashes: List[UInt64] = [0x0C4059ED409D578D, 0x30CE40C86B295193]
    var ref_ends: List[UInt64] = [12800000, 12802432]
    for c in range(2):
        var g = Tandem.from_key(SIMD[DType.uint32, 4](1, 2, 3, 4), ref_starts[c], 32)
        g.fill_normal_f64(d, 200000)
        assert_equal(fnv(BASIS, d.unsafe_bitcast[UInt8](), 200000 * 8), ref_hashes[c])
        assert_equal(g.position(), ref_ends[c])
    d.unsafe_free()
    var f = unsafe_alloc[Float32](2 * N)
    h = BASIS
    for s in starts:
        var g = Tandem(UInt128(2026) | (UInt128(7) << 64))
        g.set_position(s)
        g.fill_normal_f32(f, 2 * N - 1)
        h = fnv(h, f.unsafe_bitcast[UInt8](), (2 * N - 1) * 4)
    f.unsafe_free()
    assert_equal(h, UInt64(0xAA1EA656CE73A4FB))


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


def test_exponentials_have_the_bytes_of_tandem_c() raises:
    """The fixture of tandem-c's tests/cross_exponential.h, bit for bit: a fill of 64 from five
    positions of the key of seed 42 and the scalar draws agree with it and with each other."""
    var starts = exponential_f64_starts()
    var want = exponential_f64_want()
    var end = exponential_f64_end()
    for c in range(len(starts)):
        var a = Tandem.from_key(seed(42), starts[c])
        var b = a.copy()
        var got = unsafe_alloc[Float64](64)
        a.fill_exponential_f64(got, 64)
        for i in range(64):
            assert_equal(got.unsafe_offset(i).unsafe_load(), want[64 * c + i], String("f64 fill start ", starts[c], " element ", i))
            assert_equal(b.exponential_f64(), want[64 * c + i], String("f64 draw start ", starts[c], " element ", i))
        assert_equal(a.position(), end[c])
        assert_equal(b.position(), end[c])
        got.unsafe_free()
    var starts32 = exponential_f32_starts()
    var want32 = exponential_f32_want()
    var end32 = exponential_f32_end()
    for c in range(len(starts32)):
        var a = Tandem.from_key(seed(42), starts32[c])
        var b = a.copy()
        var got = unsafe_alloc[Float32](64)
        a.fill_exponential_f32(got, 64)
        for i in range(64):
            assert_equal(got.unsafe_offset(i).unsafe_load(), want32[64 * c + i], String("f32 fill start ", starts32[c], " element ", i))
            assert_equal(b.exponential_f32(), want32[64 * c + i], String("f32 draw start ", starts32[c], " element ", i))
        assert_equal(a.position(), end32[c])
        assert_equal(b.position(), end32[c])
        got.unsafe_free()


def test_exponential_fills_have_the_hash_of_tandem_c() raises:
    """The FNV-1a hash of 1e6 f64 then 1e6 f32 exponentials from five positions, the value that
    tandem-c's tests/test_exponential_bits.c records."""
    comptime N = 1000000
    var starts: List[UInt64] = [0, 1, 77, 12345, 1 << 30]
    var d = unsafe_alloc[Float64](N)
    var f = unsafe_alloc[Float32](N)
    var h = UInt64(0xCBF29CE484222325)
    for s in starts:
        var g = Tandem(UInt128(2026) | (UInt128(7) << 64))
        g.set_position(s)
        g.fill_exponential_f64(d, N)
        var bytes = d.unsafe_bitcast[UInt8]()
        for i in range(N * 8):
            h = (h ^ UInt64(bytes.unsafe_offset(i).unsafe_load())) * 0x100000001B3
        g.fill_exponential_f32(f, N)
        var bytes32 = f.unsafe_bitcast[UInt8]()
        for i in range(N * 4):
            h = (h ^ UInt64(bytes32.unsafe_offset(i).unsafe_load())) * 0x100000001B3
    d.unsafe_free()
    f.unsafe_free()
    assert_equal(h, UInt64(0x47F8F98297D94EE2))


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


def test_empty_exponential_fills_move_nothing() raises:
    for p in [1, 5, 33, 65, 1001]:
        var g = Tandem.from_key(seed(1), UInt64(p))
        g.fill_exponential_f64(unsafe_alloc[Float64](1), 0)
        g.fill_exponential_f32(unsafe_alloc[Float32](1), 0)
        assert_equal(g.position(), UInt64(p))


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
    test_below_matches_the_device_core()
    test_fill_below_matches_the_device_core()
    test_bound_zero_returns_zero_after_one_draw()
    test_bounded_fills_follow_the_definition()
    test_bounded_fill_without_rejection_is_the_scalar_draws()
    test_normal_matches_tandem_c()
    test_scalar_normal_f32_is_the_cos_half()
    test_normal_f32_is_box_muller_of_two_f32_draws()
    test_normal_fills_are_the_scalar_draws()
    test_empty_fills()
    test_cut_normal_f64_fills_equal_the_whole_fill()
    test_fills_match_the_cuda_fixtures()
    test_series_against_libm()
    test_cut_bounded_fills_equal_the_whole_fill()
    test_normal_fills_have_the_bytes_of_tandem_c()
    test_normals_have_unit_moments()
    test_exponentials_have_the_bytes_of_tandem_c()
    test_exponential_fills_have_the_hash_of_tandem_c()
    test_cut_exponential_fills_equal_the_whole_fill()
    test_empty_exponential_fills_move_nothing()
    test_exponentials_are_exp1()
    print("mojo derived: ok")
