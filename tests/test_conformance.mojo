# The conformance files of the specification (tests/conformance, byte copies of tandem-spec
# conformance/*.json) and every behaviour of its CHECKLIST.md: the bounded, normal, exponential
# and weighted choice fixtures with their scalar draws and cut fills, the stream and dump
# hashes, and the position bounds.
# Run: mojo run -I . -I tests tests/test_conformance.mojo
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.testing import assert_equal, assert_false, assert_true

from conformance import Case, Fnv1a, Sha256, find_case, hex_value, load
from tandem import ChoiceTable, Tandem, seed, to_f32, to_f64

comptime BELOW_U32 = 0
comptime BELOW_U64 = 1
comptime FILL_BELOW_U32 = 2
comptime FILL_BELOW_U64 = 3
comptime NORMAL_F64 = 4
comptime NORMAL_F32 = 5
comptime EXPONENTIAL_F64 = 6
comptime EXPONENTIAL_F32 = 7
comptime CHOICE = 8

comptime KEY42 = SIMD[DType.uint32, 4](0x421D21EB, 0x32D31777, 0x62E7564B, 0xDF2BDF82)


def kind_of(c: Case) raises -> Int:
    var k = c.text("kind")
    if k == "below_u32":
        return BELOW_U32
    if k == "below_u64":
        return BELOW_U64
    if k == "fill_below_u32":
        return FILL_BELOW_U32
    if k == "fill_below_u64":
        return FILL_BELOW_U64
    if k == "fill_normal_f64":
        return NORMAL_F64
    if k == "fill_normal_f32":
        return NORMAL_F32
    if k == "fill_exponential_f64":
        return EXPONENTIAL_F64
    if k == "fill_exponential_f32":
        return EXPONENTIAL_F32
    if k == "fill_choice":
        return CHOICE
    raise Error(String("unknown kind ", k))


def align(p: Int, w: Int) -> Int:
    return (p + w - 1) // w * w


def expected_end(kind: Int, start: Int, n: Int) -> Int:
    """The position after the operation by the rules of Appendix A and C: an empty bounded, f32
    normal or exponential fill moves nothing, any other empty fill aligns."""
    if kind == FILL_BELOW_U32 or kind == EXPONENTIAL_F32:
        return start if n == 0 else align(start, 32) + 32 * n
    if kind == NORMAL_F32:
        return start if n == 0 else align(start, 32) + 64 * ((n + 1) // 2)
    if kind == FILL_BELOW_U64 or kind == EXPONENTIAL_F64:
        return start if n == 0 else align(start, 64) + 64 * n
    return align(start, 64) + 64 * n


def generator(c: Case) raises -> Tandem:
    return Tandem.from_key(c.key(), UInt64(c.number("start")), UInt32(c.number("K")))


def table_of(c: Case) raises -> ChoiceTable:
    var weights = List[Float64]()
    for w in c.words("weights"):
        weights.append(bitcast[DType.float64, 1](hex_value(w)))
    return ChoiceTable(weights)


def expected_bits(c: Case) raises -> List[UInt64]:
    var out = List[UInt64]()
    for v in c.words("values"):
        out.append(hex_value(v))
    return out^


struct Buffers(Movable):
    var u32: Pointer[UInt32, MutUntrackedOrigin]
    var u64: Pointer[UInt64, MutUntrackedOrigin]
    var f32: Pointer[Float32, MutUntrackedOrigin]
    var f64: Pointer[Float64, MutUntrackedOrigin]

    def __init__(out self, n: Int):
        self.u32 = unsafe_alloc[UInt32](n + 1)
        self.u64 = unsafe_alloc[UInt64](n + 1)
        self.f32 = unsafe_alloc[Float32](n + 1)
        self.f64 = unsafe_alloc[Float64](n + 1)

    def free(self):
        self.u32.unsafe_free()
        self.u64.unsafe_free()
        self.f32.unsafe_free()
        self.f64.unsafe_free()

    def bits(self, kind: Int, i: Int) -> UInt64:
        if kind == BELOW_U32 or kind == FILL_BELOW_U32 or kind == CHOICE:
            return UInt64(self.u32.unsafe_offset(i).unsafe_load())
        if kind == BELOW_U64 or kind == FILL_BELOW_U64:
            return self.u64.unsafe_offset(i).unsafe_load()
        if kind == NORMAL_F32 or kind == EXPONENTIAL_F32:
            return UInt64(bitcast[DType.uint32, 1](self.f32.unsafe_offset(i).unsafe_load()))
        return bitcast[DType.uint64, 1](self.f64.unsafe_offset(i).unsafe_load())


def fill_piece(mut g: Tandem, kind: Int, c: Case, table: ChoiceTable, b: Buffers, a: Int, m: Int) raises:
    """One fill of m elements into elements a to a + m of the buffers."""
    if kind == FILL_BELOW_U32:
        g.fill_below_u32(b.u32.unsafe_offset(a), m, UInt32(hex_value(c.text("range"))))
    elif kind == FILL_BELOW_U64:
        g.fill_below_u64(b.u64.unsafe_offset(a), m, hex_value(c.text("range")))
    elif kind == NORMAL_F64:
        g.fill_normal_f64(b.f64.unsafe_offset(a), m)
    elif kind == NORMAL_F32:
        g.fill_normal_f32(b.f32.unsafe_offset(a), m)
    elif kind == EXPONENTIAL_F64:
        g.fill_exponential_f64(b.f64.unsafe_offset(a), m)
    elif kind == EXPONENTIAL_F32:
        g.fill_exponential_f32(b.f32.unsafe_offset(a), m)
    elif kind == CHOICE:
        g.fill_choice(b.u32.unsafe_offset(a), m, table)
    else:
        raise Error("not a fill")


def scalar_draws(mut g: Tandem, kind: Int, c: Case, table: ChoiceTable, b: Buffers, n: Int) raises:
    """Scalar draws, n of the kind, into the buffers. A Float32 normal comes in pairs."""
    if kind == BELOW_U32 or kind == FILL_BELOW_U32:
        for i in range(n):
            b.u32.unsafe_offset(i).unsafe_store(g.below_u32(UInt32(hex_value(c.text("range")))))
    elif kind == BELOW_U64 or kind == FILL_BELOW_U64:
        for i in range(n):
            b.u64.unsafe_offset(i).unsafe_store(g.below_u64(hex_value(c.text("range"))))
    elif kind == NORMAL_F64:
        for i in range(n):
            b.f64.unsafe_offset(i).unsafe_store(g.normal_f64())
    elif kind == NORMAL_F32:
        for i in range((n + 1) // 2):
            var z = g.normal2_f32()
            b.f32.unsafe_offset(2 * i).unsafe_store(z[0])
            if 2 * i + 1 < n:
                b.f32.unsafe_offset(2 * i + 1).unsafe_store(z[1])
    elif kind == EXPONENTIAL_F64:
        for i in range(n):
            b.f64.unsafe_offset(i).unsafe_store(g.exponential_f64())
    elif kind == EXPONENTIAL_F32:
        for i in range(n):
            b.f32.unsafe_offset(i).unsafe_store(g.exponential_f32())
    else:
        for i in range(n):
            b.u32.unsafe_offset(i).unsafe_store(g.choice(table))


def check_bits(b: Buffers, kind: Int, want: List[UInt64], label: String) raises:
    for i in range(len(want)):
        if b.bits(kind, i) != want[i]:
            assert_true(False, String(label, " differs at element ", i))


def rejections(c: Case) raises -> Int:
    """The draws below the Lemire threshold in the plain fill of the case: the elements that
    take the fallback. A fill without any equals the scalar draws."""
    var kind = kind_of(c)
    var n = c.number("n")
    var g = generator(c)
    var count = 0
    var r = hex_value(c.text("range"))
    if kind == FILL_BELOW_U32:
        var raw = unsafe_alloc[UInt32](n + 1)
        g.fill_u32(raw, n)
        var t = (UInt32(0) - UInt32(r)) % UInt32(r)
        for i in range(n):
            count += 1 if UInt32((UInt64(raw.unsafe_offset(i).unsafe_load()) * r) & 0xFFFFFFFF) < t else 0
        raw.unsafe_free()
    else:
        var raw = unsafe_alloc[UInt64](n + 1)
        g.fill_u64(raw, n)
        var t = (UInt64(0) - r) % r
        for i in range(n):
            var lo = UInt64((UInt128(raw.unsafe_offset(i).unsafe_load()) * UInt128(r)) & 0xFFFFFFFFFFFFFFFF)
            count += 1 if lo < t else 0
        raw.unsafe_free()
    return count


def test_cases(file: String, count: Int) raises:
    """Every case of a file: the whole fill, the scalar draws, and fills cut at elements 1, 7,
    20, 21 and n - 1 and run in order on one generator. The end positions follow the rules of
    the appendices, and the file's own `end` agrees with them."""
    var cases = load(file)
    assert_equal(len(cases), count, file)
    var rejecting = 0
    for c in cases:
        var kind = kind_of(c)
        var id = c.text("id")
        var n = c.number("n")
        var start = c.number("start")
        var want = expected_bits(c)
        assert_equal(len(want), n, id)
        # A scalar bounded draw retries on its own stream, so its end is the file's.
        var scalar_kind = kind == BELOW_U32 or kind == BELOW_U64
        var stop = c.number("end") if scalar_kind else expected_end(kind, start, n)
        if c.has("end"):
            assert_equal(c.number("end"), stop, id)
        var table = ChoiceTable([1.0]) if kind != CHOICE else table_of(c)
        if kind == CHOICE:
            assert_equal(table.capacity, hex_value(c.text("capacity")), id)
            if c.has("cut"):
                var cut = c.words("cut")
                var alias = c.words("alias")
                for i in range(len(cut)):
                    assert_equal(table.cut[i], hex_value(cut[i]), String(id, " cut ", i))
                    assert_equal(UInt64(table.alias[i]), hex_value(alias[i]), String(id, " alias ", i))
                assert_equal(len(table.cut), len(cut), id)
        var b = Buffers(n)
        var g = generator(c)
        if scalar_kind:
            scalar_draws(g, kind, c, table, b, n)
        else:
            fill_piece(g, kind, c, table, b, 0, n)
        check_bits(b, kind, want, id)
        assert_equal(Int(g.position()), stop, id)
        if scalar_kind:
            b.free()
            continue
        # A fill is the scalar draws, except that a rejected bounded draw retries elsewhere.
        var rejected = 0
        if kind == FILL_BELOW_U32 or kind == FILL_BELOW_U64:
            rejected = rejections(c) if n > 0 else 0
            if c.has("rejected"):
                assert_equal(rejected, c.number("rejected"), id)
        rejecting += 1 if rejected > 0 else 0
        if n > 0 and rejected == 0:
            var s = generator(c)
            scalar_draws(s, kind, c, table, b, n)
            check_bits(b, kind, want, String(id, " scalar"))
            assert_equal(Int(s.position()), stop, String(id, " scalar"))
        # A Float32 normal fill cut at an odd element drops a sin half, so only even cuts compose.
        for cutat in [1, 7, 20, 21, n - 1]:
            if cutat <= 0 or cutat >= n or (kind == NORMAL_F32 and cutat % 2 == 1):
                continue
            var h = generator(c)
            fill_piece(h, kind, c, table, b, 0, cutat)
            fill_piece(h, kind, c, table, b, cutat, n - cutat)
            check_bits(b, kind, want, String(id, " cut at ", cutat))
            assert_equal(Int(h.position()), stop, String(id, " cut at ", cutat))
        b.free()
    if file == "fill_below.json":
        assert_true(rejecting > 0)


def test_below_and_fill_below() raises:
    test_cases("below.json", 11)
    test_cases("fill_below.json", 77)


def test_normal() raises:
    test_cases("normal.json", 20)


def test_exponential() raises:
    test_cases("exponential.json", 12)


def test_choice() raises:
    test_cases("choice.json", 24)


def values_of(cases: List[Case], name: String) raises -> List[String]:
    return cases[find_case(cases, name)].words("values")


def test_fallback_is_keyed_by_the_global_draw_index() raises:
    """Element i of a fill from start 1 equals element i + 1 of the fill from start 0, and the
    normal fill from start 1 equals the one from start 0 shifted by a draw: the same global
    draw index gives the same fallback."""
    var fb = load("fill_below.json")
    var a = values_of(fb, "CROSS_BELOW32[4]")
    var at = values_of(fb, "CROSS_BELOW32_AT[4]")
    for i in range(63):
        assert_equal(at[i], a[i + 1], String("u32 element ", i))
    a = values_of(fb, "CROSS_BELOW64[6]")
    at = values_of(fb, "CROSS_BELOW64_AT[6]")
    for i in range(63):
        assert_equal(at[i], a[i + 1], String("u64 element ", i))
    var nm = load("normal.json")
    var z0 = values_of(nm, "CROSS_NORMAL[0]")
    var z1 = values_of(nm, "CROSS_NORMAL[1]")
    for i in range(63):
        assert_equal(z1[i], z0[i + 1], String("normal element ", i))
    # CROSS_NORMAL[3] to [5] hold a miss at element 20, so their cut fills in test_normal take
    # the fallback at and after the cut.
    for c in range(3, 6):
        assert_equal(nm[find_case(nm, String("CROSS_NORMAL[", c, "]"))].number("n"), 64)


def test_width_follows_the_interface() raises:
    """The bounded interfaces name the draw width. Range 1000 from the key of seed 42 gives the
    values of CROSS_BELOW32[3] with u32 draws and CROSS_BELOW64[3] with u64 draws, and range 0
    returns 0 after one draw of that width."""
    var fb = load("fill_below.json")
    var w32 = values_of(fb, "CROSS_BELOW32[3]")
    var w64 = values_of(fb, "CROSS_BELOW64[3]")
    var g = Tandem.from_key(KEY42, 0)
    var o32 = unsafe_alloc[UInt32](64)
    var o64 = unsafe_alloc[UInt64](64)
    g.fill_below_u32(o32, 64, 1000)
    var g2 = Tandem.from_key(KEY42, 0)
    g2.fill_below_u64(o64, 64, 1000)
    var differ = 0
    for i in range(64):
        assert_equal(UInt64(o32.unsafe_offset(i).unsafe_load()), hex_value(w32[i]))
        assert_equal(o64.unsafe_offset(i).unsafe_load(), hex_value(w64[i]))
        differ += 1 if w32[i] != w64[i] else 0
    assert_true(differ > 0)
    o32.unsafe_free()
    o64.unsafe_free()
    for p in [0, 1, 33]:
        var a = Tandem.from_key(KEY42, UInt64(p))
        assert_equal(a.below_u32(0), UInt32(0))
        assert_equal(Int(a.position()), align(p, 32) + 32)
        var b = Tandem.from_key(KEY42, UInt64(p))
        assert_equal(b.below_u64(0), UInt64(0))
        assert_equal(Int(b.position()), align(p, 64) + 64)


def test_scalar_float32_normal_is_the_cos_half() raises:
    var nm = load("normal.json")
    var want = values_of(nm, "CROSS_NORMAL32[0]")
    var g = Tandem.from_key(KEY42, 0)
    for i in range(0, 32, 2):
        assert_equal(UInt64(bitcast[DType.uint32, 1](g.normal_f32())), hex_value(want[i]))
        assert_equal(Int(g.position()), 64 * (i // 2 + 1))


def test_float32_normal_pair_rule() raises:
    """Element 2j is the cos half and 2j + 1 the sin half of draws 2j and 2j + 1. The first 33
    values of CROSS_NORMALF equal CROSS_NORMAL32[1], and a start one pair later shifts the
    output by one pair. Odd n drops the sin half of the last pair but consumes both draws."""
    var nm = load("normal.json")
    var f = values_of(nm, "CROSS_NORMALF")
    var n1 = values_of(nm, "CROSS_NORMAL32[1]")
    for i in range(33):
        assert_equal(f[i], n1[i], String("element ", i))
    var n0 = values_of(nm, "CROSS_NORMAL32[0]")
    var n2 = values_of(nm, "CROSS_NORMAL32[2]")
    for i in range(31):
        assert_equal(n2[i], n0[i + 2], String("shifted element ", i))
    for c in range(5):
        var k = nm[find_case(nm, String("CROSS_NORMAL32[", c, "]"))].copy()
        assert_equal(k.number("n"), 33)
        assert_equal(expected_end(NORMAL_F32, k.number("start"), 33), align(k.number("start"), 32) + 32 * 2 * 17)
    assert_equal(align(0, 32) + 32 * 2 * 17, 1088)
    var g = Tandem.from_key(KEY42, 0)
    var out = unsafe_alloc[Float32](34)
    g.fill_normal_f32(out, 33)
    assert_equal(Int(g.position()), 1088)
    out.unsafe_free()


def test_empty_fills() raises:
    """The seven cases with n = 0 start at 33. A uniform, f64 normal or choice fill aligns to
    the draw width, and a bounded, f32 normal or exponential fill leaves the position."""
    var empty = 0
    for file in ["fill_below.json", "normal.json", "exponential.json", "choice.json"]:
        for c in load(file):
            if c.number("n") != 0:
                continue
            var kind = kind_of(c)
            var b = Buffers(1)
            var g = generator(c)
            assert_equal(c.number("start"), 33)
            var table = ChoiceTable([1.0]) if kind != CHOICE else table_of(c)
            fill_piece(g, kind, c, table, b, 0, 0)
            assert_equal(Int(g.position()), c.number("end"), c.text("id"))
            b.free()
            empty += 1
    assert_equal(empty, 7)
    var g = Tandem.from_key(KEY42, 33)
    g.fill_u32(unsafe_alloc[UInt32](1), 0)
    assert_equal(Int(g.position()), 64)
    g = Tandem.from_key(KEY42, 33)
    g.fill_u8(unsafe_alloc[UInt8](1), 0)
    assert_equal(Int(g.position()), 40)


def test_choice_checklist() raises:
    var cases = load("choice.json")
    var a = values_of(cases, "CROSS_CHOICE[0]")
    var s = values_of(cases, "CROSS_CHOICE[1]")
    assert_equal(cases[find_case(cases, "CROSS_CHOICE[0]")].number("start"), 0)
    assert_equal(cases[find_case(cases, "CROSS_CHOICE[1]")].number("start"), 1)
    for i in range(len(s) - 1):
        assert_equal(s[i], a[i + 1], String("element ", i))
    # m = 1 returns 0 and still consumes 64 bits.
    var one = ChoiceTable([0.25])
    var g = Tandem.from_key(KEY42, 5)
    assert_equal(g.choice(one), UInt32(0))
    assert_equal(Int(g.position()), 128)
    # A zero weight, written as -0.0 too, never appears.
    var t = ChoiceTable([-0.0, 3.0, 0.0, 1.0])
    var out = unsafe_alloc[UInt32](5000)
    var r = Tandem(9)
    r.fill_choice(out, 5000, t)
    for i in range(5000):
        var v = out.unsafe_offset(i).unsafe_load()
        assert_true(v == 1 or v == 3)
    out.unsafe_free()


def raises_on(weights: List[Float64]) -> Bool:
    try:
        _ = ChoiceTable(weights)
        return False
    except:
        return True


def test_choice_rejects_invalid_weights() raises:
    var inf = bitcast[DType.float64, 1](UInt64(0x7FF0000000000000))
    assert_true(raises_on(List[Float64]()))
    assert_true(raises_on([1.0, -1.0]))
    assert_true(raises_on([1.0, inf]))
    assert_true(raises_on([1.0, -inf]))
    assert_true(raises_on([1.0, inf - inf]))
    assert_true(raises_on([0.0, -0.0]))
    assert_true(raises_on([0.0]))
    assert_false(raises_on([5e-324]))


def test_choice_matches_its_weights() raises:
    """Chi-square of 1e6 draws against nine positive weights, 8 degrees of freedom: the 0.0005
    and 0.9995 quantiles are 0.71 and 27.87. The zero weight never appears."""
    var w: List[Float64] = [0.5, 3.0, 0.0, 1.0, 7.0, 2.25, 0.1, 4.0, 1.0, 6.0]
    var n = 1000000
    var out = unsafe_alloc[UInt32](n)
    var r = Tandem(2028)
    r.fill_choice(out, n, ChoiceTable(w))
    var counts = List[Int](length=len(w), fill=0)
    for i in range(n):
        counts[Int(out.unsafe_offset(i).unsafe_load())] += 1
    out.unsafe_free()
    assert_equal(counts[2], 0)
    var total = Float64(0)
    for x in w:
        total += x
    var chi2 = Float64(0)
    for i in range(len(w)):
        if w[i] > 0.0:
            var e = Float64(n) * w[i] / total
            chi2 += (Float64(counts[i]) - e) * (Float64(counts[i]) - e) / e
    assert_true(chi2 > 0.71 and chi2 < 27.87, String("chi2 ", chi2))


# ---- Hashes ----------------------------------------------------------------------------------

def stream_bytes(kind: String, mut g: Tandem, n: Int) raises -> Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]:
    """The bytes of a uniform fill of n elements, little endian, in the layout of hashes.json."""
    var buf = unsafe_alloc[UInt8](16 * n + 16)
    var size = 0
    if kind == "UInt32":
        g.fill_u32(buf.unsafe_bitcast[UInt32](), n)
        size = 4 * n
    elif kind == "UInt64":
        g.fill_u64(buf.unsafe_bitcast[UInt64](), n)
        size = 8 * n
    elif kind == "Float64":
        g.fill_f64(buf.unsafe_bitcast[Float64](), n)
        size = 8 * n
    elif kind == "Float32":
        g.fill_f32(buf.unsafe_bitcast[Float32](), n)
        size = 4 * n
    elif kind == "UInt8":
        g.fill_u8(buf, n)
        size = n
    elif kind == "Bool":
        g.fill_bool(buf.unsafe_bitcast[Bool](), n)
        size = n
    elif kind == "UInt128":
        g.fill_u128(buf.unsafe_bitcast[UInt128](), n)
        size = 16 * n
    elif kind == "ComplexF64":
        g.fill_c64(buf.unsafe_bitcast[Float64](), n)
        size = 16 * n
    elif kind == "ComplexF32":
        g.fill_c32(buf.unsafe_bitcast[Float32](), n)
        size = 8 * n
    elif kind == "Float16":
        g.fill_f16_bits(buf.unsafe_bitcast[UInt16](), n)
        size = 2 * n
    elif kind == "Char":
        g.fill_char(buf.unsafe_bitcast[UInt32](), n)
        size = 4 * n
    else:
        raise Error(String("unknown type ", kind))
    return (buf, size)


def test_stream_hashes() raises:
    """The SHA-256 of every uniform stream of hashes.json, from the key and K of its entry. These
    cross 128-bit blocks, 1024-bit rows and chunks, and k1234_K8_u32 crosses a chunk every 8 rows."""
    var seen = 0
    for c in load("hashes.json"):
        if not c.has("file"):
            continue
        var g = Tandem.from_key(c.key(), UInt64(c.number("start")), UInt32(c.number("K")))
        var r = stream_bytes(c.text("type"), g, c.number("n"))
        assert_equal(r[1], c.number("bytes"), c.text("file"))
        var h = Sha256()
        h.update(r[0], r[1])
        assert_equal(h.hexdigest(), c.text("sha256"), c.text("file"))
        r[0].unsafe_free()
        seen += 1
    assert_equal(seen, 12)


def test_dump_hashes() raises:
    """The FNV-1a and SHA-256 of the long derived dumps: for each start a generator from the key
    and K runs the listed fills in order, and the bytes of all starts are hashed in sequence."""
    var buf = unsafe_alloc[UInt8](8 * 1000000)
    var seen = 0
    for c in load("hashes.json"):
        if c.has("file"):
            continue
        var fnv = Fnv1a()
        var sha = Sha256()
        var total = 0
        var last = UInt64(0)
        for start in c.numbers("starts"):
            var g = Tandem.from_key(c.key(), UInt64(start), UInt32(c.number("K")))
            for d in c.draws():
                var n = d[1]
                var size = 0
                if d[0] == "fill_normal_f64":
                    g.fill_normal_f64(buf.unsafe_bitcast[Float64](), n)
                    size = 8 * n
                elif d[0] == "fill_normal_f32":
                    g.fill_normal_f32(buf.unsafe_bitcast[Float32](), n)
                    size = 4 * n
                elif d[0] == "fill_exponential_f64":
                    g.fill_exponential_f64(buf.unsafe_bitcast[Float64](), n)
                    size = 8 * n
                elif d[0] == "fill_exponential_f32":
                    g.fill_exponential_f32(buf.unsafe_bitcast[Float32](), n)
                    size = 4 * n
                else:
                    raise Error(String("unknown draw ", d[0]))
                fnv.update(buf, size)
                sha.update(buf, size)
                total += size
            last = g.position()
        var id = c.text("id")
        assert_equal(total, c.number("bytes"), id)
        assert_equal(fnv.h, hex_value(c.text("fnv1a")), id)
        if c.has("sha256"):
            assert_equal(sha.hexdigest(), c.text("sha256"), id)
        if c.has("end"):
            assert_equal(Int(last), c.number("end"), id)
        seen += 1
    buf.unsafe_free()
    assert_equal(seen, 5)


# ---- Positions and blocks --------------------------------------------------------------------

def test_random_access_equals_the_sequential_fill() raises:
    """at_* at any element equals the sequential fill, across block, row and chunk boundaries:
    2048 words at K = 8 cross two chunks."""
    for k in [8, 32]:
        var g = Tandem.from_key(KEY42, 0, UInt32(k))
        var words = unsafe_alloc[UInt32](2048)
        g.fill_u32(words, 2048)
        var h = Tandem.from_key(KEY42, 0, UInt32(k))
        var longs = unsafe_alloc[UInt64](1024)
        h.fill_u64(longs, 1024)
        var r = Tandem.from_key(KEY42, 0, UInt32(k))
        for i in range(2048):
            assert_equal(r.at_u32(UInt64(i)), words.unsafe_offset(i).unsafe_load(), String("K ", k, " u32 ", i))
        for i in range(1024):
            assert_equal(r.at_u64(UInt64(i)), longs.unsafe_offset(i).unsafe_load(), String("K ", k, " u64 ", i))
        words.unsafe_free()
        longs.unsafe_free()


def test_complex_draw_spans_a_block() raises:
    """A complex draw whose real part ends a block takes its imaginary part from the next block."""
    var g = Tandem.from_key(KEY42, 64)
    var z = g.next_c64()
    assert_equal(Int(g.position()), 192)
    var re = Tandem.from_key(KEY42, 64)
    var im = Tandem.from_key(KEY42, 128)
    assert_equal(z[0], re.next_f64())
    assert_equal(z[1], im.next_f64())
    var w = Tandem.from_key(KEY42, 96)
    var y = w.next_c32()
    var re32 = Tandem.from_key(KEY42, 96)
    var im32 = Tandem.from_key(KEY42, 128)
    assert_equal(y[0], re32.next_f32())
    assert_equal(y[1], im32.next_f32())


def test_position_bounds() raises:
    """A start of 2^63 - 1 is accepted, and 2^63 and 2^64 - 1 are rejected without a change of
    state. A u64 draw at 2^63 - 1 aligns to 2^63. A fill whose end reaches 2^64 fails before
    it writes or moves, whatever its type."""
    var top = UInt64(1) << 63
    var g = Tandem.from_key(KEY42, top - 1)
    assert_equal(g.position(), top - 1)
    for bad in [top, UInt64.MAX]:
        var failed = False
        try:
            _ = Tandem.from_key(KEY42, bad)
        except:
            failed = True
        assert_true(failed)
        var before = g.copy()
        failed = False
        try:
            g.set_position(bad)
        except:
            failed = True
        assert_true(failed)
        assert_true(g == before)
    _ = g.next_u64()
    assert_equal(g.position(), top + 64)

    var guard = unsafe_alloc[UInt64](1)
    guard.unsafe_store(7)
    var guard32 = unsafe_alloc[UInt32](1)
    guard32.unsafe_store(7)
    var gf = unsafe_alloc[Float64](1)
    gf.unsafe_store(7.0)
    var gs = unsafe_alloc[Float32](1)
    gs.unsafe_store(7.0)
    var table = ChoiceTable([1.0, 2.0])
    var wide = 1 << 58
    var longs = 1 << 57
    var cases = 0
    for which in range(10):
        var h = Tandem.from_key(KEY42, top - 1)
        var failed = False
        try:
            if which == 0:
                h.fill_u64(guard, longs)
            elif which == 1:
                h.fill_u32(guard32, wide)
            elif which == 2:
                h.fill_f64(gf, longs)
            elif which == 3:
                h.fill_below_u32(guard32, wide, 10)
            elif which == 4:
                h.fill_below_u64(guard, longs, 10)
            elif which == 5:
                h.fill_normal_f64(gf, longs)
            elif which == 6:
                h.fill_normal_f32(gs, wide)
            elif which == 7:
                h.fill_exponential_f64(gf, longs)
            elif which == 8:
                h.fill_choice(guard32, longs, table)
            else:
                h.fill_u128(unsafe_alloc[UInt128](1), longs // 2)
        except:
            failed = True
        assert_true(failed, String("fill ", which))
        assert_equal(h.position(), top - 1, String("fill ", which))
        cases += 1
    assert_equal(guard.unsafe_load(), UInt64(7))
    assert_equal(guard32.unsafe_load(), UInt32(7))
    assert_equal(gf.unsafe_load(), 7.0)
    assert_equal(gs.unsafe_load(), Float32(7.0))
    assert_equal(cases, 10)
    guard.unsafe_free()
    guard32.unsafe_free()
    gf.unsafe_free()
    gs.unsafe_free()


def main() raises:
    test_below_and_fill_below()
    test_normal()
    test_exponential()
    test_choice()
    test_fallback_is_keyed_by_the_global_draw_index()
    test_width_follows_the_interface()
    test_scalar_float32_normal_is_the_cos_half()
    test_float32_normal_pair_rule()
    test_empty_fills()
    test_choice_checklist()
    test_choice_rejects_invalid_weights()
    test_choice_matches_its_weights()
    test_stream_hashes()
    test_dump_hashes()
    test_random_access_equals_the_sequential_fill()
    test_complex_draw_spans_a_block()
    test_position_bounds()
    print("mojo conformance: ok")
