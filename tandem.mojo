# Tandem8x32 in Mojo, after https://github.com/tandem-rng/spec: the building blocks, the Tandem
# generator with scalar draws, CPU fills over eight SIMD lanes, and GPU fills with one thread
# per chunk.
# Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.

from std.bit import count_leading_zeros, count_trailing_zeros, rotate_bits_left
from std.math import fma, iota, sqrt
from std.memory import bitcast, stack_allocation
from std.sys import bit_width_of, has_accelerator
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

comptime CLOCK_WEYL: UInt32 = 0x9E3779B9
comptime DOMAIN_STREAM: UInt32 = 0x9E3779B9
comptime DOMAIN_SPLIT: UInt32 = 0xBB67AE85
comptime DOMAIN_FORK: UInt32 = 0xD2511F53
comptime DOMAIN_FOLD: UInt32 = 0xCD9E8D57
comptime DOMAIN_SEED: UInt32 = 0xA54FF53A
comptime AUX_STREAM: UInt32 = 0x94D049BB
comptime RC = SIMD[DType.uint32, 8](
    0xD17CC1B7, 0xA7220A94, 0xFE13ABE8, 0xFA9A6EE0,
    0xEDB14ACC, 0x9E21C820, 0xFF28B1D5, 0xEF5DE2B0,
)


@fieldwise_init
struct Lanes[W: Int](Copyable, Movable):
    """W chunk states, word-major: every word of the state is one vector over the lanes."""

    comptime V = SIMD[DType.uint32, Self.W]
    var o0: Self.V
    var o1: Self.V
    var o2: Self.V
    var o3: Self.V
    var h0: Self.V
    var h1: Self.V
    var h2: Self.V
    var h3: Self.V

    @always_inline
    def step(mut self):
        """The step T: mix, clock, feedback.

        Vectors take the low word of each product as a plain 32-bit multiply and only the high
        word as the widening one, because taking both from one 64-bit product made LLVM emit two
        widening multiplies. The GPU kernel runs one lane per thread, and there separate multiplies
        for the low and the high word gave a kernel image that the driver rejects
        (CUDA_ERROR_INVALID_IMAGE), so one lane takes both words from the one 64-bit product."""
        comptime if Self.W == 1:
            self.step_wide()
        else:
            self.step_split()

    @always_inline
    def step_wide(mut self):
        var m0 = (self.h0 | 1).cast[DType.uint64]()
        var m1 = (self.h1 | 1).cast[DType.uint64]()
        var p0 = self.o0.cast[DType.uint64]() * m0
        var p1 = self.o2.cast[DType.uint64]() * m1
        var lo0 = p0.cast[DType.uint32]()
        var hi0 = (p0 >> 32).cast[DType.uint32]()
        var lo1 = p1.cast[DType.uint32]()
        var hi1 = (p1 >> 32).cast[DType.uint32]()
        self.finish_step(lo0, hi0, lo1, hi1)

    @always_inline
    def step_split(mut self):
        var m0 = self.h0 | 1
        var m1 = self.h1 | 1
        var lo0 = self.o0 * m0
        var lo1 = self.o2 * m1
        var hi0 = ((self.o0.cast[DType.uint64]() * m0.cast[DType.uint64]()) >> 32).cast[DType.uint32]()
        var hi1 = ((self.o2.cast[DType.uint64]() * m1.cast[DType.uint64]()) >> 32).cast[DType.uint32]()
        self.finish_step(lo0, hi0, lo1, hi1)

    @always_inline
    def finish_step(mut self, lo0: Self.V, hi0: Self.V, lo1: Self.V, hi1: Self.V):
        var n0 = self.o1 ^ hi1 ^ lo1
        var n1 = rotate_bits_left[16](lo1) ^ self.h2
        var n2 = self.o3 ^ hi0 ^ lo0
        var n3 = rotate_bits_left[16](lo0) ^ self.h3
        self.h0 ^= rotate_bits_left[7](self.h1)
        self.h1 ^= rotate_bits_left[13](self.h2)
        self.h2 ^= rotate_bits_left[22](self.h3)
        self.h3 ^= rotate_bits_left[3](self.h0)
        self.h0 = (self.h0 + CLOCK_WEYL) ^ n0
        self.o0, self.o1, self.o2, self.o3 = n0, n1, n2, n3

    @always_inline
    def f(mut self):
        """The seeding function F: eight rounds of T, a round constant, and a half swap."""

        for r in range(8):
            self.step()
            self.o0 ^= RC[r]
            self.o0, self.h0 = self.h0, self.o0
            self.o1, self.h1 = self.h1, self.o1
            self.o2, self.h2 = self.h2, self.o2
            self.o3, self.h3 = self.h3, self.o3

    @staticmethod
    @always_inline
    def keyed(key: SIMD[DType.uint32, 4], counter: Self.V, hi: Self.V, domain: UInt32, aux: UInt32) -> Self:
        """F(key, counter, domain, aux) over W counters given as low and high words."""
        var s = Self(counter, hi, Self.V(domain), Self.V(aux), Self.V(key[0]), Self.V(key[1]), Self.V(key[2]), Self.V(key[3]))
        s.f()
        return s^


comptime State = Lanes[1]


def t(mut o: SIMD[DType.uint32, 4], mut h: SIMD[DType.uint32, 4]):
    """T on one state, for the specification's vectors."""
    var s = State(o[0], o[1], o[2], o[3], h[0], h[1], h[2], h[3])
    s.step()
    o = SIMD[DType.uint32, 4](s.o0[0], s.o1[0], s.o2[0], s.o3[0])
    h = SIMD[DType.uint32, 4](s.h0[0], s.h1[0], s.h2[0], s.h3[0])


def f(mut o: SIMD[DType.uint32, 4], mut h: SIMD[DType.uint32, 4]):
    """F on one state."""
    var s = State(o[0], o[1], o[2], o[3], h[0], h[1], h[2], h[3])
    s.f()
    o = SIMD[DType.uint32, 4](s.o0[0], s.o1[0], s.o2[0], s.o3[0])
    h = SIMD[DType.uint32, 4](s.h0[0], s.h1[0], s.h2[0], s.h3[0])


def f_keyed(key: SIMD[DType.uint32, 4], counter: UInt64, domain: UInt32, aux: UInt32) -> Tuple[SIMD[DType.uint32, 4], SIMD[DType.uint32, 4]]:
    var o = SIMD[DType.uint32, 4](UInt32(counter & 0xFFFFFFFF), UInt32(counter >> 32), domain, aux)
    var h = key
    f(o, h)
    return (o, h)


def block(key: SIMD[DType.uint32, 4], c: UInt64, j: UInt32) -> SIMD[DType.uint32, 4]:
    """Block B(c, j): the exposed half of chunk c after j + 1 steps."""
    var oh = f_keyed(key, c, DOMAIN_STREAM, AUX_STREAM)
    var s = State(oh[0][0], oh[0][1], oh[0][2], oh[0][3], oh[1][0], oh[1][1], oh[1][2], oh[1][3])
    for _ in range(Int(j) + 1):
        s.step()
    return SIMD[DType.uint32, 4](s.o0[0], s.o1[0], s.o2[0], s.o3[0])


def seed(z: UInt128) -> SIMD[DType.uint32, 4]:
    """The specification's seed whitening."""
    var o = SIMD[DType.uint32, 4](0, 0, DOMAIN_SEED, 0)
    var h = SIMD[DType.uint32, 4](UInt32(z & 0xFFFFFFFF), UInt32((z >> 32) & 0xFFFFFFFF), UInt32((z >> 64) & 0xFFFFFFFF), UInt32(z >> 96))
    f(o, h)
    return o


def split(key: SIMD[DType.uint32, 4], index: UInt64) -> SIMD[DType.uint32, 4]:
    var oh = f_keyed(key, index >> 1, DOMAIN_SPLIT, 0)
    return oh[1] if (index & 1) == 1 else oh[0]


def sub(key: SIMD[DType.uint32, 4], purpose: UInt64) -> SIMD[DType.uint32, 4]:
    return f_keyed(key, purpose, DOMAIN_FOLD, 0)[0]


def fork_key(key: SIMD[DType.uint32, 4], block: UInt64, index: UInt64) -> SIMD[DType.uint32, 4]:
    """Child index of a fork at the given block."""
    var oh = f_keyed(key, block, DOMAIN_FORK, UInt32((index >> 1) & 0xFFFFFFFF))
    return oh[1] if (index & 1) == 1 else oh[0]


# ---- Generator -----------------------------------------------------------------------------

comptime DEFAULT_K: UInt32 = 32

# Purposes reserved for the fallback generators of the bounded fills.
comptime PURPOSE_BELOW32: UInt64 = 0x424C573332
comptime PURPOSE_BELOW64: UInt64 = 0x424C573634
comptime Row = SIMD[DType.uint32, 32]


@always_inline
def head_elements[W: Int](p: UInt64) -> Int:
    """Elements of W bits up to the next row boundary from the W-aligned bit position p, or 0 on a boundary."""
    return Int(((1024 - (p & 1023)) & 1023) // UInt64(W))


@always_inline
def align(pos: UInt64, w: UInt64) -> UInt64:
    """Round a bit position up to a multiple of the power of two w."""
    return (pos + w - 1) & ~(w - 1)


@always_inline
def to_f64(raw: UInt64) -> Float64:
    return Float64(raw >> 11) * 1.1102230246251565e-16  # 2^-53


@always_inline
def to_f32(raw: UInt32) -> Float32:
    return Float32(raw >> 8) * 5.9604645e-08  # 2^-24


def to_f16_bits(raw: UInt16) -> UInt16:
    """Bits of the binary16 value (raw >> 5) * 2^-11, which is exact: zero or a normal half."""
    var k = UInt32(raw >> 5)
    if k == 0:
        return 0
    var m = UInt32(31) - UInt32(count_leading_zeros(k))
    return UInt16(((m + 4) << 10) | ((k << (10 - m)) & 0x3FF))


def to_char(raw: UInt64) -> UInt32:
    """floor(raw * 1112064 / 2^64) as a code point that skips the surrogates."""
    var u = UInt32((UInt128(raw) * 1112064) >> 64)
    return u if u < 0xD800 else u + 0x800


# ---- Normals on SIMD lanes -----------------------------------------------------------------
# The algorithm and coefficients of tandem-c's normal loop, with an explicit fma for every
# multiply-add and no other contraction, so that every port gives the same bytes. A libm call
# per element would dominate the fill, and std.math's Float64 log is off by 1e-10.

@always_inline
def neg2_log_f64[W: Int](a: SIMD[DType.float64, W]) -> SIMD[DType.float64, W]:
    """-2 ln(1 - a) for a in [0, 1): 1 - a is exact. Its mantissa m in [1/sqrt 2, sqrt 2) comes
    from the exponent bits, then ln m = 2 s P(s^2) with s = (m - 1) / (m + 1)."""
    var bits = bitcast[DType.uint64, W](1.0 - a)
    var ix = bits + 0x00095F6200000000
    var nk = (1023 - (ix >> 52).cast[DType.int64]()).cast[DType.float64]()
    var m = bitcast[DType.float64, W]((ix & 0x000FFFFFFFFFFFFF) + 0x3FE6A09E00000000)
    var s = (m - 1.0) / (m + 1.0)
    var zz = s * s
    var p = SIMD[DType.float64, W](0.08312363319426472)
    p = fma(zz, p, 0.09070001083303751)
    p = fma(zz, p, 0.11111433317907482)
    p = fma(zz, p, 0.14285712049336274)
    p = fma(zz, p, 0.2000000000566491)
    p = fma(zz, p, 0.33333333333331017)
    p = fma(zz, p, 1.0)
    # ln 2 is split so that nk * ln2_hi is exact.
    return fma(nk, 3.816429394731813e-10, fma(nk, 1.3862943607382476, (s * -4.0) * p))


@always_inline
def sincos_2pi_f64[W: Int](b: SIMD[DType.float64, W]) -> Tuple[SIMD[DType.float64, W], SIMD[DType.float64, W]]:
    """(sin, cos) of 2 pi b for b in [0, 1). b - q/4 for the nearest quarter turn q is exact,
    which leaves the angle th in [-pi/4, pi/4] for the polynomials."""
    var q = (b * 4.0 + 0.5).cast[DType.int64]()
    var f = fma(-(q.cast[DType.float64]()), 0.25, b)
    var th = f * 6.283185307179586
    var w = th * th
    var hs = SIMD[DType.float64, W](1.5914650986900946e-10)
    hs = fma(w, hs, -2.5051097984389413e-08)
    hs = fma(w, hs, 2.755731600073921e-06)
    hs = fma(w, hs, -0.00019841269836630226)
    hs = fma(w, hs, 0.008333333333330813)
    hs = fma(w, hs, -0.16666666666666669)
    var hc = SIMD[DType.float64, W](2.0665708703855164e-09)
    hc = fma(w, hc, -2.7555858522576447e-07)
    hc = fma(w, hc, 2.480158263811954e-05)
    hc = fma(w, hc, -0.0013888888882156126)
    hc = fma(w, hc, 0.04166666666663108)
    hc = fma(w, hc, -0.4999999999999997)
    var sn = th * fma(w, hs, 1.0)
    var cs = fma(w, hc, 1.0)
    # Odd q swaps sin and cos, bit 1 of q negates the sin, and bit 1 of q + 1 the cos.
    var qu = bitcast[DType.uint64, W](q)
    var swap = (qu & 1).eq(1)
    var sb = bitcast[DType.uint64, W](sn)
    var cb = bitcast[DType.uint64, W](cs)
    var xb = swap.select(sb, cb) ^ (((qu + 1) << 62) & 0x8000000000000000)
    var yb = swap.select(cb, sb) ^ ((qu << 62) & 0x8000000000000000)
    return (bitcast[DType.float64, W](yb), bitcast[DType.float64, W](xb))


@always_inline
def normal2_f64[W: Int](a: SIMD[DType.float64, W], b: SIMD[DType.float64, W]) -> Tuple[SIMD[DType.float64, W], SIMD[DType.float64, W]]:
    """Both halves of a Box-Muller step, cos first: z0 = r cos(2 pi b), z1 = r sin(2 pi b),
    r = sqrt(-2 log(1 - a)). Every operation is lane-wise, so any width gives the same bits."""
    var r = sqrt(neg2_log_f64(a))
    var sc = sincos_2pi_f64(b)
    return (r * sc[1], r * sc[0])


@always_inline
def neg2_log_f32[W: Int](a: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
    """The Float32 form of neg2_log_f64."""
    var bits = bitcast[DType.uint32, W](1.0 - a)
    var ix = bits + 0x004AFB0D
    var nk = (127 - (ix >> 23).cast[DType.int32]()).cast[DType.float32]()
    var m = bitcast[DType.float32, W]((ix & 0x007FFFFF) + 0x3F3504F3)
    var s = (m - 1.0) / (m + 1.0)
    var zz = s * s
    var p = SIMD[DType.float32, W](0.14275366)
    p = fma(zz, p, 0.20000061)
    p = fma(zz, p, 0.33333334)
    p = fma(zz, p, 1.0)
    return fma(nk, 2.857213530660374e-06, fma(nk, 1.38629150390625, (s * -4.0) * p))


@always_inline
def sincos_2pi_f32[W: Int](b: SIMD[DType.float32, W]) -> Tuple[SIMD[DType.float32, W], SIMD[DType.float32, W]]:
    """The Float32 form of sincos_2pi_f64, with 2 pi as a pair of floats so that the angle is
    good to the last bit."""
    var q = (b * 4.0 + 0.5).cast[DType.int32]()
    var f = fma(-(q.cast[DType.float32]()), 0.25, b)
    var th = fma(f, -1.7484555e-7, f * 6.2831855)
    var w = th * th
    var hs = SIMD[DType.float32, W](2.72499e-06)
    hs = fma(w, hs, -0.00019840087)
    hs = fma(w, hs, 0.008333332)
    hs = fma(w, hs, -0.16666667)
    var hc = SIMD[DType.float32, W](2.4463761e-05)
    hc = fma(w, hc, -0.0013887589)
    hc = fma(w, hc, 0.04166665)
    hc = fma(w, hc, -0.5)
    var sn = th * fma(w, hs, 1.0)
    var cs = fma(w, hc, 1.0)
    var qu = bitcast[DType.uint32, W](q)
    var swap = (qu & 1).eq(1)
    var sb = bitcast[DType.uint32, W](sn)
    var cb = bitcast[DType.uint32, W](cs)
    var xb = swap.select(sb, cb) ^ (((qu + 1) << 30) & 0x80000000)
    var yb = swap.select(cb, sb) ^ ((qu << 30) & 0x80000000)
    return (bitcast[DType.float32, W](yb), bitcast[DType.float32, W](xb))


@always_inline
def normal2_f32[W: Int](a: SIMD[DType.float32, W], b: SIMD[DType.float32, W]) -> Tuple[SIMD[DType.float32, W], SIMD[DType.float32, W]]:
    """The Float32 form of normal2_f64."""
    var r = sqrt(neg2_log_f32(a))
    var sc = sincos_2pi_f32(b)
    return (r * sc[1], r * sc[0])


struct Cursor(Copyable, Movable):
    """Eight chunk states positioned at one row. Stepping inside a group costs one T per row.
    Any other move reseeds the group."""

    var lanes: Lanes[8]
    var at: UInt64
    var live: Bool

    def __init__(out self):
        self.lanes = Lanes[8](0, 0, 0, 0, 0, 0, 0, 0)
        self.at = 0
        self.live = False

    @always_inline
    def seek(mut self, key: SIMD[DType.uint32, 4], K: UInt32, row: UInt64):
        if self.live and self.at == row:
            return
        var shift = UInt64(count_trailing_zeros(K))
        var mask = UInt64(K) - 1
        if self.live and row > self.at and (row >> shift) == (self.at >> shift):
            for _ in range(Int(row - self.at)):
                self.lanes.step()
        else:
            var c0 = 8 * (row >> shift)
            var lo = SIMD[DType.uint32, 8](0, 1, 2, 3, 4, 5, 6, 7) + UInt32(c0 & 0xFFFFFFFF)
            self.lanes = Lanes[8].keyed(key, lo, SIMD[DType.uint32, 8](UInt32(c0 >> 32)), DOMAIN_STREAM, AUX_STREAM)
            for _ in range(Int(row & mask) + 1):
                self.lanes.step()
        self.at = row
        self.live = True


def below_retry_u32(key: SIMD[DType.uint32, 4], K: UInt32, n: UInt32, e: UInt64) -> UInt32:
    """Lemire's rule on the draws of key.sub(PURPOSE_BELOW32).split(e) from position 0."""
    var r = Tandem(trusted_key=key, position=0, k=K).sub(PURPOSE_BELOW32).split(e)
    var t = (UInt32(0) - n) % n
    while True:
        var m = UInt64(r.next_u32()) * UInt64(n)
        if UInt32(m & 0xFFFFFFFF) >= t:
            return UInt32(m >> 32)


def below_retry_u64(key: SIMD[DType.uint32, 4], K: UInt32, n: UInt64, e: UInt64) -> UInt64:
    var r = Tandem(trusted_key=key, position=0, k=K).sub(PURPOSE_BELOW64).split(e)
    var t = (UInt64(0) - n) % n
    while True:
        var m = UInt128(r.next_u64()) * UInt128(n)
        if UInt64(m & 0xFFFFFFFFFFFFFFFF) >= t:
            return UInt64(m >> 64)


struct Tandem(Copyable, Movable, Equatable):
    """A generator: key, bit position, chunk length K, and a cache of the current row.
    Equality covers the transport form only. Every draw aligns the position to the width of
    its type, reads, and advances, so draws of mixed widths agree with the other ports."""

    var key: SIMD[DType.uint32, 4]
    var pos: UInt64
    var k: UInt32
    var cur: Cursor
    var words: Row
    var words_row: UInt64
    var words_valid: Bool

    def __init__(out self, z: UInt128, k: UInt32 = DEFAULT_K) raises:
        """The generator of an integer seed after the specification's whitening, at position 0."""
        self = Self.from_key(seed(z), 0, k)

    def __init__(out self, *, trusted_key: SIMD[DType.uint32, 4], position: UInt64, k: UInt32):
        self.key = trusted_key
        self.pos = position
        self.k = k
        self.cur = Cursor()
        self.words = Row(0)
        self.words_row = 0
        self.words_valid = False

    @staticmethod
    def from_key(key: SIMD[DType.uint32, 4], position: UInt64, k: UInt32 = DEFAULT_K) raises -> Self:
        """A generator from its transport form. K must be a power of two in 1 to 65536."""
        if k == 0 or (k & (k - 1)) != 0 or k > 65536:
            raise Error("chunk length must be a power of two in 1..=65536")
        return Self(trusted_key=key, position=position, k=k)

    def __eq__(self, other: Self) -> Bool:
        return self.key == other.key and self.pos == other.pos and self.k == other.k

    def position(self) -> UInt64:
        return self.pos

    def set_position(mut self, position: UInt64):
        self.pos = position

    # Rows ----------------------------------------------------------------------------------

    @always_inline
    def load_row(mut self, row: UInt64):
        if self.words_valid and self.words_row == row:
            return
        self.cur.seek(self.key, self.k, row)
        self.words = row_words(self.cur.lanes)
        self.words_row = row
        self.words_valid = True

    @always_inline
    def read[w: Int](mut self, p: UInt64) -> UInt64:
        """w bits (a power of two up to 64) at the aligned position p."""
        self.load_row(p >> 10)
        var i = Int((p >> 5) & 31)
        comptime if w == 64:
            return UInt64(self.words[i]) | (UInt64(self.words[i + 1]) << 32)
        else:
            return UInt64((self.words[i] >> UInt32(p & 31)) & (UInt32.MAX >> UInt32(32 - w)))

    @always_inline
    def next[w: Int](mut self) -> UInt64:
        var p = align(self.pos, UInt64(w))
        self.pos = p + UInt64(w)
        return self.read[w](p)

    # Scalar draws --------------------------------------------------------------------------

    def next_bool(mut self) -> Bool:
        return self.next[1]() != 0

    def next_u8(mut self) -> UInt8:
        return UInt8(self.next[8]())

    def next_u16(mut self) -> UInt16:
        return UInt16(self.next[16]())

    def next_u32(mut self) -> UInt32:
        return UInt32(self.next[32]())

    def next_u64(mut self) -> UInt64:
        return self.next[64]()

    def next_u128(mut self) -> UInt128:
        var p = align(self.pos, 128)
        self.pos = p + 128
        var lo = self.read[64](p)
        return UInt128(lo) | (UInt128(self.read[64](p + 64)) << 64)

    def next_i8(mut self) -> Int8:
        return Int8(self.next_u8().cast[DType.int8]())

    def next_i16(mut self) -> Int16:
        return Int16(self.next_u16().cast[DType.int16]())

    def next_i32(mut self) -> Int32:
        return Int32(self.next_u32().cast[DType.int32]())

    def next_i64(mut self) -> Int64:
        return Int64(self.next_u64().cast[DType.int64]())

    def next_i128(mut self) -> Int128:
        return Int128(self.next_u128().cast[DType.int128]())

    def next_f32(mut self) -> Float32:
        """Uniform in [0, 1) with 24 random bits: (raw >> 8) * 2^-24."""
        return to_f32(self.next_u32())

    def next_f64(mut self) -> Float64:
        """Uniform in [0, 1) with 53 random bits: (raw >> 11) * 2^-53."""
        return to_f64(self.next_u64())

    def next_f16_bits(mut self) -> UInt16:
        """The IEEE binary16 bits of a uniform draw in [0, 1): (raw >> 5) * 2^-11."""
        return to_f16_bits(self.next_u16())

    def next_c16_bits(mut self) -> SIMD[DType.uint16, 2]:
        """Complex binary16 bits as (re, im): two draws, real part first."""
        var re = self.next_f16_bits()
        return SIMD[DType.uint16, 2](re, self.next_f16_bits())

    def next_c32(mut self) -> SIMD[DType.float32, 2]:
        var re = self.next_f32()
        return SIMD[DType.float32, 2](re, self.next_f32())

    def next_c64(mut self) -> SIMD[DType.float64, 2]:
        var re = self.next_f64()
        return SIMD[DType.float64, 2](re, self.next_f64())

    def next_char(mut self) -> UInt32:
        """A uniform Unicode scalar value as its code point, from 64 stream bits."""
        return to_char(self.next_u64())

    # Fills ---------------------------------------------------------------------------------

    @always_inline
    def fill[kind: Int, W: Int, T: DType, origin: Origin[mut=True]](mut self, dst: Pointer[Scalar[T], origin], n: Int):
        """The values n scalar draws would produce, written row by row. The cache stays on the
        last row written."""
        var p = align(self.pos, UInt64(W))
        self.pos = p + UInt64(W) * UInt64(n)
        if n > 0:
            fill_rows[kind, W, T](self.cur, self.key, self.k, p, n, dst.unsafe_origin_cast[MutAnyOrigin]())
            self.words_valid = False

    def fill_u8[origin: Origin[mut=True]](mut self, dst: Pointer[UInt8, origin], n: Int):
        self.fill[KIND_INT, 8, DType.uint8](dst, n)

    def fill_u16[origin: Origin[mut=True]](mut self, dst: Pointer[UInt16, origin], n: Int):
        self.fill[KIND_INT, 16, DType.uint16](dst, n)

    def fill_u32[origin: Origin[mut=True]](mut self, dst: Pointer[UInt32, origin], n: Int):
        self.fill[KIND_INT, 32, DType.uint32](dst, n)

    def fill_u64[origin: Origin[mut=True]](mut self, dst: Pointer[UInt64, origin], n: Int):
        self.fill[KIND_INT, 64, DType.uint64](dst, n)

    def fill_u128[origin: Origin[mut=True]](mut self, dst: Pointer[UInt128, origin], n: Int):
        """128-bit elements, each as its low then its high 64-bit draw."""
        self.pos = align(self.pos, 128)
        self.fill[KIND_INT, 64, DType.uint64](dst.unsafe_bitcast[UInt64](), 2 * n)

    def fill_i8[origin: Origin[mut=True]](mut self, dst: Pointer[Int8, origin], n: Int):
        self.fill[KIND_INT, 8, DType.int8](dst, n)

    def fill_i16[origin: Origin[mut=True]](mut self, dst: Pointer[Int16, origin], n: Int):
        self.fill[KIND_INT, 16, DType.int16](dst, n)

    def fill_i32[origin: Origin[mut=True]](mut self, dst: Pointer[Int32, origin], n: Int):
        self.fill[KIND_INT, 32, DType.int32](dst, n)

    def fill_i64[origin: Origin[mut=True]](mut self, dst: Pointer[Int64, origin], n: Int):
        self.fill[KIND_INT, 64, DType.int64](dst, n)

    def fill_i128[origin: Origin[mut=True]](mut self, dst: Pointer[Int128, origin], n: Int):
        self.pos = align(self.pos, 128)
        self.fill[KIND_INT, 64, DType.uint64](dst.unsafe_bitcast[UInt64](), 2 * n)

    def fill_f32[origin: Origin[mut=True]](mut self, dst: Pointer[Float32, origin], n: Int):
        self.fill[KIND_F32, 32, DType.float32](dst, n)

    def fill_f64[origin: Origin[mut=True]](mut self, dst: Pointer[Float64, origin], n: Int):
        self.fill[KIND_F64, 64, DType.float64](dst, n)

    def fill_f16_bits[origin: Origin[mut=True]](mut self, dst: Pointer[UInt16, origin], n: Int):
        self.fill[KIND_F16, 16, DType.uint16](dst, n)

    def fill_char[origin: Origin[mut=True]](mut self, dst: Pointer[UInt32, origin], n: Int):
        """Unicode scalar values as code points, 64 stream bits each."""
        self.fill[KIND_CHAR, 64, DType.uint32](dst, n)

    def fill_bool[origin: Origin[mut=True]](mut self, dst: Pointer[Bool, origin], n: Int):
        self.fill[KIND_BOOL, 1, DType.uint8](dst.unsafe_bitcast[UInt8](), n)

    def fill_c32[origin: Origin[mut=True]](mut self, dst: Pointer[Float32, origin], n: Int):
        """n complex values as interleaved (re, im): the f32 fill of length 2n."""
        self.fill_f32(dst, 2 * n)

    def fill_c64[origin: Origin[mut=True]](mut self, dst: Pointer[Float64, origin], n: Int):
        self.fill_f64(dst, 2 * n)

    def fill_c16_bits[origin: Origin[mut=True]](mut self, dst: Pointer[UInt16, origin], n: Int):
        self.fill_f16_bits(dst, 2 * n)

    # Random access -------------------------------------------------------------------------

    def at_u8(self, i: UInt64) -> UInt8:
        """Element i of the fill that would start here, without advancing."""
        var tmp = self.copy()
        return UInt8(tmp.read[8](align(self.pos, 8) + i * 8))

    def at_u16(self, i: UInt64) -> UInt16:
        var tmp = self.copy()
        return UInt16(tmp.read[16](align(self.pos, 16) + i * 16))

    def at_u32(self, i: UInt64) -> UInt32:
        var tmp = self.copy()
        return UInt32(tmp.read[32](align(self.pos, 32) + i * 32))

    def at_u64(self, i: UInt64) -> UInt64:
        var tmp = self.copy()
        return tmp.read[64](align(self.pos, 64) + i * 64)

    def at_f32(self, i: UInt64) -> Float32:
        return to_f32(self.at_u32(i))

    def at_f64(self, i: UInt64) -> Float64:
        return to_f64(self.at_u64(i))

    # Bounded integers and normals ----------------------------------------------------------
    # Not part of the specification. They follow the shared device core, tandem-cuda's
    # core.hpp, so every port returns the same integers. The normals are byte identical to
    # tandem-c and agree with the core's libm normals to a few ulps.

    def below_u32(mut self, n: UInt32) -> UInt32:
        """Uniform in 0..n by Lemire's multiply and reject on u32 draws. For n == 0 the result
        is 0 after one draw."""
        var m = UInt64(self.next_u32()) * UInt64(n)
        if UInt32(m & 0xFFFFFFFF) < n:
            var t = (UInt32(0) - n) % n
            while UInt32(m & 0xFFFFFFFF) < t:
                m = UInt64(self.next_u32()) * UInt64(n)
        return UInt32(m >> 32)

    def below_u64(mut self, n: UInt64) -> UInt64:
        """Uniform in 0..n by Lemire's multiply and reject on u64 draws."""
        var m = UInt128(self.next_u64()) * UInt128(n)
        if UInt64(m & 0xFFFFFFFFFFFFFFFF) < n:
            var t = (UInt64(0) - n) % n
            while UInt64(m & 0xFFFFFFFFFFFFFFFF) < t:
                m = UInt128(self.next_u64()) * UInt128(n)
        return UInt64(m >> 64)

    def fill_below_u32[origin: Origin[mut=True]](mut self, dst: Pointer[UInt32, origin], count: Int, n: UInt32):
        """Element i takes draw i of the u32 fill, and the fill consumes exactly count draws
        whatever is rejected, so rows fill independently. A rejected draw retries with Lemire's
        rule on the draws of key.sub(PURPOSE_BELOW32).split(g) at position 0, where g is the global
        draw index, the start position over 32 plus i, so chunked fills equal whole fills. Without a
        rejection the fill equals the scalar below_u32 calls. An empty fill moves nothing.

        The draws come in blocks that stay in L1. A block takes the multiply-high on SIMD lanes,
        keeps the lane-wise minimum of the low words, and rescans only a block whose minimum is
        below the threshold. Converting row by row on the integer pipes, as the u64 fill does,
        is slower here, because the 32-bit multiplies vectorize well."""
        comptime W = 16
        comptime BLOCK = 1024
        if count == 0:
            return
        var t = (UInt32(0) - n) % n if n != 0 else UInt32(0)
        var nv = SIMD[DType.uint32, W](n).cast[DType.uint64]()
        var draws = stack_allocation[BLOCK, UInt32]()
        var done = 0
        var first_draw = Int(align(self.pos, 32) // 32)
        var head = head_elements[32](align(self.pos, 32))
        while done < count:
            var m = min(BLOCK, count - done)
            if head > 0:
                m = min(m, head)
                head = 0
            self.fill_u32(draws, m)
            var j = 0
            var least = SIMD[DType.uint32, W](UInt32.MAX)
            while j + W <= m:
                var x = draws.unsafe_offset(j).unsafe_load[width=W]()
                var prod = x.cast[DType.uint64]() * nv
                least = min(least, prod.cast[DType.uint32]())
                dst.unsafe_offset(done + j).unsafe_store((prod >> 32).cast[DType.uint32]())
                j += W
            while j < m:
                var prod = UInt64(draws.unsafe_offset(j).unsafe_load()) * UInt64(n)
                dst.unsafe_offset(done + j).unsafe_store(UInt32(prod >> 32))
                least[0] = min(least[0], UInt32(prod & 0xFFFFFFFF))
                j += 1
            if least.reduce_min() < t:
                for k in range(m):
                    var prod = UInt64(draws.unsafe_offset(k).unsafe_load()) * UInt64(n)
                    if UInt32(prod & 0xFFFFFFFF) < t:
                        dst.unsafe_offset(done + k).unsafe_store(below_retry_u32(self.key, self.k, n, UInt64(first_draw + done + k)))
            done += m

    def fill_below_u64[origin: Origin[mut=True]](mut self, dst: Pointer[UInt64, origin], count: Int, n: UInt64):
        """The u64 form of fill_below_u32, with PURPOSE_BELOW64 and g = start position / 64 + i. It converts row by row, fused
        with the row generator."""
        if count == 0:
            return
        var p = align(self.pos, 64)
        self.pos = p + 64 * UInt64(count)
        var t = (UInt64(0) - n) % n if n != 0 else UInt64(0)
        fill_rows_below_u64(self.cur, self.key, self.k, p, count, dst.unsafe_origin_cast[MutAnyOrigin](), n, t)
        self.words_valid = False

    def normal2_f64(mut self) -> SIMD[DType.float64, 2]:
        """Both normals of one Box-Muller step from two f64 draws, cos half first."""
        var a = self.next_f64()
        var z = normal2_f64[1](SIMD[DType.float64, 1](a), SIMD[DType.float64, 1](self.next_f64()))
        return SIMD[DType.float64, 2](z[0][0], z[1][0])

    def normal2_f32(mut self) -> SIMD[DType.float32, 2]:
        """Both normals of one Box-Muller step from two f32 draws, cos half first."""
        var a = self.next_f32()
        var z = normal2_f32[1](SIMD[DType.float32, 1](a), SIMD[DType.float32, 1](self.next_f32()))
        return SIMD[DType.float32, 2](z[0][0], z[1][0])

    def normal_f64(mut self) -> Float64:
        """A standard normal: the cos half of normal2_f64, which consumes both draws."""
        return self.normal2_f64()[0]

    def normal_f32(mut self) -> Float32:
        """A standard normal: the cos half of normal2_f32, which consumes both draws."""
        return self.normal2_f32()[0]

    def fill_normal_f64[origin: Origin[mut=True]](mut self, dst: Pointer[Float64, origin], count: Int):
        """The flattened sequence of normal2_f64 calls, bit for bit. An odd count keeps the cos
        half of its last pair and still consumes both draws. An empty fill moves nothing.

        The draws come in blocks that stay in L1. Converting each row in registers, fused with
        the row generator, spills registers and measured slower."""
        comptime P = 4
        comptime BLOCK = 256
        var draws = stack_allocation[2 * BLOCK, Float64]()
        var pairs = count // 2 + count % 2
        var done = 0
        var head = head_elements[64](align(self.pos, 64))
        while pairs > 0:
            var m = min(BLOCK, pairs)
            if head > 1 and head % 2 == 0:
                m = min(m, head // 2)
            head = 0
            self.fill_f64(draws, 2 * m)
            var j = 0
            while j + P <= m and done + 2 * P <= count:
                var uv = draws.unsafe_offset(2 * j).unsafe_load[width=2 * P]().deinterleave()
                var z = normal2_f64[P](uv[0], uv[1])
                dst.unsafe_offset(done).unsafe_store(z[0].interleave(z[1]))
                done += 2 * P
                j += P
            if j < m:
                var padded = SIMD[DType.float64, 2 * P](0)
                for k in range(2 * (m - j)):
                    padded[k] = draws.unsafe_offset(2 * j + k).unsafe_load()
                var uv = padded.deinterleave()
                var z = normal2_f64[P](uv[0], uv[1])
                var out = z[0].interleave(z[1])
                for k in range(2 * (m - j)):
                    if done < count:
                        dst.unsafe_offset(done).unsafe_store(out[k])
                        done += 1
            pairs -= m

    def fill_normal_f32[origin: Origin[mut=True]](mut self, dst: Pointer[Float32, origin], count: Int):
        """The flattened sequence of normal2_f32 calls, as fill_normal_f64."""
        comptime P = 8
        comptime BLOCK = 256
        var draws = stack_allocation[2 * BLOCK, Float32]()
        var pairs = count // 2 + count % 2
        var done = 0
        var head = head_elements[32](align(self.pos, 32))
        while pairs > 0:
            var m = min(BLOCK, pairs)
            if head > 1 and head % 2 == 0:
                m = min(m, head // 2)
            head = 0
            self.fill_f32(draws, 2 * m)
            var j = 0
            while j + P <= m and done + 2 * P <= count:
                var uv = draws.unsafe_offset(2 * j).unsafe_load[width=2 * P]().deinterleave()
                var z = normal2_f32[P](uv[0], uv[1])
                dst.unsafe_offset(done).unsafe_store(z[0].interleave(z[1]))
                done += 2 * P
                j += P
            if j < m:
                var padded = SIMD[DType.float32, 2 * P](0)
                for k in range(2 * (m - j)):
                    padded[k] = draws.unsafe_offset(2 * j + k).unsafe_load()
                var uv = padded.deinterleave()
                var z = normal2_f32[P](uv[0], uv[1])
                var out = z[0].interleave(z[1])
                for k in range(2 * (m - j)):
                    if done < count:
                        dst.unsafe_offset(done).unsafe_store(out[k])
                        done += 1
            pairs -= m

    # Derived generators --------------------------------------------------------------------

    def split(self, index: UInt64) -> Self:
        """Child index by key alone: the same child for the same index, whatever the position."""
        return Self(trusted_key=split(self.key, index), position=0, k=self.k)

    def sub(self, purpose: UInt64) -> Self:
        """The generator of a named purpose, by key alone."""
        return Self(trusted_key=sub(self.key, purpose), position=0, k=self.k)

    def fork(mut self, n: Int) -> List[Self]:
        """Fork n children from the current block and move past it. Successive forks give fresh
        children. The parent advances once per call, for an empty batch too."""
        var b = self.pos >> 7
        self.pos = (b + 1) << 7
        var kids = List[Self](capacity=n)
        for i in range(n):
            kids.append(Self(trusted_key=fork_key(self.key, b, UInt64(i)), position=0, k=self.k))
        return kids^

    def fork(mut self) -> Self:
        """Fork one child, as fork(1) does."""
        var b = self.pos >> 7
        self.pos = (b + 1) << 7
        return Self(trusted_key=fork_key(self.key, b, 0), position=0, k=self.k)


# ---- CPU fill ------------------------------------------------------------------------------


@always_inline
def row_words(s: Lanes[8]) -> SIMD[DType.uint32, 32]:
    """The row in stream order: the 4x8 word-major state transposed to blocks of four words."""
    var ab = s.o0.interleave(s.o1)
    var cd = s.o2.interleave(s.o3)
    var lanes0to3 = ab.shuffle[0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23](cd)
    var lanes4to7 = ab.shuffle[8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31](cd)
    return lanes0to3.join(lanes4to7)


comptime KIND_INT = 0
comptime KIND_F32 = 1
comptime KIND_F64 = 2
comptime KIND_F16 = 3
comptime KIND_CHAR = 4
comptime KIND_BOOL = 5


@always_inline
def store_row[kind: Int, W: Int, T: DType](words: Row, dst: Pointer[Scalar[T], MutAnyOrigin]):
    """Convert one row to its 1024 / W elements and store them. After alignment every integer
    fill is the same little-endian byte stream, so the integer kinds only reinterpret it."""
    comptime N = 1024 // W
    comptime if kind == KIND_INT:
        dst.unsafe_store(bitcast[T, N](words))
    elif kind == KIND_F32:
        dst.unsafe_bitcast[Float32]().unsafe_store((words >> 8).cast[DType.float32]() * 5.9604645e-08)
    elif kind == KIND_F64:
        var x = bitcast[DType.uint64, 16](words)
        dst.unsafe_bitcast[Float64]().unsafe_store((x >> 11).cast[DType.float64]() * 1.1102230246251565e-16)
    elif kind == KIND_F16:
        var k = (bitcast[DType.uint16, 64](words) >> 5).cast[DType.uint32]()
        var m = 31 - count_leading_zeros(k)
        var bits = ((m + 4) << 10) | ((k << (10 - m)) & 0x3FF)
        dst.unsafe_bitcast[UInt16]().unsafe_store(k.eq(0).select(SIMD[DType.uint32, 64](0), bits).cast[DType.uint16]())
    elif kind == KIND_CHAR:
        var x = bitcast[DType.uint64, 16](words)
        var hi = x >> 32
        var lo = x & 0xFFFFFFFF
        var u = ((hi * 1112064) + ((lo * 1112064) >> 32)) >> 32
        dst.unsafe_bitcast[UInt32]().unsafe_store(u.lt(0xD800).select(u, u + 0x800).cast[DType.uint32]())
    else:
        var shifts = iota[DType.uint32, 32]()
        for j in range(32):
            dst.unsafe_bitcast[UInt8]().unsafe_offset(32 * j).unsafe_store(((SIMD[DType.uint32, 32](words[j]) >> shifts) & 1).cast[DType.uint8]())


def fill_rows[kind: Int, W: Int, T: DType](mut cur: Cursor, key: SIMD[DType.uint32, 4], K: UInt32, p: UInt64, n: Int, dst: Pointer[Scalar[T], MutAnyOrigin]):
    """Write n elements of W bits from the W-aligned bit position p, row by row. A row that the
    fill covers entirely goes straight to dst. The first and last row go through a scratch row."""
    comptime per_row = 1024 // W
    var first_el = Int(p // UInt64(W))
    var end = first_el + n
    var row = first_el // per_row
    var last_row = (end + per_row - 1) // per_row
    var c = cur.copy()
    var scratch = stack_allocation[per_row, Scalar[T]]()
    var i = 0
    while row < last_row:
        c.seek(key, K, UInt64(row))
        var words = row_words(c.lanes)
        var first = row * per_row
        if first >= first_el and first + per_row <= end:
            store_row[kind, W, T](words, dst.unsafe_offset(i))
            i += per_row
        else:
            store_row[kind, W, T](words, scratch.unsafe_origin_cast[MutAnyOrigin]())
            var lo = max(first_el, first) - first
            var hi = min(end, first + per_row) - first
            for j in range(lo, hi):
                dst.unsafe_offset(i).unsafe_store(scratch.unsafe_offset(j).unsafe_load())
                i += 1
        row += 1
    cur = c^


@always_inline
def below_row_u64(words: Row, dst: Pointer[UInt64, MutAnyOrigin], bound: UInt64, thresh: UInt64, key: SIMD[DType.uint32, 4], K: UInt32, first_element: Int):
    """The 16 bounded values of one row. The row goes through L1 so that the scalar multiplies
    run on the integer pipes while the vector pipes generate the next row. A scalar umulh per
    element beats a lane-wise emulation. The rejection tests are ORed, and only a row with a
    rejection runs the fixup."""
    var raw = stack_allocation[16, UInt64]()
    raw.unsafe_store(bitcast[DType.uint64, 16](words))
    var any = UInt64(0)
    for k in range(16):
        var prod = UInt128(raw.unsafe_offset(k).unsafe_load()) * UInt128(bound)
        any |= UInt64(UInt64(prod & 0xFFFFFFFFFFFFFFFF) < thresh)
        dst.unsafe_offset(k).unsafe_store(UInt64(prod >> 64))
    if any != 0:
        for k in range(16):
            if UInt64(UInt128(raw.unsafe_offset(k).unsafe_load()) * UInt128(bound) & 0xFFFFFFFFFFFFFFFF) < thresh:
                dst.unsafe_offset(k).unsafe_store(below_retry_u64(key, K, bound, UInt64(first_element + k)))


def fill_rows_below_u64(mut cur: Cursor, key: SIMD[DType.uint32, 4], K: UInt32, p: UInt64, n: Int, dst: Pointer[UInt64, MutAnyOrigin], bound: UInt64, thresh: UInt64):
    """fill_rows for the bounded u64 fill: element i of the fill is stream element i, bounded."""
    var first_el = Int(p // 64)
    var end = first_el + n
    var row = first_el // 16
    var last_row = (end + 15) // 16
    var c = cur.copy()
    var scratch = stack_allocation[16, UInt64]()
    var i = 0
    while row < last_row:
        c.seek(key, K, UInt64(row))
        var words = row_words(c.lanes)
        var first = row * 16
        if first >= first_el and first + 16 <= end:
            below_row_u64(words, dst.unsafe_offset(i), bound, thresh, key, K, first_el + i)
            i += 16
        else:
            var lo = max(first_el, first) - first
            below_row_u64(words, scratch.unsafe_origin_cast[MutAnyOrigin](), bound, thresh, key, K, first_el + i - lo)
            for j in range(lo, min(end, first + 16) - first):
                dst.unsafe_offset(i).unsafe_store(scratch.unsafe_offset(j).unsafe_load())
                i += 1
        row += 1
    cur = c^


def fill_u32[origin: Origin[mut=True]](key: SIMD[DType.uint32, 4], pos: UInt64, K: UInt32, dst: Pointer[UInt32, origin], n: Int) -> UInt64:
    """Write n stream words from the 32-bit aligned position pos. Returns the position after."""
    var p = align(pos, 32)
    if n > 0:
        var cur = Cursor()
        fill_rows[KIND_INT, 32, DType.uint32](cur, key, K, p, n, dst.unsafe_origin_cast[MutAnyOrigin]())
    return p + UInt64(n) * 32


# ---- GPU fill ------------------------------------------------------------------------------


@always_inline
def block_elements[kind: Int, T: DType](b: SIMD[DType.uint32, 4]) -> SIMD[T, 128 // bit_width_of[T]()]:
    """The 128 / W elements of one 16-byte block."""
    comptime N = 128 // bit_width_of[T]()
    comptime if kind == KIND_INT:
        return bitcast[T, N](b)
    elif kind == KIND_F32:
        return rebind[SIMD[T, N]]((b >> 8).cast[DType.float32]() * 5.9604645e-08)
    else:
        var x = bitcast[DType.uint64, 2](b)
        return rebind[SIMD[T, N]]((x >> 11).cast[DType.float64]() * 1.1102230246251565e-16)


def fill_rows_kernel[kind: Int, T: DType](dst: Pointer[Scalar[T], MutAnyOrigin], k0: UInt32, k1: UInt32, k2: UInt32, k3: UInt32, first_row: UInt64, nrows: UInt64, K: UInt32):
    """One thread per chunk: seed, then store its K blocks in row order."""
    comptime N = 128 // bit_width_of[T]()
    var tid = UInt64(block_idx.x) * UInt64(block_dim.x) + UInt64(thread_idx.x)
    var g = first_row / UInt64(K) + tid / 8
    var lane = tid % 8
    var c = 8 * g + lane
    var key = SIMD[DType.uint32, 4](k0, k1, k2, k3)
    var s = State.keyed(key, SIMD[DType.uint32, 1](UInt32(c & 0xFFFFFFFF)), SIMD[DType.uint32, 1](UInt32(c >> 32)), DOMAIN_STREAM, AUX_STREAM)
    var row0 = g * UInt64(K)
    for j in range(Int(K)):
        s.step()
        var row = row0 + UInt64(j)
        if row >= first_row and row < first_row + nrows:
            var at = Int((row - first_row) * UInt64(1024 // bit_width_of[T]()) + lane * UInt64(N))
            dst.unsafe_offset(at).unsafe_store(block_elements[kind, T](SIMD[DType.uint32, 4](s.o0[0], s.o1[0], s.o2[0], s.o3[0])))


def fill_gpu[kind: Int, T: DType, origin: Origin[mut=True]](ctx: DeviceContext, key: SIMD[DType.uint32, 4], first_row: UInt64, nrows: UInt64, K: UInt32, dst: Pointer[Scalar[T], origin]) raises:
    """Rows [first_row, first_row + nrows) into device memory, 1024 / W elements per row."""
    var groups = (first_row + nrows + UInt64(K) - 1) / UInt64(K) - first_row / UInt64(K)
    var threads = Int(groups * 8)
    ctx.enqueue_function[fill_rows_kernel[kind, T]](dst.unsafe_origin_cast[MutAnyOrigin](), key[0], key[1], key[2], key[3], first_row, nrows, K, grid_dim=(threads + 255) // 256, block_dim=256)


def fill_u32_gpu[origin: Origin[mut=True]](ctx: DeviceContext, key: SIMD[DType.uint32, 4], first_row: UInt64, nrows: UInt64, K: UInt32, dst: Pointer[UInt32, origin]) raises:
    """Rows [first_row, first_row + nrows) into device memory, 32 words per row."""
    fill_gpu[KIND_INT, DType.uint32](ctx, key, first_row, nrows, K, dst)


def fill_u64_gpu[origin: Origin[mut=True]](ctx: DeviceContext, key: SIMD[DType.uint32, 4], first_row: UInt64, nrows: UInt64, K: UInt32, dst: Pointer[UInt64, origin]) raises:
    """Rows [first_row, first_row + nrows) into device memory, 16 values per row."""
    fill_gpu[KIND_INT, DType.uint64](ctx, key, first_row, nrows, K, dst)


def fill_f32_gpu[origin: Origin[mut=True]](ctx: DeviceContext, key: SIMD[DType.uint32, 4], first_row: UInt64, nrows: UInt64, K: UInt32, dst: Pointer[Float32, origin]) raises:
    """Rows [first_row, first_row + nrows) into device memory, 32 values per row."""
    fill_gpu[KIND_F32, DType.float32](ctx, key, first_row, nrows, K, dst)


def fill_f64_gpu[origin: Origin[mut=True]](ctx: DeviceContext, key: SIMD[DType.uint32, 4], first_row: UInt64, nrows: UInt64, K: UInt32, dst: Pointer[Float64, origin]) raises:
    """Rows [first_row, first_row + nrows) into device memory, 16 values per row."""
    fill_gpu[KIND_F64, DType.float64](ctx, key, first_row, nrows, K, dst)
