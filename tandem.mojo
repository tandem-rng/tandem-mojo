# Tandem8x32 in Mojo: the building blocks of https://github.com/tandem-rng/spec, a CPU row
# fill over eight SIMD lanes, and a GPU row fill with one thread per chunk.
# Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.

from std.bit import count_leading_zeros, count_trailing_zeros, rotate_bits_left
from std.sys import has_accelerator
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
        """The step T: mix, clock, feedback."""
        var m0 = (self.h0 | 1).cast[DType.uint64]()
        var m1 = (self.h1 | 1).cast[DType.uint64]()
        var p0 = self.o0.cast[DType.uint64]() * m0
        var p1 = self.o2.cast[DType.uint64]() * m1
        var lo0 = p0.cast[DType.uint32]()
        var hi0 = (p0 >> 32).cast[DType.uint32]()
        var lo1 = p1.cast[DType.uint32]()
        var hi1 = (p1 >> 32).cast[DType.uint32]()
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
comptime Row = SIMD[DType.uint32, 32]


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


def fill_u32[origin: Origin[mut=True]](key: SIMD[DType.uint32, 4], pos: UInt64, K: UInt32, dst: Pointer[UInt32, origin], n: Int) -> UInt64:
    """Write n stream words from the 32-bit aligned position pos. Returns the position after."""
    var p = (pos + 31) & ~UInt64(31)
    var word = Int(p >> 5)
    var end = word + n
    var shift = UInt64(count_trailing_zeros(K))
    var mask = UInt64(K) - 1
    var row = UInt64(word >> 5)
    var last_row = UInt64((end + 31) >> 5)
    var lanes = Lanes[8](0, 0, 0, 0, 0, 0, 0, 0)
    var i = 0
    while row < last_row:
        var step_in_group = row & mask
        if row == UInt64(word >> 5) or step_in_group == 0:
            var g = row >> shift
            var c0 = 8 * g
            var lo = SIMD[DType.uint32, 8](0, 1, 2, 3, 4, 5, 6, 7) + UInt32(c0 & 0xFFFFFFFF)
            lanes = Lanes[8].keyed(key, lo, SIMD[DType.uint32, 8](UInt32(c0 >> 32)), DOMAIN_STREAM, AUX_STREAM)
            for _ in range(Int(step_in_group) + 1):
                lanes.step()
        else:
            lanes.step()
        var words = row_words(lanes)
        var first = Int(row) * 32
        if first >= word and first + 32 <= end:
            dst.unsafe_offset(i).unsafe_store(words)
            i += 32
        else:
            for w in range(32):
                if first + w >= word and first + w < end:
                    dst.unsafe_offset(i).unsafe_store(words[w])
                    i += 1
        row += 1
    return p + UInt64(n) * 32


# ---- GPU fill ------------------------------------------------------------------------------


def fill_rows_kernel(dst: Pointer[UInt32, MutAnyOrigin], k0: UInt32, k1: UInt32, k2: UInt32, k3: UInt32, first_row: UInt64, nrows: UInt64, K: UInt32):
    """One thread per chunk: seed, then store its K blocks in row order."""
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
            var at = Int((row - first_row) * 32 + lane * 4)
            dst.unsafe_offset(at).unsafe_store(SIMD[DType.uint32, 4](s.o0[0], s.o1[0], s.o2[0], s.o3[0]))


def fill_u32_gpu[origin: Origin[mut=True]](ctx: DeviceContext, key: SIMD[DType.uint32, 4], first_row: UInt64, nrows: UInt64, K: UInt32, dst: Pointer[UInt32, origin]) raises:
    """Rows [first_row, first_row + nrows) into device memory, 32 words per row."""
    var groups = (first_row + nrows + UInt64(K) - 1) / UInt64(K) - first_row / UInt64(K)
    var threads = Int(groups * 8)
    ctx.enqueue_function[fill_rows_kernel](dst.unsafe_origin_cast[MutAnyOrigin](), key[0], key[1], key[2], key[3], first_row, nrows, K, grid_dim=(threads + 255) // 256, block_dim=256)
