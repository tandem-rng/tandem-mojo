# Tandem8x32 in Mojo: the building blocks of https://github.com/tandem-rng/spec, a CPU row
# fill over eight SIMD lanes, and a GPU row fill with one thread per chunk.
# Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.

from std.bit import count_trailing_zeros, rotate_bits_left
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
