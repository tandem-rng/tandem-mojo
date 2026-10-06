# Readers for the spec's conformance files, copied byte for byte from tandem-spec
# conformance/*.json, and the hashes their dumps need. A case is one line of a file, so a case
# is read by key with no general JSON parser.
from std.bit import rotate_bits_right
from std.memory.alloc import unsafe_alloc

comptime ASCII_ZERO = 48
comptime ASCII_QUOTE = 34
comptime SHA_K = SIMD[DType.uint32, 64](
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)


def hex_value(s: String) -> UInt64:
    var v = UInt64(0)
    for b in s.as_bytes():
        var d = UInt64(b) - 48 if b <= 57 else UInt64(b) - 87
        v = (v << 4) | d
    return v


def hex_words(s: String) -> UInt32:
    return UInt32(hex_value(s))


struct Case(Copyable, Movable):
    var line: String

    def __init__(out self, line: String):
        self.line = line

    def at(self, key: String) -> Int:
        """The offset of the value of key, or -1."""
        var i = self.line.find(String('"', key, '": '))
        return -1 if i < 0 else i + key.byte_length() + 4

    def has(self, key: String) -> Bool:
        return self.at(key) >= 0

    def text(self, key: String) raises -> String:
        var p = self.at(key)
        if p < 0:
            raise Error(String("no field ", key, " in ", self.line))
        var q = self.line.find('"', p + 1)
        return String(self.line[byte = p + 1 : q])

    def number(self, key: String) raises -> Int:
        var p = self.at(key)
        if p < 0:
            raise Error(String("no field ", key, " in ", self.line))
        var v = 0
        for b in self.line[byte=p:].as_bytes():
            if b < 48 or b > 57:
                break
            v = 10 * v + Int(b) - 48
        return v

    def words(self, key: String) raises -> List[String]:
        """The quoted strings of a list value."""
        var p = self.at(key)
        if p < 0:
            raise Error(String("no field ", key, " in ", self.line))
        var end = self.line.find("]", p)
        var out = List[String]()
        var from_q = -1
        var i = p
        for b in self.line[byte = p : end].as_bytes():
            if b == ASCII_QUOTE:
                if from_q < 0:
                    from_q = i + 1
                else:
                    out.append(String(self.line[byte = from_q : i]))
                    from_q = -1
            i += 1
        return out^

    def numbers(self, key: String) raises -> List[Int]:
        """The integers of a list value."""
        var p = self.at(key)
        if p < 0:
            raise Error(String("no field ", key, " in ", self.line))
        var end = self.line.find("]", p)
        var out = List[Int]()
        var v = 0
        var open = False
        for b in self.line[byte = p + 1 : end + 1].as_bytes():
            if b >= 48 and b <= 57:
                v = 10 * v + Int(b) - 48
                open = True
            elif open:
                out.append(v)
                v = 0
                open = False
        return out^

    def key(self) raises -> SIMD[DType.uint32, 4]:
        var w = self.words("key")
        return SIMD[DType.uint32, 4](hex_words(w[0]), hex_words(w[1]), hex_words(w[2]), hex_words(w[3]))

    def draws(self) raises -> List[Tuple[String, Int]]:
        """The (kind, n) pairs of a dump."""
        var p = self.at("draws")
        var out = List[Tuple[String, Int]]()
        var from_at = p
        while True:
            var k = self.line.find('{"kind": "', from_at)
            if k < 0:
                break
            var q = self.line.find('"', k + 10)
            var kind = String(self.line[byte = k + 10 : q])
            var n = 0
            for b in self.line[byte = q + 8 :].as_bytes():
                if b < 48 or b > 57:
                    break
                n = 10 * n + Int(b) - 48
            out.append((kind, n))
            from_at = q
        return out^


def load(name: String) raises -> List[Case]:
    """The cases of tests/conformance/name, one per line that opens an object with an id or a file."""
    var out = List[Case]()
    with open(String("tests/conformance/", name), "r") as file:
        var text = file.read()
        for line in text.splitlines():
            var l = String(line)
            if l.find('{"id":') >= 0 or l.find('{"file":') >= 0:
                out.append(Case(l))
    return out^


def find_case(cases: List[Case], name: String) raises -> Int:
    """The index of the case whose id ends with the array element name."""
    var tail = String(" ", name)
    for i in range(len(cases)):
        var id = cases[i].text("id")
        if id.byte_length() >= tail.byte_length() and String(id[byte = id.byte_length() - tail.byte_length() :]) == tail:
            return i
    raise Error(String("no case ", name))


# ---- Hashes of long outputs ------------------------------------------------------------------

struct Fnv1a(Copyable, Movable):
    var h: UInt64

    def __init__(out self):
        self.h = 0xCBF29CE484222325

    def update[origin: Origin[mut=True]](mut self, data: Pointer[UInt8, origin], n: Int):
        var h = self.h
        for i in range(n):
            h = (h ^ UInt64(data.unsafe_offset(i).unsafe_load())) * 0x100000001B3
        self.h = h


struct Sha256(Copyable, Movable):
    var h: SIMD[DType.uint32, 8]
    var tail: List[UInt8]
    var total: UInt64

    def __init__(out self):
        self.h = SIMD[DType.uint32, 8](
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
        )
        self.tail = List[UInt8]()
        self.total = 0

    def block[origin: Origin[mut=True]](mut self, p: Pointer[UInt8, origin]):
        var w = SIMD[DType.uint32, 64](0)
        for i in range(16):
            var q = p.unsafe_offset(4 * i)
            w[i] = (
                (UInt32(q.unsafe_load()) << 24)
                | (UInt32(q.unsafe_offset(1).unsafe_load()) << 16)
                | (UInt32(q.unsafe_offset(2).unsafe_load()) << 8)
                | UInt32(q.unsafe_offset(3).unsafe_load())
            )
        for i in range(16, 64):
            var a = w[i - 15]
            var b = w[i - 2]
            var s0 = rotate_bits_right[7](a) ^ rotate_bits_right[18](a) ^ (a >> 3)
            var s1 = rotate_bits_right[17](b) ^ rotate_bits_right[19](b) ^ (b >> 10)
            w[i] = w[i - 16] + s0 + w[i - 7] + s1
        var a = self.h[0]
        var b = self.h[1]
        var c = self.h[2]
        var d = self.h[3]
        var e = self.h[4]
        var f = self.h[5]
        var g = self.h[6]
        var h = self.h[7]
        for i in range(64):
            var t1 = h + (rotate_bits_right[6](e) ^ rotate_bits_right[11](e) ^ rotate_bits_right[25](e)) + ((e & f) ^ (~e & g)) + SHA_K[i] + w[i]
            var t2 = (rotate_bits_right[2](a) ^ rotate_bits_right[13](a) ^ rotate_bits_right[22](a)) + ((a & b) ^ (a & c) ^ (b & c))
            h = g
            g = f
            f = e
            e = d + t1
            d = c
            c = b
            b = a
            a = t1 + t2
        self.h += SIMD[DType.uint32, 8](a, b, c, d, e, f, g, h)

    def update[origin: Origin[mut=True]](mut self, data: Pointer[UInt8, origin], n: Int):
        self.total += UInt64(n)
        var i = 0
        if len(self.tail) > 0:
            while i < n and len(self.tail) < 64:
                self.tail.append(data.unsafe_offset(i).unsafe_load())
                i += 1
            if len(self.tail) == 64:
                var t = self.tail.copy()
                self.block(Pointer(to=t[0]).unsafe_origin_cast[MutAnyOrigin]())
                self.tail.clear()
        while i + 64 <= n:
            self.block(data.unsafe_offset(i))
            i += 64
        while i < n:
            self.tail.append(data.unsafe_offset(i).unsafe_load())
            i += 1

    def hexdigest(mut self) -> String:
        var bits = self.total * 8
        var pad = List[UInt8]()
        pad.append(0x80)
        while (len(self.tail) + len(pad)) % 64 != 56:
            pad.append(0)
        for k in range(8):
            pad.append(UInt8((bits >> UInt64(56 - 8 * k)) & 0xFF))
        var keep = self.total
        self.update(Pointer(to=pad[0]).unsafe_origin_cast[MutAnyOrigin](), len(pad))
        self.total = keep
        var out = String()
        var digits = "0123456789abcdef"
        for i in range(8):
            for s in range(8):
                var nib = Int((self.h[i] >> UInt32(28 - 4 * s)) & 0xF)
                out += String(digits[byte = nib : nib + 1])
        return out
