#!/usr/bin/env python3
"""Near-minimax coefficients of the polynomials in tandem.mojo's normals, by Chebyshev
interpolation in 60-digit decimals. Prints the coefficients, lowest order first.

    python3 tools/gen_coefficients.py

atanh(s) / s as a polynomial in w = s^2 on [0, ((sqrt 2 - 1) / (sqrt 2 + 1))^2], for the
logarithm. sin(x) / x and cos(x) as polynomials in y = x^2 on [0, (pi / 4)^2], for the angle.
The degrees are the smallest with a maximum error under 2e-16 (Float64) and 3e-9 (Float32).
"""
from decimal import Decimal as D, getcontext
import math

getcontext().prec = 60
PI = D("3.14159265358979323846264338327950288419716939937510582097494")


def pw(y, k):
    return D(1) if k == 0 else y**k


def dcos(x):
    s = t = D(1)
    k = 0
    while abs(t) > D(10) ** -58:
        k += 2
        t = -t * x * x / (k * (k - 1))
        s += t
    return s


def atanh_over_s(w):
    return sum(pw(w, k) / (2 * k + 1) for k in range(80))


def sin_over_x(y):
    t = s = D(1)
    for k in range(1, 40):
        t = -t * y / ((2 * k) * (2 * k + 1))
        s += t
    return s


def cos_x(y):
    t = s = D(1)
    for k in range(1, 40):
        t = -t * y / ((2 * k - 1) * (2 * k))
        s += t
    return s


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= f * M[i][c]
    x = [D(0)] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def fit(f, hi, deg):
    n = deg + 1
    nodes = [(1 + dcos((2 * k + 1) * PI / (2 * n))) / 2 * hi for k in range(n)]
    ts = [2 * y / hi - 1 for y in nodes]
    c_t = solve([[pw(t, j) for j in range(n)] for t in ts], [f(y) for y in nodes])
    poly = [D(0)] * n
    a, b = 2 / hi, D(-1)
    for j, c in enumerate(c_t):
        for m in range(j + 1):
            poly[m] += c * D(math.comb(j, m)) * pw(a, m) * pw(b, j - m)
    return poly


def err(f, hi, poly):
    return max(abs(sum(c * pw(hi * D(i) / 400, k) for k, c in enumerate(poly)) - f(hi * D(i) / 400)) for i in range(401))


dl = ((D(2).sqrt() - 1) / (D(2).sqrt() + 1)) ** 2
qpi = (PI / 4) ** 2
for name, f, hi in (("atanh", atanh_over_s, dl), ("sin", sin_over_x, qpi), ("cos", cos_x, qpi)):
    for ty, tol in (("f64", D("2e-16")), ("f32", D("3e-9"))):
        deg = next(d for d in range(1, 20) if err(f, hi, fit(f, hi, d)) < tol)
        print(name, ty, "degree", deg, [float(c) for c in fit(f, hi, deg)])
