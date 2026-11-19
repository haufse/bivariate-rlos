"""Core library for the Section 7 numerics (asymptotic comparison).

Everything is in the manuscript notation:

* moment functions mu^{(l,m)}(a,t)            eq. (exponential-moment-h), Prop. prop:erlang-minimum
* d(t) = -d/da mu(a,t) at a = A_0(t)           eq. (d)
* lambda_W = W d / (d^T W d)                   eq. (lambda-W)
* calS^{(l,m)}(t) = int_0^1 S^{(l,m)}(h^{(l,m)}(u,t)) du,  calR likewise   eq. (rank-moment-process)
  with S^{(l,m)}, R^{(l,m)} = R_1 + R_2 from Lemma lem:finiteblockbias  (eqs. S-ellm, R1-ellm, R2-ellm)
* Sigma(t) = Gamma(t,t) = Cov{phi_t(M)}       eqs. (influence-function), (Gamma)
  computed as  Sigma_{pp'} = int int Cov{g_u^p, g_v^{p'}} du dv  with
  g_u^{(l,m)} = 1{V_1^{(l)}<=u, V_2^{(m)}<=u} - dC_1(h(u)) 1{V_1^{(l)}<=u} - dC_2(h(u)) 1{V_2^{(m)}<=u}
  (the u-integrated form of eq. (influence-function); the constant terms do not affect covariances).

Joint probabilities of the limiting Poisson process N ~ PPP(nu) are evaluated EXACTLY by counting
points in the (at most) 8 cells cut out by two thresholds per margin; the cell intensities are
inclusion-exclusion differences of the joint-exceedance function J(a,b) = a + b - L(a,b) in
q-scale (q = 1/x).  Every count is Poisson and the cells are disjoint, hence independent.

Quadrature in (s, s') = (-log u, -log v): split at s = s', write s' = s + w, and use composite
Gauss-Legendre on geometrically graded panels in s and in w (weight e^{-2s-w}); Sigma = I + I^T.

Attractor: logistic / Gumbel-Hougaard  L(q1,q2) = (q1^theta + q2^theta)^{1/theta}, theta >= 1
(theta = 1 is independence).
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np
import numba as nb

# --------------------------------------------------------------------------------------------
# index sets and pair orderings
# --------------------------------------------------------------------------------------------


def pairs(r):
    """Lexicographic order of (l,m) in [r]^2, eq. (stacking)."""
    return [(l, m) for l in range(1, r + 1) for m in range(1, r + 1)]


@lru_cache(maxsize=None)
def I_lm(l, m):
    """I_{l,m} = {(i1,i2,i3) in N0^3 : i1+i2 < l, i1+i3 < m}, eq. (I-lm)."""
    out = []
    for i1 in range(0, min(l, m)):
        for i2 in range(0, l - i1):
            for i3 in range(0, m - i1):
                out.append((i1, i2, i3))
    return tuple(out)


def _fact(i):
    return math.factorial(i)


# --------------------------------------------------------------------------------------------
# Pickands function of the logistic attractor
# --------------------------------------------------------------------------------------------


def A_log(t, theta):
    t = np.asarray(t, dtype=float)
    return (t ** theta + (1 - t) ** theta) ** (1.0 / theta)


def dA_log(t, theta):
    t = np.asarray(t, dtype=float)
    if theta == 1.0:
        return np.zeros_like(t)
    return (t ** (theta - 1) - (1 - t) ** (theta - 1)) * A_log(t, theta) ** (1 - theta)


# --------------------------------------------------------------------------------------------
# moment functions and their a-derivative (eq. exponential-moment-h)
# --------------------------------------------------------------------------------------------


def mu_lm(a, t, l, m):
    s = 0.0
    for (i1, i2, i3) in I_lm(l, m):
        k = i1 + i2 + i3
        mult = _fact(k) / (_fact(i1) * _fact(i2) * _fact(i3))
        s += mult * (1 - a) ** i1 * (a - t) ** i2 * (a + t - 1) ** i3 / (1 + a) ** (k + 1)
    return 1.0 - s


def dmu_da_lm(a, t, l, m):
    """Exact a-derivative of mu^{(l,m)}(a,t) by the product rule."""
    s = 0.0
    for (i1, i2, i3) in I_lm(l, m):
        k = i1 + i2 + i3
        mult = _fact(k) / (_fact(i1) * _fact(i2) * _fact(i3))
        f1, f2, f3, f4 = (1 - a), (a - t), (a + t - 1), (1 + a)
        p1 = f1 ** i1
        p2 = f2 ** i2
        p3 = f3 ** i3
        p4 = f4 ** (-(k + 1))
        dp1 = -i1 * f1 ** (i1 - 1) if i1 > 0 else 0.0
        dp2 = i2 * f2 ** (i2 - 1) if i2 > 0 else 0.0
        dp3 = i3 * f3 ** (i3 - 1) if i3 > 0 else 0.0
        dp4 = -(k + 1) * f4 ** (-(k + 2))
        s += mult * (dp1 * p2 * p3 * p4 + p1 * dp2 * p3 * p4 + p1 * p2 * dp3 * p4 + p1 * p2 * p3 * dp4)
    return -s


def mu_vec(a, t, r):
    return np.array([mu_lm(a, t, l, m) for (l, m) in pairs(r)])


def d_vec(a, t, r):
    """d(t) = -partial_a mu(a,t) at a, eq. (d)."""
    return np.array([-dmu_da_lm(a, t, l, m) for (l, m) in pairs(r)])


# --------------------------------------------------------------------------------------------
# integrated bias functions along h^{(l,m)}(u,t): closed forms through
#     int_0^infty x^j e^{-(1+A)x} dx = j!/(1+A)^{j+1}
# --------------------------------------------------------------------------------------------


def _gam(j, A):
    return _fact(j) / (1.0 + A) ** (j + 1)


def calS_shape(A, t, l, m, D):
    """int_0^infty e^{-(1+A)x} x^D P_{lm}(x) dx, where along the curve
    S^{(l,m)} = S(e^{-q1},e^{-q2}) * P_{lm}(x),  P = sum_i (1/i!) {a^i - sum_{j:i_j>0} (-1)^{1{j=1}} i_j a^{i-e_j}},
    a = x*(1-A, A-t, A+t-1)   (eq. S-ellm), and S(e^{-q}) = kappa e^{-L(q)} Psi(q), Psi homogeneous of degree D.
    """
    al = (1.0 - A, A - t, A + t - 1.0)
    sgn = (-1.0, 1.0, 1.0)
    tot = 0.0
    for i in I_lm(l, m):
        k = sum(i)
        ifac = _fact(i[0]) * _fact(i[1]) * _fact(i[2])
        mono = al[0] ** i[0] * al[1] ** i[1] * al[2] ** i[2]
        tot += mono * _gam(k + D, A) / ifac
        for j in range(3):
            if i[j] > 0:
                e = list(i)
                e[j] -= 1
                mono_j = al[0] ** e[0] * al[1] ** e[1] * al[2] ** e[2]
                tot -= sgn[j] * i[j] * mono_j * _gam(k - 1 + D, A) / ifac
    return tot


def calR_lm(A, v, t, l, m):
    """calR^{(l,m)}[A](t) = int_0^1 R^{(l,m)}_A(h^{(l,m)}(u,t)) du with R = R_1 + R_2 (eqs. R1-ellm, R2-ellm),
    evaluated in closed form; A = A(t), v = A'(t)."""
    al = (1.0 - A, A - t, A + t - 1.0)                 # a = x*al
    zt = (A * A - (1 - t) ** 2 - t * t, t * t - A * A, (1 - t) ** 2 - A * A)  # zeta = x^2*zt
    Ld1 = A - t * v
    Ld2 = A + (1 - t) * v
    Wt = -(l - 1) * (1 - t) * Ld1 - (m - 1) * t * Ld2  # W = x*Wt
    ht = ((l - 1) * (1 - t) + (m - 1) * t + Wt, -(m - 1) * t - Wt, -(l - 1) * (1 - t) - Wt)  # h = x*ht
    tot = 0.0
    for i in I_lm(l, m):
        k = sum(i)
        ifac = _fact(i[0]) * _fact(i[1]) * _fact(i[2])
        mono = al[0] ** i[0] * al[1] ** i[1] * al[2] ** i[2]
        # R1: (1/2) {k(1+2Ax-k) x^k al^i + sum_j i_j x^2 zt_j x^{k-1} al^{i-e_j}}
        tot += 0.5 * k * (1 - k) * mono * _gam(k, A) / ifac
        tot += 0.5 * k * 2 * A * mono * _gam(k + 1, A) / ifac
        # R2: (1/2) {Wt x x^k al^i + sum_j i_j ht_j x x^{k-1} al^{i-e_j}}
        tot += 0.5 * Wt * mono * _gam(k + 1, A) / ifac
        for j in range(3):
            if i[j] > 0:
                e = list(i)
                e[j] -= 1
                mono_j = al[0] ** e[0] * al[1] ** e[1] * al[2] ** e[2]
                tot += 0.5 * i[j] * zt[j] * mono_j * _gam(k + 1, A) / ifac
                tot += 0.5 * i[j] * ht[j] * mono_j * _gam(k, A) / ifac
    return tot


def calR_vec(A, v, t, r):
    return np.array([calR_lm(A, v, t, l, m) for (l, m) in pairs(r)])


def calR_r2_closed(a, v, t):
    """Proposition prop:integrated-R-r2, eq. (integrated-R-r2)."""
    return np.array([0.0,
                     (a + t - 1) * (1 - t) * (a - 1 - t * v) / (1 + a) ** 3,
                     t * (a - t) * (a - 1 + (1 - t) * v) / (1 + a) ** 3,
                     3 * (a - 1) * (a - t) * (a + t - 1) / (1 + a) ** 4])


# --------------------------------------------------------------------------------------------
# exact Poisson-count probabilities for <= 2 thresholds per margin
# --------------------------------------------------------------------------------------------

BIG = 1000


@nb.njit(cache=True)
def L_log(q1, q2, th):
    if q1 <= 0.0:
        return q2
    if q2 <= 0.0:
        return q1
    if th == 1.0:
        return q1 + q2
    mx = max(q1, q2)
    return mx * ((q1 / mx) ** th + (q2 / mx) ** th) ** (1.0 / th)


@nb.njit(cache=True)
def J_log(a, b, th):
    """nu-measure of joint exceedance {z1 > 1/a, z2 > 1/b} in q-scale."""
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return a + b - L_log(a, b, th)


@nb.njit(cache=True)
def prob_counts(a1, k1, a2, k2, b1, j1, b2, j2, th):
    """P(N{z1>1/a1}<=k1, N{z1>1/a2}<=k2, N{z2>1/b1}<=j1, N{z2>1/b2}<=j2), a,b >= 0 in q-scale.
    Inactive constraints: k = BIG (or threshold 0)."""
    # order margin 1
    if a1 <= a2:
        alo, klo, ahi, khi = a1, k1, a2, k2
    else:
        alo, klo, ahi, khi = a2, k2, a1, k1
    if alo <= 0.0:
        klo = BIG
    if ahi <= 0.0:
        khi = BIG
    if klo >= khi:
        klo = BIG  # implied by the larger set
    if b1 <= b2:
        blo, jlo, bhi, jhi = b1, j1, b2, j2
    else:
        blo, jlo, bhi, jhi = b2, j2, b1, j1
    if blo <= 0.0:
        jlo = BIG
    if bhi <= 0.0:
        jhi = BIG
    if jlo >= jhi:
        jlo = BIG
    # cell intensities, rows i (margin-1 band), columns j (margin-2 band)
    Jll = J_log(alo, blo, th)
    Jlh = J_log(alo, bhi, th)
    Jhl = J_log(ahi, blo, th)
    Jhh = J_log(ahi, bhi, th)
    m00 = Jll
    m01 = Jlh - Jll
    m10 = Jhl - Jll
    m11 = Jhh - Jlh - Jhl + Jll
    m02 = alo - Jlh
    m12 = (ahi - alo) - (Jhh - Jlh)
    m20 = blo - Jhl
    m21 = (bhi - blo) - (Jhh - Jhl)
    tot = L_log(ahi, bhi, th)
    # clip tiny negative round-off
    if m00 < 0.0: m00 = 0.0
    if m01 < 0.0: m01 = 0.0
    if m10 < 0.0: m10 = 0.0
    if m11 < 0.0: m11 = 0.0
    if m02 < 0.0: m02 = 0.0
    if m12 < 0.0: m12 = 0.0
    if m20 < 0.0: m20 = 0.0
    if m21 < 0.0: m21 = 0.0
    # caps per cell
    cap_row0 = min(klo, khi)
    cap_row1 = khi
    cap_col0 = min(jlo, jhi)
    cap_col1 = jhi
    s = 0.0
    c00 = min(cap_row0, cap_col0)
    if m00 == 0.0: c00 = 0
    for n00 in range(c00 + 1):
        t00 = m00 ** n00 / math.gamma(n00 + 1.0)
        c01 = min(cap_row0 - n00, cap_col1 - n00)
        if m01 == 0.0: c01 = 0
        if c01 < 0: continue
        for n01 in range(c01 + 1):
            t01 = t00 * m01 ** n01 / math.gamma(n01 + 1.0)
            c02 = cap_row0 - n00 - n01
            if m02 == 0.0: c02 = 0
            if c02 < 0: continue
            for n02 in range(c02 + 1):
                t02 = t01 * m02 ** n02 / math.gamma(n02 + 1.0)
                r0 = n00 + n01 + n02
                c10 = min(cap_row1 - r0, cap_col0 - n00)
                if m10 == 0.0: c10 = 0
                if c10 < 0: continue
                for n10 in range(c10 + 1):
                    t10 = t02 * m10 ** n10 / math.gamma(n10 + 1.0)
                    c11 = min(cap_row1 - r0 - n10, cap_col1 - n00 - n01 - n10)
                    if m11 == 0.0: c11 = 0
                    if c11 < 0: continue
                    for n11 in range(c11 + 1):
                        t11 = t10 * m11 ** n11 / math.gamma(n11 + 1.0)
                        c12 = cap_row1 - r0 - n10 - n11
                        if m12 == 0.0: c12 = 0
                        if c12 < 0: continue
                        for n12 in range(c12 + 1):
                            t12 = t11 * m12 ** n12 / math.gamma(n12 + 1.0)
                            col0 = n00 + n10
                            c20 = min(cap_col0 - col0, cap_col1 - col0 - n01 - n11)
                            if m20 == 0.0: c20 = 0
                            if c20 < 0: continue
                            for n20 in range(c20 + 1):
                                t20 = t12 * m20 ** n20 / math.gamma(n20 + 1.0)
                                c21 = cap_col1 - col0 - n20 - n01 - n11
                                if m21 == 0.0: c21 = 0
                                if c21 < 0: continue
                                for n21 in range(c21 + 1):
                                    s += t20 * m21 ** n21 / math.gamma(n21 + 1.0)
    return math.exp(-tot) * s


# --------------------------------------------------------------------------------------------
# copula partial derivatives along the curves, general (l,m)
# --------------------------------------------------------------------------------------------


def _Iarrays(r):
    """Flattened index sets for numba: for pair index p, rows of (i1,i2,i3)."""
    P = pairs(r)
    offs = [0]
    rows = []
    for (l, m) in P:
        I = I_lm(l, m)
        rows.extend(I)
        offs.append(len(rows))
    return np.array(rows, dtype=np.int64), np.array(offs, dtype=np.int64)


@nb.njit(cache=True)
def dC_lm_direct(l, m, q1, q2, th, Irows, o0, o1):
    """(dC_1, dC_2) of C^{(l,m)} at q-coordinates (q1,q2) > 0:
    C = G(q1,q2) = e^{-L} sum_{i in I} a^i/i!, a = (q1+q2-L, L-q2, L-q1);  u_i = Q_l(q_i),
    dC_1 = d_{q1}G / Q_l'(q1),  Q_l'(q) = -e^{-q} q^{l-1}/(l-1)!."""
    L = L_log(q1, q2, th)
    if th == 1.0:
        L1 = 1.0
        L2 = 1.0
    else:
        L1 = (q1 / L) ** (th - 1.0)
        L2 = (q2 / L) ** (th - 1.0)
    a = np.empty(3)
    a[0] = q1 + q2 - L
    a[1] = L - q2
    a[2] = L - q1
    if a[0] < 0.0: a[0] = 0.0
    if a[1] < 0.0: a[1] = 0.0
    if a[2] < 0.0: a[2] = 0.0
    da1 = np.empty(3)
    da1[0] = 1.0 - L1
    da1[1] = L1
    da1[2] = L1 - 1.0
    da2 = np.empty(3)
    da2[0] = 1.0 - L2
    da2[1] = L2 - 1.0
    da2[2] = L2
    P = 0.0
    dP1 = 0.0
    dP2 = 0.0
    for rI in range(o0, o1):
        i1 = Irows[rI, 0]
        i2 = Irows[rI, 1]
        i3 = Irows[rI, 2]
        ifac = math.gamma(i1 + 1.0) * math.gamma(i2 + 1.0) * math.gamma(i3 + 1.0)
        P += a[0] ** i1 * a[1] ** i2 * a[2] ** i3 / ifac
        # derivative of the monomial
        g1 = 0.0
        g2 = 0.0
        if i1 > 0:
            mono = i1 * a[0] ** (i1 - 1) * a[1] ** i2 * a[2] ** i3 / ifac
            g1 += mono * da1[0]
            g2 += mono * da2[0]
        if i2 > 0:
            mono = i2 * a[0] ** i1 * a[1] ** (i2 - 1) * a[2] ** i3 / ifac
            g1 += mono * da1[1]
            g2 += mono * da2[1]
        if i3 > 0:
            mono = i3 * a[0] ** i1 * a[1] ** i2 * a[2] ** (i3 - 1) / ifac
            g1 += mono * da1[2]
            g2 += mono * da2[2]
        dP1 += g1
        dP2 += g2
    # d_{q}G = e^{-L} (-L_i P + dP_i);  divide by Q'(q) = -e^{-q} q^{l-1}/(l-1)!
    dG1 = -L1 * P + dP1
    dG2 = -L2 * P + dP2
    c1 = -math.gamma(l * 1.0) * math.exp(q1 - L) * q1 ** (1.0 - l) * dG1
    c2 = -math.gamma(m * 1.0) * math.exp(q2 - L) * q2 ** (1.0 - m) * dG2
    return c1, c2


@nb.njit(cache=True)
def _only_parts(q1, q2, th):
    """(beta, gamma, J): nu-measures of {z1>1/q1 only}, {z2>1/q2 only}, both; cancellation-free."""
    if th == 1.0:
        return q1, q2, 0.0
    if q2 <= q1:
        r = q2 / q1
        gam = q1 * math.expm1(math.log1p(r ** th) / th)      # L - q1
        L = q1 + gam
        bet = L - q2
    else:
        r = q1 / q2
        bet = q2 * math.expm1(math.log1p(r ** th) / th)      # L - q2
        L = q2 + bet
        gam = L - q1
    J = q1 - bet
    if J < 0.0:
        J = 0.0
    return bet, gam, J


@nb.njit(cache=True)
def _pois_cdf(j, lam):
    if j < 0:
        return 0.0
    s = 0.0
    term = math.exp(-lam)
    for h in range(j + 1):
        if h > 0:
            term *= lam / h
        s += term
    return s


@nb.njit(cache=True)
def _cond_cdf(n, p, lam, j):
    """P(Bin(n,p) + Pois(lam) <= j)."""
    s = 0.0
    for c in range(n + 1):
        if c > j:
            break
        s += math.gamma(n + 1.0) / (math.gamma(c + 1.0) * math.gamma(n - c + 1.0)) * p ** c * (1 - p) ** (n - c) \
            * _pois_cdf(j - c, lam)
    return s


@nb.njit(cache=True)
def dC_lm(l, m, q1, q2, th, Irows, o0, o1):
    """(dC_1, dC_2) of C^{(l,m)} at q-coordinates (q1,q2) > 0, in the cancellation-free form
    dC_1 = Ld_1 P(N_2 <= m-1 | N_1 = l-1) + (1 - Ld_1) P(N_2 <= m-2 | N_1 = l-1),
    where N_i = N{z_i > 1/q_i}; given N_1 = l-1 the joint-exceedance count is Bin(l-1, J/q1), and the
    remaining part of N_2 is Poisson(gamma) (gamma = L - q1).  Symmetrically for dC_2.
    Equivalent to d_{q1}G / Q_l'(q1) with G = C^{(l,m)} in q-coordinates (checked in s05)."""
    bet, gam, J = _only_parts(q1, q2, th)
    L = q1 + gam
    if th == 1.0:
        L1 = 1.0
        L2 = 1.0
    else:
        L1 = (q1 / L) ** (th - 1.0)
        L2 = (q2 / L) ** (th - 1.0)
    p1 = J / q1
    p2 = J / q2
    if p1 > 1.0: p1 = 1.0
    if p2 > 1.0: p2 = 1.0
    c1 = L1 * _cond_cdf(l - 1, p1, gam, m - 1) + (1.0 - L1) * _cond_cdf(l - 1, p1, gam, m - 2)
    c2 = L2 * _cond_cdf(m - 1, p2, bet, l - 1) + (1.0 - L2) * _cond_cdf(m - 1, p2, bet, l - 2)
    return c1, c2


# --------------------------------------------------------------------------------------------
# Sigma(t) by quadrature
# --------------------------------------------------------------------------------------------


def graded_nodes(smax=40.0, npan=40, m=10, smin=1e-7):
    edges = np.concatenate([[0.0], np.geomspace(smin, smax, npan)])
    g, w = np.polynomial.legendre.leggauss(m)
    S, W = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        S.append(0.5 * (hi - lo) * g + 0.5 * (hi + lo))
        W.append(0.5 * (hi - lo) * w)
    return np.concatenate(S), np.concatenate(W)


@nb.njit(parallel=True, cache=True)
def _sigma_kernel(t, th, S, WS, Wn, WW, lv, mv, Irows, Ioff):
    npr = lv.shape[0]
    ns = S.shape[0]
    nw = Wn.shape[0]
    out = np.zeros((ns, npr, npr))
    for ii in nb.prange(ns):
        s = S[ii]
        # quantities at u (fixed s)
        cu = np.zeros((npr, 3))
        pu = np.zeros((npr, 3))
        qa_u = (1.0 - t) * s
        qb_u = t * s
        for p in range(npr):
            l = lv[p]
            m = mv[p]
            c1, c2 = dC_lm(l, m, qa_u, qb_u, th, Irows, Ioff[p], Ioff[p + 1])
            cu[p, 0] = 1.0
            cu[p, 1] = -c1
            cu[p, 2] = -c2
            pu[p, 0] = prob_counts(qa_u, l - 1, 0.0, BIG, qb_u, m - 1, 0.0, BIG, th)
            pu[p, 1] = prob_counts(qa_u, l - 1, 0.0, BIG, 0.0, BIG, 0.0, BIG, th)
            pu[p, 2] = prob_counts(0.0, BIG, 0.0, BIG, qb_u, m - 1, 0.0, BIG, th)
        acc = np.zeros((npr, npr))
        cv = np.zeros((npr, 3))
        pv = np.zeros((npr, 3))
        for jj in range(nw):
            sp = s + Wn[jj]
            wt = WS[ii] * WW[jj] * math.exp(-s - sp)
            qa_v = (1.0 - t) * sp
            qb_v = t * sp
            for p in range(npr):
                l = lv[p]
                m = mv[p]
                c1, c2 = dC_lm(l, m, qa_v, qb_v, th, Irows, Ioff[p], Ioff[p + 1])
                cv[p, 0] = 1.0
                cv[p, 1] = -c1
                cv[p, 2] = -c2
                pv[p, 0] = prob_counts(qa_v, l - 1, 0.0, BIG, qb_v, m - 1, 0.0, BIG, th)
                pv[p, 1] = prob_counts(qa_v, l - 1, 0.0, BIG, 0.0, BIG, 0.0, BIG, th)
                pv[p, 2] = prob_counts(0.0, BIG, 0.0, BIG, qb_v, m - 1, 0.0, BIG, th)
            for p in range(npr):
                l = lv[p]
                m = mv[p]
                for pp in range(npr):
                    l2 = lv[pp]
                    m2 = mv[pp]
                    K = 0.0
                    for e in range(3):
                        # event e at u: 0 = A (both), 1 = B (margin 1), 2 = C (margin 2)
                        a1 = qa_u if e != 2 else 0.0
                        k1 = l - 1 if e != 2 else BIG
                        b1 = qb_u if e != 1 else 0.0
                        j1 = m - 1 if e != 1 else BIG
                        for f in range(3):
                            a2 = qa_v if f != 2 else 0.0
                            k2 = l2 - 1 if f != 2 else BIG
                            b2 = qb_v if f != 1 else 0.0
                            j2 = m2 - 1 if f != 1 else BIG
                            pj = prob_counts(a1, k1, a2, k2, b1, j1, b2, j2, th)
                            K += cu[p, e] * cv[pp, f] * (pj - pu[p, e] * pv[pp, f])
                    acc[p, pp] += wt * K
        out[ii] = acc
    return out


def sigma(t, theta, r, smax=40.0, npan=40, m=10):
    """Sigma(t) (r^2 x r^2), rank-corrected covariance, eq. (Gamma), logistic attractor."""
    P = pairs(r)
    lv = np.array([p[0] for p in P], dtype=np.int64)
    mv = np.array([p[1] for p in P], dtype=np.int64)
    Irows, Ioff = _Iarrays(r)
    S, WS = graded_nodes(smax, npan, m)
    Wn, WW = graded_nodes(smax, npan, m)
    out = _sigma_kernel(float(t), float(theta), S, WS, Wn, WW, lv, mv, Irows, Ioff)
    I = out.sum(axis=0)
    return I + I.T


# --------------------------------------------------------------------------------------------
# second-order models (all exchangeable, GH(theta) attractor unless stated)
# S(e^{-q}) = kappa * e^{-L(q)} * Psi(q), Psi homogeneous of degree D; alpha(b) given
# --------------------------------------------------------------------------------------------


def Gamma2(t, A, v):
    """Gamma(1-t,t) = sum_j x_j^2 Ldot_j at x = (1-t,t)  (BVZ Lemma 2.5(ii))."""
    return A * (1 - 2 * t + 2 * t * t) + t * (1 - t) * (2 * t - 1) * v


def Gamma3(t, A, v):
    """sum_j x_j^3 Ldot_j at x = (1-t,t)."""
    return A * (1 - 3 * t + 3 * t * t) + t * (1 - t) * (2 * t - 1) * v


MODELS = {
    # name: dict(kappa, D, Psi(t,A,v), alpha(b) as (const, power), pot: (const, power, Sp(t,A,v)))
    'GH': dict(label='Gumbel--Hougaard(2)', theta=2.0, kappa=0.0, D=2,
               Psi=lambda t, A, v: 0.0 * t, alpha=(1.0, -1.0),
               pot=(0.5, -1.0, lambda t, A, v: -(A * A - Gamma2(t, A, v)))),
    'Joe': dict(label='Joe(2)', theta=2.0, kappa=-1.0, D=2,
                Psi=lambda t, A, v: A * A - Gamma2(t, A, v), alpha=(0.5, -1.0),
                pot=(1.0, -2.0, lambda t, A, v: -0.5 * (1 - t) ** 2 * t ** 2 / A)),
    'OPC': dict(label='outer-power Clayton (BVZ theta=1, beta=2)', theta=2.0, kappa=1.0, D=2,
                Psi=lambda t, A, v: A * A - Gamma2(t, A, v), alpha=(0.5, -1.0),
                pot=(1.0, -1.0, lambda t, A, v: -1.0 * (A * A - Gamma2(t, A, v)))),
    'Archimax': dict(label='Archimax psi_2, L0 logistic(2)', theta=2.0, kappa=1.0 / 6.0, D=3,
                     Psi=lambda t, A, v: A ** 3 - Gamma3(t, A, v), alpha=(1.0, -2.0),
                     pot=(0.5, -1.0, lambda t, A, v: -(A * A - Gamma2(t, A, v)))),
}


def calS_vec(model, t, r):
    M = MODELS[model] if isinstance(model, str) else model
    th = M['theta']
    A = float(A_log(t, th))
    v = float(dA_log(t, th))
    psi = M['Psi'](t, A, v)
    return np.array([M['kappa'] * psi * calS_shape(A, t, l, m, M['D']) for (l, m) in pairs(r)])


def pot_var(t, theta):
    """Var{A_p(t)} in BVZ Prop. 4.1(i):  -A + l1^2(1-t) + l2^2 t + 2 l1 l2 (1-A)."""
    A = A_log(t, theta)
    v = dA_log(t, theta)
    l1 = A - t * v
    l2 = A + (1 - t) * v
    return -A + l1 ** 2 * (1 - t) + l2 ** 2 * t + 2 * l1 * l2 * (1 - A)


def pot_bias_fn(model, t):
    M = MODELS[model]
    th = M['theta']
    A = A_log(t, th)
    v = dA_log(t, th)
    return M['pot'][2](t, A, v)
