"""Shared library for the Section 8 bias investigation (2026-11-18).

Notation follows the manuscript (Sections 3-5, 7):
  mu(a,t)      moment map, eq. (mu-r2), pair order (1,1),(1,2),(2,1),(2,2)
  d(t)         = -d/da mu(a,t) at a = A0(t), eq. (d)
  lambda_W(t)  = W d / (d' W d), eq. (lambda-W)
  Sigma(t)     rank covariance, eq. (Gamma); oracle values from the Sec. 8 cache
  calR(t)      eq. (integrated-R-r2)
The new objects of this note are
  beta_C(t)    = lim_k k * { int E C_hat_k(h(w,t)) dw - int C(h(w,t)) dw }   (rank bias, per pair)
  n_W(t)       second-order (nonlinearity) term of the GMM estimator
  c_W(t)       = lambda_W' beta_C + n_W   (total O(1/k) bias coefficient, fixed weights)
see the working note of the investigation (not included; the results are in the paper's supplement).
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate

HERE = Path(__file__).resolve().parent
P = HERE.parents[2]                      # repository root
SEC8 = P / "sec8_simulations"
SEC7 = P / "sec7_asymptotics"
T = np.round(np.arange(0.10, 0.9001, 0.05), 2)
PAIRS = [(1, 1), (1, 2), (2, 1), (2, 2)]


# ----------------------------------------------------------------------------- truth
def A_log(t, th=2.0):
    t = np.asarray(t, float)
    return (t ** th + (1 - t) ** th) ** (1 / th)


def dA_log(t, th=2.0):
    t = np.asarray(t, float)
    return (t ** (th - 1) - (1 - t) ** (th - 1)) * A_log(t, th) ** (1 - th)


# ----------------------------------------------------------------------------- moment map (sympy)
_a, _t = sp.symbols("a t", positive=True)
_MU = sp.Matrix([_a / (1 + _a), (_a ** 2 + 1 - _t) / (1 + _a) ** 2, (_a ** 2 + _t) / (1 + _a) ** 2,
                 (_a ** 3 - _a ** 2 + 2 * _a + 2 * _t ** 2 - 2 * _t) / (1 + _a) ** 3])
_mu = [sp.lambdify((_a, _t), e, "numpy") for e in _MU]
_mu1 = [sp.lambdify((_a, _t), sp.simplify(e.diff(_a)), "numpy") for e in _MU]
_mu2 = [sp.lambdify((_a, _t), sp.simplify(e.diff(_a, 2)), "numpy") for e in _MU]


def _vec(fs, a, t):
    a, t = np.broadcast_arrays(np.asarray(a, float), np.asarray(t, float))
    return np.stack([np.broadcast_to(np.asarray(f(a, t), float), a.shape) for f in fs], -1)


def mu(a, t):
    return _vec(_mu, a, t)


def mu_dot(a, t):
    """d/da mu = -d."""
    return _vec(_mu1, a, t)


def mu_ddot(a, t):
    return _vec(_mu2, a, t)


def calR(t, a=None, v=None):
    t = np.asarray(t, float)
    a = A_log(t) if a is None else a
    v = dA_log(t) if v is None else v
    t, a, v = np.broadcast_arrays(t, np.asarray(a, float), np.asarray(v, float))
    s = 1 + a
    return np.stack([0 * t, (a + t - 1) * (1 - t) * (a - 1 - t * v) / s ** 3,
                     t * (a - t) * (a - 1 + (1 - t) * v) / s ** 3,
                     3 * (a - 1) * (a - t) * (a + t - 1) / s ** 4], -1)


# ----------------------------------------------------------------------------- oracle Sigma and weights
def oracle_sigma(tag="GH2"):
    z = np.load(SEC8 / "cache" / f"oracle_sigma_{tag}_r2.npz")
    assert np.allclose(z["T"], T)
    return z["Sigma"]


def lam_W(W, d):
    Wd = np.einsum("...ij,...j->...i", W, d)
    return Wd / np.einsum("...i,...i->...", d, Wd)[..., None]


def weights(kind, Sigma):
    if kind == "or":
        return np.linalg.inv(Sigma)
    if kind == "id":
        return np.broadcast_to(np.eye(4), Sigma.shape).copy()
    if kind == "r1":
        W = np.zeros_like(Sigma)
        W[..., 0, 0] = 1.0
        return W
    raise ValueError(kind)


def nonlin_term(W, Sigma, a, t):
    """n_W(t) = (3/2)(lam' mu'')(lam' Sigma lam) - mu''' W Sigma lam / (d' W d)."""
    d = -mu_dot(a, t)
    lam = lam_W(W, d)
    m2 = mu_ddot(a, t)
    s2 = np.einsum("...i,...ij,...j->...", lam, Sigma, lam)
    H = np.einsum("...i,...ij,...j->...", d, W, d)
    WSl = np.einsum("...ij,...jk,...k->...i", W, Sigma, lam)
    return 1.5 * np.einsum("...i,...i->...", lam, m2) * s2 - np.einsum("...i,...i->...", m2, WSl) / H


# ----------------------------------------------------------------------------- rank-bias functional beta_C
_y1, _y2 = sp.symbols("y1 y2", positive=True)


@lru_cache(maxsize=None)
def _G_funcs(l, m, attractor):
    """CDF G(y1,y2)=P(N1<l, N2<m) of the limiting r-LOS at Frechet levels 1/y, and derivatives.

    N1 = Nc+Na, N2 = Nc+Nb, Nc~Poi(y1+y2-L), Na~Poi(L-y2), Nb~Poi(L-y1)."""
    if attractor == "GH2":
        L = sp.sqrt(_y1 ** 2 + _y2 ** 2)
    elif attractor == "IND":
        L = _y1 + _y2
    else:
        raise ValueError(attractor)
    lc, la, lb = _y1 + _y2 - L, L - _y2, L - _y1
    s = 0
    for c in range(min(l, m)):
        for a in range(l - c):
            for b in range(m - c):
                s += lc ** c * la ** a * lb ** b / (math.factorial(c) * math.factorial(a)
                                                    * math.factorial(b))
    G = sp.exp(-L) * s
    p1 = -sp.exp(-_y1) * _y1 ** (l - 1) / math.factorial(l - 1)    # du1/dy1
    p2 = -sp.exp(-_y2) * _y2 ** (m - 1) / math.factorial(m - 1)
    u1 = sp.exp(-_y1) * sum(_y1 ** j / math.factorial(j) for j in range(l))
    u2 = sp.exp(-_y2) * sum(_y2 ** j / math.factorial(j) for j in range(m))
    G1, G2 = G.diff(_y1), G.diff(_y2)
    C1, C2 = G1 / p1, G2 / p2
    C11 = (G.diff(_y1, 2) * p1 - G1 * p1.diff(_y1)) / p1 ** 3
    C22 = (G.diff(_y2, 2) * p2 - G2 * p2.diff(_y2)) / p2 ** 3
    C12 = G.diff(_y1, _y2) / (p1 * p2)
    D = (-sp.Rational(1, 2) * (1 - 2 * u1) * C1 - sp.Rational(1, 2) * (1 - 2 * u2) * C2
         + sp.Rational(1, 2) * u1 * (1 - u1) * C11 + sp.Rational(1, 2) * u2 * (1 - u2) * C22
         + (G - u1 * u2) * C12)
    return (sp.lambdify((_y1, _y2), D, "numpy"), sp.lambdify((_y1, _y2), D, "mpmath"),
            sp.lambdify((_y1, _y2), G, "numpy"))


def beta_C(t, attractor="GH2", pairs=PAIRS, method="float"):
    """beta_C^{(l,m)}(t) = int_0^infty Dfun(u(x)) e^{-x} dx along y=((1-t)x, t x).

    method="float": scipy quad (agrees with 50-digit mpmath to 1e-10 for l,m <= 2);
    method="mp": mpmath quad at 40 digits (needed for l or m = 3, where the float
    integrand loses digits near x = 0)."""
    if method == "mp":
        import mpmath as mp
        mp.mp.dps = 40
        res = []
        for (l, m) in pairs:
            Dm = _G_funcs(l, m, attractor)[1]
            tt = mp.mpf(t)
            f = lambda x: Dm((1 - tt) * x, tt * x) * mp.e ** (-x)
            x0 = mp.mpf("1e-6")
            res.append(float(x0 * f(x0) + mp.quad(f, [x0, 0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 30, 60])))
        return np.array(res)
    out = []
    for (l, m) in pairs:
        D = _G_funcs(l, m, attractor)[0]
        f = lambda x: float(D((1 - t) * x, t * x)) * math.exp(-x)
        # [0, x0]: the float integrand loses digits for x < 1e-6 (1-u ~ x^2); the piece is
        # x0*D(x0) up to O(x0^2) (D is continuous at 0).  Checked against 50-digit mpmath (s02).
        x0 = 1e-6
        v = x0 * f(x0)
        for lo_, hi_ in ((x0, 1.0), (1.0, 4.0), (4.0, 16.0), (16.0, 80.0)):
            v += integrate.quad(f, lo_, hi_, limit=400, epsabs=1e-14, epsrel=1e-12)[0]
        out.append(v)
    return np.array(out)


# ----------------------------------------------------------------------------- GMM solver (unconstrained option)
def gmm_solve(target, W, T_, lo=None, hi=None, grid=101, levels=6):
    """Zoom grid search of ||mu(a,t)-target||_W^2 over [lo,hi] (default Theta_t); vectorised."""
    target = np.asarray(target, float)
    T_ = np.broadcast_to(np.asarray(T_, float), target.shape[:-1])
    lo = np.broadcast_to(np.maximum(T_, 1 - T_) if lo is None else lo, T_.shape).astype(float).copy()
    hi = np.broadcast_to(1.0 if hi is None else hi, T_.shape).astype(float).copy()
    lo0, hi0 = lo.copy(), hi.copy()
    W = np.broadcast_to(W, target.shape[:-1] + (4, 4))
    g = np.linspace(0, 1, grid)
    for _ in range(levels):
        a = lo[..., None] + (hi - lo)[..., None] * g
        dd = mu(a, T_[..., None]) - target[..., None, :]
        f = np.einsum("...gi,...ij,...gj->...g", dd, W, dd)
        j = np.argmin(f, -1)
        step = (hi - lo) / (grid - 1)
        best = np.take_along_axis(a, j[..., None], -1)[..., 0]
        lo, hi = np.maximum(lo0, best - step), np.minimum(hi0, best + step)
    return best
