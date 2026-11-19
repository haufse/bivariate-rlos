"""s10: sensitivity of the O(1/k) bias coefficient eta_W[A](t) to the Pickands function A.

Purpose (revision after H1-H3): the bootstrap 1/k correction subtracts
eta_W[A_tilde](t)/k, where A_tilde is the Bernstein pilot.  Its extra variance is driven by the
derivative of A -> eta_W[A](t) at A0 (GH(2)), at t = 1/2.  This script computes that derivative.

Structure used (derived in tex/sec8-bias-supp-body.tex, Remark sec8-bias:rem:bck-variance):
  eta_W[A](t) = lambda_W(a)' Q[a, a1, a2](t) + nu_W(a, Sigma[A](t)),
  with a = A(t), a1 = A'(t), a2 = A''(t).  The rank-bias vector Q^{(l,m)}(t) depends on A only
  through L(y1, y2) = (y1+y2) A(y2/(y1+y2)) and its first two derivatives on the ray
  y = ((1-t)x, t x), i.e. through (a, a1, a2); lambda_W (W fixed) and the moment map depend on a
  only; nu_W depends on Sigma[A](t), which is a global functional of A.
Outputs per weights w in {r1, or (W = Sigma_GH2^{-1}(1/2), fixed), id}:
  F_a  = d eta / d a,   F_a1 = d eta / d a1,   F_a2 = d eta / d a2   (local part, nu at fixed Sigma),
  N_GH = d nu_W / d A(1/2) through Sigma along the GH(theta) family (global part, GH direction),
  slope_GH = total d eta / d A(1/2) along the GH(theta) family, and the IND chord for comparison.
Checks: Q from the local-quadratic L reproduces biaslib.beta_C for GH2 (t = 0.3, 0.5) and IND;
  finite-difference steps h = 1e-3 and 1e-4 agree; 30-digit mpmath check of F_a2 (r1).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-9 nice -n 19 \
     python s10_eta_derivative.py
Writes ../outputs/s10_output.txt and ../outputs/s10_eta_derivative.csv.  Deterministic.
"""
import math
import sys
from functools import lru_cache

import mpmath as mp
import numpy as np
import pandas as pd
import sympy as sp
from scipy import integrate

import biaslib as B

sys.path.insert(0, str(B.SEC8))
import sigma2_lib  # noqa: E402  (Sec. 8 oracle quadrature, read-only use)

OUT = B.HERE.parent / "outputs"
_y1, _y2, _a, _a1, _a2, _t0 = sp.symbols("y1 y2 a a1 a2 t0", real=True)
LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


@lru_cache(maxsize=None)
def D_local(l, m):
    """D^{(l,m)} (biaslib / eq. sec8-bias:eq:D) for L = (y1+y2) A(s), A locally quadratic at t0."""
    s = _y2 / (_y1 + _y2)
    L = (_y1 + _y2) * (_a + _a1 * (s - _t0) + _a2 * (s - _t0) ** 2 / 2)
    lc, la, lb = _y1 + _y2 - L, L - _y2, L - _y1
    S = 0
    for c in range(min(l, m)):
        for i in range(l - c):
            for j in range(m - c):
                S += lc ** c * la ** i * lb ** j / (math.factorial(c) * math.factorial(i) * math.factorial(j))
    G = sp.exp(-L) * S
    p1 = -sp.exp(-_y1) * _y1 ** (l - 1) / math.factorial(l - 1)
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
    args = (_y1, _y2, _a, _a1, _a2, _t0)
    return sp.lambdify(args, D, "numpy"), sp.lambdify(args, D, "mpmath")


def Q_local(t, a, a1, a2, pairs=B.PAIRS):
    """Q^{(l,m)}(t) = int_0^inf D((1-t)x, t x) e^{-x} dx (same quadrature as biaslib.beta_C)."""
    out = []
    for (l, m) in pairs:
        Df = D_local(l, m)[0]
        f = lambda x: float(Df((1 - t) * x, t * x, a, a1, a2, t)) * math.exp(-x)
        x0 = 1e-6
        v = x0 * f(x0)
        for lo_, hi_ in ((x0, 1.0), (1.0, 4.0), (4.0, 16.0), (16.0, 80.0)):
            v += integrate.quad(f, lo_, hi_, limit=400, epsabs=1e-14, epsrel=1e-12)[0]
        out.append(v)
    return np.array(out)


def Q_local_mp(t, a, a1, a2, pair, dps=30):
    mp.mp.dps = dps
    Dm = D_local(*pair)[1]
    tt, a, a1, a2 = mp.mpf(t), mp.mpf(a), mp.mpf(a1), mp.mpf(a2)
    f = lambda x: Dm((1 - tt) * x, tt * x, a, a1, a2, tt) * mp.e ** (-x)
    x0 = mp.mpf("1e-6")
    return x0 * f(x0) + mp.quad(f, [x0, 0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 30, 60])


def gh_local(th, t):
    """(A, A', A'') of GH(theta) at t."""
    A = (t ** th + (1 - t) ** th) ** (1 / th)
    g = t ** (th - 1) - (1 - t) ** (th - 1)
    A1 = g * A ** (1 - th)
    g1 = (th - 1) * (t ** (th - 2) + (1 - t) ** (th - 2))
    A2 = g1 * A ** (1 - th) + g * (1 - th) * A ** (-th) * A1
    return A, A1, A2


def eta_parts(t, a, a1, a2, Sig, W):
    """(rank term lambda'Q, nu_W) for fixed W (4x4) and Sigma (4x4) at a single t."""
    d = -B.mu_dot(a, t)
    lam = B.lam_W(W, d)
    Q = Q_local(t, a, a1, a2)
    nu = float(B.nonlin_term(W[None], Sig[None], np.array([a]), np.array([t]))[0])
    return float(lam @ Q), nu


def main():
    t = 0.5
    T = B.T
    j5 = int(np.argmin(np.abs(T - t)))
    Sig_gh = B.oracle_sigma("GH2")[j5]
    W = {"r1": B.weights("r1", Sig_gh[None])[0], "or": np.linalg.inv(Sig_gh), "id": np.eye(4)}

    # ---- checks of the local construction
    log("CHECKS")
    for tt in (0.3, 0.5):
        a, a1, a2 = gh_local(2.0, tt)
        q_loc, q_ref = Q_local(tt, a, a1, a2), B.beta_C(tt, "GH2")
        log(f"  GH2 t={tt}: Q local {np.round(q_loc, 6)}  biaslib {np.round(q_ref, 6)}  "
            f"max|diff| {np.max(np.abs(q_loc - q_ref)):.1e}")
    q_loc, q_ref = Q_local(0.5, 1.0, 0.0, 0.0), B.beta_C(0.5, "IND")
    log(f"  IND t=0.5: Q local {np.round(q_loc, 6)}  biaslib {np.round(q_ref, 6)}  "
        f"max|diff| {np.max(np.abs(q_loc - q_ref)):.1e}")

    a0, a10, a20 = gh_local(2.0, t)
    log(f"  GH2 at t=1/2: A = {a0:.10f}, A' = {a10:.3g}, A'' = {a20:.10f}")

    # ---- local partial derivatives (nu at fixed Sigma = Sigma_GH2)
    rows = []
    for h in (1e-3, 1e-4):
        for w, Ww in W.items():
            def eta(a, a1, a2):
                rk, nu = eta_parts(t, a, a1, a2, Sig_gh, Ww)
                return rk + nu
            Fa = (eta(a0 + h, a10, a20) - eta(a0 - h, a10, a20)) / (2 * h)
            Fa1 = (eta(a0, a10 + h, a20) - eta(a0, a10 - h, a20)) / (2 * h)
            Fa2 = (eta(a0, a10, a20 + h) - eta(a0, a10, a20 - h)) / (2 * h)
            rows.append(dict(h=h, weights=w, eta=eta(a0, a10, a20), F_a=Fa, F_a1=Fa1, F_a2=Fa2))
    loc = pd.DataFrame(rows)
    log("\nLOCAL PARTIALS of eta_W(1/2) at GH2 (Sigma held at Sigma_GH2; W fixed):")
    log(loc.round(5).to_string(index=False))

    # 30-digit check of the r1 partial in a2 (pair (1,1); eta_r1 = lam*Q + nu, lam = -(1+a)^2)
    h = mp.mpf("1e-6")
    mp.mp.dps = 30
    dq = (Q_local_mp(t, a0, a10, a20 + 1e-6, (1, 1)) - Q_local_mp(t, a0, a10, a20 - 1e-6, (1, 1))) / (2 * h)
    lam_r1 = -(1 + a0) ** 2
    log(f"  mpmath30 check F_a2(r1) = lam * dQ11/da2 = {float(lam_r1 * dq):.6f}")

    # ---- GH(theta) direction: total slope and the Sigma (nu) part
    log("\nGH(theta) DIRECTION at t=1/2 (theta = 2 -/+ dth; W fixed at the GH2 choice):")
    dth = 1e-3
    gh_rows = []
    sig_th = {th: sigma2_lib.sigma2(sigma2_lib.Gumbel(th), t) for th in (2 - dth, 2 + dth)}
    for w, Ww in W.items():
        vals, vals_fixS, As = {}, {}, {}
        for th in (2 - dth, 2 + dth):
            a, a1, a2 = gh_local(th, t)
            rk, nu = eta_parts(t, a, a1, a2, sig_th[th], Ww)
            rk2, nu2 = eta_parts(t, a, a1, a2, Sig_gh, Ww)
            vals[th], vals_fixS[th], As[th] = rk + nu, rk2 + nu2, a
        dA = As[2 + dth] - As[2 - dth]
        slope = (vals[2 + dth] - vals[2 - dth]) / dA
        slope_fixS = (vals_fixS[2 + dth] - vals_fixS[2 - dth]) / dA
        A2p, A2m = gh_local(2 + dth, t)[2], gh_local(2 - dth, t)[2]
        gh_rows.append(dict(weights=w, slope_GH=slope, slope_GH_SigmaFixed=slope_fixS,
                            N_GH=slope - slope_fixS, dA2_dA_GH=(A2p - A2m) / dA))
    gh = pd.DataFrame(gh_rows)
    log(gh.round(4).to_string(index=False))
    log("  (slope_GH_SigmaFixed should equal F_a + F_a2 * dA2_dA_GH; N_GH = nu-part through Sigma)")
    l4 = loc[loc.h == 1e-4].set_index("weights")
    for _, rr in gh.iterrows():
        pred = l4.loc[rr.weights, "F_a"] + l4.loc[rr.weights, "F_a2"] * rr.dA2_dA_GH
        log(f"  {rr.weights}: F_a + F_a2*dA2/dA = {pred:.4f}  vs  slope_GH_SigmaFixed {rr.slope_GH_SigmaFixed:.4f}")

    # IND chord (eta values from s02) for comparison with the earlier h2b reference
    s02 = pd.read_csv(OUT / "s02_theory.csv")
    s02 = s02[np.isclose(s02.t, 0.5)].set_index(["attractor", "weights"])["c_total"]
    log("\nIND chord (eta_IND - eta_GH2)/(1 - A_GH2(1/2)): " + ", ".join(
        f"{w} {(s02[('IND', w)] - s02[('GH2', w)]) / (1 - a0):.3f}" for w in W))

    out = l4.reset_index().merge(gh, on="weights")
    out["A0"], out["A2_0"] = a0, a20
    out.to_csv(OUT / "s10_eta_derivative.csv", index=False, float_format="%.10g")
    (OUT / "s10_output.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
