"""s04: exact k = infinity (finite block size b) bias of the r1 and r2 estimators, models M1-M4.

For each model, b in the Sec. 8 grid and t in T, computes the exact population moments
    mu_b^{(l,m)}(t) = 1 - int_0^infty C_b^{(l,m)}(h^{(l,m)}(x,t)) e^{-x} dx,
where C_b^{(l,m)} is the copula of (l-th largest of margin 1, m-th largest of margin 2) in a
block of b iid draws from the model copula C (multinomial cell counts, as in
an earlier check script (finiteb_gh.py, not included), here vectorised and written in terms
of s = -log a to avoid cancellation), and then the k = infinity estimators
    r1:   mu_b^{(1,1)} / (1 - mu_b^{(1,1)}),
    r2or: argmin ||mu(a,t) - mu_b(t)||_W, W = Sigma_0(t)^{-1} (Sec. 8 oracle cache),
    r2id: same with W = I.
Columns: exact = A_inf(b,t) - A0(t); lead = leading-order bias lam'(alpha(b) calS + calR/b)
(Sec. 7; for r1 lam_1 * alpha(b) * calS^{(1,1)}).

Run: OMP_NUM_THREADS=1 NUMBA_CACHE_DIR=<scratch> taskset -c 0-9 nice -n 19 \
     python s04_exact_finite_b.py > ../outputs/s04_output.txt
Deterministic.  Writes ../outputs/s04_exact_bias.csv.
"""
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import gammaln, gammaincc

import biaslib as B

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[3] / ".cache" / "numba"))
sys.path.insert(0, str(B.SEC7 / "scripts"))
import rlos_lib  # noqa: E402  (calS_vec, MODELS; read-only use)

BGRID = [5, 10, 20, 40, 80, 160, 320]


# ----------------------------------------------------------------------------- parent copulas
# each returns 1 - C(a1, a2) given s_i = -log a_i >= 0 (accurate for a_i near 1)
def omc_gh(s1, s2, th=2.0):
    return -np.expm1(-(s1 ** th + s2 ** th) ** (1 / th))


def omc_joe(s1, s2, th=2.0):
    x, y = -np.expm1(-s1), -np.expm1(-s2)        # 1 - a
    return (x ** th + y ** th - x ** th * y ** th) ** (1 / th)


def omc_opc(s1, s2, theta=1.0, beta=2.0):
    # C = [1 + {(a1^-theta - 1)^beta + (a2^-theta - 1)^beta}^{1/beta}]^{-1/theta}
    D = (np.expm1(theta * s1) ** beta + np.expm1(theta * s2) ** beta) ** (1 / beta)
    return -np.expm1(-np.log1p(D) / theta)


def _psi2(x):
    return np.where(x <= .5, 1 - x + x * x / 2, np.maximum(7 / 8 - x / 2, 0))


def omc_archimax(s1, s2):
    # C = psi2(L0(psi2^{-1}(a1), psi2^{-1}(a2))), L0 logistic(2); 1-a = -expm1(-s)
    om = [-np.expm1(-s) for s in (s1, s2)]
    xs = []
    for o in om:
        u = 1 - o
        root = np.sqrt(np.maximum(1 - 2 * o, 0))             # sqrt(2u-1)
        xs.append(np.where(u >= 5 / 8, 2 * o / (1 + root), 7 / 4 - 2 * u))
    L = np.hypot(xs[0], xs[1])
    # 1 - psi2(L): for L <= 1/2, L - L^2/2 (no cancellation)
    return np.where(L <= .5, L - L * L / 2, 1 - _psi2(L))


MODELS = {"M1": omc_gh, "M2": omc_joe, "M3": omc_opc, "M4": omc_archimax}
LEAD = {"M1": "GH", "M2": "Joe", "M3": "OPC", "M4": "Archimax"}


# ----------------------------------------------------------------------------- C_b^{(l,m)}
def s_of_v(v, l, b):
    """s = -log a with P(at most l-1 of b iid U(0,1) exceed a) = v."""
    lv = np.log(v)
    if l == 1:
        return -lv / b
    lo, hi = np.zeros_like(v), np.full_like(v, 1.0)
    f = lambda s: -(b - 1) * s + np.log(np.exp(-s) - b * np.expm1(-s)) - lv
    while np.any(f(hi) > 0):
        hi = np.where(f(hi) > 0, 2 * hi, hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def Cb(l, m, b, v1, v2, omc):
    s1, s2 = s_of_v(v1, l, b), s_of_v(v2, m, b)
    o1, o2 = -np.expm1(-s1), -np.expm1(-s2)            # 1 - a_i
    oc = omc(s1, s2)                                     # 1 - C(a1,a2)
    p11 = np.maximum(o1 + o2 - oc, 0)                    # both exceed
    p10 = np.maximum(oc - o2, 0)                         # only margin 1 exceeds
    p01 = np.maximum(oc - o1, 0)                         # only margin 2 exceeds
    lp00 = np.log1p(-oc)
    tot = np.zeros_like(v1)
    for n11 in range(min(l, m)):
        for n10 in range(l - n11):
            for n01 in range(m - n11):
                n00 = b - n11 - n10 - n01
                lc = gammaln(b + 1) - gammaln(n11 + 1) - gammaln(n10 + 1) - gammaln(n01 + 1) - gammaln(n00 + 1)
                # Fix 2026-11-18 (revision after H1-H3): for n00 = 0 and 1 - C = 1 (lp00 = -inf)
                # the product 0 * lp00 is NaN; the correct contribution is 0.  Only b = l + m - 1
                # or smaller can give n00 = 0 (e.g. r = 2 at b = 2); the Sec. 8 grid (b >= 5)
                # used here is unaffected.  Same fix as th_theory.py (follow-up analysis).
                term = lc + (n00 * lp00 if n00 else 0.0)
                with np.errstate(divide="ignore"):
                    for n_, p_ in ((n11, p11), (n10, p10), (n01, p01)):
                        if n_:
                            term = term + n_ * np.log(p_)      # -inf where p_ = 0 -> exp = 0
                tot += np.exp(term)
    return tot


def _nodes(npan=40):
    edges = np.array([0, 0.02, 0.1, 0.3, 0.7, 1.5, 3, 6, 12, 24, 45, 80.0])
    x, w = np.polynomial.legendre.leggauss(npan)
    X, Wt = [], []
    for a, c in zip(edges[:-1], edges[1:]):
        X.append(a + (c - a) * (x + 1) / 2)
        Wt.append(w * (c - a) / 2)
    return np.concatenate(X), np.concatenate(Wt)


def mu_b(model, b, t, npan=40):
    X, Wt = _nodes(npan)
    out = []
    for (l, m) in B.PAIRS:
        v1 = gammaincc(l, (1 - t) * X)
        v2 = gammaincc(m, t * X)
        out.append(1 - np.sum(Cb(l, m, b, v1, v2, MODELS[model]) * np.exp(-X) * Wt))
    return np.array(out)


def main():
    Sig = B.oracle_sigma("GH2")
    A0 = B.A_log(B.T)
    d = -B.mu_dot(A0, B.T)
    Wd = {"r2or": B.weights("or", Sig), "r2id": B.weights("id", Sig)}
    lam = {"r1": B.lam_W(B.weights("r1", Sig), d), **{k: B.lam_W(W, d) for k, W in Wd.items()}}
    R = B.calR(B.T)
    rows = []
    # quadrature check: limit moments (C_b -> C) and node doubling
    mu0 = B.mu(A0, B.T)
    chk = np.array([mu_b("M1", 10 ** 6, t) for t in (0.3, 0.5)]) - mu0[[4, 8]]
    print("CHECK b=1e6 GH moments minus mu(A0): max |.| = %.2e" % np.abs(chk).max())
    chk2 = mu_b("M3", 10, 0.3, npan=80) - mu_b("M3", 10, 0.3, npan=40)
    print("CHECK node doubling M3 b=10 t=0.3: max |.| = %.2e" % np.abs(chk2).max())
    for model in MODELS:
        M = rlos_lib.MODELS[LEAD[model]]
        S = np.array([rlos_lib.calS_vec(LEAD[model], float(t), 2) for t in B.T])
        for b in BGRID + ([1000] if model in ("M1", "M3") else []):
            mub = np.array([mu_b(model, b, float(t)) for t in B.T])
            alpha = M["alpha"][0] * b ** M["alpha"][1]
            est = {"r1": mub[:, 0] / (1 - mub[:, 0])}
            for k_, W in Wd.items():
                est[k_] = B.gmm_solve(mub, W, B.T, lo=0.5 * np.ones_like(B.T),
                                      hi=1.3 * np.ones_like(B.T), grid=41, levels=10)
            for name, a in est.items():
                lead = np.einsum("ti,ti->t", lam[name], alpha * S + (R / b if name != "r1" else 0 * R))
                lin = np.einsum("ti,ti->t", lam[name], mu0 - mub)
                for i, t in enumerate(B.T):
                    rows.append(dict(model=model, b=b, t=t, estimator=name, exact=a[i] - A0[i],
                                     exact_lin=lin[i], lead=lead[i]))
            print(model, b, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(B.HERE.parent / "outputs" / "s04_exact_bias.csv", index=False, float_format="%.10g")
    sel = df[np.isclose(df.t, 0.3) | np.isclose(df.t, 0.5)].copy()
    sel["ratio"] = sel.exact / sel.lead
    pd.set_option("display.width", 200)
    print(sel.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
