"""th_theory: deterministic theory inputs for the follow-up analysis (no Monte Carlo).

(a) exact k = infinity block bias beta_{b,W}(t) of r1, r2 (oracle Sigma_2^{-1}) and r2 (identity)
    for M1-M4 at the block lengths of modes smallb/fineb that are not in s04_exact_bias.csv.
    C_b via Cb_gen below (s04's cell-count formula, NaN-safe for n00 = 0); otherwise the same code path as bias_analysis/investigation/scripts/s04_exact_finite_b.py
    (S4.mu_b, biaslib.gmm_solve, identical settings); imported read-only.
(b) r = 3, GH(2) attractor, oracle weights Sigma_3^{-1} from the Sec. 8 cache
    sec8_simulations/cache/oracle_sigma_GH2_r3.npz on T_EVAL: sigma^2, rank term lambda' Q,
    curvature nu = sigma^2 lambda' mu'' / 2 and eta = lambda'Q + nu for r = 1, 2, 3 (the r = 1, 2
    weights are the sub-blocks, i.e. the same oracle weights as in s02).  Q from biaslib.beta_C
    with 40-digit mpmath (as s08).  Check: r = 1, 2 against s02_theory.csv.
(c) exact k = infinity bias of r3_or for M1 (GH(2)) at b in {3, 4, 5, 10, 20, 40, 80, 160}:
    C_b^{(l,m)} by the multinomial cell counts of s04 with a general-l threshold equation
    (s04's s_of_v is specific to l <= 2), then the 9-moment GMM with Sigma_3^{-1}.
    Check: the r = 2 sub-problem reproduces s04 r2or.

Outputs: outputs/th_exact_bias_extra.csv, outputs/th_r3_eta.csv, outputs/th_r3_exact_bias.csv,
outputs/th_theory.txt.  Run (from scripts/):
  OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=<scratch> taskset -c 0-9 nice -n 19 \
      python th_theory.py
"""
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import gammaincc, gammaln  # noqa: E402

import common as C   # noqa: E402

sys.path.insert(0, str(C.BIAS / "scripts"))
import biaslib as B              # noqa: E402
import s04_exact_finite_b as S4  # noqa: E402

rl = S4.rlos_lib
T = C.T
NEW_B = [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 26, 28, 32, 34, 50, 60]
PAIRS3 = rl.pairs(3)
LOG = []


def log(s):
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------------------- (a)
def exact_r12(args):
    model, b = args
    Sig = B.oracle_sigma("GH2")
    Wd = {"r2or": B.weights("or", Sig), "r2id": B.weights("id", Sig)}
    A0 = B.A_log(T)
    mub = np.array([mu_b3(model, b, float(t), pairs=B.PAIRS) for t in T])   # NaN-safe Cb_gen
    est = {"r1": mub[:, 0] / (1 - mub[:, 0])}
    for k_, W in Wd.items():
        est[k_] = B.gmm_solve(mub, W, T, lo=0.5 * np.ones_like(T), hi=1.3 * np.ones_like(T),
                              grid=41, levels=10)
    return [dict(model=model, b=b, t=t, estimator=nm, exact=a[i] - A0[i])
            for nm, a in est.items() for i, t in enumerate(T)]


# ----------------------------------------------------------------------------- (b)
def Sigma3():
    z = np.load(C.SEC8 / "cache" / "oracle_sigma_GH2_r3.npz")
    assert np.allclose(z["T"], T)
    return z["Sigma"]


def mu3(a, t):
    return np.stack([rl.mu_lm(a, t, l, m) for (l, m) in PAIRS3], -1)


def dmu3(a, t):
    return np.stack([rl.dmu_da_lm(a, t, l, m) for (l, m) in PAIRS3], -1)


def q3(t):
    return B.beta_C(float(t), "GH2", PAIRS3, method="mp")


def eta_r123(Q):
    S3 = Sigma3()
    rows = []
    idx = {1: [0], 2: [0, 1, 3, 4], 3: list(range(9))}
    h = 1e-5
    for i, t in enumerate(T):
        a = float(B.A_log(t))
        d = -dmu3(a, t)
        mdd = (dmu3(a + h, t) - dmu3(a - h, t)) / (2 * h)
        for r, ix in idx.items():
            S = S3[i][np.ix_(ix, ix)]
            W = np.linalg.inv(S)
            lam = W @ d[ix] / (d[ix] @ W @ d[ix])
            s2 = lam @ S @ lam
            nu = 0.5 * s2 * lam @ mdd[ix]
            rk = lam @ Q[i][ix]
            rows.append(dict(t=t, r=r, A0=a, sigma2=s2, rank=rk, nu=nu, eta=rk + nu))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- (c)
def s_of_v_gen(v, l, b):
    """s = -log a with P(at most l-1 of b iid U(0,1) exceed a) = v (general l)."""
    lv = np.log(v)
    if l == 1:
        return -lv / b
    js = np.arange(l)
    lc = gammaln(b + 1) - gammaln(js + 1) - gammaln(b - js + 1)

    def f(s):
        s = s[..., None]
        terms = lc - (b - js) * s + js * np.log(-np.expm1(-s))
        return np.logaddexp.reduce(terms, axis=-1) - lv
    lo, hi = np.full_like(v, 1e-300), np.full_like(v, 1.0)
    while np.any(f(hi) > 0):
        hi = np.where(f(hi) > 0, 2 * hi, hi)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def Cb_gen(l, m, b, v1, v2, omc):
    s1, s2 = s_of_v_gen(v1, l, b), s_of_v_gen(v2, m, b)
    o1, o2 = -np.expm1(-s1), -np.expm1(-s2)
    oc = omc(s1, s2)
    p11 = np.maximum(o1 + o2 - oc, 0)
    p10 = np.maximum(oc - o2, 0)
    p01 = np.maximum(oc - o1, 0)
    with np.errstate(divide="ignore"):
        lp00 = np.log1p(-oc)
    tot = np.zeros_like(v1)
    for n11 in range(min(l, m)):
        for n10 in range(l - n11):
            for n01 in range(m - n11):
                n00 = b - n11 - n10 - n01
                if n00 < 0:
                    continue
                lc = gammaln(b + 1) - gammaln(n11 + 1) - gammaln(n10 + 1) - gammaln(n01 + 1) - gammaln(n00 + 1)
                # n00 = 0 with 1 - C = 1 (lp00 = -inf) must give 0 * log(.) = 0, not NaN (the
                # inherited s04 Cb returns NaN there, which only matters for b = l, e.g. r = 2 at b = 2)
                term = lc + (n00 * lp00 if n00 else 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    for n_, p_ in ((n11, p11), (n10, p10), (n01, p01)):
                        if n_:
                            term = term + n_ * np.log(p_)
                tot += np.exp(term)
    return tot


def mu_b3(model, b, t, npan=40, pairs=PAIRS3):
    X, Wt = S4._nodes(npan)
    out = []
    for (l, m) in pairs:
        v1 = gammaincc(l, (1 - t) * X)
        v2 = gammaincc(m, t * X)
        out.append(1 - np.sum(Cb_gen(l, m, b, v1, v2, S4.MODELS[model]) * np.exp(-X) * Wt))
    return np.array(out)


def gmm3(target, W, ix, grid=41, levels=10):
    lo, hi = np.full(len(T), 0.5), np.full(len(T), 1.3)
    lo0, hi0 = lo.copy(), hi.copy()
    g = np.linspace(0, 1, grid)
    for _ in range(levels):
        a = lo[:, None] + (hi - lo)[:, None] * g                 # (nT, grid)
        mm = mu3(a, T[:, None])[..., ix]                          # (nT, grid, p)
        dd = mm - target[:, None, :]
        f = np.einsum("tgi,tij,tgj->tg", dd, W, dd)
        j = np.argmin(f, -1)
        step = (hi - lo) / (grid - 1)
        best = a[np.arange(len(T)), j]
        lo, hi = np.maximum(lo0, best - step), np.minimum(hi0, best + step)
    return best


def exact_r3(b):
    S3 = Sigma3()
    A0 = B.A_log(T)
    mub = np.array([mu_b3("M1", b, float(t)) for t in T])
    rows = []
    for nm, ix in (("r3or", list(range(9))), ("r2or_from_r3", [0, 1, 3, 4])):
        W = np.linalg.inv(S3[:, ix][:, :, ix])
        a = gmm3(mub[:, ix], W, ix)
        rows += [dict(model="M1", b=b, t=t, estimator=nm, exact=a[i] - A0[i]) for i, t in enumerate(T)]
    return rows


def main():
    t0 = time.time()
    nw = C.NWORK
    with ProcessPoolExecutor(nw) as ex:
        # (b) rank terms Q for the 9 pairs at every t (mpmath, slow) -- submit first
        fq = [ex.submit(q3, t) for t in T]
        fa = [ex.submit(exact_r12, (mo, b)) for mo in ("M1", "M2", "M3", "M4") for b in NEW_B]
        fc = [ex.submit(exact_r3, b) for b in (3, 4, 5, 10, 20, 40, 80, 160)]
        rows_a = [r for f in fa for r in f.result()]
        rows_c = [r for f in fc for r in f.result()]
        Q = np.array([f.result() for f in fq])
    ea = pd.DataFrame(rows_a)
    C.save(ea, "th_exact_bias_extra.csv")
    # check (a): smooth continuation of the s04 grid (b=5, 10 from s04 vs neighbours)
    s04 = pd.read_csv(C.BIAS / "outputs" / "s04_exact_bias.csv")
    allb = pd.concat([s04[["model", "b", "t", "estimator", "exact"]], ea], ignore_index=True)
    piv = allb[np.isclose(allb.t, 0.5)].pivot_table(index="b", columns=["model", "estimator"], values="exact")
    log("exact k=inf bias at t=0.5 (rows b; s04 grid + new b):\n" + piv.round(6).to_string())

    er3 = pd.DataFrame(rows_c)
    C.save(er3, "th_r3_exact_bias.csv")
    m = er3[er3.estimator == "r2or_from_r3"].merge(
        s04[s04.estimator == "r2or"], on=["model", "b", "t"], suffixes=("", "_s04"))
    log(f"check (c): r2or from the general-l code vs s04 r2or, b in {sorted(m.b.unique())}: "
        f"max |diff| = {np.abs(m.exact - m.exact_s04).max():.2e}")
    eta = eta_r123(Q)
    C.save(eta, "th_r3_eta.csv")
    s02 = pd.read_csv(C.BIAS / "outputs" / "s02_theory.csv")
    s02 = s02[s02.attractor == "GH2"]
    for r, w in ((1, "r1"), (2, "or")):
        a = eta[eta.r == r].sort_values("t")
        b_ = s02[s02.weights == w].sort_values("t")
        log(f"check (b): r={r} vs s02 ({w}): max|d sigma2| = {np.abs(a.sigma2.values - b_.sigma2.values).max():.2e}, "
            f"max|d eta| = {np.abs(a.eta.values - b_.c_total.values).max():.2e}")
    kk = eta[(eta.t >= 0.15 - 1e-9) & (eta.t <= 0.85 + 1e-9)]
    log("K-average eta: " + ", ".join(f"r={r}: {g.eta.mean():.3f}" for r, g in kk.groupby("r")))
    log("eta at t=0.3/0.5: " + ", ".join(
        f"r={r}: {g.set_index('t').eta[0.3]:.3f}/{g.set_index('t').eta[0.5]:.3f}" for r, g in eta.groupby("r")))
    log("r3 exact bias at t=0.5 (M1): " + ", ".join(
        f"b={b}: {g.set_index('t').exact[0.5]:.2e}" for b, g in er3[er3.estimator == "r3or"].groupby("b")))
    log(f"runtime {time.time() - t0:.0f} s with {nw} workers")
    (C.OUT / "th_theory.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_CACHE_DIR", str(C.CACHE.parent / "numba"))
    main()
