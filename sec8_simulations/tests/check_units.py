"""Unit checks of every estimator piece against the manuscript formulas.

Run from sec8_simulations/:  python tests/check_units.py
Prints one line per check with the maximal discrepancy and PASS/FAIL.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammainccinv

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import estimators as E          # noqa: E402
import models as Mo              # noqa: E402
import sigma2_lib                # noqa: E402
from pickands import raw_madogram_curve, fit_pickands_pilot  # noqa: E402

RESULTS = []


def report(name, value, tol):
    ok = bool(value <= tol)
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3e} (tol {tol:.0e})", flush=True)


rng = np.random.default_rng(1)
t = rng.uniform(0.01, 0.99, 400)
a = np.maximum(t, 1 - t) + rng.uniform(0, 1, 400) * (1 - np.maximum(t, 1 - t))
v = rng.uniform(-1, 1, 400)

# 1. moment map vs eq. (mu-r2)
mu_ms = np.stack([a / (1 + a), (a ** 2 + 1 - t) / (1 + a) ** 2, (a ** 2 + t) / (1 + a) ** 2,
                  (a ** 3 - a ** 2 + 2 * a + 2 * t ** 2 - 2 * t) / (1 + a) ** 3], -1)
report("mu_vec(r=2) vs eq. mu-r2", np.abs(E.mu_vec(a, t, 2) - mu_ms).max(), 1e-14)
report("mu_vec(r=1) vs a/(1+a)", np.abs(E.mu_vec(a, t, 1)[:, 0] - a / (1 + a)).max(), 1e-15)
# r=3 component (1,1)..(2,2) consistent with r=2 (same pairs)
m3 = E.mu_vec(a, t, 3)
idx = [0, 1, 3, 4]  # (1,1),(1,2),(2,1),(2,2) in 3x3 lexicographic order
report("mu_vec(r=3) sub-block equals r=2", np.abs(m3[:, idx] - mu_ms).max(), 1e-14)
# mu^(l,m)(a,t) is the E exp(-T), T ~ H_{a,t}: Monte Carlo check at r=3 via Ohi-Nishida
# construction T1 ~ Gamma(l) clock: common shock rate 1-a, individual a-t, a+t-1.
def mc_mu(aa, tt, l, m, N=400000, seed=3):
    g = np.random.default_rng(seed)
    # Poisson processes: common (rate 1-a) counts for both; T1 = l-th arrival of
    # (common + indiv1), T2 = m-th arrival of (common + indiv2)
    rates = np.array([1 - aa, aa - tt, aa + tt - 1])
    K = l + m + 2
    arr = [np.cumsum(g.exponential(1 / max(r_, 1e-300), size=(N, K)), axis=1) if r_ > 0
           else np.full((N, K), np.inf) for r_ in rates]
    c, i1, i2 = arr
    T1 = np.sort(np.concatenate([c, i1], 1), 1)[:, l - 1]
    T2 = np.sort(np.concatenate([c, i2], 1), 1)[:, m - 1]
    return np.exp(-np.minimum(T1, T2)).mean()
errs = []
for (aa, tt) in [(0.8, 0.4), (0.9, 0.7)]:
    mv = E.mu_vec(aa, tt, 3)
    for j, (l, m) in enumerate(E.pairs(3)):
        errs.append(abs(mc_mu(aa, tt, l, m) - mv[j]))
report("mu_vec(r=3) vs MC of Ohi-Nishida minimum (N=4e5)", max(errs), 3e-3)

# 2. D'(a,t)
report("dprime_vec(r=2) vs sigma2_lib.d_vec",
       max(np.abs(E.dprime_vec(a[i], t[i], 2) - sigma2_lib.d_vec(a[i], t[i])).max()
           for i in range(400)), 1e-13)
h = 1e-6
fd = -(E.mu_vec(a + h, t, 3) - E.mu_vec(a - h, t, 3)) / (2 * h)
report("dprime_vec(r=3) vs central difference", np.abs(E.dprime_vec(a, t, 3) - fd).max(), 1e-7)

# 3. R vector: eq. (integrated-R-r2) vs finite-sum code (_R_pair_scalar)
Rcode = np.array([[E._R_pair_scalar(t[i], a[i], v[i], l, m) for (l, m) in E.pairs(2)]
                  for i in range(400)])
report("R_vector_r2 (eq. integrated-R-r2) vs finite sums", np.abs(E.R_vector_r2(t, a, v) - Rcode).max(), 1e-14)
report("R vanishes at A=1", np.abs(E.R_vector(t, np.ones_like(t), np.zeros_like(t), 3)).max(), 1e-14)
R3 = E.R_vector(t[:50], a[:50], v[:50], 3)
report("R(r=3) sub-block equals r=2", np.abs(R3[:, idx] - E.R_vector_r2(t[:50], a[:50], v[:50])).max(), 1e-14)

# 4. GMM solver vs scipy bounded search + endpoint check (estimate_A_gmm_from_mean logic)
def scipy_gmm(target, W, tt, r):
    lo = max(tt, 1 - tt)
    f = lambda x: float((E.mu_vec(x, tt, r) - target) @ W @ (E.mu_vec(x, tt, r) - target))
    res = minimize_scalar(f, bounds=(lo, 1.0), method="bounded", options={"xatol": 1e-10})
    cands = [(res.x, f(res.x)), (lo, f(lo)), (1.0, f(1.0))]
    return min(cands, key=lambda c: c[1])
for r in (2, 3):
    T = np.linspace(0.1, 0.9, 17)
    nrep = 40
    errs, ferr = [], []
    for rep in range(nrep):
        A0 = Mo.A_logistic(T)
        target = E.mu_vec(A0, T, r) + rng.normal(0, 0.02, (17, r * r))
        Lm = rng.normal(size=(17, r * r, r * r))
        W = Lm @ np.swapaxes(Lm, 1, 2) + 0.1 * np.eye(r * r)
        est, fv = E.gmm_solve(target, W, T, r)
        for i in range(17):
            xs, fs = scipy_gmm(target[i], W[i], T[i], r)
            # compare objective values (robust to flat criteria) and arguments
            ferr.append(fv[i] - fs)
            errs.append(abs(est[i] - xs))
    report(f"gmm_solve(r={r}) objective <= scipy objective + tol (max excess)", max(ferr), 1e-10)
    report(f"gmm_solve(r={r}) argmin vs scipy (median abs diff)", float(np.median(errs)), 1e-7)

# 5. r=1 closed form equals GMM with r=1
T = np.linspace(0.1, 0.9, 17)
m11 = rng.uniform(0.4, 0.55, 17)
est, _ = E.gmm_solve(m11[:, None], np.ones((17, 1, 1)), T, 1)
report("r1 closed form vs gmm_solve(r=1)", np.abs(E.r1_closed_form(m11, T) - est).max(), 1e-8)

# 6. rank moments (r=1) equal the raw madogram moment of pickands.py
X = Mo.sample_gh(5000, np.random.default_rng(5))
top = E.block_top_r(X, 20, 2)
Mh = E.rank_scores(top)
mom = E.rank_moments(Mh, T)
raw = raw_madogram_curve(top[:, :, 0], T)
report("rank_moments (1,1) vs pickands.raw_madogram_curve", np.abs(mom[:, 0] / (1 - mom[:, 0]) - raw).max(), 1e-12)
# block_top_r against a slow reference
Xb = X[: (5000 // 20) * 20].reshape(-1, 20, 2)
ref = np.stack([np.sort(Xb[:, :, 0], 1)[:, ::-1][:, :2], np.sort(Xb[:, :, 1], 1)[:, ::-1][:, :2]], 1)
report("block_top_r vs sort reference", np.abs(top - ref).max(), 0.0)
# rank scores: F_(l)(M-hat) = rank/(k+1) with F_(l) of eq. (invgam)
k = top.shape[0]
F = lambda x, l: np.exp(-1 / x) * sum(x ** (-q) / math.factorial(q) for q in range(l))
from scipy.stats import rankdata
u = rankdata(top, axis=0) / (k + 1)
report("F_(l)(M-hat) = rank/(k+1) (eq. invgam, sample-u)",
       max(np.abs(F(Mh[:, :, l], l + 1) - u[:, :, l]).max() for l in range(2)), 1e-12)

# 7. POT: brute-force definition with F-hat denominator n
Xs = Mo.sample_gh(997, np.random.default_rng(7))
kp = 83
Fh = np.stack([(Xs[:, i][:, None] >= Xs[:, i][None, :]).sum(1) / 997 for i in range(2)], 1)
bf = np.array([np.mean((Fh[:, 0] > 1 - (1 - tt) * kp / 997) | (Fh[:, 1] > 1 - tt * kp / 997)) * 997 / kp
               for tt in T])
report("POT vs brute-force definition", np.abs(E.pot_estimator(Xs, kp, T) - bf).max(), 1e-12)

# 7b. fast bootstrap covariance == reference implementation (same draws)
pil = fit_pickands_pilot(np.linspace(0, 1, 201), Mo.A_logistic(np.linspace(0, 1, 201)), m=40)
for r in (2, 3):
    S_ref = E.rank_bootstrap_covariances_reference(pil, 300, 60, r, T, np.random.default_rng(4))
    S_new = E.rank_bootstrap_covariances(pil, 300, 60, r, T, np.random.default_rng(4))
    report(f"bootstrap Sigma fast vs reference (r={r}), max rel diff",
           np.abs(S_new - S_ref).max() / np.abs(S_ref).max(), 1e-10)

# 8. Sign of R: known-margin finite-block bias in the max-stable GH model.
#    By Lemma finiteblockbias, E mu-hat° - mu(A0) = b^{-1} R + o(1/b) (S=0).
#    The equation correction uses target mu-hat + R-hat/b, which removes the
#    bias iff E mu-hat° - mu(A0) ~ -R/b.  Report the regression slope of
#    b*(E mu-hat° - mu) on R across t and b in {4, 8}.
def known_margin_scores(top, b):
    # U = F_{l:b}(X) for uniform margins, then M° = F_(l)^{-1}(U)
    out = np.empty_like(top)
    for l in range(top.shape[2]):
        x = top[:, :, l]
        Fl = sum(math.comb(b, j) * (1 - x) ** j * x ** (b - j) for j in range(l + 1))
        out[:, :, l] = 1.0 / gammainccinv(l + 1, Fl)
    return out
Tt = np.array([0.3, 0.5, 0.7])
A0 = Mo.A_logistic(Tt)
dA0 = np.array([sigma2_lib.Gumbel(2.0).dA(x) for x in Tt])
Rtrue = E.R_vector_r2(Tt, A0, dA0)
g = np.random.default_rng(11)
lines = []
for b in (4, 8):
    acc = np.zeros((3, 4)); acc2 = np.zeros((3, 4)); nrep = 0
    for rep in range(30):
        Xg = Mo.sample_gh(200000 * b // 4, g)
        Mk = known_margin_scores(E.block_top_r(Xg, b, 2), b)
        dev = E.rank_moments(Mk, Tt) - E.mu_vec(A0, Tt, 2)
        acc += dev; acc2 += dev ** 2; nrep += 1
    mean = acc / nrep
    se = np.sqrt((acc2 / nrep - mean ** 2) / nrep)
    lines.append((b, mean, se))
    print(f"   b={b}: b*(E mu-hat° - mu) =\n{np.round(b * mean, 4)}\n   +-2se\n{np.round(2 * b * se, 4)}\n   R=\n{np.round(Rtrue, 4)}")
x = np.concatenate([Rtrue[:, 1:].ravel()] * 2)
y = np.concatenate([b * m[:, 1:].ravel() for b, m, _ in lines])
slope = float(x @ y / (x @ x))
print(f"   regression slope of b*(E mu-hat° - mu(A0)) on R: {slope:.3f} (theory: -1 if E mu-hat ~ mu - R/b)")
report("sign/size of R: |slope + 1| (known margins, GH, b=4,8)", abs(slope + 1), 0.25)

print()
print(f"{sum(ok for _, ok in RESULTS)}/{len(RESULTS)} checks passed")
