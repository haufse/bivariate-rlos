"""s03: Monte Carlo cross-checks of the quadrature covariance Sigma(t) (r=3, contains the r=1,2 blocks)
and of the POT variance formula, all at the GH(theta=2) attractor.

Run:  python s03_mc_sigma_check.py > ../outputs/s03_output.txt
Seed: numpy default_rng(20261117).  Runtime about 15 min on 12 cores.

Limit blocks are drawn exactly with Algorithm alg:truncated_rlos (spectral representation, supplement G),
with the exact stopping rule verified for every block.

(1) Influence-function MC (no finite-k effects): Sigma(t) = Cov{phi_t(M)}, eqs. (influence-function), (Gamma).
    With q_i = 1/M_i (one per order), xi = min{q_1^{(l)}/(1-t), q_2^{(m)}/t} and
    G_i^{(l,m)}(y) = int_0^y dC_i^{(l,m)}(h^{(l,m)}(e^{-s},t)) e^{-s} ds  (tabulated on a fine grid),
        phi_t^{(l,m)}(M) = const - [ (1 - e^{-xi}) - G_1(q_1^{(l)}/(1-t)) - G_2(q_2^{(m)}/t) ].
    N = 2e6 blocks per t; s.e. from 20 batches.
(2) Known-margin MC: Cov{exp(-xi^{(l,m)}(t))} vs the quadrature with the rank-correction terms switched off
    (checks the exact Poisson joint probabilities on their own).
(3) Rank-based limit-law MC (the estimator's own construction) at k = 400 and k = 1600: k Cov(hat mu^rank(t)).
    This target has O(k^{-1/2})-type finite-k corrections for the higher orders (reported, not used as a test).
(4) POT: rank-based empirical STDF estimator on GH(2) samples; k_p Var vs Var{A_p(t)}.
"""
import sys
import time
import numpy as np
from scipy.special import gammainccinv
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, '.')
import rlos_lib as rl

rng = np.random.default_rng(20261117)
TH = 2.0
R = 3
TS = (0.3, 0.5)
PAIRS = rl.pairs(R)
Irows, Ioff = rl._Iarrays(R)

wg = np.linspace(0.0, 1.0, 400001)
Fg = 0.5 * (1.0 + rl.dA_log(wg, TH))
Fg[0], Fg[-1] = 0.0, 1.0


def draw_W(size):
    return np.interp(rng.random(size), Fg, wg)


def draw_q(k, npts=60):
    """k exact draws of the limiting top-R matrix; returns q = 1/M, shape (k, 2, R)."""
    while True:
        E = rng.exponential(size=(k, npts + 1))
        G = np.cumsum(E, axis=1)
        W = draw_W((k, npts))
        z1 = 2 * W / G[:, :npts]
        z2 = 2 * (1 - W) / G[:, :npts]
        top1 = -np.sort(-z1, axis=1)[:, :R]
        top2 = -np.sort(-z2, axis=1)[:, :R]
        ok = 2.0 / G[:, npts] < np.minimum(top1[:, R - 1], top2[:, R - 1])
        if np.all(ok):
            return 1.0 / np.stack([top1, top2], axis=1)
        npts *= 2


def batch_cov(X, nb=20):
    """X: (N, p). Returns mean of batch covariances and their s.e."""
    N = X.shape[0]
    cs = np.array([np.cov(X[j * N // nb:(j + 1) * N // nb].T) for j in range(nb)])
    return cs.mean(0), cs.std(0, ddof=1) / np.sqrt(nb)


def sig_opt(S, d, r_):
    idx = [l * R + m for l in range(r_) for m in range(r_)]
    Ss, dd = S[np.ix_(idx, idx)], d[idx]
    return 1.0 / (dd @ np.linalg.solve(Ss, dd))


def sigma_known_margin(t, npan=30, m=8):
    """Quadrature with the rank-correction coefficients set to zero (known-margin covariance)."""
    lv = np.array([p[0] for p in PAIRS], dtype=np.int64)
    mv = np.array([p[1] for p in PAIRS], dtype=np.int64)
    S, WS = rl.graded_nodes(40.0, npan, m)
    I = np.zeros((9, 9))
    # Cov{1_{A_u}, 1_{A_v}} only:  direct double quadrature, reuse the library probability function
    Wn, WW = S, WS
    for ii in range(len(S)):
        s = S[ii]
        for jj in range(len(Wn)):
            sp = s + Wn[jj]
            wt = WS[ii] * WW[jj] * np.exp(-s - sp)
            for p in range(9):
                pu = rl.prob_counts((1 - t) * s, lv[p] - 1, 0.0, rl.BIG, t * s, mv[p] - 1, 0.0, rl.BIG, TH)
                for pp in range(9):
                    pv = rl.prob_counts((1 - t) * sp, lv[pp] - 1, 0.0, rl.BIG, t * sp, mv[pp] - 1, 0.0, rl.BIG, TH)
                    pj = rl.prob_counts((1 - t) * s, lv[p] - 1, (1 - t) * sp, lv[pp] - 1,
                                        t * s, mv[p] - 1, t * sp, mv[pp] - 1, TH)
                    I[p, pp] += wt * (pj - pu * pv)
    return I + I.T


def G_tables(t):
    """y-grid and G_i^{(l,m)}(y) for all pairs."""
    y = np.concatenate([[0.0], np.geomspace(1e-9, 80.0, 200000)])
    G1 = np.empty((9, y.size))
    G2 = np.empty((9, y.size))
    for p, (l, m) in enumerate(PAIRS):
        c1 = np.empty(y.size)
        c2 = np.empty(y.size)
        for j, s in enumerate(y):
            ss = max(s, 1e-12)
            c1[j], c2[j] = rl.dC_lm(l, m, (1 - t) * ss, t * ss, TH, Irows, Ioff[p], Ioff[p + 1])
        G1[p] = cumulative_trapezoid(c1 * np.exp(-y), y, initial=0.0)
        G2[p] = cumulative_trapezoid(c2 * np.exp(-y), y, initial=0.0)
    return y, G1, G2


print('Quadrature Sigma(t) uses rlos_lib.sigma(t, 2, 3, npan=30, m=8).')
for t in TS:
    t0 = time.time()
    Sq = rl.sigma(t, TH, R, npan=30, m=8)
    A = float(rl.A_log(t, TH))
    d = rl.d_vec(A, t, R)
    y, G1, G2 = G_tables(t)
    N, chunk = 2_000_000, 200_000
    PHI = np.empty((N, 9))
    KNOWN = np.empty((N, 9))
    for c0 in range(0, N, chunk):
        q = draw_q(chunk)
        for p, (l, m) in enumerate(PAIRS):
            y1 = q[:, 0, l - 1] / (1 - t)
            y2 = q[:, 1, m - 1] / t
            xi = np.minimum(y1, y2)
            KNOWN[c0:c0 + chunk, p] = np.exp(-xi)
            PHI[c0:c0 + chunk, p] = (1 - np.exp(-xi)) - np.interp(y1, y, G1[p]) - np.interp(y2, y, G2[p])
    Sphi, sephi = batch_cov(PHI)
    Skn, sekn = batch_cov(KNOWN)
    Sqk = sigma_known_margin(t)
    iu = np.triu_indices(9)
    print(f'\nt={t}  (time {time.time() - t0:.0f}s)')
    print(f'  (1) influence-function MC vs quadrature Sigma: max |z| over 45 entries = '
          f'{np.max(np.abs((Sphi - Sq) / sephi)[iu]):.2f};  max rel. diff on diagonal = '
          f'{np.max(np.abs(np.diag(Sphi) / np.diag(Sq) - 1)):.4f}')
    print(f'  (2) known-margin MC vs known-margin quadrature: max |z| = {np.max(np.abs((Skn - Sqk) / sekn)[iu]):.2f};'
          f'  max rel. diff on diagonal = {np.max(np.abs(np.diag(Skn) / np.diag(Sqk) - 1)):.4f}')
    nbt = 20
    PHIb = [np.cov(PHI[j * N // nbt:(j + 1) * N // nbt].T) for j in range(nbt)]
    for r_ in (1, 2, 3):
        vb = np.array([sig_opt(S_, d, r_) for S_ in PHIb])
        print(f'      sigma_opt^2 r={r_}: quadrature {sig_opt(Sq, d, r_):.6f};  influence MC {vb.mean():.6f} '
              f'+- {vb.std(ddof=1) / np.sqrt(nbt):.6f}')


def rank_moments(q, k, ts):
    qhat = np.empty((k, 2, R))
    for i in range(2):
        for l in range(R):
            ranks = np.argsort(np.argsort(-q[:, i, l])) + 1   # rank of M = 1/q
            qhat[:, i, l] = gammainccinv(l + 1, ranks / (k + 1.0))
    out = np.empty((len(ts), 9))
    for it, t in enumerate(ts):
        for p, (l, m) in enumerate(PAIRS):
            xi = np.minimum(qhat[:, 0, l - 1] / (1 - t), qhat[:, 1, m - 1] / t)
            out[it, p] = np.mean(np.exp(-xi))
    return out


print('\n(3) rank-based limit-law MC, k Cov(hat mu^rank): diagonal ratio MC/quadrature, (l,m) lexicographic')
Sq_t = {t: rl.sigma(t, TH, R, npan=30, m=8) for t in TS}
for k, B in ((400, 8000), (1600, 4000)):
    mus = np.array([rank_moments(draw_q(k), k, TS) for _ in range(B)])
    for it, t in enumerate(TS):
        Smc = k * np.cov(mus[:, it, :].T)
        A = float(rl.A_log(t, TH))
        d = rl.d_vec(A, t, R)
        print(f'  k={k:5d} B={B} t={t}: ' + ' '.join(f'{Smc[i, i] / Sq_t[t][i, i]:.3f}' for i in range(9))
              + f'  | sigma_opt^2 ratio r=2: {sig_opt(Smc, d, 2) / sig_opt(Sq_t[t], d, 2):.3f}, r=3: '
              f'{sig_opt(Smc, d, 3) / sig_opt(Sq_t[t], d, 3):.3f}  (s.e. of ratios approx {np.sqrt(2 / B):.3f})')

print('\n(4) POT variance, GH(2) copula samples:')


def gh_sample(n):
    a = 1.0 / TH
    Wexp = rng.exponential(size=n)
    Phi = rng.uniform(0, np.pi, size=n)
    S = (np.sin(a * Phi) / np.sin(Phi) ** (1 / a)) * (np.sin((1 - a) * Phi) / Wexp) ** ((1 - a) / a)
    E = rng.exponential(size=(n, 2))
    return np.exp(-(E / S[:, None]) ** a)


n, kp, Bp = 20000, 400, 3000
est = np.empty((Bp, len(TS)))
for b in range(Bp):
    X = gh_sample(n)
    F1 = (np.argsort(np.argsort(X[:, 0])) + 1) / n
    F2 = (np.argsort(np.argsort(X[:, 1])) + 1) / n
    for it, t in enumerate(TS):
        est[b, it] = np.sum((F1 > 1 - (1 - t) * kp / n) | (F2 > 1 - t * kp / n)) / kp
for it, t in enumerate(TS):
    v = kp * est[:, it].var(ddof=1)
    se = v * np.sqrt(2.0 / (Bp - 1))
    print(f'  t={t}: k_p*Var MC = {v:.5f} +- {se:.5f};  formula Var A_p(t) = {float(rl.pot_var(t, TH)):.5f}')
