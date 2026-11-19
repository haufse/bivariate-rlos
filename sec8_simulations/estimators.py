"""Estimator pieces for the Section 8 Monte Carlo study.

Every function implements one displayed formula of the manuscript;
the equation labels are given in the docstrings.

* order-level ranks and inverse-gamma scores: eq. (sample-u), (defs-xihat)
* rank moments: eq. (def-murank), stacked as in eq. (madogram-moments)
* moment map mu(a,t): eq. (exponential-moment-h) with I_{l,m} of eq. (I-lm);
  for r=2 it reduces to eq. (mu-r2) (checked in tests/check_units.py)
* D'(a,t) = -d/da mu(a,t): analytic derivative of eq. (exponential-moment-h)
* constrained GMM minimiser over Theta_t: eq. (hatA-new)
* integrated finite-block bias R[A](t): eq. (integrated-R-r2) for r=2;
  general r via the finite sums of Lemma finiteblockbias-supp (function
  adapted from the authors' earlier simulation code)
* equation correction: eq. (bc-moment-estimator); one-step correction:
  eq. (bc-lambda-estimator), (bc-one-step-estimator)
* fitted limiting-law rank bootstrap: eq. (rankboot-ranks),
  (rankboot-cov-estimator) (adapted from the authors' earlier simulation code)
* POT estimator: Section sec:comparison-pot, F-hat with denominator n.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import gammainccinv
from scipy.stats import rankdata

from pickands import sample_rlos


# ----------------------------------------------------------------------------- pairs
def pairs(r):
    """Lexicographic order (1,1),(1,2),...,(r,r) of eq. (madogram-moments)."""
    return [(l, m) for l in range(1, r + 1) for m in range(1, r + 1)]


@lru_cache(maxsize=None)
def _terms(l, m):
    """Index set I_{l,m} = {i1+i2<l, i1+i3<m} of eq. (I-lm), with multinomials."""
    out = []
    for i1 in range(min(l, m)):
        for i2 in range(l - i1):
            for i3 in range(m - i1):
                n = i1 + i2 + i3
                coef = math.factorial(n) / (math.factorial(i1) * math.factorial(i2)
                                            * math.factorial(i3))
                out.append((i1, i2, i3, n, coef))
    return tuple(out)


def mu_vec(a, t, r):
    """mu(a,t) of eq. (exponential-moment-h); a, t broadcast; last axis r^2."""
    a = np.asarray(a, dtype=float)
    t = np.asarray(t, dtype=float)
    a, t = np.broadcast_arrays(a, t)
    x1, x2, x3, s = 1.0 - a, a - t, a + t - 1.0, 1.0 + a
    comps = []
    for (l, m) in pairs(r):
        tot = np.zeros_like(a)
        for i1, i2, i3, n, c in _terms(l, m):
            tot = tot + c * x1 ** i1 * x2 ** i2 * x3 ** i3 / s ** (n + 1)
        comps.append(1.0 - tot)
    return np.stack(comps, axis=-1)


def dprime_vec(a, t, r):
    """D'(a,t) = -d/da mu(a,t) (eq. D-prime-explicit), analytic, last axis r^2."""
    a = np.asarray(a, dtype=float)
    t = np.asarray(t, dtype=float)
    a, t = np.broadcast_arrays(a, t)
    x1, x2, x3, s = 1.0 - a, a - t, a + t - 1.0, 1.0 + a

    def pw(x, p):
        return np.ones_like(x) if p == 0 else x ** p

    comps = []
    for (l, m) in pairs(r):
        tot = np.zeros_like(a)  # d/da of sum_terms  (mu = 1 - sum)
        for i1, i2, i3, n, c in _terms(l, m):
            P = pw(x1, i1) * pw(x2, i2) * pw(x3, i3)
            dP = np.zeros_like(a)
            if i1:
                dP = dP - i1 * pw(x1, i1 - 1) * pw(x2, i2) * pw(x3, i3)
            if i2:
                dP = dP + i2 * pw(x1, i1) * pw(x2, i2 - 1) * pw(x3, i3)
            if i3:
                dP = dP + i3 * pw(x1, i1) * pw(x2, i2) * pw(x3, i3 - 1)
            tot = tot + c * (dP / s ** (n + 1) - (n + 1) * P / s ** (n + 2))
        comps.append(tot)  # D' = -d mu/da = + d(sum)/da
    return np.stack(comps, axis=-1)


# ----------------------------------------------------------------------------- data -> moments
def block_top_r(X, b, r):
    """Top-r order statistics per disjoint block; returns (k, 2, r), descending.

    The incomplete final block is discarded (Section sec:estimator-def)."""
    k = X.shape[0] // b
    Xb = X[: k * b].reshape(k, b, 2)
    part = -np.partition(-Xb, r - 1, axis=1)[:, :r, :]   # r largest, unsorted
    top = -np.sort(-part, axis=1)                          # (k, r, 2) descending
    return np.transpose(top, (0, 2, 1)).copy()             # (k, 2, r)


def rank_scores(top):
    """M-hat = F_(l)^{-1}(rank/(k+1)) per margin and order, eq. (sample-u)."""
    k, _, r = top.shape
    u = rankdata(top, method="average", axis=0) / (k + 1.0)
    return 1.0 / gammainccinv(np.arange(1, r + 1)[None, None, :], u)


def rank_moments(Mhat, T):
    """mu-hat^rank(t) of eq. (def-murank) for t in T; returns (len(T), r^2)."""
    k, _, r = Mhat.shape
    out = np.empty((len(T), r * r))
    for i, t in enumerate(T):
        # exp(-xi) = max(exp(-1/((1-t)M1)), exp(-1/(t M2)))
        e1 = np.exp(-1.0 / ((1.0 - t) * Mhat[:, 0, :]))      # (k, r)
        e2 = np.exp(-1.0 / (t * Mhat[:, 1, :]))              # (k, r)
        out[i] = np.maximum(e1[:, :, None], e2[:, None, :]).mean(axis=0).reshape(r * r)
    return out


# ----------------------------------------------------------------------------- GMM
def gmm_solve(target, W, T, r, grid=101, levels=5):
    """Vectorised minimiser of ||mu(a,t)-target||_W^2 over Theta_t (eq. hatA-new).

    target: (..., nT, r^2); W: broadcastable to (..., nT, r^2, r^2).
    Deterministic zoom grid search: level 0 is an equispaced grid on
    Theta_t=[max(t,1-t),1] including both endpoints; each further level
    refines around the current argmin.  Final resolution
    |Theta_t| * (2/(grid-1))^(levels-1) / (grid-1) (= 1.6e-9 |Theta_t|
    for the defaults), far below Monte Carlo error.  Returns (est, fval).
    """
    target = np.asarray(target, float)
    T = np.asarray(T, float)
    lo = np.broadcast_to(np.maximum(T, 1 - T), target.shape[:-1]).copy()
    hi = np.ones_like(lo)
    W = np.broadcast_to(W, target.shape[:-1] + (r * r, r * r))
    Tb = np.broadcast_to(T, target.shape[:-1])
    g = np.linspace(0.0, 1.0, grid)
    for _ in range(levels):
        a = lo[..., None] + (hi - lo)[..., None] * g          # (..., nT, G)
        d = mu_vec(a, Tb[..., None], r) - target[..., None, :]  # (..., nT, G, r2)
        f = np.einsum("...gi,...ij,...gj->...g", d, W, d)
        j = np.argmin(f, axis=-1)
        step = (hi - lo) / (grid - 1)
        best = np.take_along_axis(a, j[..., None], -1)[..., 0]
        fbest = np.take_along_axis(f, j[..., None], -1)[..., 0]
        lo_new = np.maximum(np.maximum(Tb, 1 - Tb), best - step)
        hi_new = np.minimum(1.0, best + step)
        lo, hi = lo_new, hi_new
    return best, fbest


def r1_closed_form(m11, T):
    """r=1 rank madogram: m/(1-m) projected onto Theta_t (Section 3)."""
    T = np.asarray(T, float)
    return np.clip(m11 / (1.0 - m11), np.maximum(T, 1 - T), 1.0)


# ----------------------------------------------------------------------------- bias functional R
def R_vector_r2(t, a, v):
    """Eq. (integrated-R-r2), pair order (1,1),(1,2),(2,1),(2,2)."""
    t, a, v = np.broadcast_arrays(np.asarray(t, float), np.asarray(a, float),
                                  np.asarray(v, float))
    s = 1.0 + a
    return np.stack([
        np.zeros_like(a),
        (a + t - 1) * (1 - t) * (a - 1 - t * v) / s ** 3,
        t * (a - t) * (a - 1 + (1 - t) * v) / s ** 3,
        3 * (a - 1) * (a - t) * (a + t - 1) / s ** 4,
    ], axis=-1)


def _R_pair_scalar(t, A, dA, ell, m):
    """Integrated R^(ell,m)(t) for general (ell,m).

    Logic adapted unchanged from the authors' earlier simulation code
    (finite sums of Lemma finiteblockbias-supp after the substitution of
    eq. bc-local-substitution)."""
    p, q = 1.0 - t, t
    c, d, e, beta = 1.0 - A, A - q, A - p, 1.0 + A
    D1 = -A + q * dA
    D2 = -A - p * dA
    B0 = (ell - 1) * p * D1 + (m - 1) * q * D2
    B1 = (ell - 1) * p * (1.0 + D1) + (m - 1) * q * (1.0 + D2)
    B2 = (ell - 1) * p * D1 + (m - 1) * q * (1.0 + D2)
    B3 = (ell - 1) * p * (1.0 + D1) + (m - 1) * q * D2

    def pw(x, power):
        return 1.0 if power == 0 else x ** power

    total = 0.0
    for i1 in range(ell):
        for i2 in range(ell - i1):
            for i3 in range(m - i1):
                k = i1 + i2 + i3
                P = pw(c, i1) * pw(d, i2) * pw(e, i3)
                P1 = pw(c, i1 - 1) * pw(d, i2) * pw(e, i3) if i1 else 0.0
                P2 = pw(c, i1) * pw(d, i2 - 1) * pw(e, i3) if i2 else 0.0
                P3 = pw(c, i1) * pw(d, i2) * pw(e, i3 - 1) if i3 else 0.0
                Ccoef = k * (1 - k) * P + i1 * B1 * P1 - i2 * B2 * P2 - i3 * B3 * P3
                Ecoef = ((2.0 * A * k + B0) * P + i1 * (A * A - p * p - q * q) * P1
                         + i2 * (q * q - A * A) * P2 + i3 * (p * p - A * A) * P3)
                denom = math.factorial(i1) * math.factorial(i2) * math.factorial(i3)
                total += 0.5 / denom * (Ccoef * math.factorial(k) / beta ** (k + 1)
                                        + Ecoef * math.factorial(k + 1) / beta ** (k + 2))
    return total


def R_vector(t, a, v, r):
    """R[A](t) of eq. (bc-R-functional) for 1-d arrays t, a=A(t), v=A'(t)."""
    if r == 1:
        return np.zeros(np.shape(t) + (1,))
    if r == 2:
        return R_vector_r2(t, a, v)
    t, a, v = (np.atleast_1d(np.asarray(x, float)) for x in (t, a, v))
    return np.array([[_R_pair_scalar(ti, ai, vi, l, m) for (l, m) in pairs(r)]
                     for ti, ai, vi in zip(t, a, v)])


# ----------------------------------------------------------------------------- bootstrap
def rank_bootstrap_covariances_reference(pilot, k, reps, r, points, seed):
    """Sigma-hat_n(t) of eq. (rankboot-cov-estimator) -- reference implementation.

    Adapted from the authors' earlier simulation code (unchanged logic): k*reps exact
    limiting r-LOS draws from the fitted pilot (Algorithm alg:truncated_rlos),
    fresh ranks per bootstrap dataset, margin and order (eq. rankboot-ranks),
    common draws across t, factor k, ddof=1."""
    values = sample_rlos(pilot, r=r, size=k * reps, rng=seed).reshape(reps, k, 2, r)
    u = rankdata(values, method="average", axis=1) / (k + 1)
    values = 1 / gammainccinv(np.arange(1, r + 1), u)
    covariances = []
    for t in points:
        xi = np.minimum(1 / ((1 - t) * values[:, :, 0, :, None]),
                        1 / (t * values[:, :, 1, None, :]))
        means = np.exp(-xi).mean(axis=1).reshape(reps, r * r)
        covariances.append(k * np.atleast_2d(np.cov(means, rowvar=False, ddof=1)))
    return np.array(covariances)


def rank_bootstrap_covariances(pilot, k, reps, r, points, seed):
    """Same estimator as rank_bootstrap_covariances_reference, same random draws,
    fewer exponentials: exp(-xi^(l,m)) = max(exp(-q1^(l)/(1-t)), exp(-q2^(m)/t))
    with q = 1/M-hat, so only 2r (not r^2) exponentials per block and t.
    Agreement with the reference to rounding is checked in tests/check_units.py."""
    values = sample_rlos(pilot, r=r, size=k * reps, rng=seed).reshape(reps, k, 2, r)
    u = rankdata(values, method="average", axis=1) / (k + 1)
    q = gammainccinv(np.arange(1, r + 1), u)            # = 1 / M-hat
    q1 = np.ascontiguousarray(q[:, :, 0, :])
    q2 = np.ascontiguousarray(q[:, :, 1, :])
    covariances = []
    for t in points:
        e1 = np.exp(-q1 / (1 - t))
        e2 = np.exp(-q2 / t)
        means = np.maximum(e1[:, :, :, None], e2[:, :, None, :]).mean(axis=1).reshape(reps, r * r)
        covariances.append(k * np.atleast_2d(np.cov(means, rowvar=False, ddof=1)))
    return np.array(covariances)


def rank_bootstrap_cov_means(pilot, k, reps, r, points, seed):
    """rank_bootstrap_covariances plus the bootstrap moment vectors (bias investigation, H2).

    Same random draws and the same covariance arithmetic as rank_bootstrap_covariances
    (identical output for the same seed; checked in tests/check_bck.py).  Additionally returns
    means (reps, len(points), r^2): the rank moments mu-hat* of every bootstrap dataset, which
    are drawn from the fitted limiting law (no block bias), so that the GMM estimates computed
    from them estimate the O(1/k) bias eta_W/k of the estimator (see
    bias_analysis/, run H2)."""
    values = sample_rlos(pilot, r=r, size=k * reps, rng=seed).reshape(reps, k, 2, r)
    u = rankdata(values, method="average", axis=1) / (k + 1)
    q = gammainccinv(np.arange(1, r + 1), u)            # = 1 / M-hat
    q1 = np.ascontiguousarray(q[:, :, 0, :])
    q2 = np.ascontiguousarray(q[:, :, 1, :])
    covariances, allmeans = [], []
    for t in points:
        e1 = np.exp(-q1 / (1 - t))
        e2 = np.exp(-q2 / t)
        means = np.maximum(e1[:, :, :, None], e2[:, :, None, :]).mean(axis=1).reshape(reps, r * r)
        covariances.append(k * np.atleast_2d(np.cov(means, rowvar=False, ddof=1)))
        allmeans.append(means)
    return np.array(covariances), np.stack(allmeans, axis=1)


def inverse_weights(Sig, cond_max=1e12):
    """Pointwise eigen-diagnostics and inverse of each Sigma-hat(t).

    Returns (W_inv, pd, lam_min, cond).  pd[t] is True iff
    lambda_min>0 and cond<=cond_max (numerical version of lambda_min>0 in
    Theorem thm:fitted-rank-bootstrap); W_inv[t] is NaN where not pd."""
    ev, V = np.linalg.eigh(Sig)
    lam_min, lam_max = ev[:, 0], ev[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(lam_min > 0, lam_max / lam_min, np.inf)
    pd = (lam_min > 0) & (cond <= cond_max)
    Winv = np.full_like(Sig, np.nan)
    for i in np.flatnonzero(pd):
        Winv[i] = (V[i] / ev[i]) @ V[i].T
    return Winv, pd, lam_min, cond


# ----------------------------------------------------------------------------- POT
def pot_estimator(X, kp, T):
    """Rank-based empirical STDF A-hat_{n,p}(t)=L-hat(1-t,t), Section sec:comparison-pot.

    F-hat_{i,n} uses denominator n (as in the manuscript).  The indicator
    F-hat_1 > 1-(1-t)kp/n  <=>  rank_1 > n-(1-t)kp.  Not projected."""
    n = X.shape[0]
    R = rankdata(X, method="average", axis=0)
    T = np.asarray(T, float)
    out = np.empty(len(T))
    for i, t in enumerate(T):
        # round the thresholds to remove floating error in (1-t)*kp
        th1, th2 = np.round(n - (1 - t) * kp, 9), np.round(n - t * kp, 9)
        exc = (R[:, 0] > th1) | (R[:, 1] > th2)
        out[i] = exc.sum() / kp
    return out
