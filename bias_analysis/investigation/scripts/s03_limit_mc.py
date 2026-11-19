"""s03: Monte Carlo under the limiting r-LOS law (no block bias): finite-k bias of the GMM
madogram estimators, and checks of the O(1/k) coefficients c_W = lam' beta_C + n_W of s02.

Part A (Gaussian check of n_W): moments m ~ N(mu0, Sigma/k) exactly, fixed W; the mean of
   k{A_hat - A0 - lam'(mu0 - m)} estimates n_W (the subtracted linear term has mean zero).
Part B (limit law, GH(2) attractor): k iid blocks of the exact limiting top-2 vector
   (Poisson points 2(W,1-W)/Gamma_j, W with cdf (A0'(w)+1)/2), for k in {125,...,1000};
   estimators r1, r2 with oracle Sigma^{-1} and identity weights, each with known margins
   (M itself) and with order-level ranks (eq. sample-u).  Unconstrained solver on [0.5,1.3]
   (the constraint is reported separately: 'r2or_c' = constrained on Theta_t).
Part C (independence attractor): same with independent margins; unconstrained precursor
   bias and the atom P(A_hat(1/2) = 1) of the constrained estimator at k in {125,250,500}.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-9 nice -n 19 \
     python s03_limit_mc.py > ../outputs/s03_output.txt
Seed: numpy SeedSequence(20261118).  Writes ../outputs/s03_limit_mc.csv.
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import gammainccinv
from scipy.stats import rankdata, norm

import biaslib as B

TSUB = np.array([0.15, 0.3, 0.5])
ROOT = np.random.SeedSequence(20261118)


def sample_gh2_top2(nblocks, rng, J=24, chunk=200_000):
    """(nblocks, 2, 2) limiting top-2 (Frechet scale), logistic(2) attractor.

    Exact: points 2(W,1-W)/Gamma_j of a unit-rate Poisson process; a block is finished once
    2/Gamma_j is below the current second largest value in both margins (no later point can
    enter); unfinished blocks continue the same process with further arrivals."""
    out = np.empty((nblocks, 2, 2))
    for c0 in range(0, nblocks, chunk):
        m = min(chunk, nblocks - c0)
        top = np.zeros((m, 2, 2))
        g0 = np.zeros(m)
        todo = np.arange(m)
        while len(todo):
            G = g0[todo, None] + np.cumsum(rng.exponential(size=(len(todo), J)), axis=1)
            s = 2 * rng.uniform(size=(len(todo), J)) - 1
            W = 0.5 * (1 + s / np.sqrt(2 - s * s))
            for i, P in enumerate((2 * W / G, 2 * (1 - W) / G)):
                cand = np.concatenate([top[todo, i, :], P], axis=1)
                top[todo, i, :] = -np.sort(-cand, axis=1)[:, :2]
            g0[todo] = G[:, -1]
            done = 2 / g0[todo] < top[todo, :, 1].min(axis=1)
            todo = todo[~done]
        out[c0:c0 + m] = top
    return out


def gmm_chunked(mom, W, T_, lo, hi, chunk=4000):
    return np.concatenate([B.gmm_solve(mom[i:i + chunk], W[None], T_[None], lo=lo, hi=hi,
                                       grid=41, levels=9)
                           for i in range(0, len(mom), chunk)])


def sample_ind_top2(nblocks, rng):
    G = np.cumsum(rng.exponential(size=(nblocks, 2, 2)), axis=2)
    return 1.0 / G


def moments(M, T_):
    """M: (R, k, 2, 2) Frechet-scale top-2 per block -> (R, nT, 4) empirical madogram moments."""
    out = np.empty(M.shape[:1] + (len(T_), 4))
    for i, t in enumerate(T_):
        e1 = np.exp(-1.0 / ((1 - t) * M[:, :, 0, :]))
        e2 = np.exp(-1.0 / (t * M[:, :, 1, :]))
        out[:, i] = np.maximum(e1[:, :, :, None], e2[:, :, None, :]).mean(1).reshape(-1, 4)
    return out


def rank_transform(M):
    k = M.shape[1]
    u = rankdata(M, axis=1) / (k + 1.0)
    return 1.0 / gammainccinv(np.array([1, 2])[None, None, None, :], u)


def estimate(mom, Wd, T_, lo, hi):
    est = {}
    est["r1"] = np.clip(mom[..., 0] / (1 - mom[..., 0]), lo, hi)
    for name, W in Wd.items():
        est[name] = gmm_chunked(mom, W, T_, lo, hi)
    return est


def part_A(Sig, A0, T_, rng):
    rows = []
    for kind in ("r1", "or", "id"):
        W = B.weights(kind, Sig)
        d = -B.mu_dot(A0, T_)
        lam = B.lam_W(W, d)
        mu0 = B.mu(A0, T_)
        k, R = 250, 100000
        L = np.linalg.cholesky(Sig / k)
        z = rng.standard_normal((R, len(T_), 4))
        m = mu0 + np.einsum("tij,rtj->rti", L, z)
        if kind == "r1":
            a = m[..., 0] / (1 - m[..., 0])
        else:
            a = gmm_chunked(m, W, T_, 0.5 * np.ones_like(T_), 1.5 * np.ones_like(T_))
        y = k * (a - A0 - np.einsum("ti,rti->rt", lam, mu0 - m))
        nW = B.nonlin_term(W, Sig, A0, T_)
        for i, t in enumerate(T_):
            rows.append(dict(part="A", weights=kind, t=t, k=k, R=R, mc=y[:, i].mean(),
                             se=y[:, i].std(ddof=1) / np.sqrt(R), theory=nW[i]))
    return rows


def part_BC(att, ks, Nblocks, T_, rng, Sig):
    A0 = B.A_log(T_) if att == "GH2" else np.ones_like(T_)
    Wd = {"r2or": B.weights("or", Sig), "r2id": B.weights("id", Sig)}
    lo_u, hi_u = 0.5 * np.ones_like(T_), 1.3 * np.ones_like(T_)
    lo_c = np.maximum(T_, 1 - T_)
    rows = []
    for k in ks:
        R = Nblocks // k
        acc = {}
        t0 = time.time()
        for c0 in range(0, R, 1000):
            r = min(1000, R - c0)
            Mk = (sample_gh2_top2 if att == "GH2" else lambda n, g: sample_ind_top2(n, g))(r * k, rng)
            Mk = Mk.reshape(r, k, 2, 2)
            for marg, MM in (("known", Mk), ("rank", rank_transform(Mk))):
                mom = moments(MM, T_)
                est = estimate(mom, Wd, T_, lo_u, hi_u)
                if marg == "rank":
                    est["r2or_c"] = gmm_chunked(mom, Wd["r2or"], T_, lo_c, np.ones_like(T_))
                    est["r1_c"] = np.clip(mom[..., 0] / (1 - mom[..., 0]), lo_c, 1.0)
                for name, a in est.items():
                    acc.setdefault((marg, name), []).append(a)
        for (marg, name), lst in acc.items():
            a = np.concatenate(lst)
            e = a - A0
            for i, t in enumerate(T_):
                row = dict(part="B" if att == "GH2" else "C", attractor=att, margins=marg,
                           estimator=name, t=t, k=k, R=len(a), kbias=k * e[:, i].mean(),
                           se=k * e[:, i].std(ddof=1) / np.sqrt(len(a)),
                           kvar=k * e[:, i].var(ddof=1))
                if att == "IND" and name.endswith("_c"):
                    row["atom"] = float(np.mean(a[:, i] >= 1 - 1e-12))
                rows.append(row)
        print(f"{att} k={k} R={R} done in {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    rngA, rngB, rngC = (np.random.default_rng(s) for s in ROOT.spawn(3))
    Sig_all = B.oracle_sigma("GH2")
    idx = [int(np.argmin(np.abs(B.T - t))) for t in TSUB]
    Sig = Sig_all[idx]
    rows = part_A(Sig, B.A_log(TSUB), TSUB, rngA)
    print("part A done", flush=True)
    rows += part_BC("GH2", [125, 250, 500, 1000], 4_000_000, TSUB, rngB, Sig)
    sys.path.insert(0, str(B.SEC8))
    import sigma2_lib
    SigI = np.array([sigma2_lib.sigma2(sigma2_lib.Gumbel(1.0), float(t)) for t in TSUB])
    rows += part_BC("IND", [125, 250, 500], 2_000_000, TSUB, rngC, SigI)
    df = pd.DataFrame(rows)
    df.to_csv(B.HERE.parent / "outputs" / "s03_limit_mc.csv", index=False, float_format="%.8g")
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
