"""s06: why the equation correction lowers the variance by a relative O(1/b) amount.

Limit-law Monte Carlo (GH(2) attractor, no block bias), k = 500, rank-based moments,
oracle weights.  The equation-corrected estimator solves the GMM problem with target
m_hat + R/b.  Three versions isolate the two first-order mechanisms:
  fixed : R = calR[A0]                (deterministic shift; only the curvature of the GMM map)
  pilot : R = calR[A_r1(t), A0'(t)]   (plug-in through the unsmoothed r=1 estimate at t)
  both  : identical to 'pilot' (the curvature part is always present)
Reported: b * {1 - Var(corrected)/Var(uncorrected)} integrated over K = [0.15, 0.85] with
the Sec. 8 Simpson weights, for 1/b in {0.2, 0.1} (b = 5, 10), with paired-bootstrap MCSE.
The Sec. 8 data give 0.40 for the feasible pilot (Bernstein, derivative included).

Run: OMP_NUM_THREADS=1 taskset -c 0-9 nice -n 19 \
     python s06_bc_variance_mc.py > ../outputs/s06_output.txt
Seed: SeedSequence(20261119).  Runtime about 1-2 min.
"""
import numpy as np

import biaslib as B
import s03_limit_mc as S3


def main():
    rng = np.random.default_rng(np.random.SeedSequence(20261119))
    T_ = B.T[(B.T >= 0.15 - 1e-9) & (B.T <= 0.85 + 1e-9)]
    w = np.ones(len(T_))
    w[1:-1:2], w[2:-1:2] = 4, 2
    w *= 0.05 / 3
    Sig = B.oracle_sigma("GH2")[(B.T >= 0.15 - 1e-9) & (B.T <= 0.85 + 1e-9)]
    W = B.weights("or", Sig)
    A0, v0 = B.A_log(T_), B.dA_log(T_)
    R0 = B.calR(T_)
    k, R = 500, 6000
    lo, hi = 0.5 * np.ones_like(T_), 1.3 * np.ones_like(T_)
    est = {key: [] for key in ("unc", "fixed5", "pilot5", "fixed10", "pilot10")}
    for c0 in range(0, R, 1000):
        M = S3.rank_transform(S3.sample_gh2_top2(1000 * k, rng).reshape(1000, k, 2, 2))
        mom = S3.moments(M, T_)
        a1 = mom[..., 0] / (1 - mom[..., 0])
        Rp = B.calR(T_[None], a1, v0[None])
        est["unc"].append(S3.gmm_chunked(mom, W, T_, lo, hi))
        for b in (5, 10):
            est[f"fixed{b}"].append(S3.gmm_chunked(mom + R0 / b, W, T_, lo, hi))
            est[f"pilot{b}"].append(S3.gmm_chunked(mom + Rp / b, W, T_, lo, hi))
    est = {key: np.concatenate(v) for key, v in est.items()}
    iv = lambda a: float(a.var(0, ddof=1) @ w)
    base = iv(est["unc"])
    bs = np.random.default_rng(1)
    print(f"k={k}, R={R}, oracle weights, IVar_K(uncorrected)*k = {base * k:.4f}")
    for b in (5, 10):
        for kind in ("fixed", "pilot"):
            a = est[f"{kind}{b}"]
            val = b * (1 - iv(a) / base)
            boots = []
            for _ in range(200):
                i = bs.integers(0, R, R)
                boots.append(b * (1 - iv(a[i]) / iv(est["unc"][i])))
            print(f"b={b:3d} {kind:6s}: b*(1 - IVar ratio) = {val:.3f}  (MCSE {np.std(boots):.3f})")


if __name__ == "__main__":
    main()
