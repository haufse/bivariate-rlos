"""s08: O(1/k) bias coefficient c_W for r = 1, 2, 3 with oracle weights, GH(2) attractor.

Uses the Sec. 7 covariance grid (9x9 matrices, sec7_asymptotics/outputs/
sigma_grid_theta2.npz) at its nodes t = 0.2982 and 0.4982; the r = 2 matrix is the sub-block
of the pairs (1,1),(1,2),(2,1),(2,2).  beta_C from biaslib (general (l,m), 40-digit mpmath); mu and its
a-derivatives from rlos_lib.mu_lm by central differences (step 1e-4; checked against the
closed r = 2 derivatives).  Also reports the integrated variance-weighted ratio
c^2/sigma^2, i.e. the squared O(1/k) bias in units of the variance at k = 1.

Run: OMP_NUM_THREADS=1 NUMBA_CACHE_DIR=<scratch> taskset -c 0-9 nice -n 19 \
     python s08_r3_coefficient.py > ../outputs/s08_output.txt
Deterministic.
"""
import os
from pathlib import Path
import sys

import numpy as np

import biaslib as B

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[3] / ".cache" / "numba"))
sys.path.insert(0, str(B.SEC7 / "scripts"))
import rlos_lib  # noqa: E402

GRID = B.SEC7 / "outputs/sigma_grid_theta2.npz"


def mu_r(a, t, r):
    return np.array([rlos_lib.mu_lm(a, t, l, m) for (l, m) in rlos_lib.pairs(r)])


def main():
    z = np.load(GRID)
    pairs3 = rlos_lib.pairs(3)
    for ti in (35, 47):
        t = float(z["t"][ti])
        a = float(B.A_log(t))
        S3 = z["Sigma"][ti]
        h = 1e-4
        m0, mp_, mm = mu_r(a, t, 3), mu_r(a + h, t, 3), mu_r(a - h, t, 3)
        mdot = (mp_ - mm) / (2 * h)
        mdd = (mp_ - 2 * m0 + mm) / h ** 2
        # check against closed r=2 derivatives
        idx2 = [0, 1, 3, 4]
        err = max(np.abs(mdot[idx2] - B.mu_dot(a, t)).max(), np.abs(mdd[idx2] - B.mu_ddot(a, t)).max())
        beta3 = B.beta_C(t, "GH2", pairs3, method="mp")
        print(f"t={t:.4f}: derivative check vs closed r=2 forms {err:.1e}")
        for r, idx in ((1, [0]), (2, idx2), (3, list(range(9)))):
            S = S3[np.ix_(idx, idx)]
            d = -mdot[idx]
            W = np.linalg.inv(S)
            lam = W @ d / (d @ W @ d)
            s2 = lam @ S @ lam
            nW = 0.5 * s2 * lam @ mdd[idx]           # W = Sigma^{-1}
            rank = lam @ beta3[idx]
            c = rank + nW
            print(f"  r={r}: sigma2={s2:.5f}  lam={np.round(lam, 2)}  rank={rank:.3f}  nonlin={nW:.4f}"
                  f"  c={c:.3f}  c^2/sigma2={c * c / s2:.1f}")
        print("  beta_C (r=3 pairs, lexicographic):", np.round(beta3, 3))


if __name__ == "__main__":
    main()
