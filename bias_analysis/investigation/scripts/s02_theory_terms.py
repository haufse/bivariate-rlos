"""s02: deterministic O(1/k) bias coefficients and leading-order block biases on the Sec. 8 grid.

For the GH(2) attractor (models M1-M4) and the independence attractor (D6a/b), and the
weights r1 (W=e11 e11'), r2_or (W=Sigma^{-1}) and r2_id (W=I), computes per t in T:
  lam_W, sigma_W^2 = lam' Sigma lam,
  lam' beta_C           (rank term, beta_C from biaslib.beta_C),
  n_W                   (nonlinearity term, biaslib.nonlin_term),
  c_W = lam' beta_C + n_W  (predicted lim k E(A_hat - A0) under the limit law, fixed W),
  q_W = lam' calR, s_W(OPC) = lam' calS_OPC (leading block biases, Sec. 7 notation; the
  OPC bias at block size b is s_W/(2b) + q_W/b).
Also checks beta_C against the exact independence value 1/3 (r=1, t=1/2) and against a
50-digit mpmath quadrature at t=0.3 (GH2).

Run: OMP_NUM_THREADS=1 NUMBA_CACHE_DIR=<scratch> taskset -c 0-9 nice -n 19 \
     python s02_theory_terms.py > ../outputs/s02_output.txt
Writes ../outputs/s02_theory.csv.  Deterministic.  Runtime < 1 min.
"""
import os
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
import pandas as pd

import biaslib as B

sys.path.insert(0, str(B.SEC8))
import sigma2_lib  # noqa: E402  (Sec. 8 oracle quadrature, read-only use)

SEC7 = B.SEC7 / "scripts"
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[3] / ".cache" / "numba"))
sys.path.insert(0, str(SEC7))
import rlos_lib  # noqa: E402  (calS_vec for OPC; read-only use)


def main():
    rows = []
    Sig_gh = B.oracle_sigma("GH2")
    Sig_ind = np.array([sigma2_lib.sigma2(sigma2_lib.Gumbel(1.0), float(t)) for t in B.T])
    for att, Sig in (("GH2", Sig_gh), ("IND", Sig_ind)):
        a = B.A_log(B.T) if att == "GH2" else np.ones_like(B.T)
        d = -B.mu_dot(a, B.T)
        bet = np.array([B.beta_C(float(t), att) for t in B.T])
        R = B.calR(B.T) if att == "GH2" else np.zeros((len(B.T), 4))
        S_opc = (np.array([rlos_lib.calS_vec("OPC", float(t), 2) for t in B.T])
                 if att == "GH2" else np.zeros((len(B.T), 4)))
        for kind in ("r1", "or", "id"):
            W = B.weights(kind, Sig)
            lam = B.lam_W(W, d)
            s2 = np.einsum("ti,tij,tj->t", lam, Sig, lam)
            nW = B.nonlin_term(W, Sig, a, B.T)
            lb = np.einsum("ti,ti->t", lam, bet)
            for i, t in enumerate(B.T):
                rows.append(dict(attractor=att, weights=kind, t=t, A0=a[i], sigma2=s2[i],
                                 lam1=lam[i, 0], lam2=lam[i, 1], lam3=lam[i, 2], lam4=lam[i, 3],
                                 beta1=bet[i, 0], beta2=bet[i, 1], beta3=bet[i, 2], beta4=bet[i, 3],
                                 rank_term=lb[i], nonlin_term=nW[i], c_total=lb[i] + nW[i],
                                 q_W=lam[i] @ R[i], s_W_opc=lam[i] @ S_opc[i]))
    df = pd.DataFrame(rows)
    df.to_csv(B.HERE.parent / "outputs" / "s02_theory.csv", index=False, float_format="%.10g")
    pd.set_option("display.width", 200)
    print(df[["attractor", "weights", "t", "sigma2", "lam1", "lam2", "lam3", "lam4",
              "rank_term", "nonlin_term", "c_total", "q_W", "s_W_opc"]]
          .round(4).to_string(index=False))

    # checks
    print("\nCHECK beta_C IND r1 t=1/2 =", B.beta_C(0.5, "IND", [(1, 1)])[0], "(exact 1/3)")
    mp.mp.dps = 50
    for (l, m) in B.PAIRS:
        Dm = B._G_funcs(l, m, "GH2")[1]
        t = mp.mpf("0.3")
        f = lambda x: Dm((1 - t) * x, t * x) * mp.e ** (-x)
        x0 = mp.mpf("1e-6")   # tanh-sinh nodes below 1e-6 lose all digits (1-u ~ x^2)
        val = x0 * f(x0) + mp.quad(f, [x0, 0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 30, 60])
        print(f"CHECK beta_C GH2 t=0.3 pair {(l, m)}: scipy {B.beta_C(0.3, 'GH2', [(l, m)])[0]:.12f}"
              f"  mpmath50 {mp.nstr(val, 15)}")
    # sanity: oracle Sigma for IND equals the closed form ratio quoted in Sec. 7 (0.757 at t=1/2)
    i = int(np.argmin(np.abs(B.T - 0.5)))
    r = df[(df.attractor == "IND") & (np.isclose(df.t, 0.5))].set_index("weights")["sigma2"]
    print("CHECK IND sigma2 t=1/2: r1 %.4f  or %.4f  id %.4f  ratio or/r1 %.4f (Sec. 8 draft: 0.1905,"
          " 0.1441, 0.1506; 0.757)" % (r["r1"], r["or"], r["id"], r["or"] / r["r1"]))


if __name__ == "__main__":
    main()
