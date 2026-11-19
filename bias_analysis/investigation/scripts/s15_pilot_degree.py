"""s15: dependence of the curvature channel of the bootstrap correction on the pilot degree m.

Model M1, n = 1e4, b = 20 (k = 500), R = 200 datasets of the main study (data seeds
SeedSequence([20261117, model_id, n, rep])).  For m in {10, 20, 40, 80}: the Bernstein madogram
pilot (fit_madogram_pilot) at t = 1/2, its value and second derivative, and the oracle r = 2
estimate.  Reports E P(1/2), E P''(1/2), sqrt(k) sd P(1/2), sqrt(k) sd P''(1/2), and the
predicted kappa = [G_a Cov(A,P) + F_a2 Cov(A,P'')]/Var(A) with the s10 derivatives at A0.
Also the deterministic limit A_bar_m (pilot fitted to the exact GH(2) curve).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-9 nice -n 19 \
     python s15_pilot_degree.py
Writes ../outputs/s15_output.txt.  Deterministic.  Runtime < 1 min.
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import biaslib as B

sys.path.insert(0, str(B.SEC8))
import config as C           # noqa: E402
import estimators as E       # noqa: E402
import models as Mo          # noqa: E402
from pickands import fit_madogram_pilot, fit_pickands_pilot  # noqa: E402

OUT = B.HERE.parent / "outputs"
N, BL, R, MS = 10000, 20, 200, [10, 20, 40, 80]
T = C.T_EVAL
J5 = int(np.argmin(np.abs(T - 0.5)))


def one(rep):
    spec = Mo.MODELS["M1"]
    data_ss, _ = np.random.SeedSequence([C.SEED, spec["id"], N, rep]).spawn(2)
    X = spec["sampler"](N, np.random.default_rng(data_ss))
    Wor = np.linalg.inv(np.load(B.SEC8 / "cache" / "oracle_sigma_GH2_r2.npz")["Sigma"])
    top = E.block_top_r(X, BL, 2)
    mom = E.rank_moments(E.rank_scores(top), T)
    a = E.gmm_solve(mom, Wor, T, 2)[0][J5]
    out = []
    for m in MS:
        p = fit_madogram_pilot(top[:, :, 0], m=m)
        out.append(dict(rep=rep, m=m, A=a, P=p(0.5), P2=p.derivative(0.5, 2)))
    return out


def main():
    with ProcessPoolExecutor(10) as ex:
        df = pd.DataFrame([r for rr in ex.map(one, range(R)) for r in rr])
    s10 = pd.read_csv(OUT / "s10_eta_derivative.csv").set_index("weights")
    Ga = s10.loc["or", "F_a"] + s10.loc["or", "N_GH"]
    Fa2 = s10.loc["or", "F_a2"]
    k = N // BL
    tg = np.linspace(0, 1, 201)
    rows = []
    for m, g in df.groupby("m"):
        pb = fit_pickands_pilot(tg, B.A_log(tg), m=m)
        va = g.A.var(ddof=1)
        cp, cp2 = np.cov(g.A, g.P)[0, 1] / va, np.cov(g.A, g.P2)[0, 1] / va
        rows.append(dict(m=m, Abar=float(pb(0.5)), Abar2=float(pb.derivative(0.5, 2)),
                         mean_P=g.P.mean(), mean_P2=g.P2.mean(),
                         sqrtk_sd_P=np.sqrt(k) * g.P.std(ddof=1), sqrtk_sd_P2=np.sqrt(k) * g.P2.std(ddof=1),
                         corr_A_P2=np.corrcoef(g.A, g.P2)[0, 1],
                         kappa_a=Ga * cp, kappa_a2=Fa2 * cp2, kappa=Ga * cp + Fa2 * cp2))
    o = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    txt = (f"M1, n=1e4, b={BL} (k={k}), R={R}, t=1/2; A0 = {B.A_log(0.5):.5f}, A0'' = 2.82843; "
           f"oracle derivatives at A0: G_a = {Ga:.4f}, F_a2 = {Fa2:.4f}\n" + o.round(4).to_string(index=False))
    sl = np.polyfit(np.log(o.m), np.log(o.sqrtk_sd_P2), 1)[0]
    txt += f"\nlog-log slope of sqrt(k) sd P''(1/2) in m: {sl:.3f}\n"
    print(txt)
    (OUT / "s15_output.txt").write_text(txt)


if __name__ == "__main__":
    main()
