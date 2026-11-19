"""s07: optimized r2/r1 ratios under the Sec. 7 criterion and under two refinements.

For models M2 (Joe), M3 (OPC) and, for completeness, M1 and M4, n in {1e3,...,1e6} and integer
b in [2, 400], with k = floor(n/b) and T = K = [0.15,0.85] (Simpson on the Sec. 8 grid):
  (L) Sec. 7 leading criterion:   V_r/k + int (lead bias)^2        [lead from s04 formula]
  (E) exact finite-b bias:         V_r/k + int (exact k=inf bias)^2
  (C) exact + O(1/k) term:         V_r/k + int (exact + c_W/k)^2      [c_W from s02, oracle W]
V_r/k uses the asymptotic variance lam' Sigma lam (s02).  For each criterion the minimum over
b is taken separately for r=1 and r=2 (oracle weights), and over the Sec. 8 grid B_GRID[n]
as well as over all integers.  Output: ../outputs/s07_ratios.csv and s07_output.txt.

Run: OMP_NUM_THREADS=1 NUMBA_CACHE_DIR=<scratch> taskset -c 0-9 nice -n 19 \
     python s07_augmented_criterion.py > ../outputs/s07_output.txt
Deterministic.  Runtime about 1 min.
"""
import numpy as np
import pandas as pd

import biaslib as B
import s04_exact_finite_b as S4

OUT = B.HERE.parent / "outputs"
mK = (B.T >= 0.15 - 1e-9) & (B.T <= 0.85 + 1e-9)
wK = np.ones(mK.sum())
wK[1:-1:2], wK[2:-1:2] = 4, 2
wK *= 0.05 / 3
B_GRID = {1000: [5, 10, 20, 40], 5000: [5, 10, 20, 40, 80], 10000: [5, 10, 20, 40, 80, 160],
          20000: [5, 10, 20, 40, 80, 160, 320], 100000: [10, 20, 40, 80, 160, 320],
          1000000: [10, 20, 40, 80, 160, 320, 640]}
BS = sorted(set(list(range(2, 60)) + list(range(60, 200, 5)) + list(range(200, 701, 20))
                + [b for g in B_GRID.values() for b in g]))


def integ(x):
    return np.asarray(x)[..., mK] @ wK


def main():
    th = pd.read_csv(OUT / "s02_theory.csv")
    th = th[th.attractor == "GH2"]
    Sig = B.oracle_sigma("GH2")
    A0 = B.A_log(B.T)
    d = -B.mu_dot(A0, B.T)
    lam = {"r1": B.lam_W(B.weights("r1", Sig), d), "r2": B.lam_W(B.weights("or", Sig), d)}
    W2 = B.weights("or", Sig)
    R = B.calR(B.T)
    V = {r: integ(th[th.weights == w].sort_values("t").sigma2.to_numpy()) for r, w in (("r1", "r1"), ("r2", "or"))}
    c = {r: th[th.weights == w].sort_values("t").c_total.to_numpy() for r, w in (("r1", "r1"), ("r2", "or"))}
    print("V_r on K:", {r: round(float(v), 5) for r, v in V.items()}, "(Sec. 7: 0.0329, 0.0267)")
    rows = []
    for model in ("M3", "M2", "M1", "M4"):
        M = S4.rlos_lib.MODELS[S4.LEAD[model]]
        S = np.array([S4.rlos_lib.calS_vec(S4.LEAD[model], float(t), 2) for t in B.T])
        ex, ld = {"r1": [], "r2": []}, {"r1": [], "r2": []}
        for b in BS:
            mub = np.array([S4.mu_b(model, b, float(t)) for t in B.T])
            alpha = M["alpha"][0] * b ** M["alpha"][1]
            ex["r1"].append(mub[:, 0] / (1 - mub[:, 0]) - A0)
            ex["r2"].append(B.gmm_solve(mub, W2, B.T, lo=0.5 * np.ones_like(B.T), hi=1.3 * np.ones_like(B.T),
                                        grid=41, levels=10) - A0)
            ld["r1"].append(lam["r1"][:, 0] * alpha * S[:, 0])
            ld["r2"].append(np.einsum("ti,ti->t", lam["r2"], alpha * S + R / b))
        ex = {r: np.array(v) for r, v in ex.items()}
        ld = {r: np.array(v) for r, v in ld.items()}
        bs = np.array(BS)
        for n in (1000, 5000, 10000, 20000, 100000, 1000000):
            k = n // bs
            ok = k >= 10
            crit = {}
            for r in ("r1", "r2"):
                crit[("L", r)] = V[r] / k + integ(ld[r] ** 2)
                crit[("E", r)] = V[r] / k + integ(ex[r] ** 2)
                crit[("C", r)] = V[r] / k + integ((ex[r] + c[r][None, :] / k[:, None]) ** 2)
            for bfix in (10, 20):
                if n == 10000:
                    i = int(np.flatnonzero(bs == bfix)[0])
                    for cr in ("L", "E", "C"):
                        rows.append(dict(model=model, n=n, bset=f"b={bfix}", criterion=cr, b1=bfix, b2=bfix,
                                         imse1=crit[(cr, "r1")][i], imse2=crit[(cr, "r2")][i],
                                         ratio=crit[(cr, "r2")][i] / crit[(cr, "r1")][i]))
            for which, sel in (("all b", ok), ("Sec8 grid", ok & np.isin(bs, B_GRID[n]))):
                for cr in ("L", "E", "C"):
                    i1 = np.flatnonzero(sel)[np.argmin(crit[(cr, "r1")][sel])]
                    i2 = np.flatnonzero(sel)[np.argmin(crit[(cr, "r2")][sel])]
                    rows.append(dict(model=model, n=n, bset=which, criterion=cr, b1=bs[i1], b2=bs[i2],
                                     imse1=crit[(cr, "r1")][i1], imse2=crit[(cr, "r2")][i2],
                                     ratio=crit[(cr, "r2")][i2] / crit[(cr, "r1")][i1]))
        print(model, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "s07_ratios.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["model", "bset", "n"], columns="criterion", values="ratio", sort=False)
          .round(3).to_string())
    print(df[df.bset == "all b"].pivot_table(index=["model", "n"], columns="criterion",
                                              values=["b1", "b2"], sort=False).to_string())


if __name__ == "__main__":
    main()
