"""s09: compact evidence tables from the s05 outputs (no computation of new MC).

Table A: K-average (Simpson weights) of k*{MC mean error - exact k=inf bias} by k, for the
         four models separately, with the theoretical K-average of c_W (s02; oracle W for
         r2_or and r2_eff, identity for r2_id).  MCSE from the pointwise MCSEs (independent
         across t is NOT assumed: conservative sum of weighted MCSEs).
Run: python s09_tables.py > ../outputs/s09_output.txt
"""
import numpy as np
import pandas as pd
import biaslib as B

OUT = B.HERE.parent / "outputs"
m = pd.read_csv(OUT / "s05_kresid.csv")
mK = (B.T >= 0.15 - 1e-9) & (B.T <= 0.85 + 1e-9)
w = np.ones(mK.sum()); w[1:-1:2], w[2:-1:2] = 4, 2; w *= 0.05 / 3; w /= w.sum()
TK = np.round(B.T[mK], 2)
m = m[m.tkey.isin(TK)]
rows = []
for (model, n, b, k, est), g in m.groupby(["model", "n", "b", "k", "estimator"]):
    g = g.set_index("tkey").loc[TK]
    rows.append(dict(model=model, n=n, b=b, k=k, estimator=est,
                     kres=float(g.kresid.to_numpy() @ w), se=float(g.se_kresid.to_numpy() @ w),
                     c_theory=float(g.c_total.to_numpy() @ w)))
A = pd.DataFrame(rows)
A.to_csv(OUT / "s09_kres_Kavg.csv", index=False, float_format="%.5g")
pd.set_option("display.width", 250)
for est in ("r1", "r2_or", "r2_eff", "r2_id", "r2_or_bceq"):
    x = A[A.estimator == est]
    print(f"\n== {est}: K-average of k*(mean error - exact k=inf bias); theory {x.c_theory.iloc[0]:.3f}")
    print(x.pivot_table(index=["k", "n", "b"], columns="model", values="kres").round(2).to_string())
    print("   conservative MCSE range:", round(x.se.min(), 3), "-", round(x.se.max(), 3))
