"""s14: the constraint A >= max(t,1-t) near the ends of K, and an interior comparison of the
r = 3 and r = 2 oracle estimators.

Data: Sec. 8 shards, model M1, n = 1e4 (modes r3or/exp r3 for r3_or, full/main for r1 and r2_or),
read through the follow-up loader (read only).  For each b and t: the fraction of estimates on the
lower boundary max(t,1-t) of Theta_t (|A_hat - max(t,1-t)| < 1e-7).  On K = [0.15,0.85] and on the
interior subinterval I = [0.30,0.70] (Simpson weights on the 0.05 grid):
  IMSE ratio r3_or/r2_or (paired delta-method MCSE), IVar ratio and its asymptotic value,
  and the average of k (mean error - exact k = inf bias) against the average of eta (r = 2, 3).
Exact k = inf bias: s04 (r = 2) and th_r3_exact_bias.csv of the follow-up analysis (r = 3).

Run: OMP_NUM_THREADS=1 SEC8FU_CACHE=<scratch> \
     python s14_boundary_interior.py
Writes ../outputs/s14_output.txt, ../outputs/s14_boundary.csv.  Deterministic.
"""
import os
import sys

import numpy as np
import pandas as pd

import biaslib as B

FU = B.HERE.parents[1] / "followup"
sys.path.insert(0, str(FU / "scripts"))
os.environ.setdefault("SEC8FU_CACHE", str(B.P / ".cache" / "sec8fu"))
import common as C  # noqa: E402

OUT = B.HERE.parent / "outputs"
N = 10000
ESTS = ["r1", "r2_or", "r3_or"]
BS = [5, 10, 20, 40, 80, 160]
LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def simpson(mask):
    m = int(mask.sum())
    w = np.ones(m)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return w * 0.05 / 3


def main():
    df = pd.concat([C.load("r3or", "r3", "M1", N), C.load("full", "main", "M1", N)], ignore_index=True)
    df = df[df.estimator.isin(ESTS) & df.b.isin(BS)].drop_duplicates(["b", "rep", "estimator"])
    T = C.T
    lo = np.maximum(T, 1 - T)
    A0 = C.A_TRUE["M1"]
    rows = []
    for (b, est), g in df.groupby(["b", "estimator"]):
        X = g[C.TCOLS].to_numpy()
        hit = (np.abs(X - lo) < 1e-7).mean(0)
        rows.append(dict(b=b, k=N // b, estimator=est, **{f"hit_{t:.2f}": h for t, h in zip(T, hit)}))
    hb = pd.DataFrame(rows)
    hb.to_csv(OUT / "s14_boundary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    log(f"fraction of estimates on the lower boundary max(t,1-t), M1, n=1e4 (gap A0-max(t,1-t) at "
        f"t=0.15/0.20/0.30: {A0[1]-lo[1]:.4f}/{A0[2]-lo[2]:.4f}/{A0[4]-lo[4]:.4f}):")
    log(hb[["b", "k", "estimator", "hit_0.15", "hit_0.20", "hit_0.25", "hit_0.30", "hit_0.50", "hit_0.85"]]
        .round(3).to_string(index=False))

    eta = pd.read_csv(FU / "outputs" / "th_r3_eta.csv")
    ex3 = pd.read_csv(FU / "outputs" / "th_r3_exact_bias.csv")
    s04 = pd.read_csv(OUT / "s04_exact_bias.csv")
    mI = (T >= 0.3 - 1e-9) & (T <= 0.7 + 1e-9)
    regions = {"K": (C.mK, C.wK), "I=[0.3,0.7]": (mI, simpson(mI))}
    out = []
    for b in [5, 10, 20, 40, 80]:
        k = N // b
        E = {}
        for est in ("r2_or", "r3_or"):
            g = df[(df.b == b) & (df.estimator == est)].sort_values("rep")
            E[est] = g[C.TCOLS].to_numpy() - A0
        bt = {2: s04[(s04.model == "M1") & (s04.estimator == "r2or") & (s04.b == b)].sort_values("t").exact.to_numpy(),
              3: ex3[(ex3.estimator == "r3or") & (ex3.b == b)].sort_values("t").exact.to_numpy()}
        for name, (mask, w) in regions.items():
            L = w.sum()
            ise = {e: (E[e][:, mask] ** 2) @ w for e in E}
            rho, se = C.ratio_mcse(ise["r3_or"], ise["r2_or"])
            ivr = (E["r3_or"][:, mask].var(0, ddof=1) @ w) / (E["r2_or"][:, mask].var(0, ddof=1) @ w)
            s2 = {r: eta[eta.r == r].sort_values("t").sigma2.to_numpy() for r in (2, 3)}
            et = {r: eta[eta.r == r].sort_values("t").eta.to_numpy() for r in (2, 3)}
            ivr_as = (s2[3][mask] @ w) / (s2[2][mask] @ w)
            rec = dict(b=b, k=k, region=name, imse_ratio=rho, mcse=se, ivar_ratio=ivr, ivar_ratio_asym=ivr_as)
            for r, est in ((2, "r2_or"), (3, "r3_or")):
                kr = (k * (E[est] - bt[r])[:, mask] @ w) / L
                rec[f"kres_r{r}"] = kr.mean()
                rec[f"se_r{r}"] = kr.std(ddof=1) / np.sqrt(len(kr))
                rec[f"eta_r{r}"] = (et[r][mask] @ w) / L
            out.append(rec)
    o = pd.DataFrame(out)
    log("\nr3_or vs r2_or on K and on the interior subinterval I (M1, n=1e4, R=1000):")
    log(o.round(3).to_string(index=False))
    (OUT / "s14_output.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
