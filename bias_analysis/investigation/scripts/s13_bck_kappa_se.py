"""s13: data part of the variance inflation of the bootstrap correction, with Monte Carlo standard
errors, and the prediction with derivatives at the pilot limit A_bar_m.

Proposition sec8-bias:prop:bck-variance (revised): Var(A^bck) = Var(A - beta0) + E Var*(A*)/B, and
Var(A - beta0)/Var(A) = 1 - 2 kappa_W/k + o(1/k).  Per cell (model, n, b; t = 1/2; weights r1, or,
id) from the Sec. 8 shards (modes full and bck, read only):
    ratio  = Var(A^bck)/Var(A)       (paired, R = 1000),
    noise  = mean(bk_sd^2)/B / Var(A)   (B = 200; the bootstrap part, observed),
    D      = ratio - 1 - noise        (the data part; = -2 kappa/k + O(k^-2)),
    kappa_cov = k Cov(A, beta_hat)/Var(A)   (round 2: exact for Cov(A, beta0), free of 1/B and O(k^-2)),
    kappa_obs = -k D / 2,
with standard errors from 400 bootstrap resamples of the replications (pairs kept).  Pooling over
cells with the same k: precision-weighted mean.  Predictions (s11 covariances, s10 derivatives):
    kappa_A0   = derivatives at A0 (GH(2)), as in Remark sec8-bias:rem:bck-variance;
    kappa_Abar = F_a, F_a2 re-evaluated at the deterministic limit A_bar_m of the m = 40 Bernstein
                 pilot (pilot fitted to the exact A0 curve), N_GH unchanged (approximation).
The covariances Cov(A,P)/Var(A), Cov(A,P'')/Var(A) are recovered from the s11 columns.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SEC8FU_CACHE=<scratch> \
     python s13_bck_kappa_se.py
Writes ../outputs/s13_output.txt, ../outputs/s13_cells.csv.  Deterministic (seeded resampling).
"""
import os
import sys

import numpy as np
import pandas as pd

import biaslib as B
import s10_eta_derivative as S10

FU = B.HERE.parents[1] / "followup"
sys.path.insert(0, str(FU / "scripts"))
os.environ.setdefault("SEC8FU_CACHE", str(B.P / ".cache" / "sec8fu"))
import common as C  # noqa: E402  (follow-up loader, read-only use)
sys.path.insert(0, str(B.SEC8))
from pickands import fit_pickands_pilot  # noqa: E402

OUT = B.HERE.parent / "outputs"
MODELS, NS, BOOT, NRES = ["M1", "M2", "M3", "M4"], [10000, 20000], 200, 400
LAB = {"r1": ("r1", "r1_bck"), "or": ("r2_or", "r2_or_bck"), "id": ("r2_id", "r2_id_bck")}
LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def main():
    rng = np.random.default_rng(20261119)
    rows = []
    for mo in MODELS:
        for n in NS:
            full = C.load("full", "main", mo, n)
            bck = C.load("bck", "main", mo, n)
            dg = C.load("bck", "main", mo, n, kind="diag")
            e = pd.concat([full[full.estimator.isin([v[0] for v in LAB.values()])],
                           bck[bck.estimator.isin([v[1] for v in LAB.values()])]])
            w = e.pivot_table(index=["b", "rep"], columns="estimator", values="A_0.50").reset_index()
            w = w.merge(dg[["b", "rep"] + [f"bk_{x}_sd_0.50" for x in LAB]], on=["b", "rep"])
            for b, g in w.groupby("b"):
                k = n // b
                R = len(g)
                idx = rng.integers(0, R, size=(NRES, R))
                for wt, (u, c) in LAB.items():
                    x, y = g[u].to_numpy(), g[c].to_numpy()
                    s2 = g[f"bk_{wt}_sd_0.50"].to_numpy() ** 2

                    def stat(ii):
                        ve = x[ii].var(ddof=1)
                        return y[ii].var(ddof=1) / ve - 1 - s2[ii].mean() / BOOT / ve

                    def kcov(ii):   # round 2 (n1): kappa = k Cov(A, beta_hat)/Var(A), beta_hat = A - A_bck
                        return k * np.cov(x[ii], x[ii] - y[ii])[0, 1] / x[ii].var(ddof=1)
                    D = stat(np.arange(R))
                    Db = np.array([stat(ii) for ii in idx])
                    Kc = kcov(np.arange(R))
                    Kcb = np.array([kcov(ii) for ii in idx])
                    rows.append(dict(model=mo, n=n, b=b, k=k, weights=wt, D=D, se_D=Db.std(ddof=1),
                                     kappa_obs=-k * D / 2, se_kappa=k * Db.std(ddof=1) / 2,
                                     kappa_cov=Kc, se_kappa_cov=Kcb.std(ddof=1)))
    obs = pd.DataFrame(rows)

    # predictions from s11 (covariances) and s10 (derivatives at A0)
    s11 = pd.read_csv(OUT / "s11_cells.csv")
    s10 = pd.read_csv(OUT / "s10_eta_derivative.csv").set_index("weights")
    # derivatives at the pilot limit A_bar_m (m = 40): pilot fitted to the exact GH(2) curve
    tg = np.linspace(0, 1, 201)
    pb = fit_pickands_pilot(tg, B.A_log(tg), m=40)
    abar, abar2 = float(pb(0.5)), float(pb.derivative(0.5, 2))
    a0, _, a20 = S10.gh_local(2.0, 0.5)
    log(f"pilot limit A_bar_40: A(1/2) = {abar:.5f} (A0 {a0:.5f}), A''(1/2) = {abar2:.4f} (A0'' {a20:.4f})")
    j5 = int(np.argmin(np.abs(B.T - 0.5)))
    Sig = B.oracle_sigma("GH2")[j5]
    W = {"r1": B.weights("r1", Sig[None])[0], "or": np.linalg.inv(Sig), "id": np.eye(4)}
    Fbar = {}
    h = 1e-4
    for wt, Ww in W.items():
        def eta(a, a2):
            rk, nu = S10.eta_parts(0.5, a, 0.0, a2, Sig, Ww)
            return rk + nu
        Fbar[wt] = ((eta(abar + h, abar2) - eta(abar - h, abar2)) / (2 * h),
                    (eta(abar, abar2 + h) - eta(abar, abar2 - h)) / (2 * h),
                    eta(abar, abar2) - eta(a0, a20))
        log(f"  {wt}: F_a, F_a2 at A0 = ({s10.loc[wt, 'F_a']:.4f}, {s10.loc[wt, 'F_a2']:.4f}); at A_bar = "
            f"({Fbar[wt][0]:.4f}, {Fbar[wt][1]:.4f}); eta[A_bar] - eta[A0] (Sigma fixed) = {Fbar[wt][2]:.4f}")
    pr = []
    for _, r in s11.iterrows():
        Fa, Fa2, N = s10.loc[r.weights, "F_a"], s10.loc[r.weights, "F_a2"], s10.loc[r.weights, "N_GH"]
        kap0 = -r.k * r.plugin_pred / 2
        cp = (-r.k * (r.plugin_pred - r.plugin_pred_local_only) / 2) / N        # Cov(A,P)/Var(A)
        cp2 = (kap0 - (Fa + N) * cp) / Fa2                                      # Cov(A,P'')/Var(A)
        Fab, Fa2b, _ = Fbar[r.weights]
        pr.append(dict(model=r.model, n=r.n, b=r.b, weights=r.weights, kappa_A0=kap0,
                       kappa_Abar=(Fab + N) * cp + Fa2b * cp2, c_P=cp, c_P2=cp2))
    m = obs.merge(pd.DataFrame(pr), on=["model", "n", "b", "weights"])
    m.to_csv(OUT / "s13_cells.csv", index=False, float_format="%.6g")

    def pool(g):
        wgt = 1 / g.se_kappa ** 2
        return pd.Series(dict(cells=len(g), kappa_obs=np.sum(wgt * g.kappa_obs) / wgt.sum(),
                              se=1 / np.sqrt(wgt.sum()), kappa_A0=g.kappa_A0.mean(),
                              kappa_Abar=g.kappa_Abar.mean()))
    pd.set_option("display.width", 200)
    log("\nkappa by k (precision-weighted over models and n; SE from 400 bootstrap resamples per cell):")
    tab = m.groupby(["weights", "k"]).apply(pool)
    tab["shortfall_A0"] = 1 - tab.kappa_A0 / tab.kappa_obs
    log(tab.round(3).to_string())
    log("\npooled over all k (precision-weighted), and over k <= 500:")
    log(m.groupby("weights").apply(pool).round(3).to_string())
    log(m[m.k <= 500].groupby("weights").apply(pool).round(3).to_string())
    log("\nmedians over cells (as in the earlier text), oracle: kappa_obs "
        f"{m[m.weights == 'or'].kappa_obs.median():.3f}")

    def pool2(g):   # round 2 (n1): covariance estimator, free of the 1/B and O(k^-2) terms
        wgt = 1 / g.se_kappa_cov ** 2
        return pd.Series(dict(cells=len(g), kappa_cov=np.sum(wgt * g.kappa_cov) / wgt.sum(),
                              se=1 / np.sqrt(wgt.sum()), kappa_A0=g.kappa_A0.mean(),
                              kappa_Abar=g.kappa_Abar.mean(), median=g.kappa_cov.median()))
    log("\nROUND 2: kappa_hat = k Cov(A, beta_hat)/Var(A), by k and pooled (precision weights):")
    t2 = m.groupby(["weights", "k"]).apply(pool2)
    t2["shortfall_A0"] = 1 - t2.kappa_A0 / t2.kappa_cov
    log(t2.round(3).to_string())
    p2 = m.groupby("weights").apply(pool2)
    p2["shortfall_A0"] = 1 - p2.kappa_A0 / p2.kappa_cov
    p2["shortfall_Abar"] = 1 - p2.kappa_Abar / p2.kappa_cov
    log(p2.round(3).to_string())
    log("pooled over k >= 125:")
    log(m[m.k >= 125].groupby("weights").apply(pool2).round(3).to_string())
    (OUT / "s13_output.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
