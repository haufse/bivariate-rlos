"""h3_r3or: hypothesis H3 of the bias analysis -- r = 3 with oracle Sigma_3^{-1} weights, M1, n = 1e4.

Stated prediction: r3_or is not better than r2_or for k <~ 1400 (b >= 10).
Quantitative version (DERIVED HERE from the expansion of the bias analysis, no fitting): criterion C
  IMSE_K ~ int_K { sigma_r^2/k + (beta_{b,r} + eta_r/k)^2 },
with sigma_r^2, eta_r from th_r3_eta.csv (oracle weights; r = 2 equals s02 'or') and the exact
k = inf bias beta_{b,r} from th_r3_exact_bias.csv (r = 3) and s04 (r = 2).  Also with the MC IVar
in place of sigma^2/k.  Residual check: k*(mean error - beta_{b,3}) against eta_3 (K-average and
t = 0.3, 0.5).
Data (all paired, same datasets; a00 verifies that r1 is identical across exps main and r3):
  r3or/r3/M1/n10000 (r3_or, r3_or_bceq), full/r3/M1/n10000 (r1, r3_eff, r3_id, ...),
  full/main/M1/n10000 (r2_or, r2_eff, r2_id, ... ).
Outputs: outputs/h3_cells.csv, outputs/h3_ratios.csv, outputs/h3_output.txt, figs/h3_r3or.pdf.
"""
import sys

sys.dont_write_bytecode = True
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

BS = [5, 10, 20, 40, 80]
N = 10000
ESTS = ["r1", "r2_or", "r2_or_bceq", "r2_eff", "r2_eff_bceq", "r2_id",
        "r3_or", "r3_or_bceq", "r3_eff", "r3_eff_bceq", "r3_id"]
RATIOS = [("r3_or", "r2_or"), ("r3_or", "r1"), ("r2_or", "r1"), ("r3_or_bceq", "r2_or_bceq"),
          ("r3_eff", "r2_eff"), ("r3_eff", "r3_or"), ("r3_id", "r2_id")]
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def main():
    df = pd.concat([C.load("r3or", "r3", "M1", N), C.load("full", "r3", "M1", N),
                    C.load("full", "main", "M1", N)], ignore_index=True)
    df = df[df.estimator.isin(ESTS) & df.b.isin(BS)]
    df = df.drop_duplicates(["b", "rep", "estimator"])     # r1 is in both exps (identical, a00)
    dg = C.load("full", "r3", "M1", N, kind="diag")
    reps, E = C.cube(df, ESTS, BS)
    assert len(reps) == 1000 and np.isfinite(E).all()
    ise = (E[..., C.mK] ** 2) @ C.wK
    eta = pd.read_csv(C.OUT / "th_r3_eta.csv")
    ex3 = pd.read_csv(C.OUT / "th_r3_exact_bias.csv")
    s04 = pd.read_csv(C.BIAS / "outputs" / "s04_exact_bias.csv")
    th = {r: eta[eta.r == r].sort_values("t") for r in (1, 2, 3)}
    beta = {}
    for b in BS:
        beta[(3, b)] = ex3[(ex3.estimator == "r3or") & (ex3.b == b)].sort_values("t").exact.to_numpy()
        beta[(2, b)] = s04[(s04.model == "M1") & (s04.estimator == "r2or") & (s04.b == b)].sort_values("t").exact.to_numpy()
        beta[(1, b)] = np.zeros(len(C.T))
    cells, rats = [], []
    for ib, b in enumerate(BS):
        k = N // b
        for ie, est in enumerate(ESTS):
            ibias, ibse, iv, nf = C.ibias2_with_se(E[:, ib, ie])
            rec = dict(b=b, k=k, estimator=est, imse_K=ise[:, ib, ie].mean(),
                       mcse_K=ise[:, ib, ie].std(ddof=1) / np.sqrt(1000), ibias2_K=ibias, mcse_ibias2=ibse,
                       ivar_K=iv, noise_floor=nf)
            r = {"r1": 1, "r2_or": 2, "r3_or": 3}.get(est)
            if r:
                sg, et = th[r].sigma2.to_numpy(), th[r].eta.to_numpy()
                bt = beta[(r, b)]
                rec.update(pred_ivar=(sg[C.mK] @ C.wK) / k, pred_ibias2=((bt + et / k)[C.mK] ** 2) @ C.wK,
                           ibias2_kinf=(bt[C.mK] ** 2) @ C.wK)
                rec["pred_C"] = rec["pred_ivar"] + rec["pred_ibias2"]
                rec["pred_C_mcvar"] = iv + rec["pred_ibias2"]
                res = k * (E[:, ib, ie] - bt)
                kr = (res[:, C.mK] @ C.wK) / 0.7
                rec.update(kresid_K=kr.mean(), se_kresid_K=kr.std(ddof=1) / np.sqrt(1000),
                           eta_K=(et[C.mK] @ C.wK) / 0.7,
                           kresid_030=res[:, 4].mean(), se_030=res[:, 4].std(ddof=1) / np.sqrt(1000), eta_030=et[4],
                           kresid_050=res[:, 8].mean(), se_050=res[:, 8].std(ddof=1) / np.sqrt(1000), eta_050=et[8])
            cells.append(rec)
        for a, c in RATIOS:
            ia, ic = ESTS.index(a), ESTS.index(c)
            rho, se = C.ratio_mcse(ise[:, ib, ia], ise[:, ib, ic])
            ivr = (E[:, ib, ia][:, C.mK].var(0, ddof=1) @ C.wK) / (E[:, ib, ic][:, C.mK].var(0, ddof=1) @ C.wK)
            rats.append(dict(b=b, k=k, num=a, den=c, ratio=rho, mcse=se, ivar_ratio=ivr))
    cells = pd.DataFrame(cells); rats = pd.DataFrame(rats)
    ce = cells.set_index(["b", "estimator"])
    for col in ("pred_C", "pred_C_mcvar"):
        rats[col] = [ce.loc[(r.b, r.num), col] / ce.loc[(r.b, r.den), col]
                     if (r.num in ("r3_or", "r2_or") and r.den in ("r2_or", "r1")) else np.nan
                     for r in rats.itertuples()]
    fb = dg[dg.b.isin(BS)].groupby("b").fallback_K.mean()
    rats["r3_eff_fallback_rate"] = rats.b.map(fb)
    C.save(cells, "h3_cells.csv"); C.save(rats, "h3_ratios.csv")
    pd.set_option("display.width", 220)
    log("IMSE_K, IBias2_K, IVar_K (1e-4) and C-criterion predictions, M1, n=1e4:")
    x = cells[cells.estimator.isin(["r1", "r2_or", "r3_or"])].copy()
    for c in ("imse_K", "mcse_K", "ibias2_K", "mcse_ibias2", "ivar_K", "pred_ivar", "pred_ibias2", "ibias2_kinf", "pred_C"):
        x[c] = 1e4 * x[c]
    log(x[["b", "k", "estimator", "imse_K", "mcse_K", "ibias2_K", "mcse_ibias2", "ivar_K", "pred_ivar",
           "ibias2_kinf", "pred_ibias2", "pred_C"]].round(4).to_string(index=False))
    log("\nk*(mean error - exact k=inf bias) vs eta (oracle weights):")
    log(cells[cells.estimator.isin(["r2_or", "r3_or"])][["b", "k", "estimator", "kresid_K", "se_kresid_K", "eta_K",
                                                          "kresid_030", "se_030", "eta_030", "kresid_050", "se_050",
                                                          "eta_050"]].round(3).to_string(index=False))
    log("\npaired IMSE_K ratios (MCSE), IVar ratio, C predictions; r3_eff fallback rate:")
    log(rats.round(3).to_string(index=False))
    log("\nH3 verdict: r3_or / r2_or for b >= 10 (prediction: not < 1):")
    for r in rats[(rats.num == "r3_or") & (rats.den == "r2_or")].itertuples():
        log(f"  b={r.b} k={r.k}: {r.ratio:.3f} ({r.mcse:.3f}), z(1) = {(r.ratio - 1) / r.mcse:+.2f}; "
            f"C pred {r.pred_C:.3f}, C+MCvar {r.pred_C_mcvar:.3f}, z(C+MCvar) = {(r.ratio - r.pred_C_mcvar) / r.mcse:+.2f}; "
            f"IVar ratio {r.ivar_ratio:.3f}")
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for est, col in (("r1", "k"), ("r2_or", "#0072B2"), ("r3_or", "#D55E00"), ("r3_eff", "#E69F00")):
        g = cells[cells.estimator == est]
        ax[0].errorbar(g.b, 1e4 * g.imse_K, yerr=2e4 * g.mcse_K, fmt="o", color=col, ms=4, label=est)
        if "pred_C" in g and g.pred_C.notna().any():
            ax[0].plot(g.b, 1e4 * g.pred_C, "-", color=col, lw=.8)
            ax[0].plot(g.b, 1e4 * g.pred_C_mcvar, ":", color=col, lw=.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("b"); ax[0].set_ylabel("1e4 IMSE_K")
    ax[0].legend(fontsize=7); ax[0].set_title("M1 n=1e4; lines: C (solid), C with MC var (dotted)", fontsize=8)
    for (a, c), col in zip([("r3_or", "r2_or"), ("r3_eff", "r2_eff"), ("r3_eff", "r3_or")],
                           ("#D55E00", "#E69F00", "#009E73")):
        g = rats[(rats.num == a) & (rats.den == c)]
        ax[1].errorbar(g.b, g.ratio, yerr=2 * g.mcse, fmt="o-", color=col, ms=4, lw=.8, label=f"{a}/{c}")
        if g.pred_C_mcvar.notna().any():
            ax[1].plot(g.b, g.pred_C_mcvar, "x", color=col, ms=7)
    ax[1].axhline(1, color="grey", lw=.5); ax[1].set_xscale("log"); ax[1].set_xlabel("b")
    ax[1].legend(fontsize=7); ax[1].set_title("paired IMSE_K ratios; x: C with MC var", fontsize=8)
    for est, col in (("r2_or", "#0072B2"), ("r3_or", "#D55E00")):
        g = cells[cells.estimator == est]
        ax[2].errorbar(g.k, g.kresid_K, yerr=2 * g.se_kresid_K, fmt="o", color=col, ms=4, label=est)
        ax[2].axhline(g.eta_K.iloc[0], color=col, ls="--", lw=.8)
    ax[2].set_xscale("log"); ax[2].set_xlabel("k"); ax[2].set_ylabel("k (mean err - exact k=inf bias), K-avg")
    ax[2].legend(fontsize=7); ax[2].set_title("dashed: eta_W (theory)", fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGS / "h3_r3or.pdf")
    (C.OUT / "h3_output.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
