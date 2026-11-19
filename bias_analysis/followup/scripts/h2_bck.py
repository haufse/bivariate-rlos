"""h2_bck: hypothesis H2 of the bias analysis -- bootstrap 1/k bias correction (*_bck labels).

Paired comparison of mode bck (labels r1_bck, r2_or_bck, r2_eff_bck, r2_eff_bceq_bck, r2_id_bck)
with the uncorrected estimates of mode full (same data, same bootstrap draws; a00 checks that the
recomputed uncorrected estimates are identical), M1-M4, n in {5e3, 1e4, 2e4, 1e5}, full b grid.

Stated predictions (H2):
  P1  r = 2 IBias^2_K falls to about the k = inf values: M3 b=10 (n=1e4) 0.42e-4 -> ~0.33e-4,
      M1 b=40 (n=1e4) 0.154e-4 -> ~0;
  P2  IVar_K rises by about 0.5 %;
  P3  r2/r1 IMSE_K ratio at b = 10, n = 1e4: OPC (M3) ~1.05, Joe (M2) ~1.10 (s07 criterion E,
      oracle weights).
Extra tests (same theory, all cells): residual bias k*(mean error - exact k=inf bias) of the
bck estimators (expected ~0; uncorrected: eta_W); bootstrap bias estimate k*bias-hat vs eta_W;
r2/r1 ratio at the grid optimum vs criterion E (s07, 'Sec8 grid').
k=inf IBias^2 from s05_ibias.csv (ibias2_kinf), exact pointwise bias from s04 (+ th_theory).
Outputs: outputs/h2_cells.csv, outputs/h2_kresid.csv, outputs/h2_ratios.csv, outputs/h2_biashat.csv,
outputs/h2_output.txt, figs/h2_bck.pdf.
"""
import sys
import time

sys.dont_write_bytecode = True
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

MODELS = ["M1", "M2", "M3", "M4"]
NS = [5000, 10000, 20000, 100000]
PAIRS = [("r1", "r1_bck"), ("r2_or", "r2_or_bck"), ("r2_eff", "r2_eff_bck"),
         ("r2_eff_bceq", "r2_eff_bceq_bck"), ("r2_id", "r2_id_bck")]
UNC = [p[0] for p in PAIRS]
BCK = [p[1] for p in PAIRS]
EXW = {"r1": "r1", "r2_or": "r2or", "r2_eff": "r2or", "r2_id": "r2id"}      # exact-bias key
ETAW = {"r1": "r1", "r2_or": "or", "r2_eff": "or", "r2_id": "id"}
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def main():
    t0 = time.time()
    s04 = pd.read_csv(C.BIAS / "outputs" / "s04_exact_bias.csv")[["model", "b", "t", "estimator", "exact"]]
    ibk = pd.read_csv(C.BIAS / "outputs" / "s05_ibias.csv")
    th = pd.read_csv(C.BIAS / "outputs" / "s02_theory.csv")
    th = th[th.attractor == "GH2"]
    eta = {w: th[th.weights == w].sort_values("t").c_total.to_numpy() for w in ("r1", "or", "id")}
    sig = {w: th[th.weights == w].sort_values("t").sigma2.to_numpy() for w in ("r1", "or", "id")}
    s07 = pd.read_csv(C.BIAS / "outputs" / "s07_ratios.csv")
    Wb = C.boot_counts(1000)
    cells, kres, rat, bh = [], [], [], []
    store = {}
    for mo in MODELS:
        for n in NS:
            df = C.load_many([("full", "main", mo, n), ("bck", "main", mo, n)])
            dg = C.load("bck", "main", mo, n, kind="diag")
            bs = sorted(df.b.unique())
            reps, E = C.cube(df, UNC + BCK, bs)
            assert len(reps) == 1000 and np.isfinite(E).all()
            ise = (E[..., C.mK] ** 2) @ C.wK                        # (R, nb, ne)
            store[(mo, n)] = (bs, ise)
            for ib, b in enumerate(bs):
                k = n // b
                for ie, est in enumerate(UNC + BCK):
                    base = est.replace("_bck", "")
                    ibias, ibse, iv, nf = C.ibias2_with_se(E[:, ib, ie])
                    kin = ibk[(ibk.model == mo) & (ibk.n == n) & (ibk.b == b) & (ibk.estimator == base)]
                    rec = dict(model=mo, n=n, b=b, k=k, estimator=est, imse_K=ise[:, ib, ie].mean(),
                               mcse_K=ise[:, ib, ie].std(ddof=1) / np.sqrt(1000), ibias2_K=ibias,
                               ibias2_minus_floor=ibias - nf, mcse_ibias2=ibse, ivar_K=iv,
                               ibias2_kinf=float(kin.ibias2_kinf.iloc[0]) if len(kin) else np.nan,
                               ibias2_pred_unc=float(kin.ibias2_pred.iloc[0]) if len(kin) else np.nan)
                    if est in BCK:
                        ju = UNC.index(base)
                        rho, se = C.ratio_mcse(ise[:, ib, ie], ise[:, ib, ju])
                        # paired IVar ratio, bootstrap SE
                        e1, e0 = E[:, ib, ie][:, C.mK], E[:, ib, ju][:, C.mK]
                        vr = (e1.var(0, ddof=1) @ C.wK) / (e0.var(0, ddof=1) @ C.wK)
                        m1, m0 = Wb @ e1, Wb @ e0
                        v1 = (Wb @ e1 ** 2 - m1 ** 2) @ C.wK
                        v0 = (Wb @ e0 ** 2 - m0 ** 2) @ C.wK
                        rec.update(imse_ratio_bck_unc=rho, mcse_imse_ratio=se, ivar_ratio_bck_unc=vr,
                                   mcse_ivar_ratio=np.std(v1 / v0, ddof=1))
                    cells.append(rec)
                    # k-residuals vs exact k=inf bias (K-average of the pointwise values, Simpson/|K|)
                    if base in EXW:
                        ex = s04[(s04.model == mo) & (s04.b == b) & (s04.estimator == EXW[base])].sort_values("t")
                        if len(ex) == len(C.T):
                            e = E[:, ib, ie]
                            res = k * (e - ex.exact.to_numpy())             # (R, 17)
                            kr = (res[:, C.mK] @ C.wK) / 0.7               # K-average per rep
                            kres.append(dict(model=mo, n=n, b=b, k=k, estimator=est,
                                             kresid_K=kr.mean(), se=kr.std(ddof=1) / np.sqrt(1000),
                                             eta_K=(eta[ETAW[base]][C.mK] @ C.wK) / 0.7,
                                             kresid_05=res[:, 8].mean(), se_05=res[:, 8].std(ddof=1) / np.sqrt(1000)))
                # bootstrap bias-hat
                g = dg[dg.b == b]
                for w, wt in (("r1", "r1"), ("eff", "or"), ("id", "id"), ("or", "or")):
                    x = k * g[f"bk_{w}_K"].to_numpy()
                    bh.append(dict(model=mo, n=n, b=b, k=k, weights=w, k_biashat_K=x.mean(),
                                   se=x.std(ddof=1) / np.sqrt(len(x)), eta_K_mean=eta[wt][C.mK].mean(),
                                   k_biashat_05=(k * g[f"bk_{w}_0.50"]).mean(),
                                   eta_05=eta[wt][8], above1_rate=g.bck_above1_eff_K.mean() / C.mK.sum()))
            # r2/r1 ratios: fixed b and grid optimum, uncorrected and bck
            for tag, e1, e2s in (("unc", "r1", ["r2_or", "r2_eff", "r2_eff_bceq"]),
                                 ("bck", "r1_bck", ["r2_or_bck", "r2_eff_bck", "r2_eff_bceq_bck"])):
                i1 = (UNC + BCK).index(e1)
                for e2 in e2s:
                    i2 = (UNC + BCK).index(e2)
                    for ib, b in enumerate(bs):
                        rho, se = C.ratio_mcse(ise[:, ib, i2], ise[:, ib, i1])
                        rat.append(dict(model=mo, n=n, kind=tag, estimator=e2, b1=b, b2=b, ratio=rho,
                                        mcse=se, bset=f"b={b}"))
                    m1 = ise[:, :, i1].mean(0); m2 = ise[:, :, i2].mean(0)
                    j1, j2 = int(np.argmin(m1)), int(np.argmin(m2))
                    rho, se = C.ratio_mcse(ise[:, j2, i2], ise[:, j1, i1])
                    bb1 = (Wb @ ise[:, :, i1]).min(1); bb2 = (Wb @ ise[:, :, i2]).min(1)
                    rat.append(dict(model=mo, n=n, kind=tag, estimator=e2, b1=bs[j1], b2=bs[j2], ratio=rho,
                                    mcse=max(se, np.std(bb2 / bb1, ddof=1)), bset="grid opt"))
    cells = pd.DataFrame(cells); kres = pd.DataFrame(kres); rat = pd.DataFrame(rat); bh = pd.DataFrame(bh)
    # predictions for the ratios: criterion E (asym var + exact k=inf bias, oracle), s07
    def predE(mo, n, bset):
        if bset == "grid opt":
            s = s07[(s07.model == mo) & (s07.n == n) & (s07.bset == "Sec8 grid") & (s07.criterion == "E")]
        else:
            s = s07[(s07.model == mo) & (s07.n == n) & (s07.bset == bset) & (s07.criterion == "E")]
        return float(s.ratio.iloc[0]) if len(s) else np.nan
    rat["pred_E"] = [predE(r.model, r.n, r.bset) if r.kind == "bck" else np.nan for r in rat.itertuples()]
    # semi-empirical E with MC variance of the bck estimators (oracle): IVar_MC + IBias2_kinf
    ce = cells.set_index(["model", "n", "b", "estimator"])
    pe = []
    for r in rat.itertuples():
        if r.kind != "bck" or r.estimator != "r2_or_bck":
            pe.append(np.nan); continue
        try:
            a2 = ce.loc[(r.model, r.n, r.b2, "r2_or_bck")]; a1 = ce.loc[(r.model, r.n, r.b1, "r1_bck")]
            pe.append((a2.ivar_K + a2.ibias2_kinf) / (a1.ivar_K + a1.ibias2_kinf))
        except KeyError:
            pe.append(np.nan)
    rat["pred_E_mcvar"] = pe
    C.save(cells, "h2_cells.csv"); C.save(kres, "h2_kresid.csv"); C.save(rat, "h2_ratios.csv")
    C.save(bh, "h2_biashat.csv")
    pd.set_option("display.width", 250)

    # ---------------- P1
    log("P1: IBias^2_K (1e-4 units; raw MC, raw minus noise floor IVar/R, MCSE, k=inf prediction)")
    for mo, n, b in (("M3", 10000, 10), ("M1", 10000, 40), ("M2", 10000, 10), ("M4", 10000, 10),
                     ("M1", 10000, 5), ("M3", 10000, 20), ("M1", 10000, 160)):
        x = cells[(cells.model == mo) & (cells.n == n) & (cells.b == b)].set_index("estimator")
        for est in ("r2_eff", "r2_eff_bck", "r2_or", "r2_or_bck", "r1", "r1_bck"):
            r = x.loc[est]
            log(f"  {mo} n={n} b={b} {est:16s} raw {1e4 * r.ibias2_K:.4f}  -floor {1e4 * r.ibias2_minus_floor:.4f}"
                f"  (MCSE {1e4 * r.mcse_ibias2:.4f})  kinf {1e4 * r.ibias2_kinf:.4f}"
                f"  z={(r.ibias2_minus_floor - r.ibias2_kinf) / r.mcse_ibias2:+.2f}")
    # all-cell summary of P1: z-scores of (IBias2 - floor - kinf)/SE for the bck labels
    b_ = cells[cells.estimator.isin(BCK)].copy()
    b_["z"] = (b_.ibias2_minus_floor - b_.ibias2_kinf) / b_.mcse_ibias2
    log("\nP1 over all 104 cells per bck label: share |z|<=2, median z, and cells with k>=250:")
    for est, g in b_.groupby("estimator"):
        g2 = g[g.k >= 250]
        log(f"  {est:16s} |z|<=2: {np.mean(np.abs(g.z) <= 2):.2f} (k>=250: {np.mean(np.abs(g2.z) <= 2):.2f}), "
            f"median z {g.z.median():+.2f} (k>=250: {g2.z.median():+.2f})")
    # ---------------- P2
    log("\nP2: IVar_K ratio bck/uncorrected (bootstrap SE); prediction ~1.005")
    v = cells[cells.estimator.isin(BCK)]
    log(v.groupby("estimator").ivar_ratio_bck_unc.describe()[["mean", "min", "50%", "max"]].round(4).to_string())
    for mo, n, b in (("M3", 10000, 10), ("M1", 10000, 40), ("M2", 10000, 10)):
        x = v[(v.model == mo) & (v.n == n) & (v.b == b)]
        log(f"  {mo} n={n} b={b}: " + ", ".join(f"{r.estimator} {r.ivar_ratio_bck_unc:.4f} ({r.mcse_ivar_ratio:.4f})"
                                                for r in x.itertuples()))
    pooled = v.groupby("estimator").apply(lambda g: np.average(g.ivar_ratio_bck_unc, weights=1 / g.mcse_ivar_ratio ** 2))
    log("  precision-weighted mean over all cells: " + ", ".join(f"{k} {x:.4f}" for k, x in pooled.items()))
    # ---------------- P3
    log("\nP3: r2/r1 IMSE_K ratio, n=1e4, b=10 (pred. E criterion, oracle): uncorrected vs bck")
    for mo in MODELS:
        x = rat[(rat.model == mo) & (rat.n == 10000) & (rat.bset == "b=10")]
        for e in ("r2_or", "r2_eff", "r2_eff_bceq"):
            u = x[(x.kind == "unc") & (x.estimator == e)].iloc[0]
            bk = x[(x.kind == "bck") & (x.estimator == e + "_bck")].iloc[0]
            pz = f"; pred E {bk.pred_E:.3f}, z={(bk.ratio - bk.pred_E) / bk.mcse:+.2f}" if e == "r2_or" else ""
            pz += f"; E with MC var {bk.pred_E_mcvar:.3f}" if e == "r2_or" else ""
            log(f"  {mo} {e:12s} unc {u.ratio:.3f} ({u.mcse:.3f}) -> bck {bk.ratio:.3f} ({bk.mcse:.3f}){pz}")
    log("\nr2_or_bck / r1_bck at the grid optimum vs criterion E (s07, Sec8 grid) -- all n")
    x = rat[(rat.bset == "grid opt") & (rat.estimator == "r2_or_bck")]
    u = rat[(rat.bset == "grid opt") & (rat.estimator == "r2_or")].set_index(["model", "n"])
    for r in x.itertuples():
        uu = u.loc[(r.model, r.n)]
        log(f"  {r.model} n={r.n}: unc {uu.ratio:.3f} ({uu.mcse:.3f}, b {uu.b1}/{uu.b2}) -> bck {r.ratio:.3f} "
            f"({r.mcse:.3f}, b {r.b1}/{r.b2}); pred E {r.pred_E:.3f}; z={(r.ratio - r.pred_E) / r.mcse:+.2f}")
    log("\nr2_eff_bceq(_bck) / r1(_bck) at the grid optimum (the Sec. 8 'corrected' column):")
    for mo in MODELS:
        for n in NS:
            a = rat[(rat.model == mo) & (rat.n == n) & (rat.bset == "grid opt")]
            u_ = a[a.estimator == "r2_eff_bceq"].iloc[0]; b2 = a[a.estimator == "r2_eff_bceq_bck"].iloc[0]
            log(f"  {mo} n={n}: unc {u_.ratio:.3f} ({u_.mcse:.3f}) -> bck {b2.ratio:.3f} ({b2.mcse:.3f})")
    log("\nIMSE_K ratio bck/uncorrected at each estimator's own grid optimum (does the correction pay?):")
    for mo in MODELS:
        for n in NS:
            bs, ise = store[(mo, n)]
            m = ise.mean(0)
            parts = []
            for e0, e1 in PAIRS:
                i0, i1 = UNC.index(e0), len(UNC) + BCK.index(e1)
                j0, j1 = int(np.argmin(m[:, i0])), int(np.argmin(m[:, i1]))
                rho, se = C.ratio_mcse(ise[:, j1, i1], ise[:, j0, i0])
                parts.append(f"{e0} {rho:.3f}({se:.3f})")
            log(f"  {mo} n={n}: " + ", ".join(parts))
    # ---------------- k-residuals pooled over models
    log("\nk*(mean error - exact k=inf bias), K-average, pooled over M1-M4 and n with equal k (precision-weighted):")
    kres["w"] = 1 / kres.se ** 2
    pk = kres.groupby(["k", "estimator"]).apply(lambda g: pd.Series(dict(
        kres=np.average(g.kresid_K, weights=g.w), se=1 / np.sqrt(g.w.sum()), ncell=len(g)))).reset_index()
    tab = pk.pivot_table(index="k", columns="estimator", values="kres").round(3)
    tse = pk.pivot_table(index="k", columns="estimator", values="se").round(3)
    log(tab[["r1", "r1_bck", "r2_or", "r2_or_bck", "r2_eff", "r2_eff_bck", "r2_id", "r2_id_bck"]].to_string())
    log("pooled SE:\n" + tse[["r1", "r1_bck", "r2_or", "r2_or_bck", "r2_eff", "r2_eff_bck"]].to_string())
    log("theory eta_K (K-average): r1 %.3f, or %.3f, id %.3f" % tuple(
        (eta[w][C.mK] @ C.wK) / 0.7 for w in ("r1", "or", "id")))
    C.save(pk, "h2_kresid_pooled.csv")
    # bias-hat
    log("\nbootstrap bias estimate k*bias-hat (mean over t in K), pooled over models, by k:")
    bh["w"] = 1 / bh.se ** 2
    pb = bh.groupby(["k", "weights"]).apply(lambda g: np.average(g.k_biashat_K, weights=g.w)).unstack().round(3)
    log(pb.to_string())
    log("theory eta (plain mean over the 15 K points): " + ", ".join(
        f"{w} {eta[wt][C.mK].mean():.3f}" for w, wt in (("r1", "r1"), ("or", "or"), ("id", "id"))))
    log(f"r2_eff_bck above 1 on K (share of t-points): max over cells {bh.above1_rate.max():.4f}")

    # ---------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for est, col, mk in (("r1", "k", "o"), ("r2_or", "#0072B2", "s"), ("r2_eff", "#D55E00", "^"),
                         ("r2_id", "#009E73", "v")):
        for suf, ls, fc in (("", "-", "none"), ("_bck", "--", None)):
            g = pk[pk.estimator == est + suf]
            ax[0].errorbar(g.k, g.kres, yerr=2 * g.se, fmt=mk + ls, color=col, ms=4, lw=0.8,
                           mfc=fc if fc else col, label=est + suf)
    ax[0].axhline(0, color="grey", lw=.5); ax[0].set_xscale("log"); ax[0].set_xlabel("k")
    ax[0].set_ylabel("k (mean error - exact k=inf bias), K-avg"); ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title("residual 1/k bias: uncorrected (open) vs bck (filled)", fontsize=8)
    x = cells[cells.estimator.isin(["r2_or", "r2_or_bck", "r2_eff", "r2_eff_bck"]) & (cells.n == 10000)]
    for est, col in (("r2_or", "#0072B2"), ("r2_or_bck", "#56B4E9"), ("r2_eff", "#D55E00"), ("r2_eff_bck", "#E69F00")):
        for mo, mk in zip(MODELS, "osD^"):
            g = x[(x.estimator == est) & (x.model == mo)]
            ax[1].errorbar(g.ibias2_kinf * 1e4, g.ibias2_minus_floor * 1e4, yerr=2e4 * g.mcse_ibias2, fmt=mk,
                           color=col, ms=3, lw=.6, label=f"{est}" if mo == "M1" else None)
    lim = [1e-4, 5]
    ax[1].plot(lim, lim, "k:", lw=.6); ax[1].set_xscale("symlog", linthresh=1e-3); ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[1].set_xlabel("k=inf IBias2 (1e-4)"); ax[1].set_ylabel("MC IBias2 - IVar/R (1e-4)")
    ax[1].set_title("n=1e4, all b; markers M1 o, M2 s, M3 D, M4 ^", fontsize=8); ax[1].legend(fontsize=6)
    y = rat[(rat.n == 10000) & (rat.estimator.isin(["r2_or", "r2_or_bck"])) & (rat.bset.str.startswith("b="))]
    for mo, col in zip(MODELS, ("k", "#E69F00", "#0072B2", "#009E73")):
        for kind, ls in (("unc", ":"), ("bck", "-")):
            g = y[(y.model == mo) & (y.kind == kind)]
            ax[2].errorbar(g.b1, g.ratio, yerr=2 * g.mcse, fmt="o" + ls, color=col, ms=3, lw=.8,
                           mfc="none" if kind == "unc" else col, label=f"{mo} {kind}")
    for mo, b, p in (("M3", 10, 1.053), ("M2", 10, 1.095)):
        ax[2].plot([b], [p], "x", color="r", ms=8)
    ax[2].axhline(1, color="grey", lw=.5); ax[2].set_xscale("log"); ax[2].set_xlabel("b")
    ax[2].set_ylabel("IMSE_K ratio r2_or/r1 (paired)"); ax[2].legend(fontsize=6, ncol=2)
    ax[2].set_title("n=1e4; x = stated H2 predictions (b=10)", fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGS / "h2_bck.pdf")
    log(f"\nruntime {time.time() - t0:.0f} s")
    (C.OUT / "h2_output.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
