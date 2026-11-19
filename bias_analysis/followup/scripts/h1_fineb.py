"""h1_fineb: hypothesis H1 of the bias analysis -- r2/r1 IMSE_K ratio at each estimator's own
optimum over the merged fine b grid (modes full + smallb + fineb, rows 'main').

Prediction (criterion C of s07, all integer b, oracle weights): M3 1.137 (n=1e4), 1.024 (1e5);
M2 0.863 (1e4), 0.913 (1e5); optimal b (r1, r2) = (11, 13) at n=1e4 for M3 and M2.
M4 has no stated H1 prediction (s07 C: 1.619 with b = 2/3; reported for reference only).

Monte Carlo quantities (R = 1000, paired over reps):
  * IMSE_K(b), IBias2_K, IVar_K per estimator and b, MCSE;
  * ratio of minima min_b IMSE(r2_x) / min_b IMSE(r1) with
      - delta-method MCSE at the selected b (as summarize.py, ignores the selection),
      - nonparametric bootstrap over replications (B = 2000, resampling the same reps for all b and
        estimators), which includes the argmin selection; bootstrap estimate of the selection bias;
  * smoothed optimum: quadratic fit of log IMSE in log b on the grid points in [b*/2, 2 b*]
    (b* = grid argmin of the point estimate), minimum of the fit, bootstrapped likewise.
Theory curves on the merged grid (criterion L/E/C as s07; C with MC IVar as a semi-empirical
variant).  Outputs: outputs/h1_imse_curves.csv, outputs/h1_optimum.csv, outputs/h1_theory_curves.csv,
outputs/h1_output.txt, figs/h1_fineb_imse.pdf.
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

CELLS = [("M2", 10000), ("M3", 10000), ("M4", 10000), ("M2", 100000), ("M3", 100000)]
ESTS = ["r1", "r2_or", "r2_or_bceq", "r2_eff", "r2_eff_bceq", "r2_id", "pot"]
PRED = {("M3", 10000): 1.137, ("M3", 100000): 1.024, ("M2", 10000): 0.863, ("M2", 100000): 0.913}
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def theory_inputs():
    s04 = pd.read_csv(C.BIAS / "outputs" / "s04_exact_bias.csv")[["model", "b", "t", "estimator", "exact"]]
    ext = pd.read_csv(C.OUT / "th_exact_bias_extra.csv")
    ex = pd.concat([s04, ext], ignore_index=True).drop_duplicates(["model", "b", "t", "estimator"])
    th = pd.read_csv(C.BIAS / "outputs" / "s02_theory.csv")
    th = th[th.attractor == "GH2"]
    V = {}, {}
    sig = {w: th[th.weights == w].sort_values("t").sigma2.to_numpy() for w in ("r1", "or", "id")}
    eta = {w: th[th.weights == w].sort_values("t").c_total.to_numpy() for w in ("r1", "or", "id")}
    return ex, sig, eta


def crit_curves(model, n, bs, ex, sig, eta, ivar_mc):
    """Criteria L-free: E, C (asymptotic variance), C_mcvar (MC IVar) for r1 and r2 oracle."""
    rows = []
    for b in bs:
        k = n // b
        for est, w, exl in (("r1", "r1", "r1"), ("r2_or", "or", "r2or")):
            e = ex[(ex.model == model) & (ex.b == b) & (ex.estimator == exl)].sort_values("t")
            if len(e) != len(C.T):
                continue
            beta = e.exact.to_numpy()
            V = (sig[w][C.mK] @ C.wK) / k
            ib_E = (beta[C.mK] ** 2) @ C.wK
            ib_C = ((beta + eta[w] / k)[C.mK] ** 2) @ C.wK
            rows.append(dict(model=model, n=n, b=b, k=k, estimator=est, ivar_asym=V, ibias2_E=ib_E,
                             ibias2_C=ib_C, crit_E=V + ib_E, crit_C=V + ib_C,
                             crit_C_mcvar=ivar_mc.get((b, est), np.nan) + ib_C))
    return pd.DataFrame(rows)


def smooth_min(bs, imse, bstar):
    sel = (bs >= bstar / 2 - 1e-9) & (bs <= 2 * bstar + 1e-9)
    x, y = np.log(bs[sel]), np.log(imse[..., sel])
    X = np.stack([np.ones_like(x), x, x * x], 1)
    coef = np.linalg.lstsq(X, np.atleast_2d(y).T, rcond=None)[0]          # (3, nrep)
    c0, c1, c2 = coef
    xm = np.where(c2 > 0, -c1 / (2 * np.where(c2 > 0, c2, 1)), np.nan)
    xm = np.clip(xm, x.min(), x.max())
    return np.exp(c0 + c1 * xm + c2 * xm * xm), np.exp(xm), int(sel.sum())


def main():
    t0 = time.time()
    ex, sig, eta = theory_inputs()
    s07 = pd.read_csv(C.BIAS / "outputs" / "s07_ratios.csv")
    curves, opts, thc = [], [], []
    Wb = C.boot_counts(1000)
    fig, axes = plt.subplots(2, len(CELLS), figsize=(16, 6.2), sharex="col")
    for j, (mo, n) in enumerate(CELLS):
        df = C.load_many([(md, "main", mo, n) for md in ("full", "smallb", "fineb")
                          if C.load(md, "main", mo, n) is not None])
        bs = np.sort(df.b.unique())
        reps, E = C.cube(df, ESTS, list(bs))
        assert len(reps) == 1000 and np.isfinite(E).all()
        ise = (E[..., C.mK] ** 2) @ C.wK                            # (R, nb, ne)
        imse = ise.mean(0)
        ivar_mc = {}
        for ib, b in enumerate(bs):
            for ie, est in enumerate(ESTS):
                ibias, ibse, iv, nf = C.ibias2_with_se(E[:, ib, ie])
                rho, rse = C.ratio_mcse(ise[:, ib, ie], ise[:, ib, 0])
                ivar_mc[(b, est)] = iv
                curves.append(dict(model=mo, n=n, b=b, k=n // b, estimator=est, imse_K=imse[ib, ie],
                                   mcse_K=ise[:, ib, ie].std(ddof=1) / np.sqrt(1000),
                                   ibias2_K=ibias, mcse_ibias2=ibse, ivar_K=iv, noise_floor=nf,
                                   ratio_to_r1=rho, mcse_ratio=rse,
                                   source="fineb" if b in set(df[df.b == b].b) and
                                   b not in (2, 3, 4, 5, 10, 20, 40, 80, 160, 320) else
                                   ("smallb" if b in (2, 3, 4) else "full")))
        # theory curves on the merged grid
        tc = crit_curves(mo, n, bs, ex, sig, eta, ivar_mc)
        thc.append(tc)
        # optimum, ratio of minima
        ib1 = int(np.argmin(imse[:, 0]))
        bimse = Wb @ ise.reshape(1000, -1)                           # (nboot, nb*ne)
        bimse = bimse.reshape(len(Wb), len(bs), len(ESTS))
        bmin = bimse.min(1)                                          # (nboot, ne)
        sm_pt, sm_b = {}, {}
        for ie, est in enumerate(ESTS):
            bstar = bs[int(np.argmin(imse[:, ie]))]
            v, bb, npts = smooth_min(bs.astype(float), imse[:, ie], bstar)
            vb, _, _ = smooth_min(bs.astype(float), bimse[:, :, ie], bstar)
            sm_pt[est], sm_b[est] = (v[0], bb[0], npts), vb
        for ie, est in enumerate(ESTS):
            ibe = int(np.argmin(imse[:, ie]))
            rho, se = C.ratio_mcse(ise[:, ibe, ie], ise[:, ib1, 0])
            br = bmin[:, ie] / bmin[:, 0]
            amin = np.bincount(bimse[:, :, ie].argmin(1), minlength=len(bs))
            sm_rat = sm_pt[est][0] / sm_pt["r1"][0]
            sm_br = sm_b[est] / sm_b["r1"]
            rec = dict(model=mo, n=n, estimator=est, b_opt=bs[ibe], b_opt_r1=bs[ib1],
                       imse_opt=imse[ibe, ie], mcse_opt=ise[:, ibe, ie].std(ddof=1) / np.sqrt(1000),
                       ratio=rho, mcse_delta=se, mcse_boot=br.std(ddof=1),
                       boot_selection_bias=br.mean() - rho,
                       boot_b_mode=bs[int(np.argmax(amin))],
                       boot_b_range90=f"{np.quantile(bs[bimse[:, :, ie].argmin(1)], .05):g}-"
                                      f"{np.quantile(bs[bimse[:, :, ie].argmin(1)], .95):g}",
                       ratio_smooth=sm_rat, mcse_smooth_boot=np.nanstd(sm_br, ddof=1),
                       b_smooth=sm_pt[est][1], smooth_npts=sm_pt[est][2])
            if est == "r2_or":
                for cr in ("E", "C", "C_mcvar"):
                    c1 = tc[tc.estimator == "r1"].set_index("b")[f"crit_{cr}"]
                    c2 = tc[tc.estimator == "r2_or"].set_index("b")[f"crit_{cr}"]
                    rec[f"pred_{cr}_mergedgrid"] = c2.min() / c1.min()
                    rec[f"pred_{cr}_b"] = f"{c1.idxmin()}/{c2.idxmin()}"
                for cr in ("E", "C"):
                    s = s07[(s07.model == mo) & (s07.n == n) & (s07.bset == "all b") & (s07.criterion == cr)]
                    if len(s):
                        rec[f"pred_{cr}_allint"] = float(s.ratio.iloc[0])
                        rec[f"pred_{cr}_allint_b"] = f"{int(s.b1.iloc[0])}/{int(s.b2.iloc[0])}"
                    s = s07[(s07.model == mo) & (s07.n == n) & (s07.bset == "Sec8 grid") & (s07.criterion == cr)]
                    if len(s):
                        rec[f"pred_{cr}_sec8grid"] = float(s.ratio.iloc[0])
            opts.append(rec)
        # figure
        ax, axr = axes[0, j], axes[1, j]
        cols = {"r1": "#000000", "r2_or": "#0072B2", "r2_eff_bceq": "#D55E00"}
        cur = pd.DataFrame([c for c in curves if c["model"] == mo and c["n"] == n])
        for est, col in cols.items():
            c = cur[cur.estimator == est]
            ax.errorbar(c.b, 1e4 * c.imse_K, yerr=2e4 * c.mcse_K, fmt="o", ms=3, color=col, lw=0.8,
                        label=est + " (MC)")
            if est in ("r1", "r2_or"):
                t_ = tc[tc.estimator == est]
                ax.plot(t_.b, 1e4 * t_.crit_C, "-", color=col, lw=1, label=est + " crit. C")
                ax.plot(t_.b, 1e4 * t_.crit_C_mcvar, ":", color=col, lw=1, label=est + " C, MC var")
            if est != "r1":
                axr.errorbar(c.b, c.ratio_to_r1, yerr=2 * c.mcse_ratio, fmt="o", ms=3, color=col, lw=0.8)
        axr.axhline(1, color="grey", lw=0.5)
        axr.axhline(PRED.get((mo, n), np.nan), color="#0072B2", ls="--", lw=0.8)
        o = pd.DataFrame([r for r in opts if r["model"] == mo and r["n"] == n]).set_index("estimator")
        ax.set_title(f"{mo}, n={n}: min-ratio r2_or/r1 = {o.loc['r2_or', 'ratio']:.3f}"
                     f" ({o.loc['r2_or', 'mcse_boot']:.3f})", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log"); axr.set_xscale("log")
        axr.set_xlabel("b")
        if j == 0:
            ax.set_ylabel(r"$10^4\,$IMSE$_K$"); axr.set_ylabel("paired ratio to r1 at same b")
            ax.legend(fontsize=6)
        axr.set_ylim(0.6, 1.8)
    fig.tight_layout()
    fig.savefig(C.FIGS / "h1_fineb_imse.pdf")
    curves = pd.DataFrame(curves)
    curves["source"] = np.where(curves.b.isin([2, 3, 4]), "smallb",
                                np.where(curves.b.isin([5, 10, 20, 40, 80, 160, 320]), "full", "fineb"))
    C.save(curves, "h1_imse_curves.csv")
    opts = pd.DataFrame(opts)
    C.save(opts, "h1_optimum.csv")
    C.save(pd.concat(thc), "h1_theory_curves.csv")
    pd.set_option("display.width", 250)
    cols = ["model", "n", "estimator", "b_opt_r1", "b_opt", "ratio", "mcse_delta", "mcse_boot",
            "boot_selection_bias", "boot_b_range90", "ratio_smooth", "mcse_smooth_boot", "b_smooth"]
    log(opts[cols].round(4).to_string(index=False))
    o = opts[opts.estimator == "r2_or"]
    log("\nH1 verdict table (r2_or / r1 at own optimum):")
    for _, r in o.iterrows():
        p = PRED.get((r.model, r.n), np.nan)
        se = max(r.mcse_boot, r.mcse_delta)
        log(f"{r.model} n={r.n}: MC {r.ratio:.3f} (boot SE {r.mcse_boot:.3f}, delta {r.mcse_delta:.3f}; "
            f"selection bias {r.boot_selection_bias:+.3f}), smooth {r.ratio_smooth:.3f} ({r.mcse_smooth_boot:.3f}); "
            f"b_opt r1/r2 = {r.b_opt_r1}/{r.b_opt}; pred C all-int {r.get('pred_C_allint', np.nan):.3f} "
            f"(b {r.get('pred_C_allint_b', '')}), C merged grid {r.pred_C_mergedgrid:.3f} (b {r.pred_C_b}), "
            f"C+MCvar merged grid {r.pred_C_mcvar_mergedgrid:.3f} (b {r.pred_C_mcvar_b}), "
            f"E all-int {r.get('pred_E_allint', np.nan):.3f}; "
            f"z(stated) = {(r.ratio - p) / se:+.2f}" if np.isfinite(p) else
            f"{r.model} n={r.n}: MC {r.ratio:.3f} (boot SE {r.mcse_boot:.3f}); no stated prediction; "
            f"C merged grid {r.pred_C_mergedgrid:.3f} (b {r.pred_C_b}); C all-int {r.get('pred_C_allint', np.nan):.3f}")
    log(f"\nruntime {time.time() - t0:.0f} s")
    (C.OUT / "h1_output.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
