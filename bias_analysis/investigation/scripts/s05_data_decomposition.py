"""s05: decomposition of the Sec. 8 Monte Carlo bias and IMSE (cluster run of mode full, R=1000).

Inputs: ../outputs/s01_pointwise.csv (MC pointwise mean error and variance),
        ../outputs/s02_theory.csv (c_W, sigma_W^2), ../outputs/s04_exact_bias.csv (k=inf bias),
        the independence draws via the Sec. 8 extras file data/sec8/full_only/extra_indep.csv.
Outputs (../outputs/):
  s05_kresid.csv     k * {MC mean error - exact k=inf bias}, per model, n, b, estimator, t
  s05_cfit.csv       WLS fit of that residual on 1/k (k >= 125), per model, estimator, t,
                     with the theoretical c_W
  s05_ibias.csv      IBias^2_K: MC, k=inf exact, exact + c_W/k (predicted), per model, n, b, est.
  s05_opc_waterfall.csv  r2/r1 IMSE ratio decomposition for M3 at n = 1e4
  s05_bc_variance.csv    IVar(corrected)/IVar(uncorrected) and b*(1 - ratio)
  s05_atom.csv       independence atom at t=1/2: data vs Phi({sqrt(k) B_b + c/sqrt(k)}/sigma),
                     B_b = 0 (D6a) or the leading Clayton(1) block bias lam' calS/b (D6b)
Printed summary -> ../outputs/s05_output.txt.  Deterministic.

Run: OMP_NUM_THREADS=1 taskset -c 0-9 nice -n 19 \
     python s05_data_decomposition.py > ../outputs/s05_output.txt
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import biaslib as B

OUT = B.HERE.parent / "outputs"
EST_W = {"r1": "r1", "r2_or": "r2or", "r2_id": "r2id", "r2_eff": "r2or", "r2_inv": "r2or",
         "r2_or_bceq": "r2or_bc", "r2_id_bceq": "r2id_bc", "r2_eff_bceq": "r2or_bc"}
THW = {"r1": "r1", "r2or": "or", "r2id": "id", "r2or_bc": "or", "r2id_bc": "id"}
mK = (B.T >= 0.15 - 1e-9) & (B.T <= 0.85 + 1e-9)
wK = np.ones(mK.sum())
wK[1:-1:2], wK[2:-1:2] = 4, 2
wK *= 0.05 / 3


def integ(x):
    return float(np.asarray(x)[mK] @ wK)


def exact_with_bc(ex):
    """Add k=inf corrected estimators: target mu_b + calR[A0]/b (pilot limit taken as A0)."""
    Sig = B.oracle_sigma("GH2")
    A0 = B.A_log(B.T)
    mu0 = B.mu(A0, B.T)
    R = B.calR(B.T)
    rows = []
    for (model, b), g in ex.groupby(["model", "b"]):
        # reconstruct mu_b from the linearised bias is not possible; recompute via s04 functions
        import s04_exact_finite_b as S4
        mub = np.array([S4.mu_b(model, b, float(t)) for t in B.T])
        for kind in ("or", "id"):
            W = B.weights(kind, Sig)
            a = B.gmm_solve(mub + R / b, W, B.T, lo=0.5 * np.ones_like(B.T),
                            hi=1.3 * np.ones_like(B.T), grid=41, levels=10)
            for i, t in enumerate(B.T):
                rows.append(dict(model=model, b=b, t=t, estimator=f"r2{kind}_bc", exact=a[i] - A0[i]))
    return pd.concat([ex[["model", "b", "t", "estimator", "exact", "lead"]], pd.DataFrame(rows)],
                     ignore_index=True)


def main():
    pw = pd.read_csv(OUT / "s01_pointwise.csv")
    th = pd.read_csv(OUT / "s02_theory.csv")
    ex = exact_with_bc(pd.read_csv(OUT / "s04_exact_bias.csv"))
    pw = pw[(pw.exp == "main") & pw.model.isin(["M1", "M2", "M3", "M4"])
            & pw.estimator.isin(EST_W)].copy()
    pw["tkey"] = pw.t.round(2)
    ex["tkey"] = ex.t.round(2)
    pw["exkey"] = pw.estimator.map(EST_W)
    m = pw.merge(ex.rename(columns={"estimator": "exkey"})[["model", "b", "tkey", "exkey", "exact", "lead"]],
                 on=["model", "b", "tkey", "exkey"], how="left")
    thg = th[th.attractor == "GH2"].copy()
    thg["tkey"] = thg.t.round(2)
    m["thw"] = m.exkey.map(THW)
    m = m.merge(thg.rename(columns={"weights": "thw"})[["thw", "tkey", "c_total", "sigma2"]],
                on=["thw", "tkey"], how="left")
    m["kresid"] = m.k * (m.mean_err - m.exact)
    m["se_kresid"] = m.k * m.mcse_mean
    m.to_csv(OUT / "s05_kresid.csv", index=False, float_format="%.8g")

    # ---- WLS fit of residual = c/k (+ e), k >= 125
    fits = []
    for (model, est, t), g in m[m.k >= 125].groupby(["model", "estimator", "tkey"]):
        x = 1.0 / g.k.to_numpy()
        y = (g.mean_err - g.exact).to_numpy()
        w = 1.0 / g.mcse_mean.to_numpy() ** 2
        c = np.sum(w * x * y) / np.sum(w * x * x)
        se = 1.0 / np.sqrt(np.sum(w * x * x))
        chi2 = np.sum(w * (y - c * x) ** 2)
        fits.append(dict(model=model, estimator=est, t=t, c_fit=c, se=se, c_theory=g.c_total.iloc[0],
                         chi2=chi2, df=len(g) - 1))
    fits = pd.DataFrame(fits)
    fits.to_csv(OUT / "s05_cfit.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 220)
    print("=== WLS fit of MC mean error minus exact k=inf bias on 1/k (k>=125), selected t")
    sel = fits[fits.t.isin([0.15, 0.3, 0.5])]
    print(sel.pivot_table(index=["estimator", "t"], columns="model", values="c_fit").round(2).to_string())
    print(sel.pivot_table(index=["estimator", "t"], columns="model", values="se").round(2).to_string())
    print(sel[sel.model == "M1"][["estimator", "t", "c_theory"]].round(3).to_string(index=False))
    # pooled over t in K (equal weights): mean c_fit vs mean c_theory
    fK = fits[fits.t.between(0.15, 0.85)]
    pool = fK.groupby(["model", "estimator"]).agg(c_fit=("c_fit", "mean"), c_theory=("c_theory", "mean"),
                                                   chi2=("chi2", "sum"), df=("df", "sum")).reset_index()
    print("\n=== pooled over t in K (mean over the 15 grid points)")
    print(pool.round(3).to_string(index=False))

    # ---- IBias^2 decomposition
    rows = []
    for (model, n, b, k, est), g in m.groupby(["model", "n", "b", "k", "estimator"]):
        g = g.sort_values("tkey")
        mc = integ(g.mean_err.to_numpy() ** 2)
        ivar = integ(g["var"].to_numpy())
        exk = integ(g.exact.to_numpy() ** 2)
        pred = integ((g.exact + g.c_total / k).to_numpy() ** 2)
        lead = integ(g.lead.to_numpy() ** 2) if g.lead.notna().all() else np.nan
        avar = integ(g.sigma2.to_numpy()) / k
        rows.append(dict(model=model, n=n, b=b, k=k, estimator=est, ibias2_mc=mc,
                         ibias2_noise=ivar / 1000, ibias2_kinf=exk, ibias2_pred=pred,
                         ibias2_lead=lead, ivar_mc=ivar, ivar_asym=avar))
    ib = pd.DataFrame(rows)
    ib.to_csv(OUT / "s05_ibias.csv", index=False, float_format="%.6g")
    print("\n=== IBias^2_K x 1e4 (MC, noise level IVar/R, k=inf exact, exact + c/k) for M1 and M3")
    for model in ("M1", "M3"):
        x = ib[(ib.model == model) & ib.estimator.isin(["r1", "r2_or", "r2_or_bceq", "r2_eff", "r2_eff_bceq",
                                                        "r2_id", "r2_id_bceq"])
               & ib.n.isin([10000, 100000])].copy()
        for c in ("ibias2_mc", "ibias2_noise", "ibias2_kinf", "ibias2_pred", "ivar_mc", "ivar_asym"):
            x[c] = x[c] * 1e4
        print(model)
        print(x[["n", "b", "k", "estimator", "ibias2_mc", "ibias2_noise", "ibias2_kinf", "ibias2_pred",
                 "ivar_mc", "ivar_asym"]].round(4).to_string(index=False))

    # ---- OPC waterfall at n=1e4: r2/r1 IMSE ratio
    wrows = []
    for model in ("M3", "M2", "M4", "M1"):
        for b in (5, 10, 20, 40, 80):
            for r2 in ("r2_or", "r2_eff"):
                g1 = ib[(ib.model == model) & (ib.n == 10000) & (ib.b == b) & (ib.estimator == "r1")].iloc[0]
                g2 = ib[(ib.model == model) & (ib.n == 10000) & (ib.b == b) & (ib.estimator == r2)].iloc[0]
                step = {
                    "lead (Sec.7: asym var + leading bias)": (g2.ivar_asym + g2.ibias2_lead) / (g1.ivar_asym + g1.ibias2_lead),
                    "asym var + exact k=inf bias": (g2.ivar_asym + g2.ibias2_kinf) / (g1.ivar_asym + g1.ibias2_kinf),
                    "MC var + exact k=inf bias": (g2.ivar_mc + g2.ibias2_kinf) / (g1.ivar_mc + g1.ibias2_kinf),
                    "MC var + exact bias + c/k (theory)": (g2.ivar_mc + g2.ibias2_pred) / (g1.ivar_mc + g1.ibias2_pred),
                    "MC (var + bias^2)": (g2.ivar_mc + g2.ibias2_mc) / (g1.ivar_mc + g1.ibias2_mc),
                }
                for s_, v in step.items():
                    wrows.append(dict(model=model, b=b, r2=r2, step=s_, ratio=v,
                                      ivar_ratio_mc=g2.ivar_mc / g1.ivar_mc,
                                      ivar_ratio_asym=g2.ivar_asym / g1.ivar_asym,
                                      ibias2_ratio_mc=g2.ibias2_mc / g1.ibias2_mc,
                                      ibias2_ratio_kinf=g2.ibias2_kinf / g1.ibias2_kinf,
                                      ibias2_ratio_lead=g2.ibias2_lead / g1.ibias2_lead))
    wf = pd.DataFrame(wrows)
    wf.to_csv(OUT / "s05_opc_waterfall.csv", index=False, float_format="%.5g")
    print("\n=== r2/r1 IMSE_K ratio decomposition, n = 1e4")
    print(wf.pivot_table(index=["model", "r2", "b"], columns="step", values="ratio", sort=False).round(3).to_string())
    print(wf.drop_duplicates(["model", "b", "r2"])[["model", "b", "r2", "ivar_ratio_mc", "ivar_ratio_asym",
                                                    "ibias2_ratio_mc", "ibias2_ratio_kinf",
                                                    "ibias2_ratio_lead"]].round(3).to_string(index=False))

    # ---- variance effect of the equation correction
    vrows = []
    for (model, n, b), g in ib.groupby(["model", "n", "b"]):
        for base in ("r2_or", "r2_eff", "r2_id"):
            u = g[g.estimator == base]
            c = g[g.estimator == base + "_bceq"]
            if len(u) and len(c):
                r = c.ivar_mc.iloc[0] / u.ivar_mc.iloc[0]
                vrows.append(dict(model=model, n=n, b=b, estimator=base, ivar_ratio=r, b_times_1_minus=b * (1 - r)))
    vr = pd.DataFrame(vrows)
    vr.to_csv(OUT / "s05_bc_variance.csv", index=False, float_format="%.5g")
    print("\n=== IVar(corrected)/IVar(uncorrected) and b(1-ratio), n = 1e4")
    print(vr[vr.n == 10000].pivot_table(index=["model", "b"], columns="estimator",
                                         values=["ivar_ratio", "b_times_1_minus"]).round(3).to_string())

    # ---- independence atom
    ind = pd.read_csv(B.P / "data/sec8/full_only/extra_indep.csv")
    thi = th[(th.attractor == "IND") & np.isclose(th.t, 0.5)].set_index("weights")
    arows = []
    # Clayton(1) (D6b): leading block bias b^{-1} lam' calS_clay(1/2), calS_clay as in
    # sec7_asymptotics/scripts/s07_clayton_independence.py (theta = 1)
    import s04_exact_finite_b as S4
    sclay = np.array([S4.rlos_lib.calS_shape(1.0, 0.5, l, m, 2) * 0.25 for (l, m) in B.PAIRS])
    for (model, b, k, est), g in ind.groupby(["model", "b", "k", "estimator"]):
        if est not in ("r1", "r2_or", "r2_eff", "r2_id"):
            continue
        w = {"r1": "r1", "r2_or": "or", "r2_eff": "or", "r2_id": "id"}[est]
        c, s2 = thi.loc[w, "c_total"], thi.loc[w, "sigma2"]
        lamv = thi.loc[w, ["lam1", "lam2", "lam3", "lam4"]].to_numpy(float)
        Bb = float(lamv @ sclay) / b if model == "D6b" else 0.0
        mu_ = np.sqrt(k) * Bb + c / np.sqrt(k)
        s_ = np.sqrt(s2)
        atom = float(np.mean(g.z >= -1e-12))
        arows.append(dict(model=model, b=b, k=k, estimator=est, R=len(g), atom_mc=atom,
                          se=np.sqrt(atom * (1 - atom) / len(g)),
                          atom_pred=norm.cdf(mu_ / s_),
                          mean_z_mc=g.z.mean(),
                          mean_z_pred=mu_ * norm.cdf(-mu_ / s_) - s_ * norm.pdf(mu_ / s_),
                          mean_z_limit=-s_ * norm.pdf(0), block_bias_sqrtk=np.sqrt(k) * Bb))
    at = pd.DataFrame(arows)
    at.to_csv(OUT / "s05_atom.csv", index=False, float_format="%.5g")
    print("\n=== independence boundary at t=1/2: atom P(A_hat=1) and mean of sqrt(k)(A_hat-1)")
    print(at.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
