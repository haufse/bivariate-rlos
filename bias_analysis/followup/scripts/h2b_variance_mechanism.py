"""h2b: why the bck correction raises the variance by more than the 1/B = 0.5 % expected for H2.

At t = 1/2, per cell (model, n, b) and weights w in {r1, or, id} (labels r1, r2_or, r2_id):
  e      = estimate(0.5)                        (mode full)
  bh     = bias-hat_w(0.5) = bk_w_0.50          (mode bck diag; *_bck = e - bh)
  pilot  = Bernstein pilot(0.5)                 (mode full, label 'pilot')
Decomposition  Var(e - bh) / Var(e) - 1 = [Var(bh) - 2 Cov(e, bh)] / Var(e), with
  noise part   mean(bk_w_sd_0.50^2) / B / Var(e)           (bootstrap Monte Carlo error, ~1/B)
  slope        s = OLS slope of k*bh on pilot(0.5)          (empirical d eta / d a)
  plug-in part -2 (s/k) Cov(e, pilot) / Var(e)
Reference slope (chord, theory values of the bias analysis): eta(IND) vs eta(GH2) at t = 1/2,
  (eta_IND - eta_GH2) / (1 - 2^{-1/2}): r1 -2.98, or -8.10, id -5.57.
Outputs: outputs/h2b_variance.csv, outputs/h2b_output.txt.
"""
import sys

sys.dont_write_bytecode = True
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import common as C   # noqa: E402

B = 200
CHORD = {"r1": (-1.24 + 0.369) / (1 - 2 ** -0.5), "or": (-3.58 + 1.208) / (1 - 2 ** -0.5),
         "id": (-2.60 + 0.97) / (1 - 2 ** -0.5)}
LAB = {"r1": "r1", "or": "r2_or", "id": "r2_id"}


def main():
    rows = []
    for mo in ("M1", "M2", "M3", "M4"):
        for n in (5000, 10000, 20000, 100000):
            est = C.load("full", "main", mo, n)
            dg = C.load("bck", "main", mo, n, kind="diag")
            est = est[est.estimator.isin(["pilot", "r1", "r2_or", "r2_id"])][["b", "rep", "estimator", "A_0.50"]]
            w = est.pivot_table(index=["b", "rep"], columns="estimator", values="A_0.50").reset_index()
            m = w.merge(dg, on=["b", "rep"])
            for b, g in m.groupby("b"):
                k = n // b
                for wt, lab in LAB.items():
                    e, bh, p = g[lab].to_numpy(), g[f"bk_{wt}_0.50"].to_numpy(), g["pilot"].to_numpy()
                    ve = e.var(ddof=1)
                    rel = (e - bh).var(ddof=1) / ve - 1
                    noise = np.mean(g[f"bk_{wt}_sd_0.50"].to_numpy() ** 2) / B / ve
                    s = np.polyfit(p, k * bh, 1)[0]
                    plug = -2 * (s / k) * np.cov(e, p)[0, 1] / ve
                    plug_chord = -2 * (CHORD[wt] / k) * np.cov(e, p)[0, 1] / ve
                    rows.append(dict(model=mo, n=n, b=b, k=k, weights=wt, rel_var_increase=rel,
                                     noise_part=noise, slope_k_bh_on_pilot=s, plugin_part=plug,
                                     noise_plus_plugin=noise + plug, plugin_part_chord=plug_chord,
                                     corr_e_pilot=np.corrcoef(e, p)[0, 1]))
    df = pd.DataFrame(rows)
    C.save(df, "h2b_variance.csv")
    pd.set_option("display.width", 220)
    out = ["relative variance increase of the bck estimate at t=1/2 and its decomposition "
           "(medians over cells with k >= 250):"]
    g = df[df.k >= 250].groupby("weights")[["rel_var_increase", "noise_part", "plugin_part",
                                            "noise_plus_plugin", "plugin_part_chord",
                                            "slope_k_bh_on_pilot", "corr_e_pilot"]].median()
    out.append(g.round(4).to_string())
    out.append(f"chord slopes (theory): " + ", ".join(f"{k} {v:.2f}" for k, v in CHORD.items()))
    sel = df[(df.n == 10000) & (df.weights == "or")]
    out.append("\nn = 1e4, oracle weights, by model and b:")
    out.append(sel[["model", "b", "k", "rel_var_increase", "noise_part", "plugin_part", "noise_plus_plugin",
                    "slope_k_bh_on_pilot", "corr_e_pilot"]].round(4).to_string(index=False))
    r = np.corrcoef(df.rel_var_increase, df.noise_plus_plugin)[0, 1]
    out.append(f"\ncorrelation over all {len(df)} cells of observed vs (noise + plug-in): {r:.3f}; "
               f"median |observed - (noise+plug-in)| = {np.median(np.abs(df.rel_var_increase - df.noise_plus_plugin)):.4f}")
    txt = "\n".join(out)
    print(txt)
    (C.OUT / "h2b_output.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
