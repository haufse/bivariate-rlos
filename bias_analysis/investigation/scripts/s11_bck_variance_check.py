"""s11: check of the derived variance inflation of the bootstrap 1/k correction (H2, *_bck).

Derived (tex/sec8-bias-supp-body.tex, Remark sec8-bias:rem:bck-variance; conditional on the
expansion and on a differentiability assumption): with bias-hat = eta_W[P]/k + bootstrap noise,
P the Bernstein pilot, and eta_W[A](1/2) = F(A(1/2), A'(1/2), A''(1/2)) + nu-part,
  Var(A_hat - bias-hat)/Var(A_hat) - 1
     = noise + plug-in + quad + o(1/k),
  noise   = sigma_W^2 / (B k Var(A_hat))                               (~ 1/B),
  plug-in = -(2/k) {G_a Cov(A_hat, P) + F_a2 Cov(A_hat, P'')} / Var(A_hat),
  quad    = Var(G_a P + F_a2 P'') / {k^2 Var(A_hat)},
with G_a = F_a + N_GH (F_a, F_a2 local partials; N_GH the Sigma/nu part, GH direction only:
an approximation, see the remark), all at t = 1/2 and A0 = GH(2), from s10.  The implied
regression slope of k*bias-hat on P(1/2) is  G_a + F_a2 Cov(P, P'')/Var(P)  (P' drops out by
exchangeability of M1-M4).
This script recomputes, on the SAME datasets as the Sec. 8 runs (seeds
SeedSequence([20261117, model_id, n, rep]) -> data stream), the estimates A_hat(1/2)
(r1, r2 oracle, r2 identity) and P(1/2), P''(1/2); no bootstrap is run.  The covariances are
Monte Carlo inputs; F_a, F_a2, N_GH come from s10 (deterministic theory).  The prediction is
compared with the observed variance ratio of the bck estimates from the follow-up analysis
(bias_analysis/followup/outputs/h2b_variance.csv, read only), cell by cell.
Pairing check: the recomputed r2_or(1/2) and pilot(1/2) equal the Sec. 8 shard values.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_CACHE_DIR=<scratch> \
     python s11_bck_variance_check.py
Writes ../outputs/s11_output.txt and ../outputs/s11_cells.csv.  Deterministic (fixed seeds).
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import biaslib as B

sys.path.insert(0, str(B.SEC8))
import config as C           # noqa: E402  (Sec. 8 code, read-only use)
import estimators as E       # noqa: E402
import models as Mo          # noqa: E402
from pickands import fit_madogram_pilot  # noqa: E402

OUT = B.HERE.parent / "outputs"
FU = B.HERE.parents[1] / "followup" / "outputs"
SCR = os.environ.get("SEC8FU_CACHE", str(B.P / ".cache" / "sec8fu"))   # parquet cache of followup/scripts/common.py
# n in {1e4, 2e4} only: the full set {5e3, 1e4, 2e4, 1e5} exceeded the 10-minute budget for
# short verification numerics (a first run was stopped after 12 min).  Runtime 347 s with 10 workers.
MODELS, NS, R, BOOT = ["M1", "M2", "M3", "M4"], [10000, 20000], 1000, 200
T = C.T_EVAL
J5 = int(np.argmin(np.abs(T - 0.5)))
LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def one(args):
    mo, n, rep = args
    spec = Mo.MODELS[mo]
    data_ss, _ = np.random.SeedSequence([C.SEED, spec["id"], n, rep]).spawn(2)
    X = spec["sampler"](n, np.random.default_rng(data_ss))
    Wor = np.linalg.inv(np.load(B.SEC8 / "cache" / "oracle_sigma_GH2_r2.npz")["Sigma"])
    Ir = np.broadcast_to(np.eye(4), (len(T), 4, 4))
    rows = []
    for b in C.B_GRID[n]:
        top = E.block_top_r(X, b, 2)
        mom = E.rank_moments(E.rank_scores(top), T)
        r1 = E.r1_closed_form(mom[:, 0], T)
        p = fit_madogram_pilot(top[:, :, 0], m=C.PILOT_M)
        e_or = E.gmm_solve(mom, Wor, T, 2)[0]
        e_id = E.gmm_solve(mom, Ir, T, 2)[0]
        rows.append(dict(model=mo, n=n, b=b, k=n // b, rep=rep, r1=r1[J5], r2_or=e_or[J5],
                         r2_id=e_id[J5], P=p(0.5), P1=p.derivative(0.5, 1), P2=p.derivative(0.5, 2)))
    return rows


def main():
    t0 = time.time()
    jobs = [(mo, n, rep) for mo in MODELS for n in NS for rep in range(R)]
    with ProcessPoolExecutor(10) as ex:
        df = pd.DataFrame([r for rr in ex.map(one, jobs, chunksize=20) for r in rr])
    log(f"recomputed {len(df)} (model, n, b, rep) rows in {time.time() - t0:.0f} s")

    # pairing check against the Sec. 8 shards (parquet cache of the follow-up analysis)
    try:
        sh = pd.read_parquet(f"{SCR}/est_full_main_M1_10000.parquet")
        sh = sh[sh.estimator.isin(["r2_or", "pilot"])].pivot_table(
            index=["b", "rep"], columns="estimator", values="A_0.50").reset_index()
        m = df[(df.model == "M1") & (df.n == 10000)].merge(sh, on=["b", "rep"], suffixes=("", "_sh"))
        log(f"pairing check M1 n=1e4 ({len(m)} rows): max|r2_or - shard| = "
            f"{np.max(np.abs(m.r2_or - m.r2_or_sh)):.2e}, max|pilot - shard| = "
            f"{np.max(np.abs(m.P - m.pilot)):.2e}")
    except Exception as exc:  # cache absent: the check is skipped, not faked
        log(f"pairing check skipped: {type(exc).__name__}: {exc}")

    s10 = pd.read_csv(OUT / "s10_eta_derivative.csv").set_index("weights")
    th = pd.read_csv(OUT / "s02_theory.csv")
    sig2 = th[(th.attractor == "GH2") & np.isclose(th.t, 0.5)].set_index("weights")["sigma2"]
    obs = pd.read_csv(FU / "h2b_variance.csv")
    lab = {"r1": "r1", "or": "r2_or", "id": "r2_id"}
    rows = []
    for (mo, n, b), g in df.groupby(["model", "n", "b"]):
        k = n // b
        P, P2 = g.P.to_numpy(), g.P2.to_numpy()
        for w, col in lab.items():
            e = g[col].to_numpy()
            Fa, Fa2, N = s10.loc[w, "F_a"], s10.loc[w, "F_a2"], s10.loc[w, "N_GH"]
            Ga = Fa + N
            ve = e.var(ddof=1)
            ce_p, ce_p2 = np.cov(e, P)[0, 1], np.cov(e, P2)[0, 1]
            cp_p2 = np.cov(P, P2)[0, 1]
            lin = Ga * P + Fa2 * P2
            rows.append(dict(model=mo, n=n, b=b, k=k, weights=w,
                             noise_pred=sig2[w] / (BOOT * k * ve),
                             plugin_pred=-(2 / k) * (Ga * ce_p + Fa2 * ce_p2) / ve,
                             plugin_pred_local_only=-(2 / k) * (Fa * ce_p + Fa2 * ce_p2) / ve,
                             plugin_pred_a_only=-(2 / k) * Ga * ce_p / ve,
                             quad_pred=lin.var(ddof=1) / (k ** 2 * ve),
                             slope_pred=Ga + Fa2 * cp_p2 / P.var(ddof=1),
                             corr_P_P2=np.corrcoef(P, P2)[0, 1],
                             sd_P2_over_sd_P=P2.std(ddof=1) / P.std(ddof=1)))
    pr = pd.DataFrame(rows)
    pr["total_pred"] = pr.noise_pred + pr.plugin_pred + pr.quad_pred
    m = pr.merge(obs[["model", "n", "b", "weights", "rel_var_increase", "noise_part", "plugin_part",
                      "slope_k_bh_on_pilot"]], on=["model", "n", "b", "weights"])
    m.to_csv(OUT / "s11_cells.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    cols = ["model", "b", "k", "rel_var_increase", "total_pred", "noise_part", "noise_pred",
            "plugin_part", "plugin_pred", "quad_pred", "slope_k_bh_on_pilot", "slope_pred"]
    log("\nn = 1e4, oracle weights (observed from h2b_variance.csv vs prediction):")
    log(m[(m.n == 10000) & (m.weights == "or")][cols].round(4).to_string(index=False))
    log("\nmedians over cells with k >= 250, by weights:")
    sel = m[m.k >= 250]
    log(sel.groupby("weights")[["rel_var_increase", "total_pred", "plugin_part", "plugin_pred",
                                 "plugin_pred_local_only", "plugin_pred_a_only", "quad_pred",
                                 "slope_k_bh_on_pilot", "slope_pred", "corr_P_P2",
                                 "sd_P2_over_sd_P"]].median().round(4).to_string())
    log("\nall cells: correlation(observed, predicted) and median |observed - predicted|, by weights:")
    for w, g in m.groupby("weights"):
        log(f"  {w}: corr {np.corrcoef(g.rel_var_increase, g.total_pred)[0, 1]:.3f}, median abs diff "
            f"{np.median(np.abs(g.rel_var_increase - g.total_pred)):.4f}, median ratio obs/pred "
            f"{np.median(g.rel_var_increase / g.total_pred):.3f}")
    log("\nby k (median over models, n, oracle weights): observed, predicted, k*observed, k*plugin_pred")
    g = m[m.weights == "or"].groupby("k")[["rel_var_increase", "total_pred", "plugin_pred"]].median()
    g["k_obs_minus_noise"] = g.index * (g.rel_var_increase - 1 / BOOT)
    g["k_plugin_pred"] = g.index * g.plugin_pred
    log(g.round(4).to_string())
    log(f"\nruntime {time.time() - t0:.0f} s")
    (OUT / "s11_output.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
