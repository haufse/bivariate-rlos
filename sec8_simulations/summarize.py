"""Summaries of the Section 8 shards: IMSE with MCSE, decomposition, paired ratios,
MC-optimal b, pointwise bias/variance, diagnostics.

Usage (from sec8_simulations/):  python summarize.py --mode fast [--outdir RAW]
Writes <outdir>/<mode>/summary/*.csv and copies them to ../data/sec8/<tag>/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C   # noqa: E402
import models as Mo  # noqa: E402

T = C.T_EVAL
TCOLS = [f"A_{x:.2f}" for x in T]
KEYS = ["exp", "model", "n", "b", "k"]


def simpson_weights(lo, hi):
    m = (T >= lo - 1e-9) & (T <= hi + 1e-9)
    npts = int(m.sum())
    assert npts % 2 == 1, "Simpson needs an even number of intervals"
    w = np.ones(npts)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return m, w * 0.05 / 3


def load(mode, outdir, kind, merge=()):
    """All shards of <outdir>/<mode> plus those of the --merge-mode folders (e.g. smallb,
    whose rows carry the same experiment labels as mode full).  Refuses duplicate rows."""
    files = []
    for md in (mode, *merge):
        f = sorted((Path(outdir) / md).rglob(f"*.{kind}.csv.gz"))
        if not f:
            raise SystemExit(f"no {kind} shards under {Path(outdir) / md}")
        files += f
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = ["exp", "model", "n", "b", "rep"] + (["estimator"] if kind == "est" else [])
    dup = df.duplicated(key)
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate {kind} rows after merging {mode}+{list(merge)}")
    return df


def ratio_mcse(x, y):
    """Paired ratio mean(x)/mean(y) and delta-method MCSE."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = x.mean() / y.mean()
    return rho, np.std(x - rho * y, ddof=1) / y.mean() / np.sqrt(len(x))


def boot_counts(R, nboot=2000, seed=20261118):
    """Replication-bootstrap weights (nboot x R, rows sum to 1); same construction and seed as
    bias_analysis/followup/scripts/common.py, so the selection SEs agree."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, R, nboot]))
    idx = rng.integers(0, R, size=(nboot, R))
    Cm = np.zeros((nboot, R))
    np.add.at(Cm, (np.repeat(np.arange(nboot), R), idx.ravel()), 1.0)
    return Cm / R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--outdir", default=str(Path(os.environ.get("BRLOS_RAW", HERE.parent / "raw_results")).resolve()))
    ap.add_argument("--tasks-file", default=None,
                    help="task log for runtime.csv (default <outdir>/<mode>/tasks.csv, else "
                         "tasks_all_shards.csv, the concatenated per-shard logs of a Slurm run)")
    ap.add_argument("--results-tag", default=None,
                    help="copy summaries to data/sec8/<tag>/ instead of data/sec8/<mode>/")
    ap.add_argument("--merge-mode", action="append", default=[],
                    help="also read the shards of this mode (repeatable), e.g. --mode full "
                         "--merge-mode smallb; summaries go to <mode>/summary_<mode>+<merge>")
    args = ap.parse_args()
    est = load(args.mode, args.outdir, "est", args.merge_mode)
    diag = load(args.mode, args.outdir, "diag", args.merge_mode)
    tag = "+".join([args.mode] + args.merge_mode)
    sdir = Path(args.outdir) / args.mode / ("summary" if not args.merge_mode else f"summary_{tag}")
    sdir.mkdir(parents=True, exist_ok=True)

    A0 = np.stack([Mo.MODELS[m]["A"](T) for m in est["model"]])
    err = est[TCOLS].to_numpy() - A0
    lo = np.maximum(T, 1 - T)
    for name, (a, b) in {"K": C.K_MAIN, "K10": C.K_ALT}.items():
        m, w = simpson_weights(a, b)
        est[f"ise_{name}"] = (err[:, m] ** 2) @ w
    est[[f"e_{c}" for c in TCOLS]] = err
    # boundary hits on K (lower bound of Theta_t or 1)
    mK, _ = simpson_weights(*C.K_MAIN)
    vals = est[TCOLS].to_numpy()
    est["bound_hits_K"] = (((np.abs(vals - lo) < 1e-7) | (vals > 1 - 1e-7)) & mK).sum(1)

    # --------------------------------------------------------------- IMSE table
    rows, pw = [], []
    for key, g in est.groupby(KEYS + ["estimator"], sort=True):
        e = g[[f"e_{c}" for c in TCOLS]].to_numpy()
        rec = dict(zip(KEYS + ["estimator"], key))
        for name, (a, b) in {"K": C.K_MAIN, "K10": C.K_ALT}.items():
            m, w = simpson_weights(a, b)
            ise = g[f"ise_{name}"].to_numpy()
            ok = np.isfinite(ise)
            eb = e[ok][:, m]
            rec[f"R_{name}"] = int(ok.sum())
            rec[f"imse_{name}"] = ise[ok].mean() if ok.any() else np.nan
            rec[f"mcse_{name}"] = ise[ok].std(ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else np.nan
            rec[f"ibias2_{name}"] = (eb.mean(0) ** 2) @ w if ok.any() else np.nan
            rec[f"ivar_{name}"] = eb.var(0, ddof=1) @ w if ok.sum() > 1 else np.nan
        rec["bound_hit_rate_K"] = g["bound_hits_K"].mean() / mK.sum()
        rows.append(rec)
        for j, t in enumerate(T):
            x = e[:, j][np.isfinite(e[:, j])]
            pw.append(dict(rec_key := dict(zip(KEYS + ["estimator"], key)), t=t, R=len(x),
                           bias=x.mean(), var=x.var(ddof=1), mse=np.mean(x ** 2),
                           mcse_bias=x.std(ddof=1) / np.sqrt(len(x)),
                           mcse_mse=np.std(x ** 2, ddof=1) / np.sqrt(len(x))))
    imse = pd.DataFrame(rows)

    # paired ratios to r1 at the same b
    rat = []
    for key, g in est.groupby(KEYS):
        base = g[g.estimator == "r1"].set_index("rep")
        for estn, h in g.groupby("estimator"):
            h = h.set_index("rep").reindex(base.index)
            for name in ("K", "K10"):
                rho, se = ratio_mcse(h[f"ise_{name}"].to_numpy(), base[f"ise_{name}"].to_numpy())
                rat.append(dict(zip(KEYS, key), estimator=estn, measure=name,
                                ratio_to_r1=rho, mcse_ratio=se))
    rat = pd.DataFrame(rat)
    imse = imse.merge(rat[rat.measure == "K"].drop(columns="measure"), on=KEYS + ["estimator"])
    imse.to_csv(sdir / "imse.csv", index=False, float_format="%.6g")
    pd.DataFrame(pw).to_csv(sdir / "pointwise.csv", index=False, float_format="%.6g")

    # MC-optimal b per estimator over the (merged) b grid, and the paired ratio
    # min_b IMSE_K(estimator) / min_b IMSE_K(r1).  Two MCSEs: the delta-method MCSE at the
    # selected b's (mcse_ratio) and the replication-bootstrap SE that includes the argmin
    # selection (mcse_ratio_sel, B = 2000, seed 20261118, identical to
    # bias_analysis/followup/scripts/common.boot_counts); the tables report
    # mcse_ratio_tab = max of the two.  at_end: optimum at an end of the estimator's b grid.
    opt = []
    wb_cache = {}
    for (ex, mo, n), g in est.groupby(["exp", "model", "n"], sort=True):
        P = g.set_index(["rep", "estimator", "b"])["ise_K"].unstack(["estimator", "b"]).sort_index()
        P = P.dropna(axis=1, how="all")
        gi = imse[(imse.exp == ex) & (imse.model == mo) & (imse.n == n)]
        best = gi.loc[gi.groupby("estimator")["imse_K"].idxmin()].set_index("estimator")
        R = len(P)
        if R not in wb_cache:
            wb_cache[R] = boot_counts(R)
        Wb = wb_cache[R]

        def bmin(estn):
            x = P[estn].to_numpy()
            return (Wb @ x).min(1) if np.isfinite(x).all() else None

        b1 = best.loc["r1", "b"]
        e1 = P[("r1", b1)].to_numpy()
        m1 = bmin("r1")
        for estn, row in best.iterrows():
            grid = sorted(P[estn].columns[np.isfinite(P[estn].to_numpy()).any(0)])
            rho, se = ratio_mcse(P[(estn, row.b)].to_numpy(), e1)
            me = bmin(estn)
            ses = float(np.std(me / m1, ddof=1)) if (me is not None and m1 is not None) else np.nan
            opt.append(dict(exp=ex, model=mo, n=n, estimator=estn, b_opt=row.b,
                            imse_K=row.imse_K, mcse_K=row.mcse_K, b_opt_r1=b1,
                            ratio_to_r1_at_own_opt=rho, mcse_ratio=se, mcse_ratio_sel=ses,
                            mcse_ratio_tab=np.nanmax([se, ses]),
                            at_end=bool(row.b in (grid[0], grid[-1])),
                            grid=" ".join(str(int(b)) for b in grid)))
    pd.DataFrame(opt).to_csv(sdir / "optimal_b.csv", index=False, float_format="%.6g")

    # ------------------------------------------- reproducibility of merged follow-up modes
    # modes bck/r3or recompute the standard estimates with the same seeds and store them at
    # t = 0.5 as chk_<label>_0.50 in their diag rows (exp main_bck / r3_or); compare with the
    # stored estimates of the mode that wrote those labels.
    chk = [c for c in diag.columns if c.startswith("chk_")]
    if chk:
        base = {"main_bck": "main", "r3_or": "r3"}
        dd = diag[diag[chk].notna().any(axis=1)].copy()
        dd["exp"] = dd["exp"].map(base).fillna(dd["exp"])
        rows = []
        for c in chk:
            lab = c[4:-5]
            e = est[est.estimator == lab][KEYS + ["rep", "A_0.50"]]
            m = dd[KEYS + ["rep", c]].dropna().merge(e, on=KEYS + ["rep"])
            if len(m):
                rows.append(dict(label=lab, n_compared=len(m),
                                 max_abs_diff=float((m[c] - m["A_0.50"]).abs().max())))
        pd.DataFrame(rows).to_csv(sdir / "consistency_chk.csv", index=False)
        print("consistency of recomputed estimates (max |diff|):",
              {r["label"]: r["max_abs_diff"] for r in rows})

    # --------------------------------------------------------------- diagnostics
    d = diag.copy()
    lcols = [c for c in d.columns if c.startswith("lmin_")]
    agg = dict(R=("rep", "count"), fail_pilot=("fail_pilot", "sum"), fail_other=("fail_other", "sum"),
               fallback_K=("fallback_K", "mean"), fallback_K10=("fallback_K10", "mean"),
               pilot_ise_K=("pilot_ise_K", "mean"), pilot_sup=("pilot_sup", "mean"),
               bernstein_shift=("bernstein_shift", "mean"),
               os_trunc_eff_K=("os_trunc_eff_K", "mean"), os_trunc_id_K=("os_trunc_id_K", "mean"),
               sig_frob_rel_or=("sig_frob_rel_or", "mean"), sec_per_rep=("seconds", "mean"))
    agg = {k: v for k, v in agg.items() if v[0] in d.columns}
    dg = d.groupby(KEYS).agg(**agg).reset_index()
    for c in lcols:   # pointwise pd rate (lambda_min>0 and cond<=COND_MAX)
        t = c.split("_")[1]
        d[f"pd_{t}"] = (d[c] > 0) & (d[f"cond_{t}"] <= C.COND_MAX)
    pdr = d.groupby(KEYS)[[f"pd_{c.split('_')[1]}" for c in lcols]].mean().reset_index()
    dg = dg.merge(pdr, on=KEYS, how="left")
    # sqrt(k) max_K |eq - os| for the feasible estimator (eq. bc-equivalence)
    w = est[est.estimator.str.endswith(("_eff_bceq", "_eff_bcos"))].copy()
    if len(w):
        w["kind"] = np.where(w.estimator.str.endswith("bceq"), "eq", "os")
        piv = w.pivot_table(index=KEYS + ["rep"], columns="kind",
                            values=[c for c, mk in zip(TCOLS, mK) if mk])
        diff = (piv.xs("eq", axis=1, level=1) - piv.xs("os", axis=1, level=1)).abs().max(axis=1)
        diff = diff * np.sqrt(diff.index.get_level_values("k").to_numpy())
        q = diff.groupby(level=KEYS).agg([("eqos_median", "median"),
                                          ("eqos_q95", lambda s: s.quantile(.95))]).reset_index()
        dg = dg.merge(q, on=KEYS, how="left")
    dg.to_csv(sdir / "diagnostics.csv", index=False, float_format="%.6g")

    # runtime per configuration from tasks.csv
    def task_file(md):
        f = Path(args.outdir) / md / "tasks.csv"
        return f if f.exists() else Path(args.outdir) / md / "tasks_all_shards.csv"

    tfs = [Path(args.tasks_file)] if args.tasks_file else \
        [task_file(md) for md in [args.mode] + args.merge_mode]
    tfs = [f for f in tfs if f.exists()]
    if tfs:
        tk = pd.concat([pd.read_csv(f) for f in tfs], ignore_index=True)
        tk["reps"] = tk.rep_end - tk.rep_start
        tk.groupby(["exp", "model", "n"]).agg(reps=("reps", "sum"), cpu_seconds=("seconds", "sum"),
                                              tasks=("seconds", "size")).reset_index().to_csv(
            sdir / "runtime.csv", index=False, float_format="%.1f")

    rdir = HERE.parent / "data" / "sec8" / (args.results_tag or tag)
    rdir.mkdir(parents=True, exist_ok=True)
    for f in sdir.glob("*.csv"):
        shutil.copy(f, rdir / f.name)
    print(f"wrote {sdir} and copied to {rdir}")


if __name__ == "__main__":
    main()
