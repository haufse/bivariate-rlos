"""Extra summaries of the Section 8 shards that summarize.py does not produce
(needed by figures.py).  Reads the raw per-replication shards once; no simulation.

Usage (from sec8_simulations/):
    python extras.py --mode full --outdir <raw results root> [--results-tag full_only]
                     [--merge-mode smallb]   # also read <outdir>/smallb (same labels)

Writes to ../data/sec8/<tag>/:
  extra_decomp.csv     IBias^2_K and IVar_K with delta-method MCSE (exp, model, n, b, estimator)
  extra_pointvar.csv   pointwise Var and k*Var with MCSE; paired variance ratio to r1 with MCSE
  extra_ratio_or.csv   paired IMSE_K ratio of each r=2 estimator to the oracle r2_or (feasible/oracle)
  extra_indep.csv      per-replication sqrt(k)(A-hat(0.5)-1) for D6a/D6b (r1, r2_id, r2_eff, r2_or, pot)
  extra_sens.csv       m/B sensitivity at b=40: paired ratios vs r1 and vs the main run (m=40, B=200),
                       reps 0..R_sens-1 of the main run as baseline
  extra_r3.csv         r=3 vs r=2 (paired, same data and b): IMSE ratios r3_*/r1 and r3_*/r2_*
  extra_bccheck.csv    M1: MC shift mean(r2_or) - mean(r2_or_bceq) vs the deterministic
                       lambda_or(t)' R[A_0](t) / b (oracle Sigma_0, true A_0) -- check of
                       eq. (bc-moment-estimator) at the oracle weights
  extra_sigcount.csv   per (model, n): number of b with r2_eff/r1 (and bceq/r1) significantly
                       below / above 1 (|ratio-1| > 2 MCSE)
  extra_bck.csv        (mode bck merged) per (model, n, b): bootstrap 1/k correction *_bck vs the
                       uncorrected estimator: IBias^2_K, IVar_K (each with MCSE), paired IMSE_K and
                       IVar_K ratios bck/uncorrected with delta-method MCSE
  extra_bck_opt.csv    (mode bck merged) per (model, n): paired ratios at each estimator's own
                       MC-optimal b, restricted to the b grid of mode bck (= B_GRID[n]):
                       bck/uncorrected and estimator/r1 (uncorrected and bck)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C          # noqa: E402
import estimators as E      # noqa: E402
import models as Mo         # noqa: E402
from summarize import simpson_weights, ratio_mcse  # noqa: E402

T = C.T_EVAL
TCOLS = [f"A_{x:.2f}" for x in T]
MK, WK = simpson_weights(*C.K_MAIN)


ROOTS = []        # result roots (<outdir>/<mode> and the --merge-mode roots), set in main()


def load_folder(d):
    """Shards of the relative folder d (e.g. main/M1/n10000) from every root, merged."""
    files = [f for r in ROOTS for f in sorted((r / d).glob("*.est.csv.gz"))]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert not df.duplicated(["exp", "model", "n", "b", "rep", "estimator"]).any(), d
    return df


def folders(pattern):
    """Relative folders matching pattern (e.g. 'main/*/n*') in any root, sorted."""
    return sorted({str(p.relative_to(r)) for r in ROOTS for p in r.glob(pattern)})


def errors(df):
    A0 = Mo.MODELS[df["model"].iloc[0]]["A"](T)
    return df[TCOLS].to_numpy() - A0


def decomp_rows(df):
    """IBias^2 and IVar on K with delta-method MCSE (per-rep influence functions)."""
    out = []
    err = errors(df)
    for (b, k, est), idx in df.groupby(["b", "k", "estimator"]).groups.items():
        e = err[np.asarray(idx)][:, MK]
        e = e[np.isfinite(e).all(1)]
        R = len(e)
        if R < 3:
            continue
        m = e.mean(0)
        s2 = e.var(0, ddof=1)
        ib = (m ** 2) @ WK
        iv = s2 @ WK
        psi_b = ((e - m) * (2 * m)) @ WK
        psi_v = ((e - m) ** 2 - s2) @ WK
        ise = (e ** 2) @ WK
        out.append(dict(b=b, k=k, estimator=est, R=R, ibias2=ib, mcse_ibias2=psi_b.std(ddof=1) / np.sqrt(R),
                        ivar=iv, mcse_ivar=psi_v.std(ddof=1) / np.sqrt(R),
                        imse=ise.mean(), mcse_imse=ise.std(ddof=1) / np.sqrt(R),
                        ibias2_noise=(s2 @ WK) / R))
    return out


def pointvar_rows(df):
    out = []
    err = errors(df)
    for (b, k), g in df.groupby(["b", "k"]):
        piv = {est: pd.DataFrame(err[h.index.to_numpy()], index=h["rep"].to_numpy())
               for est, h in g.groupby("estimator")}
        base = piv["r1"]
        for est, x in piv.items():
            x = x.reindex(base.index)
            for j, t in enumerate(T):
                xv, yv = x[j].to_numpy(), base[j].to_numpy()
                ok = np.isfinite(xv) & np.isfinite(yv)
                xv, yv = xv[ok], yv[ok]
                R = len(xv)
                if R < 3:
                    continue
                dx, dy = (xv - xv.mean()) ** 2, (yv - yv.mean()) ** 2
                v, vy = xv.var(ddof=1), yv.var(ddof=1)
                rho = v / vy
                out.append(dict(b=b, k=k, estimator=est, t=t, R=R, var=v,
                                mcse_var=dx.std(ddof=1) / np.sqrt(R), kvar=k * v,
                                mcse_kvar=k * dx.std(ddof=1) / np.sqrt(R),
                                var_ratio_r1=rho,
                                mcse_var_ratio_r1=np.std(dx - rho * dy, ddof=1) / vy / np.sqrt(R)))
    return out


def ise_table(df):
    err = errors(df)
    ise = (err[:, MK] ** 2) @ WK
    return df[["b", "k", "rep", "estimator"]].assign(ise=ise)


def ratio_rows(ise, ref):
    out = []
    for (b, k), g in ise.groupby(["b", "k"]):
        p = g.pivot(index="rep", columns="estimator", values="ise")
        if ref not in p:
            continue
        for est in p.columns:
            rho, se = ratio_mcse(p[est].to_numpy(), p[ref].to_numpy())
            out.append(dict(b=b, k=k, estimator=est, ref=ref, ratio=rho, mcse_ratio=se))
    return out


BCK_PAIRS = [("r1", "r1_bck"), ("r2_eff", "r2_eff_bck"), ("r2_eff_bceq", "r2_eff_bceq_bck"),
             ("r2_or", "r2_or_bck"), ("r2_id", "r2_id_bck")]


def bck_rows(df):
    """Bootstrap 1/k correction (mode bck) vs uncorrected, per b; paired over replications."""
    err = errors(df)
    out = []
    for (b, k), g in df.groupby(["b", "k"]):
        if not g.estimator.str.endswith("_bck").any():
            continue
        E = {est: pd.DataFrame(err[h.index.to_numpy()][:, MK], index=h["rep"].to_numpy())
             for est, h in g.groupby("estimator")}
        for u, c in BCK_PAIRS:
            if u not in E or c not in E:
                continue
            y = E[u].sort_index()
            x = E[c].reindex(y.index).to_numpy()
            y = y.to_numpy()
            ok = np.isfinite(x).all(1) & np.isfinite(y).all(1)
            x, y = x[ok], y[ok]
            R = len(x)
            rec = dict(b=b, k=k, unc=u, bck=c, R=R)
            for tag, z in (("unc", y), ("bck", x)):
                m, s2 = z.mean(0), z.var(0, ddof=1)
                rec[f"ibias2_{tag}"] = (m ** 2) @ WK
                rec[f"mcse_ibias2_{tag}"] = (((z - m) * (2 * m)) @ WK).std(ddof=1) / np.sqrt(R)
                rec[f"ivar_{tag}"] = s2 @ WK
                rec[f"mcse_ivar_{tag}"] = (((z - m) ** 2 - s2) @ WK).std(ddof=1) / np.sqrt(R)
                rec[f"imse_{tag}"] = ((z ** 2) @ WK).mean()
                rec[f"mcse_imse_{tag}"] = ((z ** 2) @ WK).std(ddof=1) / np.sqrt(R)
                rec[f"noise_{tag}"] = rec[f"ivar_{tag}"] / R
            rho, se = ratio_mcse((x ** 2) @ WK, (y ** 2) @ WK)
            rec.update(imse_ratio=rho, mcse_imse_ratio=se)
            px = ((x - x.mean(0)) ** 2 - x.var(0, ddof=1)) @ WK
            py = ((y - y.mean(0)) ** 2 - y.var(0, ddof=1)) @ WK
            rv = rec["ivar_bck"] / rec["ivar_unc"]
            rec.update(ivar_ratio=rv,
                       mcse_ivar_ratio=np.std(px - rv * py, ddof=1) / rec["ivar_unc"] / np.sqrt(R))
            out.append(rec)
    return out


def bck_opt_rows(ise):
    """Ratios at each estimator's own MC-optimal b on the b grid of mode bck."""
    bs = sorted(ise[ise.estimator.str.endswith("_bck")].b.unique())
    s = ise[ise.b.isin(bs)]
    P = s.pivot_table(index="rep", columns=["estimator", "b"], values="ise")
    best = {}
    for est in P.columns.get_level_values(0).unique():
        m = P[est].mean()
        best[est] = m.idxmin()
    pairs = [(c, u) for u, c in BCK_PAIRS] + \
        [(e, "r1") for e in ("r2_eff", "r2_eff_bceq", "r2_or", "r2_id")] + \
        [(e, "r1_bck") for e in ("r2_eff_bck", "r2_eff_bceq_bck", "r2_or_bck", "r2_id_bck")]
    out = []
    for num, den in pairs:
        if num not in best or den not in best:
            continue
        rho, se = ratio_mcse(P[(num, best[num])].to_numpy(), P[(den, best[den])].to_numpy())
        out.append(dict(num=num, den=den, b_num=best[num], b_den=best[den], ratio=rho, mcse_ratio=se,
                        grid=" ".join(str(int(b)) for b in bs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--results-tag", default=None)
    ap.add_argument("--merge-mode", action="append", default=[])
    args = ap.parse_args()
    ROOTS[:] = [Path(args.outdir) / md for md in [args.mode] + args.merge_mode]
    rdir = HERE.parent / "data" / "sec8" / (args.results_tag or "+".join([args.mode] + args.merge_mode))
    rdir.mkdir(parents=True, exist_ok=True)

    dec, pv, ror, sig, bck, bopt = [], [], [], [], [], []
    base_main = {}
    for d in map(Path, folders("main/*/n*")):
        model, n = d.parent.name, int(d.name[1:])
        df = load_folder(d)
        key = dict(exp="main", model=model, n=n)
        dec += [dict(key, **r) for r in decomp_rows(df)]
        pv += [dict(key, **r) for r in pointvar_rows(df)]
        ise = ise_table(df)
        rr = ratio_rows(ise, "r2_or")
        ror += [dict(key, **r) for r in rr]
        r1r = pd.DataFrame(ratio_rows(ise, "r1"))
        for est in ("r2_eff", "r2_eff_bceq", "r2_id", "r2_or", "pot"):
            x = r1r[r1r.estimator == est]
            sig.append(dict(key, estimator=est, n_b=len(x),
                            n_better=int(((x.ratio + 2 * x.mcse_ratio) < 1).sum()),
                            n_worse=int(((x.ratio - 2 * x.mcse_ratio) > 1).sum()),
                            b_better=" ".join(str(int(b)) for b in x.b[(x.ratio + 2 * x.mcse_ratio) < 1])))
        if ise.estimator.str.endswith("_bck").any():
            bck += [dict(key, **r) for r in bck_rows(df)]
            bopt += [dict(key, **r) for r in bck_opt_rows(ise)]
        if n == 10000 and model in ("M1", "M2"):
            base_main[model] = ise
        print(f"main {model} n={n}: {len(df)} rows", flush=True)
    pd.DataFrame(dec).to_csv(rdir / "extra_decomp.csv", index=False, float_format="%.6g")
    pd.DataFrame(pv).to_csv(rdir / "extra_pointvar.csv", index=False, float_format="%.6g")
    pd.DataFrame(ror).to_csv(rdir / "extra_ratio_or.csv", index=False, float_format="%.6g")
    pd.DataFrame(sig).to_csv(rdir / "extra_sigcount.csv", index=False)
    if bck:
        pd.DataFrame(bck).to_csv(rdir / "extra_bck.csv", index=False, float_format="%.6g")
        pd.DataFrame(bopt).to_csv(rdir / "extra_bck_opt.csv", index=False, float_format="%.6g")

    # ------------------------------------------------------------- independence
    rows, dec_i = [], []
    j05 = list(T).index(0.5)
    for d in map(Path, folders("indep/*/n*")):
        df = load_folder(d)
        df = df[df.estimator.isin(["r1", "r2_id", "r2_eff", "r2_or", "pot"])].reset_index(drop=True)
        x = df[["model", "n", "b", "k", "rep", "estimator"]].copy()
        x["z"] = np.sqrt(df["k"]) * (df[TCOLS[j05]] - 1.0)
        rows.append(x)
        dec_i += [dict(exp="indep", model=d.parent.name, n=int(d.name[1:]), **r) for r in decomp_rows(df)]
    pd.concat(rows).to_csv(rdir / "extra_indep.csv", index=False, float_format="%.8g")
    pd.DataFrame(dec_i).to_csv(rdir / "extra_decomp_indep.csv", index=False, float_format="%.6g")

    # ------------------------------------------------------------- sensitivity (b = 40, n = 1e4)
    sens = []
    for d in map(Path, folders("sens_*/*/n*")):
        exp, model = d.parent.parent.name, d.parent.name
        s = ise_table(load_folder(d))
        Rs = s.rep.max() + 1
        base = base_main[model]
        base = base[(base.b == 40) & (base.rep < Rs)]
        pb = base.pivot(index="rep", columns="estimator", values="ise")
        ps = s[s.b == 40].pivot(index="rep", columns="estimator", values="ise").reindex(pb.index)
        for est in ("r2_eff", "r2_eff_bceq", "r2_id"):
            for refname, ref in (("r1", pb["r1"]), ("main_same_est", pb[est]), ("r2_or", pb["r2_or"])):
                rho, se = ratio_mcse(ps[est].to_numpy(), ref.to_numpy())
                sens.append(dict(exp=exp, model=model, b=40, R=len(pb), estimator=est, ref=refname,
                                 ratio=rho, mcse_ratio=se, imse=ps[est].mean(),
                                 mcse_imse=ps[est].std(ddof=1) / np.sqrt(len(ps))))
        for est in ("r2_eff", "r2_eff_bceq", "r2_id", "r1", "r2_or"):   # baseline rows (m=40, B=200)
            if exp == "sens_B100":
                for refname, ref in (("r1", pb["r1"]), ("r2_or", pb["r2_or"])):
                    rho, se = ratio_mcse(pb[est].to_numpy(), ref.to_numpy())
                    sens.append(dict(exp="main_m40_B200", model=model, b=40, R=len(pb), estimator=est,
                                     ref=refname, ratio=rho, mcse_ratio=se, imse=pb[est].mean(),
                                     mcse_imse=pb[est].std(ddof=1) / np.sqrt(len(pb))))
    pd.DataFrame(sens).to_csv(rdir / "extra_sens.csv", index=False, float_format="%.6g")

    # ------------------------------------------------------------- r = 3 vs r = 2
    r3 = []
    for d in map(Path, folders("r3/*/n*")):
        model = d.parent.name
        s3 = ise_table(load_folder(d))
        s2 = base_main[model]
        for b in sorted(s3.b.unique()):
            p3 = s3[s3.b == b].pivot(index="rep", columns="estimator", values="ise")
            p2 = s2[s2.b == b].pivot(index="rep", columns="estimator", values="ise").reindex(p3.index)
            for est in [e for e in ("r3_eff", "r3_eff_bceq", "r3_id", "r3_or", "r3_or_bceq")
                        if e in p3.columns]:
                refs = [("r1", p3["r1"]), ("r2_eff", p2["r2_eff"]), ("r2_id", p2["r2_id"]),
                        ("r2_or", p2["r2_or"]), ("r2_or_bceq", p2["r2_or_bceq"])]
                if "r3_or" in p3.columns:   # mode r3or merged (oracle Sigma_3^{-1}, M1 only)
                    refs.append(("r3_or", p3["r3_or"]))
                for refname, ref in refs:
                    rho, se = ratio_mcse(p3[est].to_numpy(), ref.to_numpy())
                    r3.append(dict(model=model, n=10000, b=b, k=10000 // b, estimator=est, ref=refname,
                                   ratio=rho, mcse_ratio=se))
    pd.DataFrame(r3).to_csv(rdir / "extra_r3.csv", index=False, float_format="%.6g")

    # ------------------------------------------------------------- bc check at the oracle (M1)
    z = np.load(HERE / "cache" / "oracle_sigma_GH2_r2.npz")
    Sig = z["Sigma"]
    a0 = Mo.A_logistic(T)
    h = 1e-6
    v0 = (Mo.A_logistic(T + h) - Mo.A_logistic(T - h)) / (2 * h)
    d = E.dprime_vec(a0, T, 2)
    Rv = E.R_vector(T, a0, v0, 2)
    W = np.linalg.inv(Sig)
    Wd = np.einsum("tij,tj->ti", W, d)
    lam = Wd / np.einsum("ti,ti->t", d, Wd)[:, None]
    lamR = np.einsum("ti,ti->t", lam, Rv)
    lamR_id = np.einsum("ti,ti->t", d / np.einsum("ti,ti->t", d, d)[:, None], Rv)
    pw = pd.read_csv(rdir / "pointwise.csv")
    bc = []
    for n in (10000, 100000):
        for b in sorted(pw[(pw.exp == "main") & (pw.model == "M1") & (pw.n == n)].b.unique()):
            g = pw[(pw.exp == "main") & (pw.model == "M1") & (pw.n == n) & (pw.b == b)]
            for wlab, lr in (("or", lamR), ("id", lamR_id)):
                u = g[g.estimator == f"r2_{wlab}"].set_index("t").bias
                c = g[g.estimator == f"r2_{wlab}_bceq"].set_index("t").bias
                for j, t in enumerate(T):
                    bc.append(dict(model="M1", n=n, b=b, weights=wlab, t=t, mc_shift=u[t] - c[t],
                                   theory_shift=lr[j] / b))
    pd.DataFrame(bc).to_csv(rdir / "extra_bccheck.csv", index=False, float_format="%.6g")
    print(f"wrote extras to {rdir}")


if __name__ == "__main__":
    main()
