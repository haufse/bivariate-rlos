"""Checks of the follow-up modes of the bias investigation (H2 bck, H3 r3or).

Run from sec8_simulations/:  python tests/check_bck.py   (about 1-2 min)
1. rank_bootstrap_cov_means returns bit-identical covariances to rank_bootstrap_covariances
   for the same seed (so the bck run reproduces the full run's weights), and the moment
   vectors whose covariance they are.
2. The bootstrap 1/k bias estimate on the GH(2) limit law (near-exact Bernstein pilot, no
   block bias), k = 500: k * bias-hat averaged over K vs eta_W of the bias investigation
   (bias analysis: -0.28 r=1, -0.95 r=2 identity, -1.19 r=2 Sigma^{-1}; limit-law MC at
   k=500, t=1/2: -0.44+-0.07, -1.00+-0.06, -1.21+-0.06).  Tolerance 3 MCSE + 0.15 (k^-2 terms).
3. One replication of run.replicate in mode bck (M1, n=5e3, rep 0) and mode r3or (M1, n=1e4,
   rep 0, b=80): the recomputed standard estimates (chk_* diag columns) equal the stored
   full-run shards ($BRLOS_RAW/full if present, else the two shards in data/sec8/raw_sample), the new labels
   are finite, and only the out_labels are written.
4. r=3 oracle cache: r=2 sub-block equals the r=2 cache.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import config as C       # noqa: E402
import estimators as E   # noqa: E402
import models as Mo      # noqa: E402
import run               # noqa: E402
from pickands import fit_pickands_pilot  # noqa: E402

RES = (Path(os.environ.get("BRLOS_RAW", HERE.parent / "raw_results")) / "full").resolve()
if not RES.exists():   # the two shards needed here are shipped as a small sample
    RES = (HERE.parent / "data" / "sec8" / "raw_sample" / "full").resolve()
RESULTS = []


def report(name, ok, msg):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}", flush=True)


T = C.T_EVAL
inK = (T >= C.K_MAIN[0] - 1e-9) & (T <= C.K_MAIN[1] + 1e-9)
tf = np.linspace(0, 1, 2001)
pilot = fit_pickands_pilot(tf, Mo.A_logistic(tf), m=200)

# 1. identical covariances ------------------------------------------------------------------
S_old = E.rank_bootstrap_covariances(pilot, 200, 50, 2, T, np.random.default_rng(7))
S_new, Mb = E.rank_bootstrap_cov_means(pilot, 200, 50, 2, T, np.random.default_rng(7))
S_rec = np.array([200 * np.cov(Mb[:, j], rowvar=False, ddof=1) for j in range(len(T))])
report("rank_bootstrap_cov_means covariances == rank_bootstrap_covariances (bitwise)",
       bool(np.array_equal(S_old, S_new)) and Mb.shape == (50, len(T), 4),
       f"max diff {np.abs(S_old - S_new).max():.1e}, means shape {Mb.shape}")
report("covariance of the returned means == returned covariances", np.abs(S_rec - S_new).max() < 1e-15,
       f"{np.abs(S_rec - S_new).max():.1e}")

# 2. eta on the GH(2) limit law ---------------------------------------------------------------
t0 = time.time()
k, B = 500, 2000
_, Mb = E.rank_bootstrap_cov_means(pilot, k, B, 2, T, np.random.default_rng(20261118))
Winv = np.linalg.inv(np.load(HERE / "cache" / "oracle_sigma_GH2_r2.npz")["Sigma"])
Ir = np.broadcast_to(np.eye(4), (len(T), 4, 4))
pa = pilot(T)
est = {"r1": E.r1_closed_form(Mb[..., 0], T), "id": E.gmm_solve(Mb, Ir, T, 2)[0],
       "opt": E.gmm_solve(Mb, Winv, T, 2)[0]}
theory = {"r1": -0.28, "id": -0.95, "opt": -1.19}
for w, v in est.items():
    per = (v - pa)[:, inK].mean(axis=1)            # K-average per bootstrap dataset
    val, se = k * per.mean(), k * per.std(ddof=1) / np.sqrt(B)
    report(f"k*bias-hat on K, {w} (k={k}, B={B})", abs(val - theory[w]) < 3 * se + 0.15,
           f"{val:.3f} +- {se:.3f} vs eta {theory[w]}")
print(f"   ({time.time() - t0:.0f}s)")

# 3. one replication per new mode --------------------------------------------------------------
def stored(exp, model, n, rep, b, label, col="A_0.50"):
    files = sorted((RES / exp / model / f"n{n}").glob("*.est.csv.gz"))
    for f in files:
        a, z = (int(x) for x in f.name.split(".")[0].split("_")[1:3])
        if a <= rep < z:
            d = pd.read_csv(f)
            x = d[(d.rep == rep) & (d.b == b) & (d.estimator == label)]
            return float(x[col].iloc[0]) if len(x) else None
    return None


t0 = time.time()
ex = C.experiments("bck")[0]
est_rows, diag_rows = run.replicate(ex, "M1", 5000, 0)
labs = sorted({r["estimator"] for r in est_rows})
report("bck: only the *_bck labels are written", labs == sorted(ex["out_labels"]), str(labs))
fin = all(np.isfinite([r[c] for c in r if c.startswith("A_")]).all() for r in est_rows)
report("bck: all *_bck values finite", fin, f"{len(est_rows)} rows")
if RES.exists():
    dmax = 0.0
    for d in diag_rows:
        for lab in ("r1", "r2_eff", "r2_eff_bceq", "r2_id"):
            s = stored("main", "M1", 5000, 0, d["b"], lab)
            dmax = max(dmax, abs(s - d[f"chk_{lab}_0.50"]))
    report("bck: recomputed r1/r2_id/r2_eff/r2_eff_bceq == full-run shards (rep 0, all b, t=0.5)",
           dmax < 1e-9, f"max |diff| {dmax:.1e}")
bk = pd.DataFrame(diag_rows)[["b", "k", "bk_r1_K", "bk_eff_K", "bk_or_K", "bk_id_K"]]
print("   bias-hat on K x k (one dataset):",
      (bk[["bk_r1_K", "bk_eff_K", "bk_or_K", "bk_id_K"]].mul(bk.k, axis=0)).round(2).values.tolist(),
      f"({time.time() - t0:.0f}s)")

if run.oracle_r3_path("M1").exists():
    z3 = np.load(run.oracle_r3_path("M1"))["Sigma"]
    z2 = np.load(HERE / "cache" / "oracle_sigma_GH2_r2.npz")["Sigma"]
    idx = [0, 1, 3, 4]
    report("r=3 oracle cache: r=2 sub-block == r=2 cache", np.abs(z3[:, idx][:, :, idx] - z2).max() < 1e-12,
           f"{np.abs(z3[:, idx][:, :, idx] - z2).max():.1e}")
    ex3 = dict(C.experiments("r3or")[0], b_grid=[80])
    e3, d3 = run.replicate(ex3, "M1", 10000, 0)
    labs = sorted({r["estimator"] for r in e3})
    report("r3or: only r3_or, r3_or_bceq written, finite", labs == ["r3_or", "r3_or_bceq"] and
           all(np.isfinite([r[c] for c in r if c.startswith("A_")]).all() for r in e3), str(labs))
    if RES.exists():
        s = stored("r3", "M1", 10000, 0, 80, "r3_eff")
        report("r3or: recomputed r3_eff == full-run r3 shard (rep 0, b=80, t=0.5)",
               abs(s - d3[0]["chk_r3_eff_0.50"]) < 1e-9, f"{abs(s - d3[0]['chk_r3_eff_0.50']):.1e}")
else:
    report("r=3 oracle cache present", False, str(run.oracle_r3_path("M1")))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
