"""s01: pointwise bias/variance summary of the Section 8 raw shards (no simulation).

Reads   $BRLOS_RAW/full/{main,indep}/<model>/n<n>/reps_*.est.csv.gz  (raw shards, not in the repository)
Writes  ../outputs/s01_pointwise.csv  (one row per exp, model, n, b, k, estimator, t)
        columns: R, mean_err, var, mcse_mean, mcse_var
and     $S01_NPZ (default: ../outputs/s01_errors.npz; 180 MB, not in the repository) with per-replication error arrays for the main models,
        key "<model>|<n>|<b>|<estimator>" -> (R, 17) array, rows sorted by rep)

Run (from scripts/):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-9 nice -n 19 \
      python s01_pointwise_summary.py
Deterministic (no random numbers).  Runtime about 1-2 min.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P = HERE.parents[2]                      # repository root
RAW = Path(os.environ.get("BRLOS_RAW", P / "raw_results")) / "full"
OUT = HERE.parent / "outputs"
T = np.round(np.arange(0.10, 0.9001, 0.05), 2)
TC = [f"A_{x:.2f}" for x in T]


def A_log(t, th=2.0):
    return (t ** th + (1 - t) ** th) ** (1 / th)


def truth(model):
    return np.ones_like(T) if model in ("D6a", "D6b") else (
        ((1 - 0.5) * (1 - T) + (1 - 0.9) * T + ((0.5 * (1 - T)) ** 2 + (0.9 * T) ** 2) ** 0.5)
        if model == "M5" else A_log(T))


def main():
    rows, arrs = [], {}
    for exp in ("main", "indep"):
        for mdir in sorted((RAW / exp).iterdir()):
            for ndir in sorted(mdir.iterdir()):
                files = sorted(ndir.glob("reps_*.est.csv.gz"))
                df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
                A0 = truth(mdir.name)
                for (n, b, k, est), g in df.groupby(["n", "b", "k", "estimator"]):
                    g = g.sort_values("rep")
                    e = g[TC].to_numpy() - A0
                    ok = np.isfinite(e).all(1)
                    e = e[ok]
                    R = len(e)
                    m = e.mean(0)
                    v = e.var(0, ddof=1)
                    c = e - m
                    m4 = (c ** 4).mean(0)
                    mcse_var = np.sqrt(np.maximum(m4 - v ** 2, 0) / R)
                    for i, t in enumerate(T):
                        rows.append(dict(exp=exp, model=mdir.name, n=n, b=b, k=k, estimator=est,
                                         t=t, R=R, mean_err=m[i], var=v[i],
                                         mcse_mean=np.sqrt(v[i] / R), mcse_var=mcse_var[i]))
                    if exp == "main" and ok.all():
                        arrs[f"{mdir.name}|{n}|{b}|{est}"] = e
                print(exp, mdir.name, ndir.name, len(df), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "s01_pointwise.csv", index=False, float_format="%.8g")
    np.savez_compressed(os.environ.get("S01_NPZ", str(OUT / "s01_errors.npz")), **arrs)


if __name__ == "__main__":
    main()
