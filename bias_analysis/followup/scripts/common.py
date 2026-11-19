"""Shared helpers for the Sec. 8 follow-up analysis (H1 fineb, H2 bck, H3 r3or), 2026-11-18.

Read-only access to the raw shards in $BRLOS_RAW/<mode>/<exp>/<model>/n<n>/ (not in the repository).
No simulation is run here.  Truths and K-Simpson weights are identical to
sec8_simulations/{models.py,summarize.py}
(A_logistic(theta=2) for M1-M4, A_asym_logistic(2; .5, .9) for M5; Simpson on the
T_EVAL grid restricted to K=[0.15,0.85]).

Parsed shards are cached as parquet in $SEC8FU_CACHE (default: <repo>/.cache/sec8fu).
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
FIGS = HERE.parent / "figs"
PROJ = HERE.parents[2]                       # repository root
RES = Path(os.environ.get("BRLOS_RAW", PROJ / "raw_results"))
BIAS = PROJ / "bias_analysis" / "investigation"
SEC8 = PROJ / "sec8_simulations"
CACHE = Path(os.environ.get(
    "SEC8FU_CACHE",
    str(PROJ / ".cache" / "sec8fu")))
CACHE.mkdir(parents=True, exist_ok=True)
NWORK = min(10, int(os.environ.get("SEC8FU_WORKERS", "10")))

T = np.round(np.arange(0.10, 0.9001, 0.05), 2)
TCOLS = [f"A_{x:.2f}" for x in T]
mK = (T >= 0.15 - 1e-9) & (T <= 0.85 + 1e-9)
wK = np.ones(mK.sum())
wK[1:-1:2], wK[2:-1:2] = 4, 2
wK = wK * 0.05 / 3
KEYS = ["exp", "model", "n", "b", "k"]


def A_logistic(t, th=2.0):
    t = np.asarray(t, float)
    return (t ** th + (1 - t) ** th) ** (1 / th)


def A_asym(t, th=2.0, p1=0.5, p2=0.9):
    t = np.asarray(t, float)
    return (1 - p1) * (1 - t) + (1 - p2) * t + ((p1 * (1 - t)) ** th + (p2 * t) ** th) ** (1 / th)


A_TRUE = {m: A_logistic(T) for m in ("M1", "M2", "M3", "M4")}
A_TRUE["M5"] = A_asym(T)


def _read(f):
    return pd.read_csv(f)


def load(mode, exp, model, n, kind="est"):
    """All shards of one (mode, exp, model, n) folder; cached as parquet."""
    cf = CACHE / f"{kind}_{mode}_{exp}_{model}_{n}.parquet"
    d = RES / mode / exp / model / f"n{n}"
    files = sorted(d.glob(f"*.{kind}.csv.gz"))
    if not files:
        return None
    if cf.exists() and cf.stat().st_mtime > max(f.stat().st_mtime for f in files):
        return pd.read_parquet(cf)
    with ProcessPoolExecutor(min(NWORK, len(files))) as ex:
        df = pd.concat(list(ex.map(_read, files, chunksize=4)), ignore_index=True)
    df.to_parquet(cf)
    return df


def load_many(specs, kind="est"):
    """specs: iterable of (mode, exp, model, n). Refuses duplicate rows."""
    parts = [load(*s, kind=kind) for s in specs]
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    key = ["exp", "model", "n", "b", "rep"] + (["estimator"] if kind == "est" else [])
    dup = df.duplicated(key)
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate rows in {specs}")
    return df


def errors(df):
    """Per-row error matrix (R x 17) against the model truth."""
    A0 = np.stack([A_TRUE[m] for m in df["model"]])
    return df[TCOLS].to_numpy() - A0


def ise_K(err):
    return (err[:, mK] ** 2) @ wK


def cube(df, estimators, bs):
    """Error cube E[rep, b, est, t] for one (model, n); reps sorted. Missing -> NaN."""
    reps = np.sort(df["rep"].unique())
    E = np.full((len(reps), len(bs), len(estimators), len(T)), np.nan)
    ri = {r: i for i, r in enumerate(reps)}
    bi = {b: i for i, b in enumerate(bs)}
    ei = {e: i for i, e in enumerate(estimators)}
    sub = df[df.estimator.isin(estimators) & df.b.isin(bs)]
    err = errors(sub)
    E[sub.rep.map(ri).to_numpy(), sub.b.map(bi).to_numpy(), sub.estimator.map(ei).to_numpy()] = err
    return reps, E


def ibias2_with_se(err):
    """IBias^2_K (raw, i.e. including the noise floor IVar_K/R), its delta-method MCSE,
    IVar_K, and the noise floor.  err: (R, 17)."""
    e = err[:, mK]
    R = e.shape[0]
    m = e.mean(0)
    ib = (m ** 2) @ wK
    L = (2 * (e - m) * m) @ wK
    se = L.std(ddof=1) / np.sqrt(R)
    iv = e.var(0, ddof=1) @ wK
    return ib, se, iv, iv / R


def boot_counts(R, nboot=2000, seed=20261118):
    rng = np.random.default_rng(np.random.SeedSequence([seed, R, nboot]))
    idx = rng.integers(0, R, size=(nboot, R))
    C = np.zeros((nboot, R))
    np.add.at(C, (np.repeat(np.arange(nboot), R), idx.ravel()), 1.0)
    return C / R            # rows: resampling weights summing to 1


def ratio_mcse(x, y):
    """Paired ratio mean(x)/mean(y) with delta-method MCSE (as summarize.py)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = x.mean() / y.mean()
    return rho, np.std(x - rho * y, ddof=1) / y.mean() / np.sqrt(len(x))


def save(df, name):
    p = OUT / name
    df.to_csv(p, index=False, float_format="%.6g")
    print("wrote", p)
    return p
