"""Section 8 Monte Carlo driver (single entry point).

Usage (from sec8_simulations/):
    python run.py --mode pilot|fast|full|smallb|fineb|bck|r3or [--main-n 10000] [--workers N]
                  [--outdir ../raw_results]

Resumable: every task (experiment x model x n x chunk of replications)
writes one shard pair <...>/reps_AAAAA_BBBBB.{diag,est}.csv.gz atomically;
existing shards are skipped on restart.  Progress is logged to
<outdir>/<mode>/run.log; per-task runtimes to <outdir>/<mode>/tasks.csv.

Cluster (Slurm) use:
    python run.py --mode M [--main-n N] --outdir DIR --init       # once: config.json + oracle caches
    python run.py --mode M ... --list-shards --nshards K          # mapping + estimates, no compute
    python run.py --mode M ... --outdir DIR --shard I --nshards K [--min-tier a --max-tier b]
                  [--stop-after-min X]
Tasks with min-tier <= tier <= max-tier are split into K shards by a deterministic
longest-cost-first greedy assignment that ignores which tasks are done, so the
mapping is stable across resubmissions.  Shard jobs refuse to start without
config.json and the oracle caches.  Per-shard logs: <outdir>/<mode>/logs/
run_<tag>.log and tasks_<tag>.csv.  Workers default to (and are capped at) the
allocated CPUs under Slurm and 10 otherwise.  Exit status: 0 complete,
1 a task raised or the pool broke, 2 setup error, 3 stopped by --stop-after-min.

Estimator labels (r = order of the experiment, here "r2"):
  r1                rank block-maxima madogram (closed form, projected to Theta_t)
  r{r}_id           identity weights
  r{r}_inv          Sigma-hat(t)^{-1} wherever Sigma-hat(t) is numerically pd, NaN else
  r{r}_eff          feasible efficient: inv on K_MAIN if pd on all of K_MAIN,
                    identity on all of K_MAIN otherwise (simultaneous fallback of
                    Thm thm:fitted-rank-bootstrap); identity outside K_MAIN
  r{r}_or           oracle Sigma_0(t)^{-1} (sigma2_lib quadrature; r=2 only)
  suffix _bceq      equation correction, eq. (bc-moment-estimator)
  suffix _bcos      one-step correction, eq. (bc-one-step-estimator), truncated to Theta_t
  pot               rank-based empirical STDF, k_p = floor(n/b), not projected
  pilot             the Bernstein pilot tilde A_n(t) (r=1 madogram, m resolution)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import io
import json
import platform
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C                      # noqa: E402
import estimators as E                  # noqa: E402
import models as Mo                     # noqa: E402
import sigma2_lib                       # noqa: E402
from pickands import fit_madogram_pilot  # noqa: E402

T = C.T_EVAL
TCOLS = [f"A_{x:.2f}" for x in T]
_ORACLE = {}


# ----------------------------------------------------------------------------- oracle cache
ATTRACTOR = {"M1": "GH2", "M2": "GH2", "M3": "GH2", "M4": "GH2", "M5": "AL2_05_09",
             "D6a": "IND", "D6b": "IND"}


def _attractor_obj(tag):
    return {"GH2": lambda: sigma2_lib.Gumbel(2.0),
            "AL2_05_09": lambda: sigma2_lib.AsymLogistic(2.0, 0.5, 0.9),
            "IND": lambda: sigma2_lib.Gumbel(1.0)}[tag]()


def oracle_path(model_key):
    return HERE / "cache" / f"oracle_sigma_{ATTRACTOR[model_key]}_r2.npz"


def ensure_oracle(model_key, log=print):
    """Sigma_0(t) on T_EVAL by sigma2_lib quadrature, cached per attractor.
    The cache file is written atomically (temporary file + os.replace)."""
    p = oracle_path(model_key)
    if p.exists():
        return
    p.parent.mkdir(exist_ok=True)
    attractor = _attractor_obj(ATTRACTOR[model_key])
    t0 = time.time()
    S = np.array([sigma2_lib.sigma2(attractor, float(t)) for t in T])
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:          # file handle: np.savez must not append ".npz"
        np.savez(fh, T=T, Sigma=S, attractor=attractor.name)
    os.replace(tmp, p)
    log(f"oracle Sigma_0 for {model_key} ({attractor.name}) computed in {time.time()-t0:.1f}s")


def oracle_models(mode, main_n):
    """Models whose oracle cache the mode needs (r = 2 experiments only)."""
    return sorted({mk for exp in C.experiments(mode, main_n) if exp["r"] == 2
                   for mk in exp["models"]})


def oracle_r3_path(model_key):
    """r = 3 oracle cache (mode r3or only); built by make_oracle_r3.py and committed."""
    return HERE / "cache" / f"oracle_sigma_{ATTRACTOR[model_key]}_r3.npz"


def oracle_r3_models(mode, main_n):
    return sorted({mk for exp in C.experiments(mode, main_n) if exp.get("oracle_r3")
                   for mk in exp["models"]})


def load_oracle_r3(model_key):
    key = (model_key, "r3")
    if key not in _ORACLE:
        z = np.load(oracle_r3_path(model_key))
        assert np.allclose(z["T"], T)
        _ORACLE[key] = np.linalg.inv(z["Sigma"])
    return _ORACLE[key]


def load_oracle(model_key):
    if model_key not in _ORACLE:
        p = oracle_path(model_key)
        if not p.exists():
            _ORACLE[model_key] = None
        else:
            z = np.load(p)
            assert np.allclose(z["T"], T)
            _ORACLE[model_key] = np.linalg.inv(z["Sigma"])
    return _ORACLE[model_key]


# ----------------------------------------------------------------------------- one replication
def _seed_streams(model_id, n, rep):
    root = np.random.SeedSequence([C.SEED, model_id, n, rep])
    data, boot = root.spawn(2)
    return data, boot


def _boot_seq(boot, b, r, m, B):
    return np.random.SeedSequence(boot.entropy, spawn_key=boot.spawn_key + (b, r, m, B))


def _onestep(est, W, d, Rhat, b, lo):
    """eq. (bc-lambda-estimator), (bc-one-step-estimator); truncated to Theta_t."""
    Wd = np.einsum("tij,tj->ti", W, d)
    den = np.einsum("ti,ti->t", d, Wd)
    lam = np.where(den[:, None] != 0, Wd / np.where(den == 0, 1, den)[:, None], 0.0)
    raw = est - np.einsum("ti,ti->t", lam, Rhat) / b
    trunc = np.clip(raw, lo, 1.0)
    return trunc, trunc != raw


def replicate(exp, model_key, n, rep):
    spec = Mo.MODELS[model_key]
    r, m, B = exp["r"], exp["m"], exp["B"]
    b_grid = exp["b_grid"] or C.B_GRID[n]
    data_ss, boot_ss = _seed_streams(spec["id"], n, rep)
    X = spec["sampler"](n, np.random.default_rng(data_ss))
    A0 = spec["A"](T)
    lo = np.maximum(T, 1 - T)
    inK = (T >= C.K_MAIN[0] - 1e-9) & (T <= C.K_MAIN[1] + 1e-9)
    inK10 = (T >= C.K_ALT[0] - 1e-9) & (T <= C.K_ALT[1] + 1e-9)
    tf = np.linspace(0, 1, 1001)
    kf = (tf >= C.K_MAIN[0] - 1e-12) & (tf <= C.K_MAIN[1] + 1e-12)
    Winv_or = load_oracle(model_key) if r == 2 else \
        (load_oracle_r3(model_key) if exp.get("oracle_r3") else None)
    bck = bool(exp.get("bck"))
    Ir = np.broadcast_to(np.eye(r * r), (len(T), r * r, r * r))
    est_rows, diag_rows = [], []
    lab = f"r{r}"
    for b in b_grid:
        t_start = time.time()
        k = n // b
        key = dict(exp=exp["name"], model=model_key, n=n, b=b, k=k, rep=rep)
        diag = dict(key, exp=exp.get("diag_exp", exp["name"]), fail_pilot=0, fail_other=0, msg="")
        out = {}
        try:
            top = E.block_top_r(X, b, r)
            mom = E.rank_moments(E.rank_scores(top), T)
            out["r1"] = E.r1_closed_form(mom[:, 0], T)
            if exp["name"] in ("main", "indep"):
                out["pot"] = E.pot_estimator(X, k, T)
            try:
                pilot = fit_madogram_pilot(top[:, :, 0], m=m)
            except Exception as exc:  # SLSQP failure etc.
                diag["fail_pilot"] = 1
                diag["msg"] = f"pilot: {type(exc).__name__}: {exc}"[:200]
                pilot = None
            if pilot is not None:
                pa, pv = pilot(T), pilot.derivative(T)
                out["pilot"] = pa
                Rhat = E.R_vector(T, pa, pv, r)
                dt = E.dprime_vec(pa, T, r)
                targets = np.stack([mom, mom + Rhat / b])          # (2, nT, r2)
                # identity
                e_id, _ = E.gmm_solve(targets, Ir, T, r)
                out[f"{lab}_id"], out[f"{lab}_id_bceq"] = e_id
                out[f"{lab}_id_bcos"], tr_id = _onestep(e_id[0], Ir, dt, Rhat, b, lo)
                # bootstrap inverse weights
                if bck:     # same draws and covariances, plus the bootstrap moment vectors
                    Sig, Mboot = E.rank_bootstrap_cov_means(
                        pilot, k, B, r, T, np.random.default_rng(_boot_seq(boot_ss, b, r, m, B)))
                else:
                    Sig = E.rank_bootstrap_covariances(
                        pilot, k, B, r, T, np.random.default_rng(_boot_seq(boot_ss, b, r, m, B)))
                Winv, pd, lmin, cond = E.inverse_weights(Sig, C.COND_MAX)
                Wc = np.where(pd[:, None, None], np.nan_to_num(Winv), Ir)
                e_inv, _ = E.gmm_solve(targets, Wc, T, r)
                os_inv, tr_inv = _onestep(e_inv[0], Wc, dt, Rhat, b, lo)
                nanmask = np.where(pd, 1.0, np.nan)
                out[f"{lab}_inv"] = e_inv[0] * nanmask
                out[f"{lab}_inv_bceq"] = e_inv[1] * nanmask
                out[f"{lab}_inv_bcos"] = os_inv * nanmask
                fb = not bool(pd[inK].all())
                fb10 = not bool(pd[inK10].all())
                use_inv = inK & (not fb)
                out[f"{lab}_eff"] = np.where(use_inv, e_inv[0], e_id[0])
                out[f"{lab}_eff_bceq"] = np.where(use_inv, e_inv[1], e_id[1])
                out[f"{lab}_eff_bcos"] = np.where(use_inv, os_inv, out[f"{lab}_id_bcos"])
                tr_eff = np.where(use_inv, tr_inv, tr_id)
                # oracle
                if Winv_or is not None:
                    e_or, _ = E.gmm_solve(targets, Winv_or, T, r)
                    out[f"{lab}_or"], out[f"{lab}_or_bceq"] = e_or
                # H2: bootstrap 1/k bias correction (bias_analysis/, run H2):
                # bias-hat_W = mean over the B bootstrap datasets (fitted limit law, ranks
                # recomputed) of the estimate with the same weights, minus pilot(T); the *_bck
                # estimate is estimate - bias-hat (not truncated to Theta_t).
                if bck:
                    W_eff = np.where(use_inv[:, None, None], Wc, Ir)   # weights of r{r}_eff
                    boot_est = {"r1": E.r1_closed_form(Mboot[..., 0], T),
                                "eff": E.gmm_solve(Mboot, W_eff, T, r)[0],
                                "id": E.gmm_solve(Mboot, Ir, T, r)[0]}
                    if Winv_or is not None:
                        boot_est["or"] = E.gmm_solve(Mboot, Winv_or, T, r)[0]
                    bh = {w: v.mean(axis=0) - pa for w, v in boot_est.items()}
                    out["r1_bck"] = out["r1"] - bh["r1"]
                    out[f"{lab}_eff_bck"] = out[f"{lab}_eff"] - bh["eff"]
                    out[f"{lab}_eff_bceq_bck"] = out[f"{lab}_eff_bceq"] - bh["eff"]
                    out[f"{lab}_id_bck"] = out[f"{lab}_id"] - bh["id"]
                    if "or" in bh:
                        out[f"{lab}_or_bck"] = out[f"{lab}_or"] - bh["or"]
                    j05 = int(np.argmin(np.abs(T - 0.5)))
                    for w, v in bh.items():
                        diag[f"bk_{w}_K"] = float(v[inK].mean())      # mean over the t in K
                        diag[f"bk_{w}_0.50"] = float(v[j05])
                        diag[f"bk_{w}_sd_0.50"] = float(boot_est[w][:, j05].std(ddof=1))
                    diag["bck_above1_eff_K"] = int((out[f"{lab}_eff_bck"][inK] > 1).sum())
                # diagnostics
                pf = pilot(tf)
                sieve = np.interp(tf, np.linspace(0, 1, m + 1), pilot.coefficients)
                Af = spec["A"](tf)
                diag.update(
                    fallback_K=int(fb), fallback_K10=int(fb10),
                    n_pd=int(pd.sum()),
                    pilot_ise_K=float(np.trapezoid((pf - Af)[kf] ** 2, tf[kf])),
                    pilot_ise_01=float(np.trapezoid((pf - Af) ** 2, tf)),
                    pilot_sup=float(np.max(np.abs(pf - Af))),
                    bernstein_shift=float(np.max(pf - sieve)),
                    os_trunc_id_K=int(tr_id[inK].sum()), os_trunc_eff_K=int(tr_eff[inK].sum()),
                    sig_frob_rel_or=(float(np.mean(np.linalg.norm(
                        Sig - np.linalg.inv(Winv_or), axis=(1, 2))
                        / np.linalg.norm(np.linalg.inv(Winv_or), axis=(1, 2))))
                        if Winv_or is not None else np.nan),
                )
                for i, x in enumerate(T):
                    diag[f"lmin_{x:.2f}"] = float(lmin[i])
                    diag[f"cond_{x:.2f}"] = float(cond[i])
        except Exception as exc:
            diag["fail_other"] = 1
            diag["msg"] = (diag["msg"] + " | " + f"{type(exc).__name__}: {exc}")[:400]
            diag["trace"] = traceback.format_exc()[-800:]
        diag["seconds"] = time.time() - t_start
        if exp.get("out_labels"):
            # reproducibility check against the mode that wrote the standard labels
            j05 = int(np.argmin(np.abs(T - 0.5)))
            for name in ("r1", f"{lab}_id", f"{lab}_eff", f"{lab}_eff_bceq"):
                if name in out:
                    diag[f"chk_{name}_0.50"] = float(out[name][j05])
        diag_rows.append(diag)
        for name, vals in out.items():
            if exp.get("out_labels") and name not in exp["out_labels"]:
                continue
            row = dict(key, estimator=name)
            row.update({c: float(v) for c, v in zip(TCOLS, vals)})
            est_rows.append(row)
        # sanity: NaN only allowed in the *_inv labels
        for name, vals in out.items():
            if not name.endswith(("_inv", "_inv_bceq", "_inv_bcos")) and np.any(~np.isfinite(vals)):
                diag["fail_other"] = 1
                diag["msg"] += f" | non-finite {name}"
    return est_rows, diag_rows


# ----------------------------------------------------------------------------- tasks
def shard_paths(outdir, mode, exp, model_key, n, a, z):
    d = Path(outdir) / mode / exp["name"] / model_key / f"n{n}"
    stem = f"reps_{a:05d}_{z:05d}"
    return d, d / f"{stem}.est.csv.gz", d / f"{stem}.diag.csv.gz"


def _atomic_csv(df, path):
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.10g")
    os.replace(tmp, path)


def run_task(task):
    exp, model_key, n, a, z, outdir, mode = task
    t0 = time.time()
    est, diag = [], []
    for rep in range(a, z):
        e, d = replicate(exp, model_key, n, rep)
        est += e
        diag += d
    d, pest, pdiag = shard_paths(outdir, mode, exp, model_key, n, a, z)
    d.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(diag), pdiag)
    _atomic_csv(pd.DataFrame(est), pest)      # written last = completion marker
    nfail = sum(x.get("fail_pilot", 0) + x.get("fail_other", 0) for x in diag)
    return dict(exp=exp["name"], model=model_key, n=n, rep_start=a, rep_end=z,
                seconds=time.time() - t0, pid=os.getpid(), failures=nfail,
                finished=datetime.now().isoformat(timespec="seconds"))


def build_tasks(mode, main_n, outdir):
    tasks = []
    for exp in C.experiments(mode, main_n):
        for model_key in exp["models"]:
            for n in exp["ns"]:
                cs = exp["chunk"] or C.chunk_size(n, exp["r"])
                bg = exp["b_grid"] or C.B_GRID[n]
                cost_rep = sum(n // b for b in bg) * exp["B"] * exp["r"] ** 2 / 4 + 0.05 * n
                if exp.get("bck"):   # 3 GMM solves on (B, nT) bootstrap targets per b: about
                    # 1.1 laptop-s at B=200 (timed 2026-11-18) = 3.1e4 cost units; bck mode only,
                    # so the shard maps of all other modes are unchanged
                    cost_rep += len(bg) * 3.1e4 * exp["B"] / 200
                for a in range(0, exp["R"], cs):
                    z = min(exp["R"], a + cs)
                    _, pest, _ = shard_paths(outdir, mode, exp, model_key, n, a, z)
                    tasks.append(dict(task=(exp, model_key, n, a, z, str(outdir), mode),
                                      cost=cost_rep * (z - a), done=pest.exists(),
                                      tier=C.tier(exp["name"], model_key, n, main_n)))
    return tasks


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, machine=platform.machine(), node=platform.node(),
                cpus=os.cpu_count())


# ----------------------------------------------------------------------------- cluster helpers
LAPTOP_CAP = int(os.environ.get("BRLOS_MAX_WORKERS", "10"))   # local default cap (outside Slurm)
# rough cost -> seconds calibration of the cost proxy in build_tasks: fast-mode M1 n=2e4
# tasks (10 reps, cost 7.947e6) took 253-285 s per task on the loaded laptop at 10 workers
# (fast-mode run log, 2026-11-17).  Used only for the --list-shards estimates.
LAPTOP_SEC_PER_COST = 285.0 / 7.947e6
SLURM_VARS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK",
              "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT")


def under_slurm():
    return bool(os.environ.get("SLURM_JOB_ID"))


def worker_cap():
    """(cap, origin): BRLOS_MAX_WORKERS (default 10) outside Slurm; under Slurm the CPUs allocated to this job."""
    if not under_slurm():
        return LAPTOP_CAP, "local cap BRLOS_MAX_WORKERS"
    try:
        aff = len(os.sched_getaffinity(0))
    except AttributeError:          # pragma: no cover (non-Linux)
        aff = os.cpu_count() or 1
    c = os.environ.get("SLURM_CPUS_PER_TASK")
    if c:
        return max(1, min(int(c), aff)), "SLURM_CPUS_PER_TASK"
    return max(1, aff), "Slurm cpuset"


def task_key(t):
    exp, model_key, n, a, z = t["task"][:5]
    return (exp["name"], model_key, int(n), int(a), int(z))


def assign_shards(tasks, nshards):
    """Deterministic greedy longest-cost-first assignment of tasks to nshards bins.
    Depends only on the task list (mode, main_n, tier range), never on which tasks are
    done, so the mapping is stable across resubmissions and shards never overlap."""
    order = sorted(range(len(tasks)), key=lambda i: (-tasks[i]["cost"], task_key(tasks[i])))
    loads = [0.0] * nshards
    shard_of = [0] * len(tasks)
    for i in order:
        j = min(range(nshards), key=lambda s: (loads[s], s))
        shard_of[i] = j
        loads[j] += tasks[i]["cost"]
    fp = hashlib.sha256(json.dumps(sorted((list(task_key(t)), s) for t, s in
                                          zip(tasks, shard_of))).encode()).hexdigest()[:12]
    return shard_of, loads, fp


def list_shards(tasks, nshards, workers):
    shard_of, loads, fp = assign_shards(tasks, nshards)
    tot = sum(loads) or 1.0
    print(f"# {len(tasks)} tasks in {nshards} shards, map fingerprint {fp}; estimates in "
          f"laptop-speed minutes (rough cost proxy), wall = max(core/{workers}, longest)")
    print(f"{'shard':>5} {'tasks':>5} {'todo':>5} {'rel_cost':>8} {'core_min':>8} "
          f"{'longest_min':>11} {'wall_min':>8}  content")
    for s in range(nshards):
        ts = [t for t, j in zip(tasks, shard_of) if j == s]
        core = sum(t["cost"] for t in ts) * LAPTOP_SEC_PER_COST / 60
        longest = max((t["cost"] for t in ts), default=0) * LAPTOP_SEC_PER_COST / 60
        groups = {}
        for t in ts:
            k = task_key(t)
            g = f"{k[0]}/{k[1]}/n{k[2]}"
            groups[g] = groups.get(g, 0) + 1
        content = " ".join(f"{g}:{c}" for g, c in sorted(groups.items()))
        print(f"{s:>5} {len(ts):>5} {sum(not t['done'] for t in ts):>5} {loads[s]/tot:>8.3f} "
              f"{core:>8.1f} {longest:>11.1f} {max(core / workers, longest):>8.1f}  {content}")
    print(f"# total laptop-speed core-hours {tot * LAPTOP_SEC_PER_COST / 3600:.1f}")


def _atomic_text(path, text):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=["pilot", "fast", "full", "smallb", "fineb", "bck", "r3or"])
    ap.add_argument("--main-n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=None,
                    help=f"worker processes; default and maximum: {LAPTOP_CAP} outside Slurm "
                         "(env BRLOS_MAX_WORKERS), the allocated CPUs under Slurm")
    ap.add_argument("--outdir", default=None,
                    help="results root (default $BRLOS_RAW or <repo>/raw_results; required under Slurm)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report tasks/todo only; never writes config.json or oracle caches")
    ap.add_argument("--min-tier", type=int, default=1)
    ap.add_argument("--max-tier", type=int, default=99,
                    help="run only tasks with min-tier <= priority tier <= max-tier (config.tier)")
    ap.add_argument("--init", action="store_true",
                    help="write config.json and all oracle caches of the mode, then exit "
                         "(run once before any --shard job)")
    ap.add_argument("--shard", type=int, default=None, help="run only shard I of --nshards")
    ap.add_argument("--nshards", type=int, default=None)
    ap.add_argument("--list-shards", action="store_true",
                    help="print the shard mapping and cost estimates (needs --nshards); no compute")
    ap.add_argument("--stop-after-min", type=float, default=None,
                    help="stop launching tasks whose predicted end exceeds this many minutes "
                         "after start (graceful end before a wall-time limit); exit code 3")
    args = ap.parse_args()

    cap, cap_origin = worker_cap()
    workers = cap if args.workers is None else args.workers
    if workers < 1 or workers > cap:
        raise SystemExit(f"--workers must be in 1..{cap} ({cap_origin})")
    if (args.shard is None) != (args.nshards is None) and not (args.list_shards and args.nshards):
        raise SystemExit("--shard and --nshards go together")
    if args.nshards is not None and args.nshards < 1:
        raise SystemExit("--nshards must be >= 1")
    if args.shard is not None and not 0 <= args.shard < args.nshards:
        raise SystemExit(f"--shard must be in 0..{args.nshards - 1}")
    if args.list_shards and not args.nshards:
        raise SystemExit("--list-shards needs --nshards")
    if args.init and (args.shard is not None or args.dry_run or args.list_shards):
        raise SystemExit("--init runs on its own (no --shard/--dry-run/--list-shards)")
    if args.outdir is None:
        if under_slurm():
            raise SystemExit("under Slurm --outdir must be given explicitly")
        args.outdir = str(Path(os.environ.get("BRLOS_RAW", HERE.parent / "raw_results")).resolve())

    outdir = Path(args.outdir)
    mdir = outdir / args.mode

    tasks = build_tasks(args.mode, args.main_n, outdir)
    tasks = [t for t in tasks if args.min_tier <= t["tier"] <= args.max_tier]
    if args.list_shards:                         # read-only: no directories, logs or caches
        list_shards(tasks, args.nshards, workers)
        return 0

    mdir.mkdir(parents=True, exist_ok=True)
    if args.shard is not None:
        ldir = mdir / "logs"
        ldir.mkdir(exist_ok=True)
        tag = f"{args.mode}_t{args.min_tier}-{min(args.max_tier, 99)}_s{args.shard:03d}of{args.nshards}"
        logf = open(ldir / f"run_{tag}.log", "a", buffering=1)
        tcsv = ldir / f"tasks_{tag}.csv"
    else:
        logf = open(mdir / "run.log", "a", buffering=1)
        tcsv = mdir / "tasks.csv"

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    slurm = {k: os.environ[k] for k in SLURM_VARS if k in os.environ}
    log(f"host={platform.node()} pid={os.getpid()} workers={workers} (cap {cap}, {cap_origin}) "
        + " ".join(f"{k}={v}" for k, v in slurm.items()))

    cfg = dict(mode=args.mode, main_n=args.main_n, seed=C.SEED, T_eval=T.tolist(),
               K_main=C.K_MAIN, K_alt=C.K_ALT, cond_max=C.COND_MAX,
               experiments=C.experiments(args.mode, args.main_n), b_grid=C.B_GRID)
    cfg_text = json.dumps(cfg, indent=1, default=str)
    code_hash = hashlib.sha256(b"".join((HERE / f).read_bytes() for f in
                                        ("run.py", "estimators.py", "models.py", "config.py",
                                         "pickands.py", "sigma2_lib.py"))).hexdigest()[:16]
    cpath = mdir / "config.json"
    if cpath.exists():
        old = json.loads(cpath.read_text())
        if old["config"] != json.loads(cfg_text):
            log("ERROR: existing results in this folder were produced with a different "
                "configuration; refusing to mix. Use another --outdir.")
            return 2
        if old.get("code_hash") != code_hash:
            log(f"WARNING: code hash changed ({old.get('code_hash')} -> {code_hash}); "
                "resuming anyway (record this).")
    elif args.shard is not None:
        log(f"ERROR: {cpath} missing; run 'run.py --mode {args.mode} --main-n {args.main_n} "
            "--init' first (array tasks never create it).")
        return 2
    elif args.dry_run:
        log(f"note: {cpath} does not exist yet (dry run: not created)")
    else:
        _atomic_text(cpath, json.dumps(dict(config=json.loads(cfg_text), code_hash=code_hash,
                                            versions=versions(), slurm=slurm,
                                            started=datetime.now().isoformat()), indent=1))

    missing3 = [mk for mk in oracle_r3_models(args.mode, args.main_n) if not oracle_r3_path(mk).exists()]
    if missing3:
        log(f"ERROR: r=3 oracle cache missing for {missing3} ({[oracle_r3_path(m).name for m in missing3]}); "
            "it is committed with the code (make_oracle_r3.py, needs numba)")
        return 2
    missing = [mk for mk in oracle_models(args.mode, args.main_n) if not oracle_path(mk).exists()]
    if missing and (args.shard is not None or args.dry_run):
        log(f"{'ERROR' if args.shard is not None else 'note'}: oracle cache missing for "
            f"{missing}; run --init first")
        if args.shard is not None:
            return 2
    elif missing:
        for mk in missing:
            ensure_oracle(mk, log)
    if args.init:
        log(f"init done: mode={args.mode} main_n={args.main_n} config={cpath} code={code_hash} "
            f"oracle caches={sorted({oracle_path(mk).name for mk in oracle_models(args.mode, args.main_n)})}")
        return 0

    if args.shard is not None:
        shard_of, loads, fp = assign_shards(tasks, args.nshards)
        tasks = [t for t, s in zip(tasks, shard_of) if s == args.shard]
        log(f"shard {args.shard}/{args.nshards} (tiers {args.min_tier}-{args.max_tier}): "
            f"{len(tasks)} tasks, map fingerprint {fp}, relative load "
            f"{loads[args.shard] / max(sum(loads), 1e-12):.3f}")
    todo = sorted([t for t in tasks if not t["done"]], key=lambda t: (t["tier"], -t["cost"]))
    total_cost = sum(t["cost"] for t in tasks)
    done_cost = sum(t["cost"] for t in tasks if t["done"])
    for tr in sorted({t["tier"] for t in tasks}):
        log(f"tier {tr}: {sum(t['tier'] == tr for t in tasks)} tasks, relative cost "
            f"{sum(t['cost'] for t in tasks if t['tier'] == tr) / max(total_cost, 1):.3f}")
    log(f"mode={args.mode} main_n={args.main_n} tiers={args.min_tier}-{args.max_tier} "
        f"workers={workers} code={code_hash} "
        f"tasks={len(tasks)} todo={len(todo)} (resumed {len(tasks)-len(todo)})")
    if args.dry_run:
        return 0

    extra = (dict(host=platform.node(), shard=args.shard, nshards=args.nshards,
                  slurm_job_id=slurm.get("SLURM_JOB_ID", ""),
                  slurm_array_job_id=slurm.get("SLURM_ARRAY_JOB_ID", ""),
                  slurm_array_task_id=slurm.get("SLURM_ARRAY_TASK_ID", ""))
             if args.shard is not None else {})
    t0 = time.time()
    cost_now = 0.0
    n_failed = 0
    broken = False
    stopped = False
    sec_per_cost = 0.0            # max observed seconds per cost unit (for --stop-after-min)
    pending = list(todo)
    inflight = {}
    i = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def launch():
            nonlocal stopped, broken
            while pending and len(inflight) < workers and not (stopped or broken):
                t = pending[0]
                if args.stop_after_min is not None and sec_per_cost > 0:
                    pred = 1.25 * sec_per_cost * t["cost"]
                    if time.time() - t0 + pred > 60 * args.stop_after_min:
                        stopped = True
                        log(f"STOP: next task predicted {pred/60:.1f} min would end after "
                            f"--stop-after-min {args.stop_after_min}; {len(pending)} tasks left")
                        break
                try:
                    inflight[pool.submit(run_task, t["task"])] = t
                except BrokenProcessPool as exc:
                    broken = True
                    log(f"ERROR: process pool broken ({exc}); no further tasks launched")
                    break
                pending.pop(0)

        launch()
        while inflight:
            done_f, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for f in done_f:
                t = inflight.pop(f)
                i += 1
                try:
                    res = f.result()
                except Exception as exc:
                    n_failed += 1
                    if isinstance(exc, BrokenProcessPool):
                        broken = True
                    log(f"TASK FAILED {t['task'][0]['name']} {t['task'][1]} n={t['task'][2]} "
                        f"reps {t['task'][3]}-{t['task'][4]}: {type(exc).__name__}: {exc}")
                    continue
                pd.DataFrame([dict(res, **extra)]).to_csv(tcsv, mode="a", header=not tcsv.exists(),
                                                         index=False)
                sec_per_cost = max(sec_per_cost, res["seconds"] / max(t["cost"], 1e-12))
                cost_now += t["cost"]
                el = time.time() - t0
                frac = (done_cost + cost_now) / max(total_cost, 1e-12)
                eta = el / max(cost_now, 1e-9) * (total_cost - done_cost - cost_now)
                log(f"{i}/{len(todo)} {res['exp']} {res['model']} n={res['n']} reps "
                    f"{res['rep_start']}-{res['rep_end']} {res['seconds']:.1f}s fail={res['failures']} "
                    f"| done {100*frac:.1f}% elapsed {timedelta(seconds=int(el))} "
                    f"ETA {timedelta(seconds=int(eta))} "
                    f"(~{(datetime.now()+timedelta(seconds=eta)).strftime('%a %H:%M')})")
            launch()
    left = len(pending)
    log(f"finished mode={args.mode} in {timedelta(seconds=int(time.time()-t0))}: "
        f"completed={i - n_failed} failed={n_failed} not_started={left}")
    if n_failed or broken:
        return 1
    if left:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
