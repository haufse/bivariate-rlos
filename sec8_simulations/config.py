"""Run configurations for the Section 8 study (modes: pilot, fast, full).

Everything that differs between modes is set here; the estimators and the
seeding are identical in all modes.  Seeds: SeedSequence([SEED, model_id, n,
rep]) -> spawn(2) = (data, bootstrap); the bootstrap stream for block length
b, order r, pilot resolution m and B replications is the child with
spawn_key = bootstrap.spawn_key + (b, r, m, B).  Hence all experiments that
share (model, n, rep) use the same dataset (paired comparisons across b,
estimators, r, m and B).
"""
import numpy as np

SEED = 20261117
T_EVAL = np.round(np.arange(0.10, 0.9001, 0.05), 2)   # 17 points, saved pointwise
K_MAIN = (0.15, 0.85)      # inverse-weight set and primary IMSE (plan memo, Sec. 4)
K_ALT = (0.10, 0.90)       # secondary set (fallback rates also recorded for it)
COND_MAX = 1e12            # numerical threshold for "lambda_min(Sigma-hat(t)) > 0"
PILOT_M = 40               # Bernstein/sieve resolution (Remark rankboot-madogram-pilot)
B_BOOT = 200               # bootstrap replications B_n

B_GRID = {
    1000: [5, 10, 20, 40],
    5000: [5, 10, 20, 40, 80],
    10000: [5, 10, 20, 40, 80, 160],
    20000: [5, 10, 20, 40, 80, 160, 320],
    100000: [10, 20, 40, 80, 160, 320],
}


# Small-b extension (mode "smallb", 2026-11-18): the MC-optimal b of the full run sat at the
# grid end b=5 for M1, M4, M5 (and POT in M2).  b in {2,3,4} completes the grid; r=2 needs
# b>=2, r=3 needs b>=3.  Rows carry the experiment labels "main"/"r3" so that summarize.py
# --merge-mode smallb merges them with mode "full".  Seeds: same data stream per
# (model, n, rep) as "full" (paired across b, as everywhere in the design); the bootstrap
# streams have spawn keys (b, r, m, B) with b<=4, which no earlier run used (all b>=5).
SMALL_B = [2, 3, 4]


def _exp(name, models, ns, R, r=2, B=B_BOOT, m=PILOT_M, b_grid=None, chunk=None):
    return dict(name=name, models=list(models), ns=list(ns), R=int(R), r=int(r), B=int(B),
                m=int(m), b_grid=b_grid, chunk=chunk)


def experiments(mode, main_n=10000):
    """List of experiments for a mode.  ``b_grid=None`` means B_GRID[n]."""
    if mode == "pilot":
        return [_exp("main", ["M1", "M2", "M4"], [10000], R=200, B=100,
                     b_grid=[10, 20, 40, 80])]
    if mode == "fast":
        # reduced vs full: R=100 (not 1000), B=100 (not 200), M1-M4 only, n in the
        # three candidate main sizes only (10-core budget)
        return [_exp("main", ["M1", "M2", "M3", "M4"], [5000, 10000, 20000], R=100, B=100)]
    if mode == "smallb":
        # same models, n, R, B, m, K and estimators as the corresponding "full" experiments
        return [
            _exp("main", ["M1", "M2", "M3", "M4"], [1000, 5000, 10000, 20000, 100000], R=1000,
                 b_grid=SMALL_B),
            _exp("main", ["M5"], [1000, 5000, 10000, 20000], R=1000, b_grid=SMALL_B),
            _exp("r3", ["M1", "M2"], [main_n], R=1000, r=3, b_grid=[b for b in SMALL_B if b >= 3]),
        ]
    # ---- follow-up runs of the bias investigation (bias_analysis/, runs H1-H3).  Same seeds: data stream per
    # (model, n, rep) as in "full" (paired).  Est rows carry the labels "main"/"r3" so that
    # summarize.py/extras.py --merge-mode merge them with "full"; modes bck and r3or write only
    # their new estimator labels (out_labels), so no row duplicates an existing one, and label
    # their diag rows diag_exp.  Extra keys exist only in these modes, so the configs of
    # pilot/fast/full/smallb are unchanged.
    if mode == "fineb":
        # H1: finer b grid near the MC optimum (new b values only; bootstrap spawn keys
        # (b, 2, 40, 200) with these b were never used before)
        return [
            _exp("main", ["M2", "M3", "M4"], [10000], R=1000,
                 b_grid=[6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 24, 28, 32]),
            _exp("main", ["M2", "M3"], [100000], R=1000, b_grid=[14, 17, 23, 26, 34, 50, 60]),
        ]
    if mode == "bck":
        # H2: bootstrap 1/k correction, labels *_bck.  Full B_GRID of the main cells; the
        # bootstrap draws are those of "full" (same spawn keys), so the uncorrected estimates
        # are recomputed identically (recorded as chk_* columns in the diag rows).
        return [dict(_exp("main", ["M1", "M2", "M3", "M4"], [5000, 10000, 20000, 100000], R=1000),
                     bck=True, diag_exp="main_bck",
                     out_labels=["r1_bck", "r2_eff_bck", "r2_eff_bceq_bck", "r2_or_bck",
                                 "r2_id_bck"])]
    if mode == "r3or":
        # H3: r = 3 with the oracle Sigma_3^{-1} weights (cache/oracle_sigma_GH2_r3.npz)
        return [dict(_exp("r3", ["M1"], [main_n], R=1000, r=3, b_grid=[5, 10, 20, 40, 80]),
                     oracle_r3=True, diag_exp="r3_or", out_labels=["r3_or", "r3_or_bceq"])]
    if mode == "full":
        return [
            _exp("main", ["M1", "M2", "M3", "M4"], [1000, 5000, 10000, 20000, 100000], R=1000),
            _exp("main", ["M5"], [1000, 5000, 10000, 20000], R=1000),
            _exp("r3", ["M1", "M2"], [main_n], R=1000, r=3),
            _exp("indep", ["D6a", "D6b"], [10000], R=2000, b_grid=[20, 40, 80]),
            _exp("sens_m20", ["M1", "M2"], [10000], R=500, m=20, b_grid=[40]),
            _exp("sens_m80", ["M1", "M2"], [10000], R=500, m=80, b_grid=[40]),
            _exp("sens_B100", ["M1", "M2"], [10000], R=500, B=100, b_grid=[40]),
            _exp("sens_B500", ["M1", "M2"], [10000], R=500, B=500, b_grid=[40]),
        ]
    raise ValueError(mode)


def tier(exp_name, model, n, main_n):
    """Priority tier (1 = first).  Tiers allow the full study to be run in
    budget-sized stages with --max-tier; results are identical either way."""
    if exp_name == "main" and model in ("M1", "M2", "M3", "M4") and n == main_n:
        return 1
    if (exp_name == "main" and model in ("M1", "M2", "M3", "M4") and n in (5000, 10000, 20000)) \
            or exp_name in ("indep",) or exp_name.startswith("sens"):
        return 2
    if exp_name == "r3" or model == "M5" or n == 1000:
        return 3
    return 4   # n = 1e5


def chunk_size(n, r):
    """Replications per task/shard (checkpoint granularity)."""
    base = {1000: 50, 5000: 25, 10000: 20, 20000: 10, 100000: 4}[n]
    return max(1, base // (2 if r >= 3 else 1))
