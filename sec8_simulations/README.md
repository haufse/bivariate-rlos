# Section 8: Monte Carlo study

Run every command from this folder. No file outside the repository is needed, except the raw
per-replication shards (`$BRLOS_RAW`, default `../raw_results`) for `summarize.py` and `extras.py`.

## Files

| file | content |
|---|---|
| `run.py` | single driver: `python run.py --mode pilot\|fast\|full\|smallb\|fineb\|bck\|r3or [--main-n 10000] [--workers W] [--outdir RAW]` |
| `config.py` | modes, grids (n, b), R, B, m, K, seed |
| `models.py` | samplers M1-M5, D6a/b and the true Pickands functions |
| `estimators.py` | ranks, rank moments, moment maps, GMM solver, rank bootstrap covariance, bias corrections, POT |
| `pickands.py` | L2 sieve fit, Bernstein pilot and exact sampling from the fitted limiting law |
| `sigma2_lib.py` | quadrature of the oracle covariance Sigma_0(t) (r = 2) |
| `make_oracle_r3.py` | builds `cache/oracle_sigma_GH2_r3.npz` with `sec7_asymptotics/scripts/rlos_lib.py` (needs numba; the cache is committed) |
| `cache/` | oracle covariances on the evaluation grid (GH(2) attractor, r = 2 and r = 3); other attractors are written by `run.py --init` |
| `summarize.py` | from the raw shards: IMSE with MCSE, IBias^2/IVar, paired ratios, MC-optimal b, pointwise summaries, diagnostics, runtime |
| `extras.py` | further summaries from the raw shards (MCSE of IBias^2/IVar, pointwise variance, feasible/oracle ratios, independence draws, m/B sensitivity, r = 3, bias-correction checks, bootstrap 1/k correction) |
| `figures.py` | all Section 8 figures (`../figures/`) and tables (`../tables/`) from the summary CSVs only |
| `headline.py` | numbers quoted in the text of Section 8 -> `../data/sec8/main_b5/headline_numbers.csv` (column `new_b5`) |
| `tests/check_*.py` | checks against the manuscript formulas (`check_units.py`, 21 checks, ~10 s), of the bootstrap 1/k correction and an exact reproduction of stored estimates (`check_bck.py`, 11 checks, ~20 s), bootstrap vs quadrature covariance and sampler attractors (`check_oracle.py`, ~2 min); stored outputs `tests/check_*_output.txt` |

## Design

Models (all with Pickands function of the logistic/Gumbel-Hougaard(2) attractor except M5):
M1 Gumbel-Hougaard(2), M2 Joe(2), M3 outer-power Clayton with BVZ (theta, beta) = (1, 2),
M4 BVZ Archimax psi_2, M5 asymmetric logistic (2; 0.5, 0.9); boundary cases D6a independence and
D6b Clayton(1). Only ranks are used. Evaluation grid t = 0.10, 0.15, ..., 0.90; IMSE by
composite Simpson on K = [0.15, 0.85] (primary) and [0.10, 0.90].

| mode | content | R | B | tasks | cluster CPU h |
|---|---|---|---|---|---|
| `full` | M1-M4 at n in {1e3, 5e3, 1e4, 2e4, 1e5}, M5 at n <= 2e4, b in `B_GRID[n]`; r = 3 for M1, M2 at n = 1e4; D6a/D6b (R = 2000); m and B sensitivity (R = 500) | 1000 | 200 | 2650 | 32.4 |
| `fineb` | M2-M4 at n = 1e4, b in {6,...,9, 11,...,16, 18, 24, 28, 32}; M2, M3 at n = 1e5, b in {14, 17, 23, 26, 34, 50, 60} | 1000 | 200 | 650 | 17.7 |
| `bck` | bootstrap 1/k correction (labels `*_bck`), M1-M4, n in {5e3, 1e4, 2e4, 1e5}, `B_GRID[n]` | 1000 | 200 | 1760 | 42.3 |
| `r3or` | r = 3 with oracle Sigma_3^{-1} weights, M1, n = 1e4, b in {5, 10, 20, 40, 80} | 1000 | 200 | 100 | 1.1 |
| `smallb` | b in {2, 3, 4} (small-b remark only), M1-M4 (and M5 at n <= 2e4), r = 3 for M1, M2 | 1000 | 200 | 2250 | 134.3 |
| `pilot`, `fast` | reduced runs used during development (not used in the paper) | 200 / 100 | 100 | | |

Pilot resolution m = 40 throughout (sensitivity runs: 20, 80). The paper's main design is
`full + fineb + bck + r3or` (b >= 5; summaries `../data/sec8/main_b5`); `smallb` enters only the
small-b remark (`../data/sec8/all_b2`). The data of a replication (model, n, rep) are the same
in every mode (paired across b and estimators); see the seed policy in the top-level README.

Estimator labels in the shards: `r1` rank block-maxima madogram; `r2_id` identity weights;
`r2_inv` Sigma-hat(t)^{-1} where numerically positive definite; `r2_eff` feasible efficient
estimator with the simultaneous identity fallback on K; `r2_or` oracle Sigma_0(t)^{-1};
suffix `_bceq` equation bias correction, `_bcos` one-step correction truncated to Theta_t,
`_bck` bootstrap 1/k correction; `pot` empirical STDF at k_p = floor(n/b); `pilot` the
Bernstein pilot. All curves are unprojected.

## Running

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BRLOS_RAW=/path/to/raw_results          # results root (default ../raw_results)
python run.py --mode full --main-n 10000 --outdir $BRLOS_RAW --init        # config.json + oracle caches
python run.py --mode full --main-n 10000 --outdir $BRLOS_RAW --workers 10   # local run (resumable)
```

Outside Slurm the number of workers is capped at `BRLOS_MAX_WORKERS` (default 10). The run
resumes: finished shards are skipped (the `.est` file is the completion marker), and the driver
refuses to resume into a folder made with a different configuration.

On a Slurm cluster (templates in `../hpc/slurm/`):

```bash
python run.py --mode M --main-n 10000 --outdir $BRLOS_RAW --init                      # once per mode
python run.py --mode M --main-n 10000 --nshards K --list-shards                        # mapping, no compute
python run.py --mode M --main-n 10000 --outdir $BRLOS_RAW --shard $SLURM_ARRAY_TASK_ID --nshards K --stop-after-min 140
```

Tasks are split into shards by a deterministic longest-cost-first assignment that does not
depend on which tasks are done, so resubmitting a shard reruns only its missing tasks. Exit codes:
0 complete, 1 a task raised, 2 setup error, 3 stopped by `--stop-after-min`. Per-shard task logs
(`<mode>/logs/tasks_*.csv`) are concatenated into `<mode>/tasks_all_shards.csv` for `runtime.csv`.

Post-processing: see the top-level README (`summarize.py`, `extras.py`, then `figures.py`,
`headline.py`). `summarize.py` refuses duplicate (exp, model, n, b, rep, estimator) rows. The
rows of modes `bck` and `r3or` that recompute standard estimates are compared with the shards of
mode `full` (`consistency_chk.csv`; max |diff| = 0 for the paper's results).
`figures.py` cross-checks the b >= 5 optima of all 416 main cells against `imse.csv`.

## Provenance

The raw results used in the paper were computed on a Slurm cluster in November 2026 with
successive development versions of this code: mode `full` first, then `smallb`, then
`fineb`/`bck`/`r3or`; each later version added its mode without changing the experiments of the
earlier ones (identical configurations and shard mappings were checked). The code hashes
recorded in the runs' `config.json` therefore refer to those versions; the files here are the
final version, changed afterwards only in comments, default paths and the worker-cap
environment variable. `check_bck.py` reproduces stored estimates of mode `full`
from the shipped raw sample to within the 10-digit CSV rounding (4.9e-11) or exactly.
