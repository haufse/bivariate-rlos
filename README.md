# bivariate-rlos

Code and simulation results accompanying the paper

> **Estimation of Extremal Dependence from Bivariate r-Largest Order Statistics**
> Erik Haufs, Axel Bücher, Johan Segers, Michaël Lalancette (journal and DOI: *to be added*)

The repository contains

* the asymptotic numerics of the comparison section of the paper (Section 7: Table 7.2,
  supplementary Tables L.1-L.2 of the comparison appendix, Figures 7.1-7.2),
* the Monte Carlo study (Section 8 and the additional Monte Carlo results in the supplement),
* the analysis of the finite-sample bias of order 1/k (Section 8.5 and its supplement),
* aggregated simulation results from which every Section 8 table and figure can be rebuilt,
* the final figures and LaTeX tables as used in the paper.

## Layout

| path | content |
|---|---|
| `sec8_simulations/` | Monte Carlo study: samplers (`models.py`), estimators (`estimators.py`, `pickands.py`, `sigma2_lib.py`), driver (`run.py`, `config.py`), post-processing (`summarize.py`, `extras.py`, `figures.py`, `headline.py`), oracle covariance caches (`cache/`), checks (`tests/`). See `sec8_simulations/README.md`. |
| `sec7_asymptotics/` | quadrature library `scripts/rlos_lib.py` (rank covariances of r-LOS, bias terms), scripts `s01`-`s12` for Section 7, their outputs (`outputs/`, including the covariance grids `sigma_grid_theta*.npz`). See `sec7_asymptotics/README.md`. |
| `bias_analysis/` | finite-sample bias of order 1/k: `investigation/` (theory terms, exact finite-b bias, limit-law Monte Carlo, decomposition of the Section 8 results) and `followup/` (analysis of the follow-up runs H1-H3). See `bias_analysis/README.md`. |
| `data/sec8/` | aggregated Section 8 results (CSV): `main_b5` (design of the paper, b >= 5), `all_b2` (including b in {2,3,4}, used for the small-b remark), `full_only` (mode `full` alone), `raw_sample` (two raw shards used by a reproduction check). |
| `figures/`, `tables/` | figures (vector PDF, 406 pt = `\textwidth` of the imsart class) and LaTeX table snippets as in the paper. |
| `hpc/slurm/` | generic Slurm templates for the cluster runs. |
| `environment/` | software environments. |
| `tests/` | pytest entry point. |

## Software

Two Python environments were used (files in `environment/`):

* **Section 8 simulations and post-processing**: Python 3.12.7, NumPy 2.2.2, SciPy 1.15.1,
  pandas 3.0.2 (`requirements-sec8.txt`); figures additionally matplotlib 3.10.0 and a TeX
  installation with `newtxtext`/`newtxmath` (TeX Live 2022) (`requirements-postprocessing.txt`).
* **Section 7 numerics and bias analysis**: Python 3.9.20, NumPy 1.26.4, SciPy 1.13.0,
  SymPy 1.13.3, mpmath 1.3.0, numba 0.60.0, pandas 2.2.3, matplotlib 3.9.2, pyarrow 19.0.0
  (`environment-sec7.yml`, `environment-analysis.yml`).

All commands below are run with one BLAS thread:
`export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Scripts that use numba take `NUMBA_NUM_THREADS` (8 was used) and, optionally,
`NUMBA_CACHE_DIR`.

## Seeds

* Section 8: `numpy.random.SeedSequence([20261117, model_id, n, rep]).spawn(2)` gives the data
  and bootstrap streams of a replication; the bootstrap stream for block length b, order r, pilot
  resolution m and B bootstrap samples is the child with spawn key `bootstrap.spawn_key + (b, r, m, B)`.
  All experiments with the same (model, n, rep) use the same dataset (paired comparisons across
  b, estimators, r, m and B). Bootstrap standard errors of the selection step use seed 20261118.
* Section 7: the quadratures are deterministic; the Monte Carlo check `s03` uses seed 20261117.
* Bias analysis: `SeedSequence(20261118)` for the limit-law Monte Carlo; other scripts state their seeds.

## Reproducing the paper

### Section 8 tables and figures from the aggregated results (minutes, laptop)

```bash
cd sec8_simulations
python figures.py      # ~45 s: all Section 8 and supplementary figures -> ../figures, tables -> ../tables
python headline.py     # ~1 s: numbers quoted in the text -> ../data/sec8/main_b5/headline_numbers.csv
```

| paper object | file | produced by |
|---|---|---|
| Fig. `fig:sim-imse`, `fig:sim-decomposition`, `fig:sim-variance`, `fig:sim-bias-correction` | `figures/sec8-{imse-vs-b,decomposition,variance-vs-t,pointwise-bias}.pdf` | `figures.py` |
| Table `tab:sim-optimal` | `tables/sec8-table-optimal.tex` | `figures.py` |
| Table `sec8-bias:tab:opc` (numbers; typeset inline in the paper) | `tables/sec8-table-opc.tex` | `figures.py` (theory rows from `bias_analysis/investigation/outputs/s07_ratios.csv`) |
| Table `sec8-bias:tab:k` (numbers; typeset inline) | `bias_analysis/investigation/outputs/s09_kres_Kavg.csv`, `s09_output.txt` (pooled over M1-M4 by k) | `bias_analysis/investigation/scripts/s09_tables.py` |
| supplementary figures `fig:sim-independence`, `fig:sim-weights`, `fig:sim-r3`, `fig:sim-r3-oracle`, `fig:sim-M5`, `fig:sim-M5-decomposition`, `fig:sim-imse-n*`, `fig:sim-pointwise-*`, `fig:sim-bck`, `fig:sim-smallb` | `figures/sec8-supp-*.pdf` | `figures.py` |
| supplementary tables `tab:sim-{independence,fallback,sensitivity,optimal-M5,bck,smallb,diagnostics,decomposition}` | `tables/sec8-table-*.tex` | `figures.py` |

`figures/sec8-supp-decomposition-n*.pdf` are produced as well but not used in the paper.
Figure F3 (`sec8-variance-vs-t.pdf`) also reads the Section 7 covariance grid
`sec7_asymptotics/outputs/sigma_grid_theta2.npz`.

### Section 7 tables and figures (seconds to minutes, laptop)

```bash
cd sec7_asymptotics/scripts           # all Section 7 scripts are run from here
python s05_assemble.py > ../outputs/s05_output.txt                 # ~50 s: constants from the covariance grids
python s06_tables.py --r3-verified > ../outputs/s06_output.txt     # ~1 s: Tables 7.2, L.1, L.2
python s08_fig71_efficiency.py --delta 0.15 > ../outputs/s08_output-d015.txt   # Figure 7.1
python s09z_fig72_zoom_variants.py --variant C --delta 0.15        # Figure 7.2
```

| paper object | file |
|---|---|
| Table 7.2 `tab:comparison-ameise` | `tables/sec7-table-main-d015.tex` |
| Table `tab:comparison-bstar` (comparison appendix) | `tables/sec7-table-bstar-d015.tex` |
| Table `tab:comparison-supp` (comparison appendix) | `tables/sec7-table-supp-d015.tex` |
| Figure 7.1 `fig:efficiency-weighting` | `figures/fig71-efficiency-d015.pdf` |
| Figure 7.2 `fig:error-vs-blocksize` | `figures/fig72-error-vs-blocksize-d015-zoomC.pdf` |

The tables of Section 7 are typeset inline in the paper; the table bodies (all entries and
footnote markers) are identical to the generated snippets, the captions and footnote texts
were edited for the paper. Table 7.1 (`tab:comparison-rates`) contains rates only and is not
generated. The trimming set is T = [0.15, 0.85] (`d015`); `s06` also writes the alternative
T = [0.01, 0.99] to `sec7_asymptotics/outputs/`. The covariance grids
`outputs/sigma_grid_theta*.npz` were computed with `s04_sigma_grid.py` on a Slurm cluster
(32-50 s per value of theta; template `hpc/slurm/sec7_s04.sbatch`); checks: `s01`, `s02`,
`s03` (Monte Carlo, single core, on the cluster), `s07`, `s10`, `s11` (~2.5 min), `s12`.

### Bias of order 1/k (Section 8.5 and supplement)

See `bias_analysis/README.md`. The deterministic and limit-law scripts run from the repository;
scripts that read per-replication results (`investigation/s01`, `followup/*` except
`th_theory.py`) need the raw shards (below). Their outputs are included.

### Raw Monte Carlo results (cluster)

The per-replication shards (`<raw>/<mode>/<exp>/<model>/n<n>/reps_*.{est,diag}.csv.gz`) take
about 490 MB and are not part of this repository. They are regenerated by

```bash
cd sec8_simulations
export BRLOS_RAW=/path/to/raw_results                    # default: <repo>/raw_results
for M in full fineb bck r3or smallb; do python run.py --mode $M --main-n 10000 --outdir $BRLOS_RAW --init; done
# then one array job per mode (hpc/slurm/sec8_array.sbatch), or locally: python run.py --mode M --outdir $BRLOS_RAW
python summarize.py --mode full --merge-mode fineb --merge-mode bck --merge-mode r3or --outdir $BRLOS_RAW --results-tag main_b5   # ~3.5 min, 6 GB RAM
python extras.py    --mode full --merge-mode fineb --merge-mode bck --merge-mode r3or --outdir $BRLOS_RAW --results-tag main_b5   # ~1 min
python summarize.py --mode full --merge-mode smallb --merge-mode fineb --merge-mode bck --merge-mode r3or --outdir $BRLOS_RAW --results-tag all_b2
python extras.py    --mode full --merge-mode smallb --merge-mode fineb --merge-mode bck --merge-mode r3or --outdir $BRLOS_RAW --results-tag all_b2
python summarize.py --mode full --outdir $BRLOS_RAW --results-tag full_only
python extras.py    --mode full --outdir $BRLOS_RAW --results-tag full_only
```

Cluster cost of the runs used in the paper (CPU hours, sum of the task times): `full` 32.4
(2650 tasks), `fineb` 17.7 (650), `bck` 42.3 (1760), `r3or` 1.1 (100), `smallb` 134.3 (2250);
228 CPU hours in total, of which 93.5 for the main design (b >= 5). Shards of the modes smallb,
fineb, bck and r3or ran with 16 CPUs and 2-4 GB per CPU (see `hpc/slurm/sec8_array.sbatch`).

## Checks

```bash
pytest -q tests/                 # Sec. 8 unit checks, bootstrap-correction checks with exact
                                 # reproduction of stored estimates, headline numbers (~1-2 min)
BRLOS_SLOW=1 pytest -q tests/    # plus the bootstrap-vs-quadrature covariance check (~2 min)
```

The check scripts can also be run directly, e.g. `cd sec8_simulations && python tests/check_units.py`.

## Citation and licence

See `CITATION.cff`. Licence: MIT (see `LICENSE`).
