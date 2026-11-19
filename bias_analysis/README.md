# Finite-sample bias of order 1/k (Section 8.5 and supplement)

Analysis only: no new Monte Carlo of the Section 8 design is run here (the limit-law Monte Carlo
`s03`, `s06` and the checks `s11`, `s13`, `s15` simulate small auxiliary experiments with fixed
seeds). Environment: `environment/environment-analysis.yml`. Run each script from its own
`scripts/` folder; outputs go to the sibling `outputs/` (and `followup/figs/`). All outputs used
for the paper are included.

Hypotheses H1-H3 named in the docstrings are those of the follow-up runs `fineb`, `bck` and
`r3or`; the results are in Section 8.5 and the supplement of the paper.

## `investigation/`

| script | content | needs raw shards |
|---|---|---|
| `biaslib.py` | moment maps, weights, rank-bias coefficients beta_C, n_W, c_W (shared library) | |
| `s01_pointwise_summary.py` | pointwise bias and variance of the Sec. 8 shards (mode full) -> `s01_pointwise.csv` | yes |
| `s02_theory_terms.py` | deterministic O(1/k) coefficients and leading block biases -> `s02_theory.csv` | |
| `s03_limit_mc.py` | Monte Carlo under the limiting r-LOS law, seed `SeedSequence(20261118)` | |
| `s04_exact_finite_b.py` | exact bias for infinitely many blocks of size b, M1-M4 -> `s04_exact_bias.csv` | |
| `s05_data_decomposition.py` | decomposition of the Monte Carlo bias and IMSE (from `s01`, `s02`, `s04` and `data/sec8/full_only`) | |
| `s06_bc_variance_mc.py` | variance effect of the equation correction | |
| `s07_augmented_criterion.py` | optimized r2/r1 ratios under the Sec. 7 criterion and two refinements -> `s07_ratios.csv` (theory rows of `sec8-bias:tab:opc`) | |
| `s08_r3_coefficient.py` | O(1/k) coefficient for r = 1, 2, 3 (uses `sec7_asymptotics/outputs/sigma_grid_theta2.npz`) | |
| `s09_tables.py` | K-averaged residuals by k (`s09_kres_Kavg.csv`) | |
| `s10_eta_derivative.py` | sensitivity of eta_W[A] to A | |
| `s11_bck_variance_check.py` | variance inflation of the bootstrap 1/k correction | yes (parquet cache of `followup/scripts/common.py`) |
| `s12_exact_independence.py` | exact finite-k rank moments at independence | |
| `s13_bck_kappa_se.py`, `s14_boundary_interior.py` | data part of the variance inflation; boundary effects near the ends of K | yes |
| `s15_pilot_degree.py` | dependence on the pilot degree m | |

Order: `s02`, `s04` before `s05`, `s07`, `s10`; `s01` before `s05`; `s05` before `s09`;
`s10` before `s11`-`s13`.

## `followup/`

Analysis of the follow-up runs `fineb` (H1), `bck` (H2) and `r3or` (H3) of the Monte Carlo
study. `bash scripts/run_all.sh` runs everything (needs the raw shards in `$BRLOS_RAW`;
`PY` selects the interpreter; about 5 min on 10 cores). `th_theory.py` (deterministic theory
inputs) does not need the raw shards. `h4_optimal_merged.py` produces the Monte Carlo optimal
b over the merged grid; `sec8_simulations/figures.py` cross-checks its table against
`outputs/h4_optimal_merged.csv` and `outputs/h4_table_optimal_merged.tex`.

## Raw shards

Scripts marked "yes" read `$BRLOS_RAW/<mode>/<exp>/<model>/n<n>/reps_*.csv.gz` (default
`<repo>/raw_results`), which are not part of the repository (about 490 MB; see the top-level
README for how to regenerate them). Parsed shards are cached as parquet in `$SEC8FU_CACHE`
(default `<repo>/.cache/sec8fu`).

## Note on `investigation/outputs/s07_ratios.csv`

The included `s07_ratios.csv` and `s07_output.txt` were regenerated with the final
`s04_exact_finite_b.py`, which contains a fix for b = 2 (the product 0 * log 0 gave NaN, so
b = 2 was skipped). Compared with the version produced before that fix, 8 of 168 rows differ: M1,
n in {1e3, 5e3, 1e4, 2e4}, block-size set "all b", criteria E and C, where the r = 2 optimum is
now b = 2 instead of b = 3. These rows lie outside the design b >= 5 of the paper and are not used
in it; the rows behind Table `sec8-bias:tab:opc` (M2, M3, n = 1e4) are unchanged.
