# Aggregated Section 8 results

Produced by `sec8_simulations/summarize.py` and `extras.py` from the raw shards (see the
top-level README for the commands); read by `figures.py` and `headline.py`.

| folder | modes merged | used for |
|---|---|---|
| `main_b5/` | full + fineb + bck + r3or (b >= 5) | all Section 8 and supplementary figures and tables; `headline_numbers.csv` (column `new_b5`: numbers in the text) |
| `all_b2/` | full + smallb + fineb + bck + r3or (b >= 2) | small-b remark (`tables/sec8-table-smallb.tex`, `figures/sec8-supp-smallb-imse-vs-b.pdf`) |
| `full_only/` | full | consistency column of `headline.py`; independence draws for `bias_analysis/investigation/scripts/s05_data_decomposition.py` |
| `raw_sample/full/` | two raw shards of mode full | exact reproduction check in `sec8_simulations/tests/check_bck.py` |

Main files (keys `exp, model, n, b, k, estimator`; K = [0.15, 0.85], K10 = [0.10, 0.90]):
`imse.csv` (IMSE, MCSE, IBias^2, IVar, paired ratio to r1 with MCSE), `optimal_b.csv`
(MC-optimal b per estimator, paired ratio at the own optimum with delta-method and
selection-bootstrap standard errors), `pointwise.csv` (pointwise mean error and variance),
`diagnostics.csv` (fallback, boundary, truncation rates, pilot error, Frobenius error, runtime per
replication), `runtime.csv` (CPU seconds per cell), `consistency_chk.csv` (recomputed vs stored
estimates), and the `extra_*.csv` files of `extras.py` (documented in its docstring).
