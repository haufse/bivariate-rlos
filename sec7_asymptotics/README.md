# Section 7: asymptotic comparison (numerics)

Run every script from `scripts/` (they use the relative paths `../outputs`, `../../figures`,
`../../tables`). Environment: `environment/environment-sec7.yml` or `environment-analysis.yml`
(numba is needed for `rlos_lib.py`). Prefix used:
`NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

| script | purpose | output | runtime |
|---|---|---|---|
| `rlos_lib.py` | rank covariance Sigma(t) of r-LOS moments by quadrature (any r), moment maps, bias terms (calS, calR) of the models | library | |
| `kquad.py` | t-quadrature on T = [delta, 1-delta] from the grid of `s04` (barycentric interpolation for delta = 0.15) | library | |
| `figstyle.py` | figure style (imsart text width 406 pt, newtx Times, Okabe-Ito palette) | library | |
| `s01_symbolic_checks.py` | symbolic/high-precision checks of the formulas for calR, calS, mu, d | `outputs/s01_output.txt` | not recorded |
| `s02_finite_block_bias_check.py` | exact finite-b moment bias vs its limit (Richardson) | `outputs/s02_output.txt` | not recorded |
| `s03_mc_sigma_check.py` | Monte Carlo check of the quadrature covariance (r = 1, 2, 3), seed 20261117 | `outputs/s03_output.txt` | not recorded in total (parts (1)-(2): 61-70 s per t); 1 core, run on a cluster (`hpc/slurm/sec7_s03.sbatch`) |
| `s04_sigma_grid.py <theta ...>` | covariance grids Sigma(t) (9 x 9, r = 3) at GH(theta) | `outputs/sigma_grid_theta*.npz` | 32-50 s per theta on a cluster node (`hpc/slurm/sec7_s04.sbatch`) |
| `s05_assemble.py` | variance and bias constants V_r, B^SS, B^SR, B^RR, POT constants for several T | `outputs/constants_theta2.json`, `outputs/s05_output.txt` | ~50 s |
| `s06_tables.py --r3-verified` | optimized leading criterion, optimal block sizes; LaTeX tables | `../tables/sec7-table-{main,bstar,supp}-d015.tex` (paper), `outputs/sec7-table-*-d001.tex` (alternative T), `outputs/ameise_summary.json`, `outputs/s06_output.txt` | ~1 s |
| `s07_clayton_independence.py` | independence / Clayton case in closed form | `outputs/s07_output.txt` | not recorded |
| `s08_fig71_efficiency.py --delta 0.15` | Figure 7.1 | `../figures/fig71-efficiency-d015.pdf`, `outputs/fig71_data-d015.json`, `outputs/s08_output-d015.txt` | ~1 s |
| `s09z_fig72_zoom_variants.py --variant C --delta 0.15` | Figure 7.2 (zoomed panels with ratio rows) | `../figures/fig72-error-vs-blocksize-d015-zoomC.pdf` | ~2 s |
| `s10_display_formulas.py` | closed forms used in displays | `outputs/s10_output.txt` | not recorded |
| `s11_crosscheck.py` | second, independent route for the table constants (r = 2 via `sec8_simulations/sigma2_lib.py`) | `outputs/s11_output.txt` | ~2.5 min |
| `s12_conditioning.py` | conditioning and robustness of V_r/V_1 | `outputs/s12_output.txt` | not recorded |

Notes

* The paper uses T = [0.15, 0.85] (the set K of the Monte Carlo study), i.e. the `d015` files.
* `s06_tables.py` writes the footnote markers dagger, double dagger and section sign (`\dag`,
  `\ddag`, `\S`, set with `\rlap` after the entry, one note per line), as in the paper. Earlier
  versions of the generator used superscript letters a, b, c; only the markers were changed. The
  caption and footnote wording of the paper was edited by hand and differs from the generator.
* `s06_output.txt` also prints moderate-n "full criterion" numbers for the Archimax model that
  are marked WITHDRAWN: they are not supported by the theory (terms below the controlled
  remainder) and are not used in the paper.
* The script for the unzoomed variant of Figure 7.2 and a comparison with an earlier draft are
  not included.
