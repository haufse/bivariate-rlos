#!/usr/bin/env bash
# Sec. 8 follow-up analysis (H1 fineb, H2 bck, H3 r3or): one command, analysis only (no simulation).
# Run from anywhere:  bash run_all.sh      (about 5 min on 10 cores; th_theory.py dominates)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SEC8FU_CACHE=${SEC8FU_CACHE:-../../../.cache/sec8fu} NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-../../../.cache/numba}
RUN="$PY"   # e.g. RUN="taskset -c 0-9 nice -n 19 $PY" to restrict cores
$RUN a00_checks.py
$RUN th_theory.py > ../outputs/th_theory_stdout.txt 2>&1
$RUN h1_fineb.py
$RUN h2_bck.py
$RUN h2b_variance_mechanism.py
$RUN h3_r3or.py
$RUN h4_optimal_merged.py
for f in ../figs/*.pdf; do pdftoppm -r 150 -png "$f" "${f%.pdf}"; done
