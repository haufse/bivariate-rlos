#!/bin/bash
# Template: Section 7 Monte Carlo check of the covariance quadrature (s03_mc_sigma_check.py),
# single core, seed 20261117 fixed in the script; stdout -> sec7_asymptotics/outputs/s03_output.txt.
#SBATCH --job-name=brlos-sec7-s03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
##SBATCH --account=<your_account>
##SBATCH --partition=<your_partition>
set -u
REPO=${REPO:-$HOME/bivariate-rlos}
PY=${PY:-python}
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
cd "$REPO/sec7_asymptotics/scripts"
$PY -u s03_mc_sigma_check.py > ../outputs/s03_output.txt
rc=$?
rm -rf "$NUMBA_CACHE_DIR"
exit $rc
