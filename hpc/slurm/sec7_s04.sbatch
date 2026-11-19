#!/bin/bash
# Template: Section 7 covariance grids (sec7_asymptotics/scripts/s04_sigma_grid.py), one theta per
# array element; writes sec7_asymptotics/outputs/sigma_grid_theta<theta>.npz (committed already).
# Usage:  sbatch --array=0-9 --cpus-per-task=<C> --mem=<M> --time=<T> sec7_s04.sbatch
#SBATCH --job-name=brlos-sec7-s04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --account=<your_account>
##SBATCH --partition=<your_partition>
set -u
THETAS=(2.0 1.0 1.25 1.5 3.0 1.1 1.75 2.5 4.0 6.0)
TH=${THETAS[$SLURM_ARRAY_TASK_ID]}
REPO=${REPO:-$HOME/bivariate-rlos}
PY=${PY:-python}                              # interpreter with environment/environment-sec7.yml (numba)
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$NUMBA_CACHE_DIR"
cd "$REPO/sec7_asymptotics/scripts"
$PY s04_sigma_grid.py "$TH"
rc=$?
rm -rf "$NUMBA_CACHE_DIR"
exit $rc
