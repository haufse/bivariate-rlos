#!/bin/bash
# Template: one-off initialisation of a Section 8 run directory (config.json + oracle caches).
# Usage:  sbatch --job-name=brlos-<mode>-init sec8_init.sbatch <mode>
#         <mode> in fast, full, smallb, fineb, bck, r3or
# Adapt the commented #SBATCH lines (account, partition) to your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
##SBATCH --account=<your_account>
##SBATCH --partition=<your_partition>
set -eu
MODE=$1
REPO=${REPO:-$HOME/bivariate-rlos}            # checkout of this repository
PY=${PY:-python}                              # interpreter with environment/requirements-sec8.txt
RAW=${BRLOS_RAW:-$REPO/raw_results}           # results root (large; use a scratch/work file system)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO/sec8_simulations"
$PY run.py --mode "$MODE" --main-n 10000 --outdir "$RAW" --init
