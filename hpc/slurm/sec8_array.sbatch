#!/bin/bash
# Template: one Slurm array element = one shard of a Section 8 mode.
# Usage (resources on the command line; values used for the paper in the comments):
#   sbatch --job-name=brlos-<mode> --array=0-<K-1> --cpus-per-task=16 --mem-per-cpu=<M> \
#          --time=02:30:00 sec8_array.sbatch <mode> <K> [<min-tier> <max-tier>]
#   full:   e.g. --array=0-3 (K=4), 16 CPUs, --mem-per-cpu=1G, optionally by tier ranges (config.tier)
#   smallb: K=8, 16 CPUs, --mem-per-cpu=4G (estimated 2.5-3 GB per worker at n=1e5, b=2)
#   fineb:  K=2; bck: K=3; r3or: K=1; 16 CPUs, --mem-per-cpu=2G
# Run sec8_init.sbatch for the mode first.  Finished tasks are skipped, so resubmitting a shard
# (e.g. with --array=<failed ids>) computes only its missing tasks.  Exit 3 = stopped before the
# wall-time limit by --stop-after-min (resubmit).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --account=<your_account>
##SBATCH --partition=<your_partition>
set -u
MODE=$1; NSHARDS=$2; MINT=${3:-1}; MAXT=${4:-99}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
REPO=${REPO:-$HOME/bivariate-rlos}
PY=${PY:-python}
RAW=${BRLOS_RAW:-$REPO/raw_results}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO/sec8_simulations"
exec $PY run.py --mode "$MODE" --main-n 10000 --outdir "$RAW" \
    --min-tier "$MINT" --max-tier "$MAXT" --shard "$SHARD" --nshards "$NSHARDS" \
    --workers "$SLURM_CPUS_PER_TASK" --stop-after-min 140
