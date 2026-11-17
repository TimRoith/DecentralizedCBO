#!/bin/bash
#SBATCH --job-name=dcbo
#SBATCH -D /path/to/DecentralizedCBO
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH -o ./slogs/%x.%A_%a.%N.out
#SBATCH -e ./slogs/%x.%A_%a.%N.err
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=24:00:00

# usage: sbatch run_shards.sbatch <config.yaml>
#
# Each array task takes every Nth (method, problem, seed) job, so the array
# covers the config exactly once. Runs are cached individually, so a task that
# hits the wall clock can simply be resubmitted.
#
# Set -D above to this directory on your cluster, and adapt the module/conda
# lines below to your site.

set -euo pipefail

module load slurm_setup
module load miniforge3
source ~/.conda_init
conda activate cbx

# One worker process per CPU, so keep each one single-threaded.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONFIG="${1:-benchmark_d100.yaml}"
echo "host:   $(hostname)"
echo "config: ${CONFIG}"
echo "shard:  ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

mkdir -p results slogs
cd experiments
python -u run_benchmark.py "${CONFIG}" "shard=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
