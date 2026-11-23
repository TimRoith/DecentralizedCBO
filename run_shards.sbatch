#!/bin/bash
#SBATCH --job-name=dcbo
#SBATCH -D /path/to/DecentralizedCBO
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH -o %x.%A_%a.out
#SBATCH -e %x.%A_%a.err
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=24:00:00

# usage: sbatch run_shards.sbatch <config.yaml>

set -euo pipefail

# Load your conda environment here, e.g.
# module load <conda module>
# conda activate cbx

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONFIG="${1:-benchmark_d100.yaml}"
echo "host:   $(hostname)"
echo "config: ${CONFIG}"
echo "shard:  ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

mkdir -p results
cd experiments
python -u run_benchmark.py "${CONFIG}" "shard=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
