#!/bin/bash
#SBATCH --job-name=heavyhex-cnn
#SBATCH --partition=YOUR_PARTITION        # <-- EDIT: your cluster's GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# Heavy-hex (3,3) CNN training on the lab server (2x GTX 2080 Ti).
# Edit the partition above (see README), then submit from the repo root:
#   mkdir -p slurm/logs
#   sbatch slurm/train.sbatch                       # default single config
#   sbatch slurm/train.sbatch --all --mwpm          # full KCS grid + baseline
# Extra arguments are forwarded to train.py.

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate KC2

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} args=$*"
python train.py "$@"
