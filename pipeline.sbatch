#!/bin/bash
#SBATCH --job-name=heavyhex-pipeline
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# 통합 파이프라인: 규약 게이트 -> 데이터셋 생성 -> 학습.
# 각 단계가 성공해야 다음 단계로 넘어간다 (set -e).
# 이미 생성된 데이터셋 파일은 make_dataset.py가 알아서 건너뛰므로 재실행해도 안전.
#
# repo 루트에서 제출할 것:
#   sbatch pipeline.sbatch                                 # 전체 그리드 생성 + 기본 학습
#   sbatch pipeline.sbatch --all --mwpm                    # 전체 그리드 학습 + MWPM 기준선
#   DATASET_ARGS="--smoke" sbatch pipeline.sbatch --smoke  # 빠른 end-to-end 확인
#
# 추가 인자는 train.py로, DATASET_ARGS는 make_dataset.py로 전달돼.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-KC2}"    # <-- EDIT: 본인 conda/mamba 환경 이름
                                 #     (또는 CONDA_ENV=이름 sbatch pipeline.sbatch 로 덮어쓰기)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

exec 9>/tmp/heavyhex_pipeline.lock
flock -n 9 || { echo "이미 실행 중인 파이프라인이 있음 — 중단"; exit 1; }

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} env=$CONDA_ENV"
echo "dataset_args=${DATASET_ARGS:-<default>} train_args=$*"

echo "=== [1/3] 규약 게이트 ==="
python verification/verify_equivalence.py

echo "=== [2/3] 데이터셋 생성 ==="
python dataset_generation/make_dataset.py ${DATASET_ARGS:-}

echo "=== [3/3] 학습 ==="
python train.py "$@"
