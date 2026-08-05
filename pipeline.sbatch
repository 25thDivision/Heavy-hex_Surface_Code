#!/bin/bash
#SBATCH --job-name=heavyhex-pipeline
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# 통합 파이프라인: 규약 게이트 -> 데이터셋 생성 -> 학습 -> QPU 검증.
# 각 단계가 성공해야 다음 단계로 넘어간다 (set -e).
# 이미 생성된 데이터셋 파일은 make_dataset.py가 알아서 건너뛰므로 재실행해도 안전.
#
# repo 루트에서 제출할 것:
#   sbatch pipeline.sbatch                                 # 전체 그리드 생성 + 기본 학습
#   sbatch pipeline.sbatch --all --mwpm                    # 전체 그리드 학습 + MWPM 기준선
#   DATASET_ARGS="--smoke" sbatch pipeline.sbatch --smoke  # 빠른 end-to-end 확인
#   SWEEP_CONFIG=다른스윕.json sbatch pipeline.sbatch       # 다른 스윕 파일 지정
#   SWEEP_CONFIG=none sbatch pipeline.sbatch               # 스윕 끄기
#   HW_ARGS="--dry-run" sbatch pipeline.sbatch             # QPU 단계는 리허설만
#   HW_ARGS="--backend ibm_boston --shots 20000" sbatch pipeline.sbatch
#   SKIP_HW=1 sbatch pipeline.sbatch                       # QPU 단계 끄기
#
# 추가 인자는 train.py로, DATASET_ARGS는 make_dataset.py로,
# HW_ARGS는 hardware/run_hw.py로 전달돼. run_hw.py는 서브커맨드 없이
# 실행하면 all(제출->대기->분석 원샷)이고, HW_ARGS="analyze --job-id X"
# 처럼 단계를 명시하면 그 단계만 돈다.
# 스윕: repo 루트에 train_sweep.json이 있으면 데이터셋 생성과 학습이
# 자동으로 그 스윕 조합을 돈다 (-n/-p 같은 선택 인자를 주면 그쪽 우선).
# SWEEP_CONFIG=경로 로 다른 스윕 파일을, =none 으로 비활성화를 지정.
# 기본 하이퍼파라미터/샘플 수는 train_options.json이 있으면 자동 적용.
# (스키마는 train.py 도크스트링 참고)
# QPU 단계는 keys.json이 있을 때만 실행 — 실제 QPU 잡을 제출하므로
# (기본 ibm_yonsei, 50k shots) 반복 제출을 원치 않으면 SKIP_HW=1을 쓸 것.
# analyze는 --ckpt 없이 돌므로 checkpoint/*.pt 전부를 평가해
# results/hardware/hw_<job_id>.csv로 남긴다.

set -eo pipefail

CONDA_ENV="${CONDA_ENV:-KC2}"    # <-- EDIT: 본인 conda/mamba 환경 이름
                                 #     (또는 CONDA_ENV=이름 sbatch pipeline.sbatch 로 덮어쓰기)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

set -u

cd "${SLURM_SUBMIT_DIR:-$PWD}"

exec 9>/tmp/heavyhex_pipeline.lock
flock -n 9 || { echo "이미 실행 중인 파이프라인이 있음 — 중단"; exit 1; }

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} env=$CONDA_ENV"
echo "dataset_args=${DATASET_ARGS:-<default>} train_args=$* hw_args=${HW_ARGS:-<default>} sweep=${SWEEP_CONFIG:-<none>}"

echo "=== [1/4] 규약 게이트 ==="
python verification/verify_equivalence.py

echo "=== [2/4] 데이터셋 생성 ==="
python dataset_generation/make_dataset.py ${DATASET_ARGS:-} \
    ${SWEEP_CONFIG:+--config "$SWEEP_CONFIG"}

echo "=== [3/4] 학습 ==="
python train.py "$@" ${SWEEP_CONFIG:+--config "$SWEEP_CONFIG"}

echo "=== [4/4] QPU 검증 (submit -> wait -> analyze) ==="
if [[ "${SKIP_HW:-0}" == "1" ]]; then
    echo "SKIP_HW=1 — QPU 단계 건너뜀"
elif [[ ! -f keys.json ]]; then
    echo "keys.json 없음 — QPU 단계 건너뜀 (자격 증명이 있어야 제출 가능)"
else
    python hardware/run_hw.py ${HW_ARGS:-}
fi
