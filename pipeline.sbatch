#!/bin/bash
#SBATCH --job-name=qec-pipeline
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# 통합 파이프라인: 규약 게이트 -> 데이터셋 생성 -> 학습 -> QPU 검증.
# 한 번의 제출로 두 코드 축을 순서대로 돈다:
#   heavyhex -> ibm_boston  |  surface -> ibm_miami
# 각 단계가 성공해야 다음 단계로 넘어간다 (set -e).
# 이미 생성된 데이터셋 파일은 make_dataset.py가 알아서 건너뛰므로 재실행해도 안전.
#
# repo 루트에서 제출할 것:
#   sbatch pipeline.sbatch                                 # 두 코드 전부: 생성 + 기본 학습
#   sbatch pipeline.sbatch --all --mwpm                    # 전체 그리드 학습 + MWPM 기준선
#   DATASET_ARGS="--smoke" sbatch pipeline.sbatch --smoke  # 빠른 end-to-end 확인
#   CODES=heavyhex sbatch pipeline.sbatch                  # 한 코드만 (heavyhex|surface)
#   HH_BACKEND=ibm_yonsei sbatch pipeline.sbatch           # heavyhex 백엔드 변경
#   SURF_BACKEND=ibm_miami sbatch pipeline.sbatch          # surface 백엔드 변경 (기본값)
#   SWEEP_CONFIG=다른스윕.json sbatch pipeline.sbatch       # 다른 스윕 파일 지정
#   SWEEP_CONFIG=none sbatch pipeline.sbatch               # 스윕 끄기
#   HW_ARGS="--dry-run" sbatch pipeline.sbatch             # QPU 단계는 리허설만
#   HW_ARGS="--shots 20000" sbatch pipeline.sbatch         # 두 백엔드 공통 추가 인자
#   SKIP_HW=1 sbatch pipeline.sbatch                       # QPU 단계 끄기
#
# 추가 인자는 train.py로, DATASET_ARGS는 make_dataset.py로 전달돼
# (--code는 파이프라인이 코드별로 자동으로 붙이니 직접 주지 말 것).
# QPU 단계는 코드별로 hardware/run_hw.py --backend <백엔드> --code <코드>
# (submit->대기->analyze 원샷)를 실행하고, HW_ARGS는 그 뒤에 공통으로
# 붙는다 (--shots, --dry-run 등). analyze 단독 재실행처럼 서브커맨드를
# 명시하고 싶으면 pipeline이 아니라 run_hw.py를 직접 실행할 것.
# 스윕: repo 루트에 train_sweep.json이 있으면 데이터셋 생성과 학습이
# 자동으로 그 스윕 조합을 돈다 (-n/-p 같은 선택 인자를 주면 그쪽 우선).
# SWEEP_CONFIG=경로 로 다른 스윕 파일을, =none 으로 비활성화를 지정.
# 기본 하이퍼파라미터/샘플 수는 train_options.json이 있으면 자동 적용.
# (스키마는 train.py 도크스트링 참고)
# QPU 단계는 keys.json이 있을 때만 실행 — 실제 QPU 잡을 제출하므로
# 반복 제출을 원치 않으면 SKIP_HW=1을 쓸 것. 두 백엔드는 순차 실행
# (boston 제출->완료 대기->분석 후 miami 제출)이라 대기 시간이 길 수 있다.
# analyze는 --ckpt 없이 돌므로 해당 코드의 checkpoint/{MODEL}_*.pt 전부를
# 평가해 results/hardware/hw_<job_id>.csv로 남긴다.

set -eo pipefail

CONDA_ENV="${CONDA_ENV:-KC2}"    # <-- EDIT: 본인 conda/mamba 환경 이름
                                 #     (또는 CONDA_ENV=이름 sbatch pipeline.sbatch 로 덮어쓰기)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

set -u

export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

exec 9>/tmp/heavyhex_pipeline.lock
flock -n 9 || { echo "이미 실행 중인 파이프라인이 있음 — 중단"; exit 1; }

CODES="${CODES:-heavyhex surface}"
HH_BACKEND="${HH_BACKEND:-ibm_boston}"
SURF_BACKEND="${SURF_BACKEND:-ibm_miami}"

backend_of() {  # 코드 -> QPU 백엔드
    case "$1" in
        heavyhex) echo "$HH_BACKEND" ;;
        surface)  echo "$SURF_BACKEND" ;;
        *) echo "unknown code '$1' (heavyhex|surface)" >&2; return 1 ;;
    esac
}

gate_of() {     # 코드 -> 규약 게이트 스크립트
    case "$1" in
        heavyhex) echo "verification/verify_equivalence.py" ;;
        surface)  echo "verification/verify_rotatedSurface3.py" ;;
    esac
}

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} env=$CONDA_ENV"
echo "codes=$CODES (heavyhex->$HH_BACKEND, surface->$SURF_BACKEND)"
echo "dataset_args=${DATASET_ARGS:-<default>} train_args=$* hw_args=${HW_ARGS:-<default>} sweep=${SWEEP_CONFIG:-<none>}"

echo "=== [1/4] 규약 게이트 ==="
for code in $CODES; do
    echo "--- gate: $code ---"
    python "$(gate_of "$code")"
done

echo "=== [2/4] 데이터셋 생성 ==="
for code in $CODES; do
    echo "--- dataset: $code ---"
    python dataset_generation/make_dataset.py --code "$code" \
        ${DATASET_ARGS:-} ${SWEEP_CONFIG:+--config "$SWEEP_CONFIG"}
done

echo "=== [3/4] 학습 ==="
for code in $CODES; do
    echo "--- train: $code ---"
    python train.py --code "$code" "$@" \
        ${SWEEP_CONFIG:+--config "$SWEEP_CONFIG"}
done

echo "=== [4/4] QPU 검증 (코드별 submit -> wait -> analyze) ==="
if [[ "${SKIP_HW:-0}" == "1" ]]; then
    echo "SKIP_HW=1 — QPU 단계 건너뜀"
elif [[ ! -f keys.json ]]; then
    echo "keys.json 없음 — QPU 단계 건너뜀 (자격 증명이 있어야 제출 가능)"
else
    for code in $CODES; do
        backend="$(backend_of "$code")"
        echo "--- QPU: $code @ $backend ---"
        python hardware/run_hw.py --backend "$backend" --code "$code" \
            ${HW_ARGS:-}
    done
fi
