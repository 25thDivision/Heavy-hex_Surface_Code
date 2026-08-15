#!/bin/bash
#SBATCH --job-name=qec-pipeline
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# 통합 파이프라인: 규약 게이트 -> [QPU 프로파일] -> 데이터셋 -> 학습 -> QPU 검증.
# 한 번의 제출로 두 코드 축을 순서대로 돈다:
#   heavyhex -> ibm_boston  |  surface -> ibm_miami
# 각 단계가 성공해야 다음 단계로 넘어간다 (set -e).
# 이미 생성된 데이터셋은 스킵되고, 학습은 epoch-resume으로 이어서 돈다.
#
# LOOPS: 파이프라인 전체를 N회 반복 (라운드마다 자기 자신을 재제출).
#   각 루프: QPU 이력이 있으면 최신 5개(QPU_RUNS)를 평균한 qpu 프로파일을
#   만들어 realistic 그리드에 더해 학습하고, 코드별로 QPU를 QPU_RUNS회
#   제출·분석한다. 그 런들이 다음 루프의 평균 입력이 된다.
#   첫 루프는 이력이 없으므로 realistic만 학습하고 QPU를 돌린다.
#
# repo 루트에서 제출할 것:
#   sbatch pipeline.sbatch --mwpm --solution               # 1회 (기본)
#   LOOPS=5 sbatch pipeline.sbatch --mwpm --solution       # 5루프 반복
#   QPU_RUNS=3 sbatch pipeline.sbatch ...                  # 루프당 제출 수 변경
#   CODES=heavyhex sbatch pipeline.sbatch                  # 한 코드만 (heavyhex|surface)
#   HH_BACKEND=... / SURF_BACKEND=...                      # 백엔드 변경
#   DATASET_ARGS="--smoke" sbatch pipeline.sbatch --smoke  # 빠른 end-to-end 확인
#   SWEEP_CONFIG=다른스윕.json / SWEEP_CONFIG=none          # 별도 스윕 / 스윕 끄기
#   HW_ARGS="--dry-run" / HW_ARGS="--shots 20000"          # QPU 공통 인자
#   SKIP_HW=1 sbatch pipeline.sbatch                       # QPU 단계 끄기
#
# 추가 인자는 train.py로, DATASET_ARGS는 make_dataset.py로 전달돼
# (--code는 파이프라인이 코드별로 자동으로 붙이니 직접 주지 말 것).
# 학습에 --solution을 주면 QPU analyze에도 자동 전파된다.
# 스윕: config.json의 "sweep" 섹션(cnn+gnn)이 자동 적용되고, qpu 프로파일이
# 있는 루프에서는 같은 스윕으로 -n qpu/<이름> 학습이 한 번 더 돈다.
# QPU 단계는 keys.json이 있을 때만 실행 — 루프당 코드별 QPU_RUNS회의 실제
# 잡이 제출되므로 반복 제출을 원치 않으면 SKIP_HW=1을 쓸 것.
# analyze는 코드별 CNN/GNN 체크포인트 전부를 평가해
# results/hardware/<backend>_<code>_<timestamp>.csv로 남긴다.

set -eo pipefail

CONDA_ENV="${CONDA_ENV:-HHC}"    # <-- EDIT: 본인 conda/mamba 환경 이름
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
LOOPS="${LOOPS:-1}"
LOOP="${LOOP:-1}"
QPU_RUNS="${QPU_RUNS:-5}"

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
echo "loop=$LOOP/$LOOPS qpu_runs=$QPU_RUNS codes=$CODES (heavyhex->$HH_BACKEND, surface->$SURF_BACKEND)"
echo "dataset_args=${DATASET_ARGS:-<default>} train_args=$* hw_args=${HW_ARGS:-<default>} sweep=${SWEEP_CONFIG:-<none>}"

echo "=== [1/5] 규약 게이트 ==="
for code in $CODES; do
    echo "--- gate: $code ---"
    python "$(gate_of "$code")"
done

echo "=== [2/5] QPU 캘리브레이션 프로파일 (이력 있으면 최신 ${QPU_RUNS}개 평균) ==="
declare -A PROFILE
for code in $CODES; do
    backend="$(backend_of "$code")"
    if out=$(python dataset_generation/make_qpu_avg_profile.py \
                 --code "$code" --backend "$backend" \
                 --n-runs "$QPU_RUNS" --suffix "loop$LOOP" 2>&1); then
        echo "$out"
        PROFILE[$code]=$(sed -n 's/^profile \(qpu\/[^:]*\):.*/\1/p' <<< "$out" | tail -1)
        echo "--- $code: 프로파일 ${PROFILE[$code]} 사용 ---"
    else
        echo "$out"
        echo "--- $code: QPU 이력 없음/생성 불가 — 이번 루프는 realistic만 ---"
        PROFILE[$code]=""
    fi
done

echo "=== [3/5] 데이터셋 생성 ==="
for code in $CODES; do
    echo "--- dataset: $code (realistic) ---"
    python dataset_generation/make_dataset.py --code "$code" \
        ${DATASET_ARGS:-} ${SWEEP_CONFIG:+--config "$SWEEP_CONFIG"}
    if [[ -n "${PROFILE[$code]}" ]]; then
        echo "--- dataset: $code (${PROFILE[$code]}) ---"
        python dataset_generation/make_dataset.py --code "$code" \
            -n "${PROFILE[$code]}" ${DATASET_ARGS:-}
    fi
done

echo "=== [4/5] 학습 (epoch-resume 이어서) ==="
for code in $CODES; do
    echo "--- train: $code (realistic 그리드) ---"
    python train.py --code "$code" "$@" \
        ${SWEEP_CONFIG:+--config "$SWEEP_CONFIG"}
    if [[ -n "${PROFILE[$code]}" ]]; then
        echo "--- train: $code (${PROFILE[$code]}) ---"
        python train.py --code "$code" "$@" \
            --config "${SWEEP_CONFIG:-config.json}" -n "${PROFILE[$code]}"
    fi
done

echo "=== [5/5] QPU 검증 (코드별 submit -> wait -> analyze x ${QPU_RUNS}) ==="
# 학습을 --solution(참조 구현)으로 돌렸다면 analyze의 모델 로딩도 같은
# 구현을 써야 한다 — 학습 인자에서 자동으로 이어받는다 (없으면 skeleton).
HW_SOLUTION=""
case " $* " in *" --solution "*) HW_SOLUTION="--solution" ;; esac
if [[ "${SKIP_HW:-0}" == "1" ]]; then
    echo "SKIP_HW=1 — QPU 단계 건너뜀"
elif [[ ! -f keys.json ]]; then
    echo "keys.json 없음 — QPU 단계 건너뜀 (자격 증명이 있어야 제출 가능)"
else
    for code in $CODES; do
        backend="$(backend_of "$code")"
        for i in $(seq 1 "$QPU_RUNS"); do
            echo "--- QPU: $code @ $backend (run $i/$QPU_RUNS) ${HW_SOLUTION:+(--solution)} ---"
            python hardware/run_hw.py --backend "$backend" --code "$code" \
                $HW_SOLUTION ${HW_ARGS:-}
        done
    done
fi

if (( LOOP < LOOPS )); then
    echo "=== 루프 $LOOP 완료 — 다음 루프($((LOOP + 1))/$LOOPS) 재제출 ==="
    if command -v sbatch >/dev/null 2>&1; then
        sbatch --export=ALL,LOOP=$((LOOP + 1)) pipeline.sbatch "$@"
    else
        echo "WARNING: sbatch를 찾을 수 없어 재제출 불가 — LOOP=$((LOOP + 1))로 직접 제출할 것"
    fi
else
    echo "=== 전체 $LOOPS 루프 완료 ==="
fi
