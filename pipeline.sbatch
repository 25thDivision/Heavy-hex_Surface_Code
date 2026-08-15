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
#
# 고정 설정은 전부 config.json의 "pipeline" 섹션에서 읽는다:
#   conda_env  : conda/mamba 환경 이름
#   codes      : 돌릴 코드 축 (기본 ["heavyhex", "surface"])
#   backends   : 코드 -> QPU 백엔드 (기본 boston/miami)
#   qpu_runs   : 루프당 코드별 QPU 제출 횟수 (기본 5)
#   shots      : QPU 샷 수 (기본 50000)
#   train_args : train.py에 항상 붙는 인자 (기본 "--mwpm --solution")
#
# sbatch 변수는 두 개뿐:
#   LOOPS=N   파이프라인 전체를 N회 반복 (기본 1; 루프 끝에 자기 재제출)
#   SMOKE=1   리허설: 데이터셋/학습 스모크 + QPU는 dry-run 1회
#
# 사용 (repo 루트에서):
#   LOOPS=5 sbatch pipeline.sbatch      # 실전 5루프
#   sbatch pipeline.sbatch              # 1회
#   SMOKE=1 sbatch pipeline.sbatch      # 빠른 end-to-end 리허설 (제출 없음)
#   sbatch pipeline.sbatch -n qpu/<이름>   # 추가 train 인자도 뒤에 붙일 수 있음
#
# 루프 동작: 각 루프는 QPU 이력이 있으면 최신 qpu_runs개의 캘리브레이션을
# 평균한 qpu 프로파일을 만들어 realistic 그리드에 더해 학습하고(epoch-resume
# 이어서), 코드별로 QPU를 qpu_runs회 제출·분석한다. 그 런들이 다음 루프의
# 평균 입력이 된다. 첫 루프는 이력이 없으면 realistic만 학습한다.
# qpu 프로파일이 있는 루프는 하드웨어 MWPM 기준선(DEM 가중치)도 그 프로파일을
# 쓴다 (--mwpm-profile 자동 전달). 학습 인자에 --solution이 있으면 analyze에도
# 자동 전파된다. QPU 단계는 keys.json이 있을 때만 실행되고, 리포트는
# results/hardware/<backend>_<code>_<timestamp>.csv로 남는다.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

pcfg() {  # config.json pipeline 섹션 읽기 (활성화 전이므로 시스템 python3)
    python3 - "$1" "$2" <<'PY'
import json, sys
v = json.load(open("config.json")).get("pipeline", {}).get(sys.argv[1])
if v is None:
    v = sys.argv[2]
print(" ".join(v) if isinstance(v, list) else v)
PY
}

CONDA_ENV="$(pcfg conda_env HHC)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

set -u
export PYTHONUNBUFFERED=1

CODES="$(pcfg codes 'heavyhex surface')"
QPU_RUNS="$(pcfg qpu_runs 5)"
SHOTS="$(pcfg shots 50000)"
TRAIN_ARGS="$(pcfg train_args '')"
LOOPS="${LOOPS:-1}"
LOOP="${LOOP:-1}"

SMOKE_ARGS="" HW_EXTRA=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_ARGS="--smoke"
    HW_EXTRA="--dry-run"
    QPU_RUNS=1
fi

backend_of() {  # 코드 -> QPU 백엔드 (config.json pipeline.backends)
    python3 - "$1" <<'PY'
import json, sys
b = json.load(open("config.json")).get("pipeline", {}).get("backends", {})
print(b.get(sys.argv[1],
            {"heavyhex": "ibm_boston", "surface": "ibm_miami"}[sys.argv[1]]))
PY
}

gate_of() {     # 코드 -> 규약 게이트 스크립트
    case "$1" in
        heavyhex) echo "verification/verify_equivalence.py" ;;
        surface)  echo "verification/verify_rotatedSurface3.py" ;;
    esac
}

# 이전 루프 잡이 락을 놓기 전에 다음 루프가 스케줄될 수 있으므로,
# 즉시 실패(-n) 대신 최대 2시간 대기 — 루프 체인이 레이스로 죽지 않게.
exec 9>/tmp/heavyhex_pipeline.lock
flock -w 7200 9 || { echo "2시간 내 락 획득 실패 — 중단"; exit 1; }

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} env=$CONDA_ENV"
echo "loop=$LOOP/$LOOPS qpu_runs=$QPU_RUNS shots=$SHOTS codes=$CODES smoke=${SMOKE:-0}"
echo "train_args=$TRAIN_ARGS $SMOKE_ARGS $*"

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
    python dataset_generation/make_dataset.py --code "$code" $SMOKE_ARGS
    if [[ -n "${PROFILE[$code]}" ]]; then
        echo "--- dataset: $code (${PROFILE[$code]}) ---"
        python dataset_generation/make_dataset.py --code "$code" \
            -n "${PROFILE[$code]}" $SMOKE_ARGS
    fi
done

echo "=== [4/5] 학습 (epoch-resume 이어서) ==="
for code in $CODES; do
    echo "--- train: $code (realistic 그리드) ---"
    python train.py --code "$code" $TRAIN_ARGS $SMOKE_ARGS "$@"
    if [[ -n "${PROFILE[$code]}" ]]; then
        echo "--- train: $code (${PROFILE[$code]}) ---"
        python train.py --code "$code" $TRAIN_ARGS $SMOKE_ARGS "$@" \
            --config config.json -n "${PROFILE[$code]}"
    fi
done

echo "=== [5/5] QPU 검증 (코드별 submit -> wait -> analyze x ${QPU_RUNS}) ==="
HW_SOLUTION=""
case " $TRAIN_ARGS $* " in *" --solution "*) HW_SOLUTION="--solution" ;; esac
if [[ ! -f keys.json ]]; then
    echo "keys.json 없음 — QPU 단계 건너뜀 (자격 증명이 있어야 제출 가능)"
else
    for code in $CODES; do
        backend="$(backend_of "$code")"
        for i in $(seq 1 "$QPU_RUNS"); do
            echo "--- QPU: $code @ $backend (run $i/$QPU_RUNS) ---"
            python hardware/run_hw.py --backend "$backend" --code "$code" \
                --shots "$SHOTS" $HW_SOLUTION $HW_EXTRA \
                ${PROFILE[$code]:+--mwpm-profile "${PROFILE[$code]}"}
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
