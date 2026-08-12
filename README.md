# Heavy-Hex (3,3) Surface Code — CNN 디코더 프로젝트

Stim 시뮬레이션 → CNN 디코더 학습 → MWPM 기준선 → IBM QPU 검증까지
이어지는 파이프라인이야. **전체 파이프라인은 이미 완성돼 있고, 딱 하나
[model/cnn_skeleton.py](model/cnn_skeleton.py)의 CNN 모델만 비어 있어.**
이 파일만 채우면 아래 모든 단계가 그대로 돌아가.

## 1. 목표

(3,3) heavy-hex surface code (17 data + 8 dual-use ancilla, 16 stabilizer)의
측정 syndrome을 디코딩하는 **dual-head CNN**을 학습시키는 과제야.
파이프라인은 고정돼 있고, 너가 하는 일은 **CNN 설정(모델 구조 +
하이퍼파라미터)을 바꿔서 val LER을 낮추는 것.**

- **목표 지표는 하나: val/test head-LER** (logical head의 LER).
  체크포인트는 best val LER 기준으로 저장되고, 최종 성적은 그 체크포인트의
  test head-LER이야.
- **MWPM 기준선(PyMatching)은 넘어야 할 목표**: 설정을 바꿔 가며 MWPM에
  얼마나 다가가는가/넘어서는가가 과제의 서사고, 네 결과는
  **`LER/MWPM ratio` 한 숫자**로 비교돼 (같은 데이터에 대한 MWPM LER 대비
  CNN head-LER 비율; 1.0 미만이면 MWPM을 이긴 것).
- 나머지 지표들은 **진단용**이야 (순위에 안 들어감):
  - `ECR (diagnostic, sim-only)` — per-qubit head의 에러 검출률.
    per-qubit 라벨이 필요해서 시뮬레이션 전용.
  - `parity_LER (diagnostic)` — per-qubit head 예측 마스크의 LOGICAL_Z
    parity로 유도한 LER. 모델이 정말 정정을 배웠는지, logical 분류를
    지름길로 배웠는지 보는 용도. head-LER은 좋은데 parity_LER이 나쁘면
    aux loss 가중치(`--aux-weight`) 조정을 검토해 볼 것.
- (QPU 접근이 가능하면) 학습한 디코더를 **실제 IBM QPU** 데이터에
  적용해서 LER 보고하기

### 채워야 할 부분 (전부 `model/cnn_skeleton.py` 안에 있음)

1. `HeavyHexCNN.__init__` — conv feature 블록 + shared FC 레이어
2. `HeavyHexCNN.__init__` / `forward` — 두 head(17-logit per-qubit,
   1-logit logical)와 forward 경로
3. `compute_loss` — LER 우선 loss: `BCE(logical)` 주 손실 +
   `aux_weight * BCE(per-qubit)` 보조 손실 (기본 `aux_weight=0.5`)

인터페이스는 바꾸지 말아주세요 — 학습/평가/QPU 스크립트가 이 함수들을
그대로 호출해서 바꾸면 망가져용.

### Results

- 학습 곡선 (`results/train/`에 쌓이는 epoch별 CSV를 플롯하면 됌)
- config별 best 모델의 검증 **head-LER** (+ `parity_LER (diagnostic)`,
  `ECR (diagnostic, sim-only)` 진단 컬럼)
- **MWPM 대비 `LER/MWPM ratio`** (`train.py --mwpm`이 표로 출력해줘;
  MWPM을 안 돌린 실행에서는 N/A로 표기됨)
- (QPU 접근이 가능하면) `hardware/run_hw.py analyze`의 QPU LER 리포트

### 공정 비교 규칙 (결과 비교는 이 조건에서만 유효)

- **고정 (변경 금지)**:
  - 데이터셋 생성 설정 (`dataset_generation/`, 노이즈 그리드, 샘플 수)
  - train/val/test 분할 (독립 생성된 train/test 파일; test 파일이 검증셋)
  - 평가 스크립트 (`evaluation/`, `train.py`, `hardware/run_hw.py`,
    `baseline/`)
  - 체크포인트 선택 기준 (**best val LER**)
- **변경 허용**:
  - [model/cnn_skeleton.py](model/cnn_skeleton.py)의 모델 구조 전부
  - 하이퍼파라미터 (learning rate, batch size, aux loss 가중치 등)
- **최종 성적**: best-val-LER 체크포인트의 **test head-LER**
  (및 `LER/MWPM ratio`)

## 2. 환경 설정

```bash
conda create -n 환경이름 python=3.12 pip
conda activate 환경이름
pip install -r requirements.txt
```

- GPU는 CUDA GPU면 아무거나 괜찮고, 서버에는 GTX 2080 Ti 2장이 있어 (6. slurm 참고)
- 만약 conda가 뭔지 모르겠으면 물어봐줘

## 3. 실행 순서

```bash
# 0) 규약 게이트 — "ALL PASS" 확인하고 나서 데이터 생성으로 넘어가기
python verification/verify_equivalence.py

# 1) 데이터셋 생성
python dataset_generation/make_dataset.py --smoke   # 빠른 확인용
python dataset_generation/make_dataset.py           # train_sweep.json 있으면 그 조합,
                                                    # 없으면 전체 그리드 (용량/시간 꽤 큼)
#    작게 줄이고 싶으면 예 (기본 샘플 수는 train_options.json의 dataset 섹션으로도 조절 가능):
python dataset_generation/make_dataset.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 \
       -p 0.005 --train-samples 1000000 --test-samples 100000

# 2) 모델 학습 (채우기 전에는 NotImplementedError로 멈출 수 있음)
python train.py --smoke                        # end-to-end 확인
python train.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 -p 0.005 --mwpm
python train.py --all --mwpm                   # 전체 그리드 + 기준선 표
python train.py                                # train_sweep.json 있으면 자동 스윕
python train.py --config none                  # 스윕 끄고 기본 단일 설정

# 3) MWPM 기준선 표만 따로 보고 싶을 때
python baseline/mwpm.py

# 4) QPU 검증 (keys.json 필요, 5. 참고)
python hardware/run_hw.py submit --backend ibm_yonsei --dry-run   # 리허설
python hardware/run_hw.py submit --backend ibm_yonsei             # 실제 제출
python hardware/run_hw.py analyze --job-id <ID>   # checkpoint/*.pt 전부 평가
python hardware/run_hw.py analyze --job-id <ID> \
       --ckpt checkpoint/CNN_d3_c3_p0.005_dp0.001_mf0.01_rf0.01_gd0.008.pt
python hardware/run_hw.py all                     # 제출→완료 대기→분석 원샷
```

analyze는 잡이 아직 안 끝났으면 알아서 폴링하며 기다렸다가 진행해.

### 설정 파일 두 개 (repo 루트)

- **[train_options.json](train_options.json)** — *기본값* 조절: `train` 섹션은
  학습 하이퍼파라미터(epochs, batch_size, lr, ...), `dataset` 섹션은
  데이터셋 샘플 수(train_samples/test_samples), 최상위 `cycles`는 양쪽 공용.
  파일이 있으면 train.py / make_dataset.py가 자동으로 읽어 기본값을
  대체하고, CLI 인자를 명시하면 그쪽이 이겨.
- **[train_sweep.json](train_sweep.json)** — *스윕* 정의: `runs`의 각 항목이
  (노이즈, p, error_type + 하이퍼파라미터 오버라이드) 한 벌이야.
  repo 루트에 이 파일이 있으면 make_dataset.py는 필요한 조합의 데이터셋을,
  train.py는 항목별 학습을 **자동으로** 돌려. 선택 인자(-n/-p/-e/--all/
  --smoke)를 명시하면 스윕 대신 그쪽이 돌고, `--config none`으로 끄거나
  `--config 다른파일.json`으로 바꿀 수 있어. 같은 (노이즈, p, cycles)의
  하이퍼파라미터 변형에는 `"name"`을 줘서 결과 파일 이름을 구분해.

우선순위: `train_options.json` < CLI 인자 < 스윕 run 항목.

### 산출물이 저장되는 위치

- 데이터셋: `dataset/<노이즈태그>/{train,test}_d3_c3_p<p>_X.npz`
  (예: `dataset/dp0.001_mf0.01_rf0.01_gd0.008/train_d3_c3_p0.005_X.npz`;
   `d3` = code distance (dx/dz 중 큰 값; (3,3)은 3, (3,5)/(5,3)이면 5),
   `c3` = QEC cycle 수 3)
- 학습 로그: `results/train/CNN_d3_c<cycles>_p<p>_<노이즈태그>.csv`
- 체크포인트: `checkpoint/CNN_d3_c<cycles>_p<p>_<노이즈태그>.pt`
  (epoch마다 검증 LER을 재서 **최저 val LER**일 때만 갱신돼)
- QPU 런: `hardware/runs/<job_id>/` — raw 결과와 함께 그 시점의 QPU
  환경 기록(캘리브레이션 스냅샷, 제출한 회로 등)이 통째로 남아
- QPU LER 리포트: `results/hardware/hw_<job_id>.csv`
  (`hardware/run_hw.py analyze`가 표로 출력한 내용을 저장)

## 4. 규약으로 고정된 부분 (건드리지 말 것)

Stim 회로와 QPU 회로는 check-value 수준에서 **비트 단위로** 일치해야 해:

- cycle당 측정 순서 = `heavyhex_depth7_opt_for_37q.CYCLE_ORDER`
  (16개 check; syn 비트 인덱스 = `cycle*16 + j`)
- stabilizer support = `heavyhex_37q.CHECK_DEFS`
- Stim은 ancilla를 `MR`(측정+리셋)로 처리하고, QPU는 no-reset
  ancilla의 raw 비트를 `heavyhex_37q.check_values()`의 XOR chain으로
  복원해 — 두 스트림은 check-value 수준에서 동일해
  (`dataset_generation/heavyhex33_stim.py` docstring 참고)
- detector: Z-check는 cycle 0부터 (결정론적 0에 앵커), X-check는 cycle 간
  XOR, 마지막에 final-data 기반 Z-detector 8개;
  observable = logical Z = data [69, 87, 105]의 parity
- 입력 텐서: `(2*num_cycles, 4, 5)` — ancilla 8개를 4×5 다이아몬드 격자에
  임베딩 (`ANC_COORD`, rung 정의에서 유도), 채널은
  `[Z-plane, X-plane] × cycle`

`verification/verify_equivalence.py`가 이 전부를 검사해줘 (Stim 결정론, 무노이즈 Aer 불변량, 단일 에러 서명 일치).

### 라벨에 대한 주의 (중요한 미묘함)

dual-use ancilla가 X-check도 측정하기 때문에 |0⟩_L이 X-stabilizer
고유상태로 사영돼. 그래서 final data의 **개별** 비트는 무작위이고,
결정론적인 건 Z-stabilizer / logical-Z parity뿐이야. 그래서 per-qubit
라벨은 final-data의 **측정 flip**(무노이즈 기준 대비 에러 프레임,
`stim.FlipSimulator`로 샘플링)을 쓰고, logical 라벨은 그 [69, 87, 105]
parity를 써. 이건 실제 측정 비트의 parity와 정확히 같아. ECR(진단용)은
이 per-qubit 라벨이 있어야 해서 **시뮬레이션 전용**이고, LER(목표 지표)과
parity_LER(진단용)은 시뮬레이션/QPU 양쪽에서 계산할 수 있어.

## 5. keys.json (QPU 접근) (이거 만들어야 함!)

```bash
cp keys.example.json keys.json     # 그리고 네 거로 적으면 됨
```

```json
{"ibm_token": "<IBM Quantum Platform API key>",
 "ibm_instance": "<IBM Cloud instance CRN>"}
```

`keys.json`은 gitignore 되어 있어 — **토큰을 커밋하거나 코드에 하드코딩하지
말 것.** 기본 백엔드는 `ibm_yonsei`이고, `--backend ibm_boston`으로 바꿀 수
있어. 일단 `ibm_yonsei` 해보고 결과가 도저히 안 나오면 `ibm_boston`으로 해 볼 것~

백엔드별 37q 패치의 물리 큐빗 임베딩은
`heavyhex_circuits/heavyhex_37q.py`의 `EMBEDDINGS`에 등록돼 있어
(boston/aachen/pittsburgh는 Heron 번호 그대로, yonsei는 Eagle r3 레이아웃
매핑). `validate_backend`가 임베딩을 자동 선택해서 커플링을 검사하고,
목록에 없는 백엔드는 매핑을 추가하라는 에러가 나.

## 6. Slurm (서버)

sbatch 스크립트는 repo 루트에 두 개 있어 (파티션은 서버의 `main`으로 이미
설정돼 있음). 먼저 [train.sbatch](train.sbatch)의 `CONDA_ENV`를 본인 환경
이름으로 바꾸고 (또는 제출할 때 `CONDA_ENV=이름`으로 덮어쓰기), **repo
루트에서** 제출해줘 — 로그가 [slurm_logs/](slurm_logs/)에 쌓여.

```bash
# 학습만 (데이터셋이 이미 있을 때)
sbatch train.sbatch --all --mwpm            # 인자는 train.py로 그대로 전달돼

# 통합 파이프라인: 규약 게이트 -> 데이터셋 생성 -> 학습 -> QPU 검증
sbatch pipeline.sbatch --all --mwpm
DATASET_ARGS="--smoke" sbatch pipeline.sbatch --smoke   # 빠른 end-to-end 확인
# train_sweep.json이 있으면 데이터셋 생성+학습이 자동으로 스윕을 돈다:
SWEEP_CONFIG=다른스윕.json sbatch pipeline.sbatch       # 다른 스윕 파일 지정
SWEEP_CONFIG=none sbatch pipeline.sbatch                # 스윕 끄기
HW_ARGS="--dry-run" sbatch pipeline.sbatch              # QPU 단계는 리허설만
SKIP_HW=1 sbatch pipeline.sbatch                        # QPU 단계 끄기
```

QPU 단계는 keys.json이 있을 때만 돌고, `hardware/run_hw.py all`
(제출→대기→분석)을 실행해 — **실제 QPU 잡이 제출되니** 반복 제출을
원치 않으면 `SKIP_HW=1`을 쓸 것. HW_ARGS로 `--backend`, `--shots` 등을
넘길 수 있어.

이미 생성된 데이터셋 파일은 make_dataset.py가 알아서 건너뛰니까
pipeline.sbatch를 다시 제출해도 데이터 생성이 중복되지 않아.

pipeline.sbatch는 flock으로 중복 실행을 차단해 — 파이프라인 job 2개가
동시에 돌면 같은 npz 파일을 동시에 써서 데이터가 깨질 수 있기 때문.

## 7. 저장소 구조

```
heavyhex_circuits/      고정된 회로 자산 (재작성하지 말고 import해서 쓸 것)
  heavyhex_37q.py                 (3,3) 코드 정의: CHECK_DEFS, DATA_PHYS,
                                  LOGICAL_Z, check_values, validate_backend,
                                  EMBEDDINGS (백엔드별 큐빗 임베딩)
  heavyhex_depth7_opt_for_37q.py  최적화된 depth-7 QPU 회로
                                  (HeavyHex37QDepthOpt, CYCLE_ORDER)
  heavyhex_general.py / heavyhex_depth_opt.py / diamond_generator.py
                                  (3,5)/(5,3) 패치용 생성기; (3,3)은 위의
                                  두 파일이 더 최적화돼 있으니 그쪽을 쓸 것
  fetch_coupling.py               백엔드 coupling map 추출 (가장 먼저 실행)
  dd_utils.py                     dynamical decoupling (기본 XX4)
dataset_generation/     Stim 회로 생성기(heavyhex33_stim.py) +
                        데이터셋 생성(make_dataset.py)
verification/           규약 게이트 스크립트 (§4)
model/                  채워야 할 파일 (cnn_skeleton.py) + 데이터 로더
evaluation/             지표 — head-LER(목표 지표) + ECR/parity_LER(진단용)
baseline/               MWPM (PyMatching) 기준선
train.py                학습 진입점 (완성본, 수정할 필요 없어)
train_options.json      기본 하이퍼파라미터 / 데이터셋 샘플 수 (자동 적용, §3)
train_sweep.json        스윕 정의 — 있으면 자동 적용, --config none으로 끔 (§3)
train.sbatch            slurm 학습 job (§6)
pipeline.sbatch         slurm 통합 파이프라인:
                        게이트 -> 데이터셋 -> 학습 -> QPU 검증 (§6)
slurm_logs/             slurm job 로그 (내용물은 gitignore)
hardware/               IBM 제출 + 분석 파이프라인 (runs/에 런별 기록)
results/                train/ 학습 CSV, hardware/ QPU LER CSV (gitignore)
```

QPU 회로 흐름 (원본 저장소 README에서): `fetch_coupling` →
surface code 회로 생성 → `transpile` → DD — `hardware/run_hw.py`가 정확히
이 순서로 되어 있어.

참고: `heavyhex_circuits/` 안의 회로 테스트 중 coupling map을 쓰는 것들
(test_general.py, heavyhex_depth_opt.py 데모)은 fetch_coupling.py로
coupling JSON을 먼저 생성한 뒤 실행해야 해 (test_37q.py /
test_depth7_opt_for_37q.py는 Aer만 쓰므로 바로 실행 가능).
