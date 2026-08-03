# Heavy-Hex (3,3) Surface Code — CNN 디코더 프로젝트

Stim 시뮬레이션 → CNN 디코더 학습 → MWPM 기준선 → IBM QPU 검증까지
이어지는 파이프라인이야. **전체 파이프라인은 이미 완성돼 있고, 딱 하나
[model/cnn_skeleton.py](model/cnn_skeleton.py)의 CNN 모델만 비어 있어.**
이 파일만 채우면 아래 모든 단계가 그대로 돌아가.

## 1. 목표

(3,3) heavy-hex surface code (17 data + 8 dual-use ancilla, 16 stabilizer)의
측정 syndrome을 디코딩하는 게 목표:

- **dual-head CNN** 학습:
  - per-qubit head (17 logits) — **ECR**로 평가
  - logical head (1 logit) — **LER**로 평가
- 시뮬레이션 데이터에서 **MWPM 기준선**(PyMatching) 이기거나 근접시키기
- 학습한 디코더를 **실제 IBM QPU** 데이터에 적용해서 LER 보고하기

### 채워야 할 부분 (전부 `model/cnn_skeleton.py` 안에 있음)

1. `HeavyHexCNN.__init__` — conv feature 블록 + shared FC 레이어
2. `HeavyHexCNN.__init__` / `forward` — 두 head(17-logit per-qubit,
   1-logit logical)와 forward 경로
3. `compute_loss` — LER 우선 loss: `BCE(logical)` 주 손실 +
   `aux_weight * BCE(per-qubit)` 보조 손실 (기본 `aux_weight=0.5`)

인터페이스는 바꾸지 말아주세요 — 학습/평가/QPU 스크립트가 이 함수들을
그대로 호출해서 바꾸면 망가져용.

### Results

- 학습 곡선 (`results/`에 쌓이는 epoch별 CSV를 플롯하면 됌)
- config별 best 모델의 검증 **ECR / LER**
- **MWPM 대비표** (`train.py --mwpm`이 표로 출력해줘)
- (QPU 접근이 가능하면) `hardware/run_hw.py analyze`의 QPU LER 리포트

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
python dataset_generation/make_dataset.py           # 전체 그리드 (용량/시간 꽤 큼)
#    작게 줄이고 싶으면 예:
python dataset_generation/make_dataset.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 \
       -p 0.005 --train-samples 1000000 --test-samples 100000

# 2) 모델 학습 (채우기 전에는 NotImplementedError로 멈출 수 있음)
python train.py --smoke                        # end-to-end 확인
python train.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 -p 0.005 --mwpm
python train.py --all --mwpm                   # 전체 그리드 + 기준선 표

# 3) MWPM 기준선 표만 따로 보고 싶을 때
python baseline/mwpm.py

# 4) QPU 검증 (keys.json 필요, 5. 참고)
python hardware/run_hw.py submit --backend ibm_yonsei --dry-run   # 리허설
python hardware/run_hw.py submit --backend ibm_yonsei             # 실제 제출
python hardware/run_hw.py analyze --job-id <ID> \
       --ckpt checkpoint/CNN_d3_c3_p0.005_dp0.001_mf0.01_rf0.01_gd0.008.pt
```

### 산출물이 저장되는 위치

- 데이터셋: `dataset/<노이즈태그>/{train,test}_d3_c3_p<p>_X.npz`
  (예: `dataset/dp0.001_mf0.01_rf0.01_gd0.008/train_d3_c3_p0.005_X.npz`;
   `d3` = code distance (dx/dz 중 큰 값; (3,3)은 3, (3,5)/(5,3)이면 5),
   `c3` = QEC cycle 수 3)
- 학습 로그: `results/CNN_d3_c<cycles>_p<p>_<노이즈태그>.csv`
- 체크포인트: `checkpoint/CNN_d3_c<cycles>_p<p>_<노이즈태그>.pt`
  (epoch마다 검증 LER을 재서 **최저 val LER**일 때만 갱신돼)
- QPU 런: `hardware/runs/<job_id>/` — raw 결과와 함께 그 시점의 QPU
  환경 기록(캘리브레이션 스냅샷, 제출한 회로 등)이 통째로 남아

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
parity를 써. 이건 실제 측정 비트의 parity와 정확히 같아. ECR은 이
per-qubit 라벨이 있어야 해서 **시뮬레이션 전용**이고, LER은
시뮬레이션/QPU 양쪽에서 계산할 수 있어.

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

## 6. Slurm (서버)

sbatch 스크립트는 repo 루트에 두 개 있어 (파티션은 서버의 `main`으로 이미
설정돼 있음). 먼저 [train.sbatch](train.sbatch)의 `CONDA_ENV`를 본인 환경
이름으로 바꾸고 (또는 제출할 때 `CONDA_ENV=이름`으로 덮어쓰기), **repo
루트에서** 제출해줘 — 로그가 [slurm_logs/](slurm_logs/)에 쌓여.

```bash
# 학습만 (데이터셋이 이미 있을 때)
sbatch train.sbatch --all --mwpm            # 인자는 train.py로 그대로 전달돼

# 통합 파이프라인: 규약 게이트 -> 데이터셋 생성 -> 학습
sbatch pipeline.sbatch --all --mwpm
DATASET_ARGS="--smoke" sbatch pipeline.sbatch --smoke   # 빠른 end-to-end 확인
```

이미 생성된 데이터셋 파일은 make_dataset.py가 알아서 건너뛰니까
pipeline.sbatch를 다시 제출해도 데이터 생성이 중복되지 않아.

pipeline.sbatch는 flock으로 중복 실행을 차단해 — 파이프라인 job 2개가
동시에 돌면 같은 npz 파일을 동시에 써서 데이터가 깨질 수 있기 때문.

## 7. 저장소 구조

```
heavyhex_circuits/      고정된 회로 자산 (재작성하지 말고 import해서 쓸 것)
  heavyhex_37q.py                 (3,3) 코드 정의: CHECK_DEFS, DATA_PHYS,
                                  LOGICAL_Z, check_values, validate_backend
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
evaluation/             ECR / LER 지표
baseline/               MWPM (PyMatching) 기준선
train.py                학습 진입점 (완성본, 수정할 필요 없어)
train.sbatch            slurm 학습 job (§6)
pipeline.sbatch         slurm 통합 파이프라인: 게이트 -> 데이터셋 -> 학습 (§6)
slurm_logs/             slurm job 로그 (내용물은 gitignore)
hardware/               IBM 제출 + 분석 파이프라인 (runs/에 런별 기록)
```

QPU 회로 흐름 (원본 저장소 README에서): `fetch_coupling` →
surface code 회로 생성 → `transpile` → DD — `hardware/run_hw.py`가 정확히
이 순서로 되어 있어.

참고: `heavyhex_circuits/` 안의 회로 테스트 중 coupling map을 쓰는 것들
(test_general.py, heavyhex_depth_opt.py 데모)은 fetch_coupling.py로
coupling JSON을 먼저 생성한 뒤 실행해야 해 (test_37q.py /
test_depth7_opt_for_37q.py는 Aer만 쓰므로 바로 실행 가능).
