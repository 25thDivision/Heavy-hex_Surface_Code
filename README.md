# Heavy-Hex / Rotated-Surface Code — 디코더 프로젝트

Stim 시뮬레이션 → 디코더(CNN/GNN) 학습 → MWPM 기준선 → IBM QPU 검증까지
이어지는 파이프라인이야. **전체 파이프라인은 이미 완성돼 있고,
[model/cnn_skeleton.py](model/cnn_skeleton.py) (그리고 GNN을 쓰려면
model/gnn_skeleton.py)의 모델만 비어 있어.** 이 파일만 채우면 아래 모든
단계가 그대로 돌아가.

> 구버전 문서(heavyhex/CNN 단일 축 시절)는
> [README_old.md](README_old.md)로 남겨 뒀어 — 이 문서가 그 전체 내용에
> **모델 축(`--model {cnn,gnn}`)과 코드 축(`--code {heavyhex,surface}`)**
> 등 신규 기능을 통합한 최신판이야.

## 1. 목표

(3,3) heavy-hex surface code (17 data + 8 dual-use ancilla, 16 stabilizer)의
측정 syndrome을 디코딩하는 **dual-head 디코더**를 학습시키는 과제야.
파이프라인은 고정돼 있고, 너가 하는 일은 **모델 설정(모델 구조 +
하이퍼파라미터)을 바꿔서 val LER을 낮추는 것.**

- **목표 지표는 하나: val/test head-LER** (logical head의 LER).
  체크포인트는 best val LER 기준으로 저장되고, 최종 성적은 그 체크포인트의
  test head-LER이야.
- **MWPM 기준선(PyMatching)은 넘어야 할 목표**: 설정을 바꿔 가며 MWPM에
  얼마나 다가가는가/넘어서는가가 과제의 서사고, 네 결과는
  **`LER/MWPM ratio` 한 숫자**로 비교돼 (같은 데이터에 대한 MWPM LER 대비
  head-LER 비율; 1.0 미만이면 MWPM을 이긴 것).
- 나머지 지표들은 **진단용**이야 (순위에 안 들어감):
  - `ECR (diagnostic, sim-only)` — per-qubit head의 에러 검출률.
    per-qubit 라벨이 필요해서 시뮬레이션 전용.
  - `parity_LER (diagnostic)` — per-qubit head 예측 마스크의 LOGICAL_Z
    parity로 유도한 LER. 모델이 정말 정정을 배웠는지, logical 분류를
    지름길로 배웠는지 보는 용도. head-LER은 좋은데 parity_LER이 나쁘면
    aux loss 가중치(`--aux-weight`) 조정을 검토해 볼 것.
- (QPU 접근이 가능하면) 학습한 디코더를 **실제 IBM QPU** 데이터에
  적용해서 LER 보고하기

### 모델 축: `--model {cnn,gnn}`

같은 데이터셋/평가/하드웨어 파이프라인 위에서 두 아키텍처를 선택할 수 있어:

- **cnn** — 4×5 다이아몬드 격자 텐서를 그대로 받는 dual-head CNN
  ([model/cnn_skeleton.py](model/cnn_skeleton.py)).
- **gnn** — detector를 노드로 하는 그래프 신경망
  ([model/gnn_skeleton.py](model/gnn_skeleton.py)). **데이터셋 파일은 CNN과
  공용**이고 (npz 포맷 무변경), 그래프 변환은 모델 쪽에서 일어나.

모델 로딩은 [model/__init__.py](model/__init__.py)의 레지스트리
(`get_model_module(model_name, use_solution)`)로 일원화돼 있고, 결과/체크포인트
파일 이름은 `{MODEL}_{tag}` 형식이야 (예: `CNN_heavyhex_d3_c3_...`,
`GNN_heavyhex_d3_c3_...`). config.json `sweep` 섹션의 run 항목에 `"model"` 키를 주면
한 스윕에서 cnn/gnn을 같은 데이터로 이어 학습해 **CNN/GNN/MWPM이 한 표에**
나오는 요약을 얻을 수 있어.

#### GNN 그래프 표현 (그리고 CNN과의 입력 비대칭)

[model/graph.py](model/graph.py)의 `GraphBuilder`가 (code, cycles)별 정적
그래프를 만들어:

- **노드 = Stim detector**, 순서는 `heavyhex33_stim._append_detectors`와
  동일 (MWPM이 소비하는 순서 그대로). c=3 기준 48개:
  Z-check 8×3, X-check 8×2, **final-Z 8**.
- **노드 피처**: [detector 값, check 타입 one-hot(Z/X/final),
  cycle 정규화(final=1.0), ANC_COORD 기반 (row,col) 정규화 좌표].
- **엣지 (정적, (code, cycles)별 사전 계산)**:
  공간(같은 cycle에서 두 check가 data 큐빗 공유),
  시간(같은 check의 인접 cycle),
  final-Z(final-Z 노드끼리 support 공유 + 자신의 마지막 cycle Z-check).
- **입력 증강**: final-Z detector 값(= final-data Z-support parity ^ 마지막
  cycle Z-check)은 `(2C,4,5)` 텐서에 없는 정보라, `--model gnn`일 때
  train.py / run_hw.py가 `augment_features`로 **채널 1개를 추가**한
  `(B, 2C+1, 4, 5)` 텐서를 만들어 forward에 넘겨줘 (시뮬레이션은 npz의
  `labels`(측정 flip)에서, 하드웨어는 측정된 final data 비트에서 계산 —
  둘 다 MWPM의 detector 재구성과 비트 단위로 동일).

**final-Z 노드는 라벨 누출이 아니야**: final-Z detector는 final 측정의
Z-stabilizer parity이고, logical Z(data [69,87,105])는 Z-stabilizer 곱으로
표현될 수 없어서 목표 지표(head-LER)의 라벨이 유도되지 않아. MWPM이 쓰는
입력과 정확히 같은 정보야. 다만 **CNN 텐서에는 final-data 유도 신드롬이
없으므로 CNN과 GNN의 입력은 비대칭**이고, CNN/GNN 비교를 읽을 때 이 차이를
감안해야 해 (CNN은 in-run 신드롬만, GNN은 in-run + final-round 신드롬).

GNN 모델 자체는 순수 torch로 구현해 (torch_geometric 등 그래프 라이브러리
금지) — 노드 수가 작아서 dense adjacency matmul(`self.adj @ h`) 기반 MPNN
3~4층 + pooling + shared FC → dual head면 충분해.

### 코드 축: `--code {heavyhex,surface}`

- **heavyhex** (기본) — 이 저장소의 원래 (3,3) heavy-hex 코드 (17 data +
  8 dual-use ancilla, QPU는 bridge 포함 37큐빗).
- **surface** — rotated surface code d=3 (9 data + 8 ancilla = 17큐빗,
  Z-stab 4 + X-stab 4). 아래 rotatedSurface3 섹션 참고.

데이터셋은 `dataset/<code>/<노이즈태그>/...`에 저장되고, 결과/체크포인트
태그에도 코드 이름이 들어가 (`CNN_heavyhex_...`, `GNN_surface_...`).

#### rotated surface code d=3 (rotatedSurface3) 규약

코드 정의는 [circuits/rotatedSurface/rotatedSurface3.py](circuits/rotatedSurface/rotatedSurface3.py)에 있어:

- **격자**: data는 odd-odd 좌표 (x,y)∈{1,3,5}², ancilla 8개는 plaquette
  중심 (짝수 좌표). logical Z = **한 열(x=1)의 data 3개**
  {(1,1),(1,3),(1,5)}, logical X = 한 행(y=1). 메모리-Z 프로토콜, 기본
  cycles=3.
- **CX 순서 (상수로 고정, hook-safe)**: 사이클마다 모든 stabilizer가 4개
  공유 레이어에서 CX를 실행하고, 레이어별 코너는
  `Z_CORNER_ORDER`("Z자": 아랫줄→윗줄) / `X_CORNER_ORDER`("N자":
  왼쪽열→오른쪽열)를 따라. 이 조합에서 X-ancilla hook 에러의 잔여쌍은
  **수직**(logical X 방향으로 진행 없음), Z-ancilla hook은 **수평**이라
  한 번의 fault가 유효 distance를 깎지 않아. 스케줄은 레이어당 data 충돌
  없음이 import 시점에 assert돼.
- **no-reset ancilla + XOR chain**: 하드웨어 회로는 ancilla를 리셋하지
  않고 `rotatedSurface3.check_values()`(per-ancilla XOR chain)로 check 값을 복원.
  추상 Stim([dataset_generation/rotatedSurface3_stim.py](dataset_generation/rotatedSurface3_stim.py))은
  MR을 쓰며 **check-value 수준 등가성** 규약이 heavyhex와 동일하게 적용돼.
- **detector 규약** (heavyhex와 동일 구조): Z-check는 cycle 0부터
  (|0⟩_L의 결정론적 0에 앵커), X-check는 cycle≥1 XOR, 마지막에 final-Z
  detector 4개, observable = logical Z. 라벨은 FlipSimulator 측정 flip
  (per-qubit 9비트).
- **CNN 텐서**: `(2*cycles, 4, 4)` — ancilla 8개를 (d+1)×(d+1)=4×4
  plaquette-꼭짓점 격자에 임베딩 (`rotatedSurface3.ANC_GRID`), 채널은 heavyhex와
  동일한 [Z-plane, X-plane]×cycle.
- **GNN**: P1의 detector-node 표현이 그대로 적용돼 (c=3 기준 노드 24개 =
  Z 4×3 + X 4×2 + final-Z 4).
- **게이트**: `python verification/verify_rotatedSurface3.py`가 ALL PASS여야
  surface 데이터셋 생성/제출 가능 — Stim 결정론, 무노이즈 Aer 불변량,
  단일 data 에러 서명 일치에 더해 **hook error 검사**(사이클 중간
  ancilla 에러 주입 시 data 전파 서명이 CX 순서 규약의 예측 잔여쌍과
  Stim/Aer 양쪽에서 비트 단위 일치)까지 확인해.

#### ibm_miami 하드웨어 (surface)

ibm_miami는 **12행×10열 row-major square lattice**(120큐빗, CZ basis)로
확인됐고, rotatedSurface3는 **45도 임베딩**(u=(x+y)/2, v=(y−x)/2 + 오프셋, 5×5
블록)으로 올라가 — 모든 stabilizer CX가 lattice-인접이라 **SWAP이 전혀
삽입되지 않아** (dry-run 검증: 2Q 게이트 수 = CX 수 그대로 CZ 72개).
임베딩 오프셋은 `rotatedSurface3.EMBED_OFFSETS`, 검증은
`rotatedSurface3.validate_backend_surface`가 담당하고, coupling map이 예상(square
lattice)과 다르면 에러로 중단돼.

```bash
python hardware/run_hw.py submit --backend ibm_miami --code surface --dry-run
python hardware/run_hw.py submit --backend ibm_miami --code surface   # 실제 제출
python hardware/run_hw.py analyze --job-id <ID>    # code는 job.json에서 자동 인식
```

analyze는 job.json의 code를 자동으로 읽어 rotatedSurface3용 check-value 복원 /
4×4 텐서화 / MWPM / 체크포인트(`*_surface_*.pt`, CNN/GNN 자동 인식) 분기를 태워.
멀티 PUB 잡은 PUB 순서대로 이어붙여 분석하고, 리포트에 전체 합산 LER 옆
`ler_std_over_pubs`(PUB별 LER 표준편차) 컬럼이 붙어 (단일 PUB/구버전
데이터는 N/A).

**기존 산출물 이관**: 코드 축 도입 전에 만든 로컬 데이터셋은
`dataset/<노이즈태그>/`에 바로 있었어. 아래 한 줄로 heavyhex 산하로 옮기면 돼
(자동 마이그레이션은 없어):

```bash
mkdir -p dataset/heavyhex && mv dataset/dp* dataset/heavyhex/
```

기존 체크포인트/결과 CSV(`CNN_d3_...` — 코드 태그 없는 이름)는 그대로 두면
새 이름(`CNN_heavyhex_d3_...`)과 공존해. `hardware/run_hw.py analyze`의
자동 평가(`checkpoint/CNN_*.pt` glob)는 둘 다 잡으니, 헷갈리면 옛 파일을
새 이름 규칙으로 rename해 둘 것.

### 채워야 할 부분

CNN: 전부 [model/cnn_skeleton.py](model/cnn_skeleton.py) 안에 있음.

1. `HeavyHexCNN.__init__` — conv feature 블록 + shared FC 레이어
2. `HeavyHexCNN.__init__` / `forward` — 두 head(17-logit per-qubit,
   1-logit logical)와 forward 경로
3. `compute_loss` — LER 우선 loss: `BCE(logical)` 주 손실 +
   `aux_weight * BCE(per-qubit)` 보조 손실 (기본 `aux_weight=0.5`)

GNN: model/gnn_skeleton.py (P1 마일스톤에서 추가; 같은 인터페이스).

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
  - `model/<name>_skeleton.py`의 모델 구조 전부
  - 하이퍼파라미터 (learning rate, batch size, aux loss 가중치 등)
- **최종 성적**: best-val-LER 체크포인트의 **test head-LER**
  (및 `LER/MWPM ratio`)

### QPU 캘리브레이션 평균 프로파일 (`qpu/...`, mode: qpu_avg_v1)

실제 QPU 제출 기록(hardware/runs/의 스냅샷)에서 **캘리브레이션 평균
노이즈 프로파일**을 만들어, 하드웨어와 같은 구조의 Stim 회로로
데이터셋을 생성/학습할 수 있어. heavyhex와 surface **양쪽 코드 모두**
지원돼:

```bash
# 1) 프로파일 생성: 같은 백엔드·같은 코드의 non-dry-run 최신 N개(기본 5) 평균
python dataset_generation/make_qpu_avg_profile.py                    # heavyhex
python dataset_generation/make_qpu_avg_profile.py --code surface --backend ibm_miami
# -> config.json의 noise_profiles 섹션에 qpu/<backend>_<code>_avg<N>_<YYYYMMDD> 로 등록
#    (다른 섹션은 건드리지 않아)
#    (<YYYYMMDD> = 평균에 포함된 run들의 submitted_at 중 최신 날짜)

# 2) 코드별 게이트 ALL PASS 필수 -> 데이터셋 -> 학습
python verification/verify_equivalence.py    # heavyhex ([E] 포함)
python verification/verify_rotatedSurface3.py           # surface ([G] 포함)
python dataset_generation/make_dataset.py --code surface -n qpu/<이름> --smoke
python train.py --model gnn --code surface -n qpu/<이름> -p 0.005 --mwpm
```

- **추출값**: run별 target.pkl(폴백: properties.json)에서 큐빗별 readout
  error, 1Q error(sx 우선), 물리 엣지별 2Q error를 뽑아 run 간 산술 평균.
  패치 매핑은 코드별로: heavyhex는 `embedding_for(backend)`로 **37q 패치
  물리 라벨**, surface는 `rotatedSurface3.embedding_for_surface(backend)`로 **rotatedSurface3
  로컬 인덱스 0..16**(ALL_COORDS 순서: data 0–8, ancilla 9–16)에 기록.
  run 선택은 job.json의 backend/submitted_at/dry_run/**code**로 판별
  (code 필드가 없는 옛 run은 heavyhex로 간주, dry-run 제외, 백엔드 혼합
  금지, 부족하면 있는 만큼 평균 + 경고).
- **이름 규약과 계보 보호**: 키는
  `qpu/<backend>_<code>_avg<N>_<YYYYMMDD>[_<suffix>]`. 같은 키가 이미
  등록돼 있는데 provenance의 run id 목록이 지금 선택과 **다르면 덮어쓰지
  않고 에러로 중단**하며 차이를 출력해 (그 이름이 박힌 데이터셋/체크포인트의
  의미가 조용히 바뀌는 것 방지). 같은 run 목록이면 provenance만 갱신.
  구분이 필요하면 `--suffix`로 명시적 이름을 만들어. provenance(run id,
  submitted_at, 소스 파일, code, 생성 시각)가 프로파일에 남고 로컬
  경로는 기록하지 않아. **옛 해시 형식 키**(`qpu/<backend>_avg<N>_<해시8>`)는
  mode 기반이라 계속 동작하지만 전부 heavyhex이며, 새 규약으로 재생성을
  권장해.
- **회로**: `qpu/` 프로파일은 추상 회로 대신 코드별 **하드웨어형 Stim
  회로**로 생성돼 — heavyhex는
  [heavyhex37_qpu_stim.py](dataset_generation/heavyhex37_qpu_stim.py)
  (37q, bridge 포함, no-reset, depth-7 fold 미러), surface는
  [rotatedSurface3_qpu_stim.py](dataset_generation/rotatedSurface3_qpu_stim.py)(17q, no-reset,
  고정 4-레이어 hook-safe CX 스케줄). 게이트/측정마다 해당 물리 큐빗·엣지의
  평균 캘리브레이션 에러가 붙어 (CX→DEPOLARIZE2, H→DEPOLARIZE1(sx 프록시),
  측정 직전→X_ERROR readout). detector는 raw 측정 기록의 XOR 전개로
  정의되지만 순서는 각 코드의 `_append_detectors`와 동일해서 텐서/MWPM
  재구성 로직은 그대로야. `qpu/` 프로파일은 기본 그리드(ALL_NOISE)에
  **포함되지 않고**(`-n qpu/<이름>` 명시 선택), 프로파일의 code와 다른
  `--code`로 쓰려 하면 에러로 거부돼.
- **미반영 노이즈 항목** (평균 캘리브레이션 모델의 한계):
  캘리브레이션 드리프트(run 간/내 변화), T1/T2 idle 감쇠·delay 에러
  (DD 동작 포함), 측정 crosstalk/상관 readout 에러, coherent(비-Pauli)
  에러 — H는 sx 프록시 1회의 depolarizing으로 근사돼. 또한 **2Q 에러는
  네이티브 게이트 벤치마크 값을 CX 위치에 부착하는 프록시**야
  (heavyhex 백엔드의 ECR, miami의 CZ — 동일한 근사). QPU 실측과의
  잔차는 이 항목들에서 나온다고 보면 돼.

### 최종 비교표 산출 방법 (MWPM | CNN/GNN × Stim | CNN/GNN × QPU-cal)

한 코드(예: heavyhex)에 대해 세 축을 같은 지표(head-LER,
`LER/MWPM ratio`)로 모으는 절차:

```bash
# 1) [시뮬레이션 축] 표준 노이즈 프로파일 데이터셋에서 CNN/GNN + MWPM
#    -> 스윕 한 번으로 model 컬럼이 있는 통합 표가 나온다.
#    config.json의 "sweep" 섹션을 이렇게 두고 (기본으로 들어 있는 형태에
#    noise/rates만 고정):
#      "sweep": {"defaults": {"noise": "realistic/dp0.001_mf0.01_rf0.01_gd0.008",
#                             "rates": [0.005], "mwpm": true},
#                "runs": [{"model": "cnn"}, {"model": "gnn"}]}
python dataset_generation/make_dataset.py && python train.py

# 2) [QPU-cal 축] 캘리브레이션 평균 프로파일 데이터셋에서 같은 스윕
#    (사전에 make_qpu_avg_profile.py로 qpu/<이름> 등록, 게이트 ALL PASS)
#    sweep의 noise만 qpu/<이름>으로 바꿔 반복
# 3) [실기기 축(선택)] run_hw.py analyze가 raw/MWPM/CNN/GNN을 한 표로 출력
python hardware/run_hw.py analyze --job-id <ID>   # CNN+GNN 전부 (파일명 접두사로 자동 인식)
```

각 실행의 summary 표(모델·noise·p별 best epoch의 ler / mwpm_ler /
LER/MWPM ratio)를 세로로 이어 붙이면 MWPM 기준선 대비 CNN/GNN ×
{Stim 프로파일, QPU-cal 프로파일, (가능하면) 실기기}의 최종 비교표가 돼.
**surface(miami)도 `--code surface`로 동일하게 3자 비교가 가능**해 —
전제는 miami에 non-dry-run 제출 기록이 쌓여 있어서
`make_qpu_avg_profile.py --code surface`로 프로파일을 만들 수 있어야
한다는 것 (verify_rotatedSurface3 [G] 포함 ALL PASS 후 데이터셋 생성).

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
python verification/verify_equivalence.py   # heavyhex (+ qpu 프로파일 회로)
python verification/verify_rotatedSurface3.py          # surface (rotatedSurface3, hook 검사 포함)

# 1) 데이터셋 생성 (--code 기본값은 heavyhex)
python dataset_generation/make_dataset.py --smoke   # 빠른 확인용
python dataset_generation/make_dataset.py           # config.json에 sweep 섹션 있으면 그 조합,
                                                    # 없으면 전체 그리드 (용량/시간 꽤 큼)
#    작게 줄이고 싶으면 예 (기본 샘플 수는 config.json의 dataset 섹션으로도 조절 가능):
python dataset_generation/make_dataset.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 \
       -p 0.005 --train-samples 1000000 --test-samples 100000

# 2) 모델 학습 (채우기 전에는 NotImplementedError로 멈출 수 있음)
python train.py --model cnn --smoke            # end-to-end 확인
python train.py --model gnn --smoke            # GNN
python train.py --model cnn --code surface --smoke   # rotated surface d=3
python train.py --model cnn -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 -p 0.005 --mwpm
python train.py --all --mwpm                   # 전체 그리드 + 기준선 표
python train.py                                # config.json sweep 섹션 있으면 자동 스윕
python train.py --config none                  # 스윕 끄고 기본 단일 설정

# 3) MWPM 기준선 표만 따로 보고 싶을 때
python baseline/mwpm.py

# 4) QPU 검증 (keys.json 필요, 5. 참고)
python hardware/run_hw.py submit --backend ibm_yonsei --dry-run   # 리허설
python hardware/run_hw.py submit --backend ibm_yonsei             # 실제 제출
python hardware/run_hw.py analyze --job-id <ID>   # 해당 코드의 CNN+GNN 체크포인트 전부 평가
python hardware/run_hw.py analyze --job-id <ID> --model gnn       # GNN만으로 제한
python hardware/run_hw.py analyze --job-id <ID> \
       --ckpt checkpoint/CNN_heavyhex_d3_c3_p0.005_dp0.001_mf0.01_rf0.01_gd0.008.pt
python hardware/run_hw.py all                     # 제출→완료 대기→분석 원샷
```

analyze는 잡이 아직 안 끝났으면 알아서 폴링하며 기다렸다가 진행해.
학습은 **epoch-resume**이 기본이야: 같은 (model, code, noise, p) 조합을
다시 돌리면 `<tag>.resume.pt`에서 이어서 학습해 (epoch 번호는 전역,
CSV는 append). **early stopping이 이미 발동한 조합은 재실행 시 학습을
통째로 건너뛰고**(데이터 로드 전에 판정) 저장된 best로 요약만 출력해 —
파이프라인 루프 2+에서 realistic 그리드가 사실상 즉시 통과되는 이유야.
더 학습하고 싶으면 `--continue-stopped`(patience 새로 시작, 다시
멈출 때까지), 처음부터는 `--fresh`.
Early stopping은 **min_delta 기반**이야: best 체크포인트는 어떤 미세
개선에도 갱신되지만, patience 리셋은 val LER이 `min_delta`(기본 0.0015
— val 100k 기준 통계 노이즈 2σ) 넘게 좋아졌을 때만 돼서, 노이즈 수준의
"개선"이 학습을 30 epoch 꽉 채우게 만들지 않아. config `train` 섹션의
`min_delta`/`--min-delta`로 조절.
`amp: true`(config train 섹션)면 CUDA에서 fp16 autocast + GradScaler로
학습해 (가중치·loss는 fp32 유지, 스모크 검증에서 LER 궤적 동일 확인).
체크포인트에는 `best_epoch`(저장된 weight의 epoch)와 `total_epochs`
(누적 학습 epoch)가 기록되고, 학습 summary와 하드웨어 리포트 CSV에
같은 컬럼이 표시돼.
`--ckpt` 없이 돌리면 해당 코드의 `checkpoint/*.pt`를 전부 평가하는데,
각 파일의 아키텍처는 `CNN_`/`GNN_` **파일명 접두사로 자동 인식**돼서
한 리포트에 두 모델 행이 함께 나와. `--model {cnn,gnn}`은 한
아키텍처로 제한하고 싶을 때만 주면 돼.

### 설정 파일 하나 (repo 루트의 [config.json](config.json))

모든 사용자 설정이 config.json 한 파일, 섹션 네 개로 모여 있어:

- **`noise_profiles`** — 노이즈 프로파일 레지스트리. 4-파라미터 프로파일은
  여기서 직접 추가/수정하며 탐색하면 되고, `qpu/...` 항목은
  make_qpu_avg_profile.py가 **이 섹션만** 프로그램적으로 갱신해
  (다른 섹션은 보존).
- **`train` / `dataset` / 최상위 `cycles`** — *기본값* 조절: 학습
  하이퍼파라미터(epochs, batch_size, lr, ...), 데이터셋 샘플 수
  (train_samples/test_samples), `cycles`는 양쪽 공용. train.py /
  make_dataset.py가 자동으로 읽고, CLI 인자를 명시하면 그쪽이 이겨.
- **`pipeline`** — 파이프라인 고정 설정: `conda_env`, `codes`,
  `backends`(코드→QPU 백엔드), `qpu_pubs`(잡 하나에 담는 PUB 수, 기본 5),
  `profile_runs`(프로파일 평균 창 = 최근 몇 개 루프, 기본 5), `shots`
  (PUB당 샷), `train_args`(train.py에 항상 붙는 인자). sbatch 환경변수를
  외울 필요 없게 여기에 모아뒀어 (남은 변수는 LOOPS/SMOKE 둘뿐).
- **`sweep`** — *스윕* 정의: `runs`의 각 항목이 (노이즈, p, error_type,
  model + 하이퍼파라미터 오버라이드) 한 벌이야. **기본 sweep이 들어
  있어** — cnn/gnn 두 run(+mwpm)이라, 인자 없이 돌리면 같은 데이터로 두
  모델을 이어 학습해 CNN/GNN/MWPM 통합 표가 나와. 섹션이 있으면
  make_dataset.py는 필요한 조합의 데이터셋을, train.py는 항목별 학습을
  **자동으로** 돌려. 선택 인자(-n/-p/-e/--all/--smoke)를 명시하면 스윕
  대신 그쪽이 돌고, `--config none`으로 끄거나 `--config 다른파일.json`
  (별도 스윕 JSON 또는 다른 config 파일)으로 바꿀 수 있어. 같은 (노이즈,
  p, cycles)의 하이퍼파라미터 변형에는 `"name"`을 줘서 결과 파일 이름을
  구분해.

우선순위: `config.json` 기본값 < CLI 인자 < 스윕 run 항목.

### 산출물이 저장되는 위치

- 데이터셋: `dataset/<code>/<노이즈태그>/{train,test}_d3_c3_p<p>_X.npz`
  (예: `dataset/heavyhex/dp0.001_mf0.01_rf0.01_gd0.008/train_d3_c3_p0.005_X.npz`;
   `d3` = code distance (dx/dz 중 큰 값; (3,3)은 3, (3,5)/(5,3)이면 5),
   `c3` = QEC cycle 수 3)
- 학습 로그: `results/train/{MODEL}_<code>_d3_c<cycles>_p<p>_<노이즈태그>.csv`
- 체크포인트: `checkpoint/{MODEL}_<code>_d3_c<cycles>_p<p>_<노이즈태그>.pt`
  (epoch마다 검증 LER을 재서 **최저 val LER**일 때만 갱신돼)
- QPU 런: `hardware/runs/<backend>_<타임스탬프>/` (job id는 job.json에
  기록; `analyze --job-id`는 job.json 스캔으로 폴더를 찾으므로 옛
  job-id 이름 폴더도 계속 동작) — raw 결과와 함께 그 시점의 QPU
  환경 기록(캘리브레이션 스냅샷, 제출한 회로 등)이 통째로 남아
- QPU LER 리포트: `results/hardware/<backend>_<코드>_<타임스탬프>.csv`
  (코드는 heavyhex/rotatedSurface 표기; 타임스탬프는 제출 시각.
   backend/timestamp/job_id 컬럼이 표 안에도 들어가)
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
`circuits/heavyhex/heavyhex_37q.py`의 `EMBEDDINGS`에 등록돼 있어
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
sbatch train.sbatch --model gnn --all       # GNN 학습

# 통합 파이프라인: 게이트 -> QPU 수거 -> QPU 제출 -> [프로파일] -> 데이터셋 -> 학습.
# 고정 설정(환경/코드/백엔드/샷/PUB 수/프로파일 창/train 인자)은 전부
# config.json의 "pipeline" 섹션에 있고, sbatch 변수는 둘뿐이야:
LOOPS=5 sbatch pipeline.sbatch     # 실전: 파이프라인 5루프 반복.
#   QPU는 루프당 코드별 "1잡"으로 제출돼 — 같은 ISA 회로를 qpu_pubs개
#   PUB(Primitive Unified Bloc, (회로,파라미터,샷) 실행 단위)으로 담아
#   큐 엔트리 1개로 shots x qpu_pubs 를 확보 (기관 계정의 fair-share
#   후순위 문제를 큐 엔트리 최소화로 회피; 루프당 엔트리 = 백엔드 수 2).
#   제출 후 완료를 기다리지 않고 프로파일/데이터셋/학습을 진행하고,
#   "수거"는 다음 루프 시작부에서: pending_jobs.json의 잡 중 DONE만
#   분석(analyze), 나머지는 이월, 실패는 collect_failed.log에 기록.
#   프로파일은 최근 profile_runs개 루프의 "실행 시점" 캘리브레이션
#   (properties_run.json — 수거 때 job의 running 타임스탬프 기준으로
#   저장) 평균이고, 하드웨어 MWPM 가중치도 최신 qpu 프로파일을 쓴다.
sbatch pipeline.sbatch             # 1회만
SMOKE=1 sbatch pipeline.sbatch     # 리허설: 스모크 학습 + QPU dry-run(제출 없음)
# 마지막 루프의 잡이 미수거로 남으면 다음 sbatch 때 수거되거나 수동으로:
#   python hardware/run_hw.py collect --solution
DATASET_ARGS="--smoke" sbatch pipeline.sbatch --smoke   # 빠른 end-to-end 확인
# config.json에 sweep 섹션이 있으면 데이터셋 생성+학습이 자동으로 스윕을 돈다:
SWEEP_CONFIG=다른스윕.json sbatch pipeline.sbatch       # 별도 스윕 파일 지정
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
circuits/               코드별 고정 회로 자산 (재작성하지 말고 import해서 쓸 것)
  heavyhex/             (3,3) heavy-hex
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
  rotatedSurface/       rotated surface code d=3
    rotatedSurface3.py              코드 정의 + no-reset 하드웨어 회로 +
                                    ibm_miami 45도 임베딩/검증기
dataset_generation/     Stim 회로 생성기(heavyhex33_stim.py, rotatedSurface3_stim.py,
                        heavyhex37_qpu_stim.py, rotatedSurface3_qpu_stim.py) +
                        데이터셋 생성(make_dataset.py, --code 축) +
                        QPU 프로파일 생성기(make_qpu_avg_profile.py,
                        --code 축)
verification/           규약 게이트 스크립트 (§4) — verify_equivalence.py
                        (heavyhex + qpu 회로), verify_rotatedSurface3.py (surface)
model/                  채워야 할 파일 (cnn_skeleton.py, gnn_skeleton.py) +
                        모델 레지스트리(__init__.py) + 데이터 로더(data.py) +
                        GNN 그래프 인프라(graph.py)
evaluation/             지표 — head-LER(목표 지표) + ECR/parity_LER(진단용)
baseline/               MWPM (PyMatching) 기준선
train.py                학습 진입점 (완성본, 수정할 필요 없어; --model/--code)
config.json             통합 설정 — noise_profiles(프로파일 레지스트리) /
                        train·dataset(기본값) / sweep(자동 적용, §3)
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

참고: `circuits/heavyhex/` 안의 회로 테스트 중 coupling map을 쓰는 것들
(test_general.py, heavyhex_depth_opt.py 데모)은 fetch_coupling.py로
coupling JSON을 먼저 생성한 뒤 실행해야 해 (test_37q.py /
test_depth7_opt_for_37q.py는 Aer만 쓰므로 바로 실행 가능).
