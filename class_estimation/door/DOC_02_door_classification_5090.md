# 02_door_classification_5090.py 코드 설명서

> 굴착기 도어 분류 모델 학습 스크립트 (RTX 5090 최적화, RGBD 4채널 입력)

## 개요

ZED X Mini 카메라로 촬영한 **RGB + Depth** 이미지를 사용하여 굴착기 도어 종류를 분류하는 딥러닝 모델을 학습하는 스크립트입니다.

### 핵심 특징
- **입력**: RGBD 4채널(R, G, B, Depth) 이미지 + 물리 치수 보조 피처 3개
- **모델**: ResNet18 Transfer Learning (ImageNet 사전학습)
- **출력**: ONNX 형식 모델 파일 (ZED Box Mini 추론용)
- **최적화**: RTX 5090 GPU 환경에 맞춘 배치 크기/워커 수 자동 조정

---

## 전체 실행 흐름 (9단계)

```
데이터셋 스캔 → 전처리 설정 → Train/Test 분할 → 모델 생성
    → 옵티마이저 설정 → 학습 실행 → 최종 평가 → ONNX 변환
```

---

## 1단계: 설정 및 인자 파싱 (1~73행)

### 명령줄 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--cpu` | GPU 대신 CPU로 강제 실행 | False |
| `--dataset_dir` | 데이터셋 폴더 경로 | `door/datasets/` |
| `--full_train` | 전체 데이터를 학습에 사용 (배포용) | False |

### 주요 설정 변수

| 변수 | 값 | 설명 |
|------|-----|------|
| `BATCH_SIZE` | `None` (자동) | GPU 메모리에 맞춰 자동 조정 |
| `IMAGE_SIZE` | 448 | 입력 이미지 크기 (448×448) |
| `NUM_EPOCHS` | 60 | 최대 학습 에포크 |
| `EARLY_STOPPING_PATIENCE` | 10 | 개선 없을 시 조기 종료 기준 |
| `TEST_SIZE` | 0.3 | 테스트 데이터 비율 (30%) |
| `NUM_WORKERS` | 8 | DataLoader 워커 수 |

### 출력 파일 (artifacts 폴더)

| 파일 | 용도 |
|------|------|
| `best_door_model_5090.pth` | 학습된 PyTorch 모델 가중치 |
| `best_door_model_5090.onnx` | ONNX 변환 모델 (추론 배포용) |
| `training_indices_door_5090.json` | 학습/테스트 데이터 분할 정보 |
| `class_names_door_5090.json` | 클래스 이름 목록 |

---

## 2단계: 데이터셋 스캔 (92~135행)

### `scan_dataset(dataset_dir)` → `(classes, image_paths, labels)`

데이터셋 폴더를 스캔하여 클래스별 이미지 경로를 수집합니다.

**동작 방식:**
1. `dataset_dir` 아래의 모든 하위 폴더를 알파벳순으로 정렬
2. 각 폴더 안에서 `rgb_*.png` 패턴의 파일을 탐색
3. 이미지가 없는 폴더는 자동으로 건너뜀
4. 유효한 폴더에 순서대로 클래스 인덱스(0, 1, 2, ...) 부여

**폴더 구조 예시:**
```
datasets/
├── E25_door_LH_FRT/     → 클래스 0
│   ├── rgb_0001.png
│   ├── depth_0001.png   (16-bit PNG, mm 단위)
│   └── mask_0001.png    (SAM 전경 마스크, 선택사항)
├── E30_door_LH_RR/      → 클래스 1
│   ├── rgb_0001.png
│   └── depth_0001.png
└── empty_folder/        → [건너뜀: 이미지 없음]
```

**반환값:**
- `classes`: 유효한 클래스 이름 리스트 (예: `["E25_door_LH_FRT", "E30_door_LH_RR"]`)
- `image_paths`: 모든 RGB 이미지의 전체 경로 리스트
- `labels`: 각 이미지에 대응하는 클래스 인덱스 리스트

---

## 3단계: 전처리 설정 (140~156행)

`depth_utils.py`의 `RGBDTransform` 클래스를 사용합니다.

### `RGBDTransform(image_size, is_train)` 

RGB 이미지와 Depth 이미지를 동기화하여 변환합니다.

**학습용 (is_train=True) 변환 파이프라인:**
```
원본 이미지 (W×H)
  ↓ Letterbox Resize (장변 기준, 종횡비 보존)
  ↓ 중앙 배치 + 검정 패딩 → 448×448
  ↓ 수평 반전 (50% 확률)          ← RGB, Depth 동시 적용
  ↓ 회전 (±15°)                   ← RGB, Depth 동시 적용
  ↓ 스케일 변환 (90~110%, 50%)    ← RGB, Depth 동시 적용
  ↓ 밝기/대비/채도/색조 변환       ← RGB에만 적용
  ↓ 가우시안 노이즈 (30% 확률)     ← RGB에만 적용
  ↓ 가우시안 블러 (20% 확률)       ← RGB에만 적용
  ↓ 정규화
  → [4, 448, 448] 텐서
```

**검증용 (is_train=False) 변환:**
```
원본 이미지 → Letterbox Resize → 패딩 → 정규화 → [4, 448, 448] 텐서
```

**정규화 기준:**
- RGB: ImageNet 표준 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Depth: mean=0.5, std=0.25

---

## 4단계: Train/Test 분할 및 DataLoader (159~294행)

### 데이터 분할

- **일반 모드**: `sklearn.train_test_split`으로 70/30 분할 (Stratified → 클래스 비율 유지)
- **전체 학습 모드** (`--full_train`): 모든 데이터를 학습에 사용, 테스트 셋 없음

### `adjust_batch_size_5090(data_size, force_cpu)` → `int`

GPU 메모리에 맞춰 배치 사이즈를 자동 결정합니다.

| 데이터 수 | 기본 배치 | GPU ≥ 24GB | GPU ≥ 16GB | GPU ≥ 8GB |
|----------|----------|-----------|-----------|----------|
| < 100 | 16 | 16 | 16 | 16 |
| 100~499 | 32 | 32 | 32 | 16 |
| 500~1999 | 64 | 64 | 32 | 16 |
| ≥ 2000 | 64 | 128 | 64 | 16 |

### `RGBDDataset` (depth_utils.py)

PyTorch Dataset 클래스로, 하나의 샘플에 대해 3가지를 반환합니다:

| 반환값 | 형태 | 설명 |
|--------|------|------|
| `rgbd_tensor` | `[4, 448, 448]` | RGBD 4채널 이미지 텐서 |
| `aux_tensor` | `[3]` | 물리 치수 보조 피처 (가로mm, 세로mm, 종횡비) |
| `label` | scalar | 클래스 인덱스 |

**보조 피처 계산 흐름 (`compute_aux_features`):**
```
Depth 이미지 (mm 단위)
  ↓ 전경 분리 (SAM 마스크 또는 Otsu 이진화)
  ↓ 카메라 내부 파라미터로 2D→3D 점 변환
  ↓ PCA로 주성분 2축 추출
  ↓ 각 축 범위 계산
  → [가로_mm, 세로_mm, 종횡비]
```

---

## 5단계: 모델 정의 (297~332행)

### `create_resnet_model(num_classes, pretrained)` → `RGBDAuxResNet18`

`RGBDAuxResNet18` 모델을 생성합니다.

### `RGBDAuxResNet18` 모델 구조 (depth_utils.py)

```
입력 1: RGBD 이미지 [B, 4, 448, 448]
  │
  ↓ ResNet18 backbone (4ch 입력, ImageNet 가중치)
  ↓ Global Average Pooling
  → 이미지 피처 [B, 512]
                                ╲
                                 ╲ Concat
                                 ╱        → [B, 544]
                                ╱
입력 2: 보조 피처 [B, 3]       ╱
  │
  ↓ Linear(3→32) + ReLU
  → 보조 피처 [B, 32]

결합 피처 [B, 544]
  ↓ Dropout(0.3)
  ↓ Linear(544→256) + ReLU
  ↓ Dropout(0.2)
  ↓ Linear(256→num_classes)
  → 출력 [B, num_classes]
```

**4채널 입력 처리:**
- RGB 3채널: ImageNet 사전학습 가중치 그대로 사용
- Depth 1채널: RGB 3채널 가중치의 평균으로 초기화 → 안정적인 학습 시작

---

## 6단계: 손실 함수 및 옵티마이저 (335~361행)

### 클래스 불균형 처리

클래스별 데이터 수가 다를 때 가중치를 부여합니다:

```
가중치 = 전체 데이터 수 / (클래스 수 × 해당 클래스 데이터 수)
```

데이터가 적은 클래스일수록 높은 가중치 → 균형 잡힌 학습

### 옵티마이저 구성

| 항목 | 설정 |
|------|------|
| 손실 함수 | CrossEntropyLoss (클래스 가중치 적용) |
| 옵티마이저 | Adam (lr=0.001) |
| 스케줄러 | ReduceLROnPlateau (patience=3, factor=0.5) |

스케줄러 동작: 3 에포크 동안 loss가 개선되지 않으면 학습률을 절반으로 줄임

---

## 7단계: 학습/평가 함수 (365~447행)

### `train_one_epoch(model, dataloader, criterion, optimizer, device)` → `(loss, accuracy)`

한 에포크 동안 모델을 학습합니다.

**동작 순서:**
1. 모델을 학습 모드로 설정
2. 미니배치 단위로 반복:
   - 이미지/보조 피처/라벨을 GPU로 전송 (`non_blocking=True`로 비동기 전송)
   - 순전파 → 손실 계산 → 역전파 → 파라미터 업데이트
3. 에포크 전체의 평균 손실과 정확도 반환

### `evaluate(model, dataloader, criterion, device, class_names)` → `(loss, accuracy, class_accuracies)`

모델을 평가합니다 (클래스별 정확도 포함).

**동작 순서:**
1. 모델을 평가 모드로 설정 (Dropout 비활성화)
2. `torch.no_grad()` 내에서 그래디언트 계산 없이 추론
3. 전체 평균 손실/정확도 + 클래스별 정확도를 딕셔너리로 반환

---

## 8단계: 학습 실행 (449~527행)

### 학습 루프

```python
for epoch in range(NUM_EPOCHS):    # 최대 60 에포크
    train_one_epoch(...)           # 학습
    evaluate(...)                  # 평가 (일반 모드만)
    scheduler.step(loss)           # 학습률 조정
    
    if 개선됨:
        모델 저장 (.pth)
        patience 리셋
    else:
        patience += 1
    
    if patience >= 10:             # Early Stopping
        학습 중단
```

### Early Stopping 기준

| 모드 | 기준 지표 | 설명 |
|------|----------|------|
| 일반 모드 | Validation Accuracy | 검증 정확도가 10 에포크 연속 최고치 갱신 안 되면 중단 |
| 전체 학습 모드 | Train Loss | 학습 손실이 10 에포크 연속 최저치 갱신 안 되면 중단 |

### GPU 메모리 관리
- 10 에포크마다 `torch.cuda.empty_cache()` 호출하여 불필요한 캐시 정리

---

## 9단계: 최종 평가 (530~564행)

저장된 최적 모델을 불러와서 최종 성능을 측정합니다.

- **일반 모드**: 테스트 셋으로 최종 Loss, Accuracy, 클래스별 Accuracy 출력
- **전체 학습 모드**: 학습 데이터로 참고용 성능 확인 (과적합 지표)

---

## 10단계: ONNX 변환 (567~613행)

학습된 모델을 ONNX 형식으로 변환하여 ZED Box Mini에서 추론할 수 있도록 합니다.

**변환 설정:**

| 항목 | 값 |
|------|-----|
| ONNX Opset | 13 |
| 입력 이름 | `images` [B, 4, 448, 448], `aux_features` [B, 3] |
| 출력 이름 | `output` [B, num_classes] |
| Dynamic Axes | batch_size 차원만 가변 |
| Constant Folding | 활성화 (추론 최적화) |

**에러 처리:** ONNX 변환 실패 시 학습된 `.pth` 파일은 유지

---

## 의존 모듈: `depth_utils.py` 핵심 함수

### Depth 입출력

| 함수 | 설명 |
|------|------|
| `load_depth_png(path)` | 16-bit PNG → 정규화된 float [0,1] (5m 기준 클리핑) |
| `save_depth_png(depth_meters, path)` | float depth(m) → 16-bit PNG(mm) |

### 전경 분리 및 물리 치수 계산

| 함수 | 설명 |
|------|------|
| `_segment_foreground(depth_raw_mm)` | Otsu 이진화 + 형태학적 처리로 부품 영역 분리 |
| `compute_aux_features(depth_raw_mm, ...)` | 3D 점군 → PCA → 물리 가로/세로/종횡비 (mm) |

### 모델

| 클래스/함수 | 설명 |
|------------|------|
| `create_rgbd_resnet18(num_classes)` | 기본 4채널 RGBD ResNet18 생성 (보조 피처 없음) |
| `RGBDAuxResNet18` | RGBD + 보조 피처 결합 ResNet18 분류 모델 |
| `RGBDTransform` | RGB+Depth 동기화 변환 (Letterbox + Augmentation) |
| `RGBDDataset` | RGB+Depth+Mask 로딩 Dataset (보조 피처 자동 계산) |

---

## 실행 예시

```bash
# 기본 실행 (GPU 자동 감지, 70/30 분할)
python 02_door_classification_5090.py

# 전체 데이터로 학습 (배포용 모델 생성)
python 02_door_classification_5090.py --full_train

# CPU로 실행
python 02_door_classification_5090.py --cpu

# 커스텀 데이터셋 경로 지정
python 02_door_classification_5090.py --dataset_dir /path/to/custom_datasets
```
