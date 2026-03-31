# 굴착기 도어 분류 시스템 — 사용 설명서

> `class_estimation/door/` 폴더의 전체 Python 스크립트 사용법 안내

---

## 목차

1. [전체 파이프라인 개요](#1-전체-파이프라인-개요)
2. [폴더 구조](#2-폴더-구조)
3. [스크립트별 사용법](#3-스크립트별-사용법)
   - [00 STL→USD 변환](#00-stlusd-변환)
   - [01 데이터셋 생성](#01-데이터셋-생성)
   - [02 모델 학습 (4종)](#02-모델-학습)
   - [03 모델 평가 (4종)](#03-모델-평가)
   - [04 실시간 추론 / 파인튜닝](#04-실시간-추론--파인튜닝)
   - [05 마스크 생성](#05-마스크-생성)
   - [06 증강 데이터셋 생성](#06-증강-데이터셋-생성)
   - [99 치수 측정](#99-치수-측정)
4. [유틸리티 모듈 참조](#4-유틸리티-모듈-참조)
5. [실전 예시 레시피](#5-실전-예시-레시피)

---

## 1. 전체 파이프라인 개요

```
[CAD 준비]         [실물 촬영]
00_convert → 01_generate    01_capture
      ↓                        ↓
  datasets_cad/             datasets/
      ↓                        ↓
      └──── 05_generate_masks ──┘  ← SAM2로 전경 마스크 생성
                    ↓
            06_generate_augmented  ← 증강 데이터셋 생성
             ↓            ↓
         datasets_aug  datasets_aug2
                    ↓
      ┌─────── 02_학습 (4종 모델) ───────┐
      │  Baseline RGBD                   │
      │  Texture Aug RGBD                │
      │  Edge-only                       │
      │  RGBE Hybrid                     │
      └──────────────────────────────────┘
                    ↓
          artifacts/ (pth, onnx, json)
                    ↓
      ┌─────── 03_평가 ─────────────────┐
      │  Test 분할 / 크로스 도메인 평가    │
      │  혼동행렬, 오분류 시각화           │
      └──────────────────────────────────┘
                    ↓
      ┌─────── 04_실시간 추론 ──────────┐
      │  ZED 카메라 + 웹 인터페이스       │
      └──────────────────────────────────┘
```

---

## 2. 폴더 구조

```
class_estimation/door/
├── 00~99_*.py              # 실행 스크립트 (아래 상세 설명)
├── depth_utils.py          # RGBD 공통 (모델, Dataset, Transform)
├── edge_utils.py           # Edge-only 공통
├── rgbe_utils.py           # RGBE 공통
├── camera_utils.py         # ZED 카메라 관리
├── dimension_utils.py      # 부품 치수 측정
│
├── datasets/               # 실물 촬영 데이터 (원본)
├── datasets_cad/           # Isaac Sim CAD 합성 데이터
├── datasets_aug/           # 증강 데이터 (extreme)
├── datasets_aug2/          # 증강 데이터 (moderate)
│
├── artifacts/              # 현재 작업 결과물 (모델, 평가, 이미지)
├── artifacts_224/          # 224×224 해상도 학습 결과 보관
├── artifacts_448/          # 448×448 해상도 학습 결과 보관
│
├── sam_models/             # MobileSAM 가중치
├── templates/              # Flask 웹 템플릿
├── static/                 # 웹 정적 파일
└── final/                  # 최종 배포용 파일
```

**데이터셋 폴더 내부 구조** (8개 도어 클래스):
```
datasets/
├── E25_door_LH_FRT/
│   ├── rgb_0001.png       # RGB 이미지 (1920×1080)
│   ├── depth_0001.png     # 16-bit Depth (mm 단위)
│   └── mask_0001.png      # SAM 전경 마스크 (선택)
├── E25_door_LH_RR/
├── E25_door_RH/
├── E30_door_LH_FRT/
├── E30_door_LH_RR/
├── E30_E38_door_RH/
├── E38_door_LH_FRT/
└── E38_door_LH_RR/
```

---

## 3. 스크립트별 사용법

### 00. STL→USD 변환

**파일**: `00_convert_stl_to_usd.py`  
**용도**: Door STL 메쉬 파일을 Isaac Sim용 USD 포맷으로 변환

```bash
# Isaac Sim Python으로 실행해야 함
~/isaac-sim/python.sh 00_convert_stl_to_usd.py
```

| 항목 | 경로 |
|------|------|
| 입력 | `cad/door_stl/*.stl` |
| 출력 | `~/isaac-sim/assets/door/*.usd` |

---

### 01. 데이터셋 생성

#### 01_capture_dataset.py — 실물 촬영

**용도**: ZED 카메라로 도어 부품을 촬영하여 RGB+Depth 데이터셋 구축 (Flask 웹 UI)

```bash
python 01_capture_dataset.py
# 브라우저에서 http://0.0.0.0:5000 접속
```

| 항목 | 경로 |
|------|------|
| 출력 | `datasets/{클래스명}/rgb_NNNN.png`, `depth_NNNN.png` |

#### 01_generate_door_cad_dataset.py — CAD 합성

**용도**: Isaac Sim으로 CAD 모델 기반 합성 데이터셋 자동 생성 (Domain Randomization 적용)

```bash
# Isaac Sim Python으로 실행
~/isaac-sim/python.sh 01_generate_door_cad_dataset.py
```

| 항목 | 값 |
|------|-----|
| 클래스당 이미지 수 | 500장 |
| 출력 | `datasets_cad/{클래스명}/rgb_NNNN.png`, `depth_NNNN.png` |

---

### 02. 모델 학습

4종류의 모델을 학습할 수 있습니다. 모두 동일한 인자 패턴을 따릅니다.

| 스크립트 | 모델 유형 | 입력 채널 | 설명 |
|----------|----------|----------|------|
| `02_door_classification_5090.py` | **Baseline RGBD** | R,G,B,D (4ch) | 기본 RGBD 모델 |
| `02_door_classification_texture_aug_5090.py` | **Texture Aug RGBD** | R,G,B,D (4ch) | 텍스처 불변 증강 추가 |
| `02_door_classification_edge_5090.py` | **Edge-only** | Edge×3 (3ch) | Canny edge만 사용 |
| `02_door_classification_rgbe_5090.py` | **RGBE Hybrid** | R,G,B,Edge (4ch) | RGB + Canny edge |
| `02_door_cad_classification_5090.py` | **CAD RGBD** | R,G,B,D (4ch) | CAD 합성 데이터용 |

#### 공통 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--cpu` | False | CPU로 강제 실행 |
| `--dataset_dir` | 모델별 다름 | 데이터셋 경로 |
| `--full_train` | False | 전체 데이터로 학습 (배포용) |
| `--image_size` | 448 | 입력 이미지 해상도 (448 또는 224) |

#### 사용 예시

```bash
# ① Baseline RGBD 학습 (실물 데이터, 448×448)
python 02_door_classification_5090.py

# ② 224×224 해상도로 학습
python 02_door_classification_5090.py --image_size 224

# ③ 전체 데이터로 배포용 학습 (Train/Test 분할 없음)
python 02_door_classification_5090.py --full_train

# ④ CAD 합성 데이터로 학습
python 02_door_cad_classification_5090.py

# ⑤ Edge-only 모델 학습
python 02_door_classification_edge_5090.py

# ⑥ RGBE Hybrid 모델 학습
python 02_door_classification_rgbe_5090.py

# ⑦ 텍스처 증강 모델 학습
python 02_door_classification_texture_aug_5090.py

# ⑧ 커스텀 데이터셋으로 학습
python 02_door_classification_5090.py --dataset_dir datasets_aug
```

#### 출력 파일 (artifacts/)

| 파일 | 설명 |
|------|------|
| `best_door_model_5090.pth` | PyTorch 모델 가중치 |
| `best_door_model_5090.onnx` | ONNX 변환 모델 (추론 배포용) |
| `class_names_door_5090.json` | 클래스 이름 목록 |
| `training_indices_door_5090.json` | Train/Test 분할 정보 |

> 다른 모델은 파일명의 `model` 부분이 변경됨:
> `cad_model`, `edge_model`, `rgbe_model`, `texture_aug_model`

---

### 03. 모델 평가

학습된 모델의 성능을 정량 평가합니다. 혼동행렬, 오분류 시각화를 포함합니다.

| 스크립트 | 대상 모델 | 기본 모델 경로 |
|----------|----------|--------------|
| `03_door_class_evaluation_5090.py` | Baseline / Texture Aug | `artifacts/best_door_model_5090.pth` |
| `03_door_class_evaluation_edge_5090.py` | Edge-only | `artifacts/best_door_edge_model_5090.pth` |
| `03_door_class_evaluation_rgbe_5090.py` | RGBE Hybrid | `artifacts/best_door_rgbe_model_5090.pth` |
| `03_door_cad_evaluation_5090.py` | CAD RGBD | `artifacts/best_door_cad_model_5090.pth` |
| `03_5_trt_evaluation.py` | TensorRT 평가 | `--dataset_dir` 필수 |

#### 공통 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | 모델별 기본 경로 | 평가할 모델 파일 (.pth) |
| `--cpu` | False | CPU로 실행 |
| `--dataset_dir` | None | 평가 데이터셋 경로 (미지정 시 학습 시 test_paths 사용) |
| `--use_all` | False | dataset_dir 전체 이미지를 평가에 사용 |
| `--num_samples` | None (전체) | 평가할 샘플 수 제한 |
| `--image_size` | 448 | 입력 이미지 해상도 |
| `--list_models` | False | 사용 가능한 모델 목록 출력 후 종료 |

#### 사용 예시

```bash
# ① 기본 평가 (학습 시 저장된 Test 분할 사용)
python 03_door_class_evaluation_5090.py

# ② artifacts_224의 모델로 평가 (224 해상도)
python 03_door_class_evaluation_5090.py \
    --model artifacts_224/best_door_model_5090.pth \
    --image_size 224

# ③ artifacts_448의 모델로 평가 (448 해상도)
python 03_door_class_evaluation_5090.py \
    --model artifacts_448/best_door_model_5090.pth \
    --image_size 448

# ④ 증강 데이터셋으로 크로스 도메인 평가 (강건성 테스트)
python 03_door_class_evaluation_5090.py \
    --dataset_dir datasets_aug --use_all

# ⑤ datasets_aug2로 Texture Aug 모델 크로스 도메인 평가
python 03_door_class_evaluation_5090.py \
    --model artifacts/best_door_texture_aug_model_5090.pth \
    --dataset_dir datasets_aug2 --use_all

# ⑥ Edge 모델로 증강 데이터 평가 (224 해상도)
python 03_door_class_evaluation_edge_5090.py \
    --model artifacts_224/best_door_edge_model_5090.pth \
    --dataset_dir datasets_aug --use_all \
    --image_size 224

# ⑦ RGBE 모델 평가
python 03_door_class_evaluation_rgbe_5090.py \
    --model artifacts_448/best_door_rgbe_model_5090.pth \
    --dataset_dir datasets_aug2 --use_all

# ⑧ 사용 가능한 모델 목록 확인
python 03_door_class_evaluation_5090.py --list_models

# ⑨ CAD 모델 평가
python 03_door_cad_evaluation_5090.py
```

#### 출력 파일 (artifacts/)

| 파일 | 설명 |
|------|------|
| `evaluation_results_door_5090.json` | 정확도, Precision, Recall, F1 수치 |
| `evaluation_results_door_5090.png` | 클래스별 정확도 바 차트 |
| `confusion_matrix_door_5090.png` | 혼동행렬 히트맵 |
| `evaluation_wrong_predictions_door_5090.png` | 오분류 샘플 시각화 |

---

### 04. 실시간 추론 / 파인튜닝

#### 실시간 추론 (3종)

ZED 카메라 영상을 실시간으로 분류하며 Flask 웹 UI에 결과를 표시합니다.

| 스크립트 | 모델 유형 | 기본 모델 |
|----------|----------|----------|
| `04_door_realtime_inference.py` | Baseline RGBD | `best_door_model_5090.pth` |
| `04_door_realtime_inference_edge.py` | Edge-only | `best_door_edge_model_5090.pth` |
| `04_door_realtime_inference_rgbe.py` | RGBE Hybrid | `best_door_rgbe_model_5090.pth` |

#### 공통 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | 모델별 기본 경로 | 추론에 사용할 모델 (.pth) |
| `--class_names` | None (자동 추론) | 클래스명 JSON 파일 |
| `--list_models` | False | 모델 목록 출력 |
| `--no_sam` | False | MobileSAM 비활성화 |

#### 사용 예시

```bash
# ① 기본 RGBD 모델로 실시간 추론
python 04_door_realtime_inference.py
# → http://0.0.0.0:5001 접속

# ② 224 해상도로 학습한 모델로 추론
python 04_door_realtime_inference.py \
    --model artifacts_224/best_door_model_5090.pth

# ③ 448 해상도 모델로 추론
python 04_door_realtime_inference.py \
    --model artifacts_448/best_door_model_5090.pth

# ④ Edge 모델로 실시간 추론
python 04_door_realtime_inference_edge.py \
    --model artifacts_224/best_door_edge_model_5090.pth

# ⑤ RGBE 모델로 실시간 추론
python 04_door_realtime_inference_rgbe.py \
    --model artifacts_448/best_door_rgbe_model_5090.pth

# ⑥ SAM 없이 추론 (가벼운 환경)
python 04_door_realtime_inference.py --no_sam

# ⑦ 사용 가능한 모델 목록 확인
python 04_door_realtime_inference.py --list_models
```

#### CAD→실물 파인튜닝

**파일**: `04_door_cad_finetune_real_5090.py`  
**용도**: CAD 합성 데이터로 사전학습한 모델을 실물 데이터로 파인튜닝

```bash
# CAD 사전학습 가중치 → 실물 데이터로 파인튜닝 검증
python 04_door_cad_finetune_real_5090.py

# CPU로 실행
python 04_door_cad_finetune_real_5090.py --cpu
```

---

### 05. 마스크 생성

**파일**: `05_generate_masks.py`  
**용도**: SAM2 (Segment Anything Model 2)로 전경 마스크를 일괄 생성

```bash
# 모든 데이터셋에 마스크 생성
python 05_generate_masks.py

# 실물 데이터만
python 05_generate_masks.py --dataset real

# CAD 데이터만
python 05_generate_masks.py --dataset cad

# 미리보기 (처음 5장만 시각화)
python 05_generate_masks.py --preview 5

# 기존 마스크 덮어쓰기
python 05_generate_masks.py --overwrite

# CPU에서 실행
python 05_generate_masks.py --device cpu
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `all` | `cad`, `real`, `all` 중 선택 |
| `--preview` | 0 | 미리보기 장수 (0=비활성) |
| `--overwrite` | False | 기존 마스크 덮어쓰기 |
| `--device` | `cuda` | 실행 디바이스 |

**출력**: `datasets/{클래스}/mask_NNNN.png`

---

### 06. 증강 데이터셋 생성

**파일**: `06_generate_augmented_dataset.py`  
**용도**: 원본 RGB에 극단 증강을 적용하여 새로운 데이터셋 생성 (depth/mask는 그대로 복사)

```bash
# 기본: datasets → datasets_aug (extreme 증강)
python 06_generate_augmented_dataset.py

# moderate 증강으로 datasets_aug2 생성
python 06_generate_augmented_dataset.py --dst datasets_aug2 --level moderate

# mild 증강
python 06_generate_augmented_dataset.py --dst datasets_mild --level mild

# 마스크 영역만 증강 (배경 유지)
python 06_generate_augmented_dataset.py --mask_only
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--src` | `datasets` | 원본 데이터셋 경로 |
| `--dst` | `datasets_aug` | 출력 데이터셋 경로 |
| `--level` | `extreme` | 증강 강도 (`mild`, `moderate`, `extreme`) |
| `--seed` | 42 | 랜덤 시드 |
| `--mask_only` | False | 전경 마스크 영역만 증강 |

---

### 99. 치수 측정

**파일**: `99_dimension_measurement.py`  
**용도**: ZED 카메라로 실시간 부품 치수(mm) 측정 + 웹 시각화

```bash
# 기본 실행 (SAM + 웹 포트 5002)
python 99_dimension_measurement.py
# → http://0.0.0.0:5002 접속

# SAM 없이 (depth 기반 전경 분리만 사용)
python 99_dimension_measurement.py --no_sam

# 커스텀 포트 및 측정 간격
python 99_dimension_measurement.py --port 8080 --interval 0.5

# 이동 평균 윈도우 크기 변경
python 99_dimension_measurement.py --avg_window 20
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--no_sam` | False | MobileSAM 비활성화 |
| `--port` | 5002 | 웹 서버 포트 |
| `--interval` | 1.0 | 측정 간격 (초) |
| `--avg_window` | 10 | 이동 평균 윈도우 크기 |

---

## 4. 유틸리티 모듈 참조

### depth_utils.py — RGBD 공통

| API | 설명 |
|-----|------|
| `load_depth_png(path)` | 16-bit PNG → float32 [0,1] |
| `save_depth_png(depth_m, path)` | float32(m) → 16-bit PNG(mm) |
| `compute_aux_features(depth_mm, intrinsics, fg_mask)` | 물리 치수 [가로, 세로, 종횡비] mm |
| `RGBDAuxResNet18(num_classes)` | RGBD(4ch) + Aux(3) → 분류 모델 |
| `RGBDTransform(image_size, is_train)` | RGB+Depth 동기화 변환 |
| `RGBDDataset(paths, labels, transform)` | (rgbd, aux, label) 반환 Dataset |

### edge_utils.py — Edge 전용

| API | 설명 |
|-----|------|
| `EdgeAuxResNet18(num_classes)` | Canny edge(3ch) + Aux(3) → 분류 |
| `EdgeTransform(image_size, is_train)` | RGB → Canny edge 텐서 변환 |
| `EdgeDataset(paths, labels, transform)` | (edge, aux, label) 반환 Dataset |

### rgbe_utils.py — RGBE 공통

| API | 설명 |
|-----|------|
| `RGBETransform(image_size, is_train)` | RGB + Canny edge 4채널 변환 |
| `RGBEDataset(paths, labels, transform)` | (rgbe, aux, label) 반환 Dataset |
| 모델은 `depth_utils.RGBDAuxResNet18` 재사용 | (RGBE도 4채널이므로 동일 구조) |

### camera_utils.py — ZED 카메라 관리

| API | 설명 |
|-----|------|
| `CameraManager(fps)` | ZED/OpenCV 자동 선택, 백그라운드 캡처 |
| `.start()` / `.stop()` | 캡처 시작/종료 |
| `.get_frame()` / `.get_depth()` | 최신 RGB/Depth 반환 |
| `.snapshot(temp_dir)` | 현재 프레임 저장 |
| `compute_blur_score(image)` | 블러 정도 점수 반환 |

### dimension_utils.py — 치수 측정

| API | 설명 |
|-----|------|
| `DimensionEngine(sam_path, use_sam, avg_window)` | 프레임별 치수 측정 엔진 |
| `.measure(frame, depth)` | rect/pca 치수 + 평균/표준편차 반환 |
| `measure_min_area_rect(mask, depth, intrinsics)` | 외곽선 기반 치수 |
| `measure_pca(mask, depth, intrinsics)` | PCA 기반 치수 |

---

## 5. 실전 예시 레시피

### 레시피 1: 처음부터 모델 학습 → 평가 → 추론

```bash
# 1) 실물 촬영 (웹 UI)
python 01_capture_dataset.py

# 2) SAM 마스크 생성
python 05_generate_masks.py --dataset real

# 3) 모델 학습 (기본 448×448)
python 02_door_classification_5090.py

# 4) 평가
python 03_door_class_evaluation_5090.py

# 5) 실시간 추론
python 04_door_realtime_inference.py
```

### 레시피 2: 224×224 경량 모델 학습 및 배포

```bash
# 1) 학습 (224 해상도, 전체 데이터)
python 02_door_classification_5090.py --image_size 224 --full_train

# 2) 결과를 artifacts_224에 보관
cp artifacts/best_door_model_5090.* artifacts_224/

# 3) 224 모델로 실시간 추론
python 04_door_realtime_inference.py \
    --model artifacts_224/best_door_model_5090.pth
```

### 레시피 3: 증강 데이터로 모델 강건성 테스트

```bash
# 1) 증강 데이터셋 생성
python 06_generate_augmented_dataset.py --dst datasets_aug --level extreme
python 06_generate_augmented_dataset.py --dst datasets_aug2 --level moderate

# 2) 448 Baseline 모델로 증강 데이터 평가
python 03_door_class_evaluation_5090.py \
    --model artifacts_448/best_door_model_5090.pth \
    --dataset_dir datasets_aug --use_all

# 3) 224 Baseline 모델과 비교
python 03_door_class_evaluation_5090.py \
    --model artifacts_224/best_door_model_5090.pth \
    --dataset_dir datasets_aug --use_all \
    --image_size 224
```

### 레시피 4: 모든 모델 × 모든 데이터셋 크로스 평가

```bash
# Baseline RGBD (448)
python 03_door_class_evaluation_5090.py \
    --model artifacts_448/best_door_model_5090.pth \
    --dataset_dir datasets_aug --use_all

# Texture Aug RGBD (448)
python 03_door_class_evaluation_5090.py \
    --model artifacts_448/best_door_texture_aug_model_5090.pth \
    --dataset_dir datasets_aug --use_all

# Edge-only (448)
python 03_door_class_evaluation_edge_5090.py \
    --model artifacts_448/best_door_edge_model_5090.pth \
    --dataset_dir datasets_aug --use_all

# RGBE Hybrid (448)
python 03_door_class_evaluation_rgbe_5090.py \
    --model artifacts_448/best_door_rgbe_model_5090.pth \
    --dataset_dir datasets_aug --use_all
```

### 레시피 5: CAD 사전학습 → 실물 파인튜닝

```bash
# 1) CAD 합성 데이터로 학습
python 02_door_cad_classification_5090.py

# 2) CAD 모델을 실물 데이터로 파인튜닝 검증
python 04_door_cad_finetune_real_5090.py

# 3) 파인튜닝 결과 평가
python 03_door_cad_evaluation_5090.py
```

### 레시피 6: 부품 치수 측정

```bash
# ZED 카메라 연결 후 실시간 치수 측정
python 99_dimension_measurement.py
# → http://0.0.0.0:5002 에서 width, height, aspect ratio 실시간 확인
```

---

## 부록: 모델 파일 대응표

| 모델 유형 | .pth 파일명 | .onnx 파일명 | 학습 스크립트 | 평가 스크립트 | 추론 스크립트 |
|----------|------------|-------------|-------------|-------------|-------------|
| Baseline RGBD | `best_door_model_5090.pth` | `best_door_model_5090.onnx` | `02_door_classification_5090.py` | `03_door_class_evaluation_5090.py` | `04_door_realtime_inference.py` |
| Texture Aug | `best_door_texture_aug_model_5090.pth` | `best_door_texture_aug_model_5090.onnx` | `02_door_classification_texture_aug_5090.py` | `03_door_class_evaluation_5090.py` (--model 지정) | `04_door_realtime_inference.py` (--model 지정) |
| Edge-only | `best_door_edge_model_5090.pth` | `best_door_edge_model_5090.onnx` | `02_door_classification_edge_5090.py` | `03_door_class_evaluation_edge_5090.py` | `04_door_realtime_inference_edge.py` |
| RGBE Hybrid | `best_door_rgbe_model_5090.pth` | `best_door_rgbe_model_5090.onnx` | `02_door_classification_rgbe_5090.py` | `03_door_class_evaluation_rgbe_5090.py` | `04_door_realtime_inference_rgbe.py` |
| CAD RGBD | `best_door_cad_model_5090.pth` | `best_door_cad_model_5090.onnx` | `02_door_cad_classification_5090.py` | `03_door_cad_evaluation_5090.py` | `04_door_realtime_inference.py` (--model 지정) |
| Finetune | `best_door_finetune_model_5090.pth` | `best_door_finetune_model_5090.onnx` | `04_door_cad_finetune_real_5090.py` | `03_door_class_evaluation_5090.py` (--model 지정) | `04_door_realtime_inference.py` (--model 지정) |
