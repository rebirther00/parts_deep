# 굴착기 Door 분류 프로젝트 — Plan & Design

## 1. 프로젝트 개요

굴착기 3개 모델(E25, E30, E38)의 Door 3종(door_RH, door_LH_FRT, door_LH_RR)을 분류하는 딥러닝 모델을 개발한다.
총 **9개 클래스**를 대상으로 하며, 실물 Door를 ZED X Mini 카메라로 촬영하여 데이터셋을 구축한다.

| 모델 | door_RH | door_LH_FRT | door_LH_RR |
|------|---------|-------------|------------|
| E25  | E25_door_RH | E25_door_LH_FRT | E25_door_LH_RR |
| E30  | E30_door_RH | E30_door_LH_FRT | E30_door_LH_RR |
| E38  | E38_door_RH | E38_door_LH_FRT | E38_door_LH_RR |

### 1.1 데이터 확보 현황

| 구분 | 상태 |
|------|------|
| 실물 Door | 각 Door별 1장씩 확보 완료 (총 9장) |
| CAD 모델 | 추후 확보 예정 |

---

## 2. 이미지 획득 방식 결정

### 2.1 방식 비교

| 항목 | 수동 캡처 (Snapshot) | 비디오 스트리밍 + 프레임 추출 |
|------|---------------------|-------------------------------|
| 속도 | 느림 (건당 버튼 클릭) | 빠름 (연속 촬영 후 일괄 추출) |
| 뷰포인트 다양성 | 의도적 배치 필요 | 자연스러운 연속 뷰포인트 |
| 이미지 품질 제어 | 매 촬영 시 확인 가능 | 후처리 선별 필요 |
| 대량 확보 효율 | 낮음 | 높음 |
| 실물 1장 상황 적합도 | 부적합 | **적합** |

### 2.2 결정: 비디오 스트리밍 + 프레임 추출 (하이브리드)

**비디오 스트리밍 + 프레임 추출** 방식을 채택한다.

**근거:**
- 실물이 Door당 1장뿐이므로, 하나의 Door를 다양한 각도/거리/조명에서 촬영해야 한다
- 비디오 녹화하면서 Door 주위를 이동하면 자연스러운 뷰포인트 다양성 확보 가능
- 설정 간격(예: 매 N 프레임)으로 프레임을 자동 추출하고, 그 중 양호한 이미지만 선별
- 수백 장을 일일이 캡처 버튼으로 찍는 것보다 훨씬 효율적
- 블러/흐릿한 프레임은 품질 필터로 자동 제거 가능

**추가로 수동 스냅샷 기능도 제공**하여, 특정 각도에서 정밀 촬영이 필요할 때 사용할 수 있게 한다.

---

## 3. 시스템 구성

### 3.1 하드웨어

| 항목 | 사양 |
|------|------|
| 카메라 | ZED X Mini (GMSL2) |
| 컴퓨팅 | Jetson (tegra 커널) |
| SDK | ZED SDK + pyzed |

### 3.2 소프트웨어 스택

```
ZED X Mini Camera
    │
    ▼
ZED SDK (pyzed.sl)
    │
    ▼
Flask Server (Python)
    ├── 실시간 스트리밍 (MJPEG)
    ├── 비디오 녹화 & 프레임 추출
    ├── 이미지 품질 필터링
    ├── 이미지 선별(큐레이션) UI
    └── 데이터셋 저장 관리
    │
    ▼
Dataset (기존 명세 준수)
    │
    ▼
Training / Evaluation (ResNet18)
```

---

## 4. 데이터셋 명세 (기존 포맷 준수)

### 4.1 디렉터리 구조

```
door/
├── datasets/
│   ├── dataset_info.json
│   ├── E25_door_RH/
│   │   ├── rgb_0000.png
│   │   ├── rgb_0001.png
│   │   ├── ...
│   │   └── metadata.json
│   ├── E25_door_LH_FRT/
│   ├── E25_door_LH_RR/
│   ├── E30_door_RH/
│   ├── E30_door_LH_FRT/
│   ├── E30_door_LH_RR/
│   ├── E38_door_RH/
│   ├── E38_door_LH_FRT/
│   └── E38_door_LH_RR/
└── artifacts/
    ├── class_names.json
    ├── best_parts_model.pth
    └── training_indices_parts.json
```

### 4.2 파일 명명 규칙

| 파일 유형 | 패턴 | 예시 |
|----------|------|------|
| RGB 이미지 | `rgb_{frame:04d}.png` | `rgb_0000.png`, `rgb_0127.png` |
| 메타데이터 | `metadata.json` | 클래스별 1개 |

> **참고:** 기존 프로젝트의 bounding_box 파일(`.npy`, `_labels.json`)은 Isaac Sim의 Replicator가 자동 생성하는 것이다.
> 실물 촬영에서는 bounding box 없이 **RGB 이미지만** 저장한다.
> 학습 시 `--bbox_crop` 옵션을 사용하지 않으면 bounding box 없이도 학습 가능하다.

### 4.3 dataset_info.json 형식

```json
{
  "dataset_name": "Excavator Door Classification Dataset",
  "num_classes": 9,
  "images_per_class": 0,
  "total_images": 0,
  "classes": {
    "E25_door_RH": "E25_door_RH",
    "E25_door_LH_FRT": "E25_door_LH_FRT",
    "E25_door_LH_RR": "E25_door_LH_RR",
    "E30_door_RH": "E30_door_RH",
    "E30_door_LH_FRT": "E30_door_LH_FRT",
    "E30_door_LH_RR": "E30_door_LH_RR",
    "E38_door_RH": "E38_door_RH",
    "E38_door_LH_FRT": "E38_door_LH_FRT",
    "E38_door_LH_RR": "E38_door_LH_RR"
  },
  "data_source": "real_camera",
  "camera": "ZED X Mini",
  "created_at": "",
  "note": "실물 Door 촬영 데이터. CAD 기반 합성 데이터는 추후 추가 예정."
}
```

### 4.4 클래스별 metadata.json 형식

```json
{
  "class_name": "E25_door_RH",
  "display_name": "E25 Door RH",
  "data_source": "real_camera",
  "camera": "ZED X Mini",
  "resolution": [1920, 1080],
  "num_images": 0,
  "capture_method": "video_frame_extraction",
  "frame_extraction_interval": 5,
  "quality_filter": true
}
```

---

## 5. Flask 서버 UI 설계

### 5.1 화면 구성

```
┌─────────────────────────────────────────────────────────┐
│                    Door Dataset Builder                   │
├──────────┬──────────────────────────────────────────────┤
│          │                                              │
│ 클래스   │        실시간 카메라 미리보기                    │
│ 선택     │        (MJPEG 스트리밍)                        │
│          │                                              │
│ ☐ E25_RH │                                              │
│ ☐ E25_FRT│                                              │
│ ☐ E25_RR │     ┌──────────────────────────┐             │
│ ☐ E30_RH │     │                          │             │
│ ☐ E30_FRT│     │     카메라 영상 출력       │             │
│ ☐ E30_RR │     │                          │             │
│ ☐ E38_RH │     └──────────────────────────┘             │
│ ☐ E38_FRT│                                              │
│ ☐ E38_RR │  [● 녹화 시작] [📸 스냅샷]  [프레임간격: 5]    │
│          │                                              │
├──────────┴──────────────────────────────────────────────┤
│  녹화 상태: 대기 중  │ 추출 프레임: 0  │ 저장 이미지: 0    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  추출된 프레임 미리보기 (그리드)                           │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │
│  │ img │ │ img │ │ img │ │ img │ │ img │ ...           │
│  │  ✓  │ │  ✗  │ │  ✓  │ │  ✓  │ │  ✗  │              │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘              │
│                                                         │
│  [선택 이미지 저장]  [전체 선택]  [전체 해제]              │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                 데이터셋 현황 대시보드                     │
│                                                         │
│  E25_door_RH: ██████░░░░ 120/500                        │
│  E25_door_LH_FRT: ████░░░░░░ 80/500                    │
│  E25_door_LH_RR: ███░░░░░░░ 60/500                     │
│  ...                                                    │
│  총: 720 / 4500                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 주요 기능

| 기능 | 설명 |
|------|------|
| **실시간 스트리밍** | ZED X Mini → MJPEG 스트리밍으로 브라우저에 표시 |
| **클래스 선택** | 9개 클래스 중 현재 촬영 대상 선택 |
| **비디오 녹화** | 녹화 시작/정지, 녹화 중 프레임을 메모리/임시 폴더에 저장 |
| **프레임 추출** | 설정 간격(N 프레임마다)으로 프레임 추출 |
| **품질 필터링** | 블러 감지(Laplacian variance)로 흐릿한 프레임 자동 제거 |
| **이미지 선별** | 추출된 프레임을 그리드로 표시, 사용자가 선택/해제 |
| **데이터셋 저장** | 선별된 이미지를 `datasets/<class_name>/rgb_NNNN.png`로 자동 번호 매김 저장 |
| **대시보드** | 클래스별 이미지 수, 전체 진행률 표시 |
| **스냅샷** | 수동 1장 캡처 기능 (특정 각도 촬영 시) |

### 5.3 프레임 추출 & 품질 필터링 흐름

```
비디오 녹화 중
    │
    ▼
매 N 프레임마다 프레임 추출
    │
    ▼
블러 감지 (Laplacian Variance)
    ├── threshold 이상 → 후보 프레임 (양호)
    └── threshold 미만 → 자동 제거 (흐릿)
    │
    ▼
후보 프레임 그리드 표시
    │
    ▼
사용자 수동 선별 (체크/언체크)
    │
    ▼
선별 이미지 → datasets/<class_name>/rgb_NNNN.png 저장
```

---

## 6. 프로젝트 파일 구조

```
parts_deep/class_estimation/door/
│
├── DOOR_CLASSIFICATION_PLAN.md                # 본 문서 (Plan & Design)
├── requirements.txt                           # 의존성 패키지
├── camera_utils.py                            # 카메라 유틸리티
│
├── 01_capture_dataset.py                      # Flask 서버 + ZED 카메라 이미지 획득
├── 02_door_classification_5090.py             # 분류 모델 학습 (ResNet18, RTX 5090)
├── 03_door_class_evaluation_5090.py           # 모델 평가 (RTX 5090)
│
├── templates/
│   └── index.html                             # Flask UI 템플릿
│
├── static/
│   ├── css/
│   │   └── style.css                          # UI 스타일
│   └── js/
│       └── app.js                             # UI 동작 (AJAX, 이미지 선별)
│
├── datasets/
│   ├── dataset_info.json
│   ├── E25_door_LH_FRT/                       # 110장 (촬영 완료)
│   ├── E25_door_LH_RR/                        # 미촬영
│   ├── E25_door_RH/                           # 미촬영
│   ├── E30_door_LH_FRT/                       # 108장 (촬영 완료)
│   ├── E30_door_LH_RR/                        # 미촬영
│   ├── E30_door_RH/                           # 미촬영
│   ├── E38_door_LH_FRT/                       # 119장 (촬영 완료)
│   ├── E38_door_LH_RR/                        # 미촬영
│   └── E38_door_RH/                           # 미촬영
│
└── artifacts/                                 # .gitignore 대상
    ├── best_door_model_5090.pth               # 학습된 모델 가중치
    ├── best_door_model_5090.onnx              # ONNX 변환 모델
    ├── class_names_door_5090.json             # 클래스 이름
    ├── training_indices_door_5090.json        # Train/Test 분할 정보
    ├── evaluation_results_door_5090.json      # 평가 결과
    └── evaluation_results_door_5090.png       # 평가 시각화
```

---

## 7. 실행 파이프라인

### Phase 1: 데이터셋 구축 (본 단계)

```
Step 1) Flask 서버 실행
        $ python 01_capture_dataset.py
        → 브라우저에서 http://localhost:5000 접속

Step 2) 각 Door 촬영
        → 클래스 선택 → 녹화 → 프레임 추출 → 선별 → 저장
        → 9개 클래스 반복

Step 3) 데이터셋 확인
        → 대시보드에서 클래스별 이미지 수 확인
        → 목표: 클래스당 최소 200~500장
```

### Phase 2: 학습 & 평가 (데이터셋 구축 완료 후)

```
Step 4) 학습
        $ python 02_door_classification.py

Step 5) 평가
        $ python 03_door_class_evaluation.py
```

### Phase 3: CAD 기반 합성 데이터 추가 (추후)

```
Step 6) CAD 모델 확보 후 Isaac Sim으로 합성 데이터 생성
Step 7) 실물 + 합성 데이터 혼합 학습
```

---

## 8. Flask 서버 API 설계

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | 메인 UI 페이지 |
| `/video_feed` | GET | MJPEG 스트리밍 |
| `/api/start_recording` | POST | 녹화 시작 (class_name, frame_interval 전달) |
| `/api/stop_recording` | POST | 녹화 정지 → 프레임 추출 결과 반환 |
| `/api/snapshot` | POST | 수동 스냅샷 1장 캡처 |
| `/api/extracted_frames` | GET | 추출된 프레임 목록 (썸네일 포함) |
| `/api/save_selected` | POST | 선별된 프레임을 데이터셋으로 저장 |
| `/api/delete_frame` | DELETE | 추출 프레임 삭제 |
| `/api/dataset_status` | GET | 클래스별 이미지 수 현황 |
| `/api/classes` | GET | 사용 가능한 클래스 목록 |

---

## 9. 촬영 가이드라인

데이터 품질을 위해 아래 사항을 준수한다.

### 9.1 촬영 환경

- **배경**: 가능한 단순한 배경 (공장 바닥, 작업대 등)
- **조명**: 균일한 조명, 역광 회피
- **거리**: 0.5m ~ 3m (Door 전체가 프레임에 들어오도록)

### 9.2 촬영 전략

| 항목 | 권장 사항 |
|------|----------|
| **각도 다양성** | 정면, 좌/우 45°, 좌/우 90°, 상/하 30° 등 |
| **거리 다양성** | 근접(0.5m), 중간(1.5m), 원거리(3m) |
| **회전** | Door 주위를 천천히 한 바퀴 돌며 녹화 |
| **프레임 추출 간격** | 5~10 프레임 권장 (30fps 기준 0.17~0.33초) |
| **클래스당 목표** | 최소 200장, 권장 500장 |

### 9.3 품질 기준

- Laplacian Variance 기반 블러 감지 (threshold: 100)
- 흐릿한 프레임 자동 제거
- 심하게 잘린 이미지 수동 제거

---

## 10. 기술 스택 & 의존성

| 패키지 | 용도 |
|--------|------|
| `pyzed` (ZED SDK) | ZED X Mini 카메라 제어 |
| `flask` | 웹 서버 & API |
| `opencv-python` | 이미지 처리, 블러 감지, MJPEG 인코딩 |
| `numpy` | 이미지 배열 처리 |
| `torch`, `torchvision` | 분류 모델 학습 (Phase 2) |
| `scikit-learn` | 데이터 분할, 평가 지표 (Phase 2) |
| `matplotlib` | 학습 결과 시각화 (Phase 2) |

---

## 11. 기존 프로젝트와의 차이점

| 항목 | 기존 (parts classification) | 신규 (door classification) |
|------|---------------------------|---------------------------|
| 데이터 소스 | Isaac Sim 합성 이미지 | 실물 카메라 촬영 |
| 카메라 | 시뮬레이션 카메라 | ZED X Mini |
| 클래스 수 | 4 (boom/arm × 25/30) | 9 (E25/E30/E38 × 3 door) |
| Bounding Box | 자동 생성 (Replicator) | 없음 (RGB only) |
| 배경 | Domain Randomization | 실제 촬영 환경 |
| 이미지 해상도 | 1024×1024 | ZED X Mini 출력 (1920×1080 또는 설정값) |
| 데이터 수집 도구 | Python 스크립트 | Flask 웹 UI |
| 이미지 파일명 | `rgb_NNNN.png` | `rgb_NNNN.png` (동일) |
| 메타데이터 | `metadata.json` | `metadata.json` (동일, 필드 확장) |

---

## 12. 진행 이력

### 2026-03-13: 실제 이미지 취득 및 3종 분류 모델 학습/평가 완료

#### 12.1 데이터셋 취득

`01_capture_dataset.py` Flask 웹 UI를 통해 ZED X Mini 카메라로 3종 도어(LH_FRT)를 촬영하여 데이터셋 구축 완료.

| 클래스 | 이미지 수 | 해상도 |
|--------|-----------|--------|
| E25_door_LH_FRT | 110장 | 1920×1080 |
| E30_door_LH_FRT | 108장 | 1920×1080 |
| E38_door_LH_FRT | 119장 | 1920×1080 |
| **총합** | **337장** | |

- 나머지 6개 클래스(LH_RR, RH)는 실물 미확보로 데이터 없음 (폴더만 생성)
- 우선 **3종 분류**로 연습 진행

#### 12.2 학습 스크립트 구현 (`02_door_classification_5090.py`)

기존 `02_parts_classification_5090.py`를 참조하여 door 전용 학습 스크립트 작성.

| 항목 | 설정 |
|------|------|
| 모델 | ResNet18 (ImageNet 사전학습, Transfer Learning) |
| 데이터 분할 | Train 80% (269장) / Test 20% (68장), stratify 적용 |
| 배치 사이즈 | 64 (RTX 5090 자동 조정) |
| 에포크 | 60 (Early Stopping patience=10) |
| 옵티마이저 | Adam (lr=0.001) + ReduceLROnPlateau |
| 데이터 증강 | RandomHorizontalFlip, RandomRotation(15°), **RandomPerspective(0.2)**, ColorJitter |
| 클래스 가중치 | CrossEntropyLoss에 불균형 보정 적용 |
| ONNX 변환 | 학습 완료 후 자동 변환 (ZED Box Mini 추론용, opset 18) |

기존 스크립트와의 주요 차이점:
- `bbox_crop` 옵션 제거 (실제 이미지에는 bbox 데이터 없음)
- 이미지가 0인 클래스 폴더 자동 제외 로직 추가
- `RandomPerspective` 증강 추가 (실제 촬영 시점 변화 시뮬레이션)
- 학습 완료 후 ONNX 변환 단계 추가 (9단계)

#### 12.3 평가 스크립트 구현 (`03_door_class_evaluation_5090.py`)

기존 `03_parts_class_evaluation_5090.py`를 참조하여 door 전용 평가 스크립트 작성.

- Test 셋(68장) 예측 결과 출력
- 결과 시각화 그리드 이미지 생성
- 틀린 예측 그리드 이미지 생성
- 혼동 행렬 출력
- Precision, Recall, F1-Score 계산

#### 12.4 학습/평가 결과

**학습 결과:**
- Epoch 17에서 Validation Accuracy **100.00%** 달성
- Epoch 27에서 Early Stopping 발동
- 학습 시간: 약 1.7분 (RTX 5090)

**평가 결과:**

| 지표 | 값 |
|------|-----|
| 전체 정확도 | **100.00%** (68/68) |
| E25_door_LH_FRT | 100.00% (22/22) |
| E30_door_LH_FRT | 100.00% (22/22) |
| E38_door_LH_FRT | 100.00% (24/24) |
| Precision (Macro) | 100.00% |
| Recall (Macro) | 100.00% |
| F1-Score (Macro) | 100.00% |
| 오류 | 0개 |

> **참고**: 3종 분류에서 100% 정확도는 데이터가 클래스 간 시각적 차이가 뚜렷하기 때문.
> 6종/9종으로 확장 시 정확도가 낮아질 수 있으며, 추가 데이터 수집 및 증강이 필요할 수 있음.

#### 12.5 생성된 Artifacts

```
artifacts/
├── best_door_model_5090.pth          # 학습된 모델 가중치 (44MB)
├── best_door_model_5090.onnx         # ONNX 변환 모델 (ZED Box Mini용)
├── best_door_model_5090.onnx.data    # ONNX 외부 데이터
├── class_names_door_5090.json        # 클래스 이름 목록
├── training_indices_door_5090.json   # Train/Test 분할 정보
├── evaluation_results_door_5090.json # 평가 결과 JSON
└── evaluation_results_door_5090.png  # 평가 결과 시각화 이미지
```

#### 12.6 실행 방법

```bash
# conda 환경 활성화
conda activate isaac311

# 학습 실행
python class_estimation/door/02_door_classification_5090.py

# 평가 실행
python class_estimation/door/03_door_class_evaluation_5090.py

# CPU로 실행 시
python class_estimation/door/02_door_classification_5090.py -cpu
```

---

### 2026-03-13 (2차): 실시간 추론 서버 구현 및 TensorRT 호환성 이슈 해결

#### 12.7 실시간 추론 서버 구현 (`04_door_realtime_inference.py`)

ZED X Mini 카메라 + 학습된 모델을 사용하여 3종 도어를 실시간 분류하는 Flask 웹 서버 구현.

| 항목 | 내용 |
|------|------|
| 서버 포트 | 5001 (데이터셋 빌더 5000과 분리) |
| 추론 간격 | 0.5초 |
| UI 구성 | 카메라 스트리밍 + 분류 결과 오버레이 + 클래스별 확률 바 |
| API | `/api/inference_result` (추론 결과 JSON) |

#### 12.8 TensorRT 변환 시 모델 출력 불일치 문제 발견

최초 구현 시 TensorRT 기반 추론 엔진을 사용했으나, **ONNX → TensorRT 변환 시 모델 출력이 완전히 달라지는 치명적 문제** 발견.

**검증 과정:**

1. `03_5_trt_evaluation.py` 스크립트를 작성하여 TRT 엔진으로 테스트셋 평가
2. TRT 결과: **32.35% 정확도** — E30, E38을 모두 E25로 분류
3. 동일한 입력으로 PyTorch vs TRT 출력 직접 비교:
   - 입력 데이터 차이: **0** (완전 동일)
   - 출력 차이: **최대 11.55** (완전히 다른 값)
4. ONNX 파일 자체 검증: 가중치 내장 정상, 빈 파라미터 없음 (43.13 MB)
5. opset 13 → 17 변경 후에도 동일한 문제 재현
6. PyTorch 직접 평가: **98.53%** 정확도 확인 → ONNX/TRT 변환 과정의 문제

**원인 추정:**
- Jetson Orin (aarch64, TensorRT 10.3) + PyTorch 2.10 ONNX export 호환성 문제
- ResNet18의 커스텀 fc 레이어(Dropout + Linear + ReLU + Dropout + Linear)가 TRT 최적화 과정에서 잘못 변환된 것으로 추정

**해결:**
- TensorRT → **PyTorch (.pth) 직접 추론**으로 전환
- CPU 모드 사용 (Jetson용 PyTorch CUDA wheel이 아닌 x86 wheel이므로 GPU 사용 불가)
- 추론 시간: 86ms/장 (CPU) — 0.5초 간격 추론에 충분

#### 12.9 추론 파이프라인 평가 스크립트 (`03_5_trt_evaluation.py`)

실시간 추론(04) 실행 전, 테스트 데이터셋에 대해 동일한 파이프라인으로 정확도를 검증하는 스크립트.

| 항목 | 값 |
|------|-----|
| 전체 정확도 | **98.53%** (67/68) |
| E25_door_LH_FRT | 100.0% (22/22) |
| E30_door_LH_FRT | 95.5% (21/22) |
| E38_door_LH_FRT | 100.0% (24/24) |
| 오분류 | 1건 (E30 → E25, 59.3% 확률) |
| 추론 엔진 | PyTorch 2.10.0 (CPU) |
| 평균 추론 시간 | 86.18 ms/장 |

#### 12.10 업데이트된 파일 구조

```
parts_deep/class_estimation/door/
│
├── 01_capture_dataset.py              # 데이터셋 캡처 서버 (포트 5000)
├── 02_door_classification_5090.py     # 학습 (RTX 5090)
├── 03_door_class_evaluation_5090.py   # PyTorch 평가 (RTX 5090)
├── 03_5_trt_evaluation.py             # 추론 파이프라인 평가 (Jetson)
├── 04_door_realtime_inference.py      # 실시간 추론 서버 (포트 5001)
├── camera_utils.py                    # 카메라 유틸리티
│
├── templates/
│   ├── index.html                     # 데이터셋 빌더 UI
│   └── inference.html                 # 실시간 추론 UI
│
└── artifacts/
    ├── best_door_model_5090.pth       # 학습된 모델 (44MB)
    ├── best_door_model_5090.onnx      # ONNX 모델 (TRT 비호환, 참조용)
    ├── class_names_door_5090.json     # 클래스 이름
    ├── training_indices_door_5090.json
    ├── inference_evaluation_results.json  # 추론 파이프라인 평가 결과
    └── evaluation_results_door_5090.*
```

#### 12.11 Jetson 환경 (ZED Box Mini) 실행 방법

```bash
# zed_env 가상환경 활성화
source ~/workspace/zed_env/bin/activate

# 추론 전 모델 검증
python class_estimation/door/03_5_trt_evaluation.py

# 실시간 추론 서버 실행
python class_estimation/door/04_door_realtime_inference.py
# → 브라우저에서 http://<장비IP>:5001 접속
```

#### 12.12 주요 교훈

1. **ONNX → TensorRT 변환은 반드시 출력 검증 필요** — 변환 성공해도 출력이 다를 수 있음
2. **03_5 같은 파이프라인 검증 스크립트가 필수** — 실시간 추론 전 테스트셋으로 정확도 확인
3. **Jetson 환경에서는 NVIDIA 공식 PyTorch wheel 사용 권장** — pip의 x86 wheel은 CUDA 미지원

---

## 13. 향후 확장 계획

### 13.1 정확도 개선 (우선)

1. **Unknown 클래스 추가**: 비도어 이미지(바닥, 벽, 공구, 사람 등)를 `datasets/Unknown/`에 수집하여 4클래스로 재학습
   - 현재 문제: 도어가 아닌 물체를 보여줘도 E38 등 특정 클래스를 100% 확률로 분류 (Closed-set 한계)
   - 목표: 100장 이상의 비도어 이미지를 수집하여 "도어 아님" 판별 가능하게 개선
2. **조명 환경 대응 강화**: 다양한 조명 조건(자연광, 형광등, 역광, 저조도 등)에서 추가 촬영
   - 현재 문제: 학습 데이터와 다른 조명 환경에서 E30의 분류 정확도 저하 (E25/E38으로 오분류)
   - 해결: 조명별 추가 데이터 수집 + ColorJitter 증강 파라미터 강화 (brightness, contrast 범위 확대)
3. **E30 데이터 보강**: E30 도어를 다양한 각도/거리/조명에서 50~100장 추가 촬영
   - E25↔E30 간 시각적 유사성으로 인한 혼동 해소

### 13.2 클래스 확장

4. **나머지 6종 데이터 수집**: LH_RR, RH 도어 실물 확보 후 촬영 → 9클래스 + Unknown = 10클래스 분류

### 13.3 데이터 확장

5. **CAD 기반 합성 데이터**: CAD 모델 확보 후 Isaac Sim으로 추가 합성 데이터 생성
6. **데이터 증강 강화**: RandomPerspective 비율 증가, 밝기/대비/채도 변화 확대, 노이즈 추가

### 13.4 모델 및 배포 최적화

7. **모델 고도화**: ResNet18 → EfficientNet 등 경량 모델 비교 실험
8. **Jetson GPU 추론**: NVIDIA 공식 Jetson PyTorch wheel 설치 후 GPU 추론 전환 (86ms → ~10ms 예상)
9. **TensorRT 재시도**: Jetson 전용 PyTorch wheel + ONNX opset 조합 테스트로 TRT 호환성 확보
