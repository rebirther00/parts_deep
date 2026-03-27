# Door 분류 RGBD 마이그레이션 계획서

**작성일**: 2026-03-23  
**목적**: RGB 3채널 → RGBD 4채널 입력으로 전환하여 합성↔실물 Domain Gap 해소

---

## 1. 왜 Depth를 추가하는가

### 현재 문제
- CAD 합성 데이터로 학습한 모델이 실물 이미지에서 36.4% 정확도
- RGB의 색상/질감/조명/배경이 합성 vs 실물 간 완전히 다름

### Depth의 이점
| 특성 | RGB | Depth |
|------|-----|-------|
| 합성↔실물 일관성 | 낮음 (색상, 질감 다름) | **높음 (3D 형상 동일)** |
| E30 vs E38 구분 (47mm 차이) | 카메라 거리에 따라 왜곡 | **미터 단위 실측 가능** |
| 배경 분리 | 불가능 | **거리 기반 자동 분리** |
| 조명 영향 | 큼 | **없음** |

---

## 2. 전체 파일 목록 및 용도

| # | 파일 | 용도 | Depth 수정 |
|---|------|------|------------|
| 0 | `00_convert_stl_to_usd.py` | STL → USD 변환 (mm→m 스케일링 + 원점 센터링) | 불필요 |
| 1 | `01_capture_dataset.py` | Flask 웹 서버로 ZED X Mini에서 실물 이미지 수집 | **RGB+Depth 동시 저장** |
| 2 | `01_generate_door_cad_dataset.py` | Isaac Sim + Omni Replicator로 합성 데이터 생성 | **Depth 맵 추가 생성** |
| 3 | `02_door_classification_5090.py` | 실물 데이터 학습 (ResNet18, RTX 5090 최적화) | **RGBD 4채널 입력** |
| 4 | `02_door_cad_classification_5090.py` | CAD 합성 데이터 학습 (ResNet18, 혼동 행렬 포함) | **RGBD 4채널 입력** |
| 5 | `03_door_class_evaluation_5090.py` | 실물 모델 평가 (시각화, Precision/Recall/F1) | **RGBD 4채널 입력** |
| 6 | `03_door_cad_evaluation_5090.py` | CAD 모델 평가 + 크로스 도메인 (`--use_all`) | **RGBD 4채널 입력** |
| 7 | `03_5_trt_evaluation.py` | 추론 파이프라인 사전 검증 (04와 동일 전처리) | **RGBD 4채널 입력** |
| 8 | `04_door_realtime_inference.py` | ZED X Mini 실시간 추론 Flask 서버 (port 5001) | **RGBD 4채널 입력** |
| 9 | `04_door_cad_finetune_real_5090.py` | CAD 사전학습 → 실물 Fine-tuning 검증 | **RGBD 4채널 입력** |
| 10 | `camera_utils.py` | ZED X Mini / OpenCV 카메라 관리 모듈 | **Depth 프레임 반환 추가** |

---

## 3. 데이터 명세

### 3-1. 파일 저장 형식

```
datasets/                          (또는 datasets_cad/)
├── E25_door_LH_FRT/
│   ├── rgb_0000.png               # RGB 이미지 (1920x1080 또는 1024x1024)
│   ├── depth_0000.png             # Depth 이미지 (16bit unsigned, mm 단위)
│   ├── rgb_0001.png
│   ├── depth_0001.png
│   ├── ...
│   └── metadata.json
├── E25_door_LH_RR/
│   └── ...
└── dataset_info.json
```

### 3-2. Depth 이미지 규격

| 항목 | 실물 (ZED X Mini) | 합성 (Isaac Sim) |
|------|-------------------|------------------|
| 형식 | 16bit PNG | 16bit PNG |
| 단위 | **밀리미터 (mm)** | **밀리미터 (mm)** |
| 해상도 | 1920x1080 (RGB와 동일) | 1024x1024 (RGB와 동일) |
| 유효 범위 | 0~10000 (10m) | 0~10000 (10m) |
| 무효값 | 0 (측정 불가) | 0 (무한 거리) |

### 3-3. Depth 정규화 (학습/추론 시)

```python
# Depth PNG 로드 후 정규화 (0~1 범위)
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16, mm 단위
depth_m = depth_raw.astype(np.float32) / 1000.0            # mm → m
depth_norm = np.clip(depth_m / 5.0, 0.0, 1.0)             # 0~5m 범위를 0~1로
```

- 5m 클리핑: 도어 촬영 거리가 1~3m이므로 5m면 충분
- 0값(무효)은 0.0으로 유지 → 모델이 "측정 불가" 영역을 학습

### 3-4. 모델 입력 텐서

```
기존: [B, 3, 224, 224]  → R, G, B
변경: [B, 4, 224, 224]  → R, G, B, D(정규화)
```

---

## 4. 파일별 수정 상세

### 4-1. `camera_utils.py` — Depth 반환 추가

**현재**: `_grab_frame()` → RGB 프레임만 반환  
**변경**: RGB + Depth 프레임을 동시에 반환

```python
# 현재
def _grab_frame(self):
    self._zed.retrieve_image(self._zed_image, sl.VIEW.LEFT)
    return cv2.cvtColor(self._zed_image.get_data(), cv2.COLOR_BGRA2BGR)

# 변경
def _grab_frame(self):
    self._zed.retrieve_image(self._zed_image, sl.VIEW.LEFT)
    rgb = cv2.cvtColor(self._zed_image.get_data(), cv2.COLOR_BGRA2BGR)
    self._zed.retrieve_measure(self._zed_depth, sl.MEASURE.DEPTH)
    depth = self._zed_depth.get_data()  # float32, mm 단위
    return rgb, depth
```

**핵심 변경점:**
- `sl.Mat()` 인스턴스 `self._zed_depth` 추가
- `get_frame()` → `(rgb, depth)` 튜플 반환
- `latest_frame` → `latest_rgb`, `latest_depth` 분리
- OpenCV 폴백: depth는 None 반환 (depth 카메라 없는 환경)

### 4-2. `01_capture_dataset.py` — RGB+Depth 동시 저장

**현재**: `rgb_XXXX.png`만 저장  
**변경**: `rgb_XXXX.png` + `depth_XXXX.png` 쌍으로 저장

**핵심 변경점:**
- `_save_temp()`: depth를 16bit PNG로 저장
- `api_save_selected()`: rgb + depth 쌍으로 복사
- `_next_rgb_index()` → `_next_pair_index()`: rgb/depth 쌍 인덱스 관리

### 4-3. `01_generate_door_cad_dataset.py` — Isaac Sim Depth 생성

**현재**: `BasicWriter(rgb=True)` → RGB만 생성  
**변경**: `BasicWriter(rgb=True, distance_to_camera=True)` → RGB + Depth 생성

**핵심 변경점:**
- Writer 설정에 `distance_to_camera=True` 추가
- Isaac Sim의 depth는 float32(m 단위) → uint16(mm 단위) 변환 후 저장
- 출력: `rgb_XXXX.png` + `depth_XXXX.png`

### 4-4. `02_door_classification_5090.py` / `02_door_cad_classification_5090.py` — RGBD 학습

**현재**: RGB 3채널 입력, ResNet18(in_channels=3)  
**변경**: RGBD 4채널 입력, ResNet18(in_channels=4)

**핵심 변경점:**

```python
# ResNet18 첫 Conv 레이어 수정
def create_resnet_model(num_classes, pretrained=True):
    model = models.resnet18(weights=weights)
    
    # 첫 Conv: 3ch → 4ch (기존 RGB 가중치 보존 + D채널 초기화)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3] = old_conv.weight  # RGB 가중치 복사
        model.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)  # D: RGB 평균으로 초기화
    
    # FC 헤드는 동일
    ...
```

**Dataset 클래스 변경:**
```python
def __getitem__(self, idx):
    rgb = Image.open(rgb_path).convert('RGB')
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16
    depth_norm = np.clip(depth.astype(np.float32) / 5000.0, 0, 1)  # mm→0~1
    
    # RGB transform 적용 후 Depth 채널 결합
    rgb_tensor = self.rgb_transform(rgb)          # [3, 224, 224]
    depth_tensor = torch.from_numpy(depth_norm)    # [H, W]
    depth_tensor = F.interpolate(...)              # [1, 224, 224]
    
    return torch.cat([rgb_tensor, depth_tensor], dim=0), label  # [4, 224, 224]
```

**Augmentation 주의:**
- `RandomRotation`, `Letterbox Resize + Pad`: RGB와 Depth에 **동일하게 적용** 필요
- `ColorJitter`: **RGB에만 적용**, Depth에는 적용하지 않음
- `Normalize`: RGB는 ImageNet 정규화, Depth는 별도 (mean=0.5, std=0.25)

### 4-5. `03_*.py` 평가 스크립트들 — RGBD 4채널 입력

**변경**: 02와 동일한 Dataset/모델 구조로 변경
- `DoorDataset`에서 RGB+Depth 로딩
- `create_resnet_model`에서 4채널 입력

### 4-6. `03_5_trt_evaluation.py` — 추론 파이프라인 검증

**현재**: RGB만 로딩 + Resize((224,224))로 강제 리사이즈  
**변경**: RGBD 4채널 로딩 + 종횡비 유지

### 4-7. `04_door_realtime_inference.py` — 실시간 RGBD 추론

**현재**: `camera.get_frame()` → RGB → model(3ch)  
**변경**: `camera.get_frame()` → (RGB, Depth) → model(4ch)

```python
def infer(self, rgb: np.ndarray, depth: np.ndarray) -> tuple:
    rgb_tensor = self.rgb_transform(pil_img)       # [3, 224, 224]
    depth_tensor = self.depth_transform(depth)     # [1, 224, 224]
    inp = torch.cat([rgb_tensor, depth_tensor]).unsqueeze(0)  # [1, 4, 224, 224]
    
    with torch.no_grad():
        output = self.model(inp)
    ...
```

### 4-8. `04_door_cad_finetune_real_5090.py` — Fine-tuning

**변경**: 02와 동일한 RGBD Dataset/모델 구조

---

## 5. 구현 순서 (의존성 기반)

```
Phase 1: 기반 모듈
  ├─ camera_utils.py (Depth 반환)
  └─ 공통 유틸: depth 정규화, RGBD Dataset 클래스

Phase 2: 데이터 생성
  ├─ 01_capture_dataset.py (실물 RGB+Depth 수집)
  └─ 01_generate_door_cad_dataset.py (합성 RGB+Depth 생성)

Phase 3: 학습
  ├─ 02_door_cad_classification_5090.py (CAD RGBD 학습)
  └─ 02_door_classification_5090.py (실물 RGBD 학습)

Phase 4: 평가
  ├─ 03_door_cad_evaluation_5090.py
  ├─ 03_door_class_evaluation_5090.py
  └─ 03_5_trt_evaluation.py

Phase 5: 추론 + Fine-tuning
  ├─ 04_door_realtime_inference.py
  └─ 04_door_cad_finetune_real_5090.py
```

---

## 6. 공통 모듈 신규 생성 (중복 방지)

`depth_utils.py` — Depth 관련 공통 함수

```python
def load_depth(path, max_depth_m=5.0):
    """16bit PNG depth → 정규화된 float32 [0, 1]"""
    
def save_depth(depth_float, path):
    """float32 depth (m) → 16bit PNG (mm)"""

def create_rgbd_resnet18(num_classes, pretrained=True):
    """4채널 입력 ResNet18 생성 (RGB 가중치 보존)"""

class RGBDDataset(Dataset):
    """RGB + Depth 쌍을 로딩하는 공통 Dataset"""
```

이렇게 하면 02, 03, 04 스크립트에서 중복 코드를 최소화할 수 있습니다.

---

## 7. 클래스 변경 반영 (9클래스 → 8클래스)

기존 9클래스에서 `E30_door_RH` + `E38_door_RH` → `E30_E38_door_RH`로 통합됨.

| 파일 | 클래스 정의 방식 | 반영 상태 |
|------|------------------|-----------|
| `01_capture_dataset.py` | CLASSES 리스트 하드코딩 | **수정 완료** (9→8클래스) |
| `01_generate_door_cad_dataset.py` | DOOR_CLASSES dict 하드코딩 | 이미 8클래스 |
| `02_*.py` | 폴더 스캔 + `class_names_*.json` 저장 | 자동 반영 |
| `03_*.py` | `class_names_*.json` 로드 | 자동 반영 |
| `03_5_trt_evaluation.py` | `class_names_*.json` 로드 | 자동 반영 |
| `04_door_realtime_inference.py` | `class_names_*.json` 로드 | 자동 반영 |
| `datasets/` 폴더 | E30_door_RH, E38_door_RH 삭제 → E30_E38_door_RH 생성 | 완료 |

---

## 8. 리스크 및 고려사항

| 리스크 | 대응 |
|--------|------|
| ZED Depth 노이즈 (반사 금속 표면) | Depth 중앙값 필터링, 무효값(0) 처리 |
| Isaac Sim Depth vs ZED Depth 단위 차이 | 둘 다 mm로 통일 후 동일 정규화 |
| Augmentation 시 RGB/Depth 동기화 | 동일 랜덤 시드로 transform 적용 |
| OpenCV 폴백 시 Depth 없음 | Depth=None이면 0으로 채움 (graceful degradation) |
| 기존 RGB only 모델과 호환성 | 별도 artifact 이름으로 저장 (혼용 방지) |

---

## 9. 예상 효과

| 시나리오 | RGB only | RGBD (예상) |
|----------|----------|-------------|
| CAD 합성 데이터 학습 (테스트셋) | 98.5% | 98%+ |
| 크로스 도메인 (합성→실물) | 36.4% | **70~85%** |
| Fine-tuning 후 | 72.2% | **85~95%** |
| E30 vs E38 FRT 구분 | ~50% | **80~90%** (Depth로 실측 크기 비교) |
