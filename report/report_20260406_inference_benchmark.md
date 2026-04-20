# RGBE Hybrid vs RGBD + Aux MLP 추론 속도 벤치마크 보고서

**과제명**: 건설장비 부품인식 및 위치추정 기술 개발
**연구 기간**: 1차년도
**작성일**: 2026년 4월 6일
**작성자**: 한국건설기계연구원 스마트건설장비연구실

---

## 1. 실험 목적

도어 부품 분류 모델 후보인 **RGBE Hybrid (RGB + Canny Edge)** 모델과 **RGBD + Aux MLP (RGB + Depth + 물리치수)** 모델의 추론 속도를 정량적으로 비교한다. 두 모델은 분류 성능(Accuracy, F1, Recall) 측면에서는 유사하지만, 입력 채널 구성과 전처리 파이프라인이 다르므로 실시간/엣지 디바이스 배포 시 처리 지연이 달라질 수 있다. 본 보고서는 데스크탑 환경(GPU/CPU)에서 사전 측정을 수행하여, 추후 엣지 디바이스(ZED Box Mini 등) 실측 시 비교 기준으로 활용한다.

---

## 2. 측정 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| OS | Linux 6.17.0-14-generic |
| 입력 해상도 | 448 × 448 |
| 측정 스크립트 | `class_estimation/door_paper/benchmark_inference.py` |
| 측정 방식 | warmup 후 단일 forward 시간을 다회 반복하여 mean / std / p50 / p95 / p99 산출 |

> 모델 가중치는 랜덤 초기화 상태로 측정한다. **추론 속도는 가중치 값이 아니라 모델 구조와 텐서 형상에만 의존**하므로 학습된 가중치 유무는 결과에 영향을 주지 않는다.

---

## 3. 비교 대상 모델

| 항목 | RGBE Hybrid | RGBD + Aux MLP |
|------|:-----------:|:--------------:|
| Backbone | ResNet18 | ResNet18 |
| 입력 채널 | 4 (RGB + Canny Edge) | 4 (RGB + Depth) |
| 추가 분기 | 없음 | Aux MLP (3 → 32) |
| 추가 입력 | – | 물리 치수 3개 (width, height, aspect ratio) |
| 파라미터 수 | **11,313,032** | **11,321,352** |
| 차이 | – | **+8,320 (+0.074%)** |

---

## 4. 측정 결과

### 4.1 GPU Forward 추론 속도 (배치별, 300회 평균)

| 배치 크기 | RGBE Hybrid (mean ± std) | RGBD + Aux MLP (mean ± std) | 차이 |
|:---------:|:-----------------------:|:---------------------------:|:----:|
| 1  | 0.835 ms ± 0.046 (1,197 FPS) | 0.833 ms ± 0.047 (1,200 FPS) | **−0.25%** |
| 4  | 1.524 ms ± 0.057 (2,624 FPS) | 1.539 ms ± 0.058 (2,599 FPS) | +0.94% |
| 8  | 2.738 ms ± 0.100 (2,922 FPS) | 2.752 ms ± 0.100 (2,907 FPS) | +0.50% |
| 16 | 5.025 ms ± 0.169 (3,184 FPS) | 5.037 ms ± 0.169 (3,176 FPS) | +0.24% |

→ **모든 배치 크기에서 모델 간 forward 시간 차이는 1% 이내**. Aux MLP가 추가되어도 GPU 추론 속도에는 사실상 영향이 없다 (Linear 2층은 GPU에서 무시 가능한 수준).

### 4.2 CPU Forward 추론 속도 (Edge 디바이스 시뮬레이션, 50회 평균)

| 모델 | mean ± std | p50 | FPS |
|------|:----------:|:---:|:---:|
| RGBE Hybrid    | 1,391.7 ms ± 270 | 1,474 ms | 0.7 |
| RGBD + Aux MLP |   812.7 ms ± 423 |   762 ms | 1.2 |

> CPU 단독 측정은 시스템 부하의 영향으로 **분산이 매우 큼** (std가 평균의 30~50% 수준). 데스크탑 CPU는 두 모델 모두 실시간 추론이 불가능하므로, 이 수치는 정성적 참고용으로만 사용한다. **실제 엣지 보드(Jetson Orin / Hailo / NPU 등)에서 재측정 필수**.

### 4.3 CPU 전처리 속도 (단일 이미지, 1920×1080 입력 가정)

| 단계 | mean ± std | p50 | FPS |
|------|:----------:|:---:|:---:|
| RGBE 전처리 (Resize + Canny)        |   43.3 ms ± 26.9 |  63.4 ms |  23.1 |
| RGBD 전처리 (Resize + Depth sync)  |    6.6 ms ±  4.9 |   5.5 ms | 151.7 |
| RGBD Aux 계산 (SVD 기반 물리치수)   |  172.9 ms ± 13.0 | 166.2 ms |   5.8 |
| **RGBE 전처리 합계**                 | **43.3 ms**     |    –    | **23.1** |
| **RGBD 전처리 합계 (Resize + Aux)** | **179.5 ms**    |    –    | **5.6**  |

→ **전처리에서 RGBD가 RGBE 대비 약 4.1배 느림**. 주된 병목은 SVD 기반 물리치수(width/height/aspect) 추출이다.

### 4.4 종합 파이프라인 (CPU 전처리 + GPU Forward, 단일 프레임 기준)

| 모델 | 전처리 (CPU) | Forward (GPU, B=1) | **합계** | 이론 FPS |
|------|:-----------:|:-----------------:|:--------:|:--------:|
| RGBE Hybrid    |  43.3 ms |  0.835 ms |  **44.1 ms** | **22.6 FPS** |
| RGBD + Aux MLP | 179.5 ms |  0.833 ms | **180.3 ms** | **5.5 FPS**  |

→ **종합 파이프라인 기준 RGBE가 약 4.1배 빠름**. GPU 추론은 두 모델 모두 1ms 미만으로 동일하므로, 실제 처리량 차이는 전적으로 CPU 전처리 단계에서 발생한다.

---

## 5. 분석 및 고찰

### 5.1 핵심 인사이트

1. **모델 자체의 GPU 추론 속도는 사실상 동일** (~0.83 ms @ B=1).
   - Aux MLP의 파라미터 증가량은 0.074%에 불과하며, GPU 연산 시간에 측정 가능한 영향을 주지 않음.
2. **병목은 전처리, 특히 RGBD의 Aux feature 계산**.
   - `compute_aux_features()` 내부의 SVD 연산이 단일 이미지 기준 약 173 ms 소요.
   - 반면 RGBE의 Canny Edge 검출은 OpenCV 가속 덕분에 약 43 ms로 비교적 빠름.
3. **엣지 디바이스에서 RGBE가 유리할 가능성이 큼**.
   - CPU 전처리 부담이 1/4 수준이므로, 동일한 NPU/GPU 환경이라면 RGBE가 더 높은 종합 FPS를 달성할 것으로 예상.

### 5.2 RGBD의 개선 여지

RGBD를 그대로 채택하더라도 다음 최적화 적용 시 격차를 크게 줄일 수 있다.

- Aux feature 계산을 **별도 스레드/비동기 큐**로 분리하여 GPU 추론과 파이프라이닝
- SVD를 **단순 PCA 근사 또는 Bounding-Box 기반 휴리스틱**으로 대체
- 입력 포인트 수를 다운샘플링하여 SVD 부담 경감
- 카메라가 이미 Depth 데이터를 제공하므로, 센서 자체 후처리 활용 가능 여부 검토

### 5.3 측정 한계

- 데스크탑 CPU는 다른 시스템 작업의 영향으로 분산이 큼 → **엣지 보드 실측 필수**.
- 본 측정은 단일 이미지 기준 (배치 1)이며, 실제 검수 라인의 처리량 요구사항에 따라 배치 크기별 재측정 필요.
- 카메라 캡처 지연(USB 전송, ISP 처리 등)은 포함되지 않음.

---

## 6. 결론 및 권장 사항

| 평가 항목 | RGBE Hybrid | RGBD + Aux MLP | 우위 |
|-----------|:-----------:|:--------------:|:----:|
| GPU Forward (B=1)         | 0.835 ms |  0.833 ms | 동일 |
| CPU 전처리 (1920×1080)    |   43 ms  |   180 ms  | **RGBE 4.1배** |
| 종합 파이프라인 (단일 프레임) |   44 ms  |   181 ms  | **RGBE 4.1배** |
| 분류 성능 (앞선 보고서 기준) |  유사    |   유사    | – |

**권장 사항**

1. 엣지 디바이스(ZED Box Mini 등)가 준비되는 즉시 동일 스크립트(`benchmark_inference.py`)를 이식하여 실측 진행.
2. 분류 성능이 유사한 상황에서, **속도가 중요한 라인 환경에서는 RGBE Hybrid를 우선 채택** 권장.
3. RGBD를 채택해야 하는 경우(예: Depth 데이터의 추가 활용 필요), Aux feature 계산을 비동기/근사 방식으로 최적화하는 후속 개발 진행.
4. 두 모델의 **앙상블** 적용 시 정확도 향상 효과는 있을 수 있으나, 전처리/추론 비용이 단순 합산되어 실시간성이 더 떨어지므로 신중히 검토.

---

## 7. 산출물

| 항목 | 경로 |
|------|------|
| 벤치마크 스크립트 | `class_estimation/door_paper/benchmark_inference.py` |
| 본 보고서        | `report/report_20260406_inference_benchmark.md` |

### 스크립트 주요 옵션

```bash
# GPU 기본 측정 (배치별 forward)
python benchmark_inference.py --batch_sizes 1,4,8,16 --num_runs 300 --warmup 30

# 전처리 포함 측정
python benchmark_inference.py --batch_sizes 1 --num_runs 200 --include_preprocess

# CPU(Edge 시뮬레이션)
python benchmark_inference.py --device cpu --batch_sizes 1 --num_runs 50 --include_preprocess
```
