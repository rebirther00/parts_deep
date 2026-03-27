# 부품 인식 AI 학습 데이터 관리 DBMS 스키마 설계

## 1. 설계 개요

### 1.1 목적
굴착기 부품 인식 AI 시스템에서 수집·생성되는 RGB+Depth 이미지, 메타데이터, 학습 이력, 평가 결과를 체계적으로 관리하기 위한 관계형 데이터베이스 스키마를 설계한다.

### 1.2 설계 원칙
- **하이브리드 저장**: 이미지 파일은 파일시스템, 메타데이터는 RDBMS에 저장
- **이력 관리**: 학습 세션, 평가 결과의 전체 이력을 추적 가능하게 설계
- **확장성**: 새로운 부품 클래스, 센서, 모델 아키텍처 추가 시 스키마 변경 최소화
- **데이터 무결성**: 외래 키 제약 및 CHECK 제약을 통한 일관성 보장

### 1.3 대상 DBMS
- 개발/테스트: SQLite 3 (경량, 서버리스)
- 운영: PostgreSQL 15+ (확장성, 동시성, JSON 지원)

---

## 2. ER 다이어그램

```
┌─────────────┐       ┌──────────────┐       ┌──────────────────┐
│  datasets   │──1:N──│   classes    │──1:N──│     images       │
│─────────────│       │──────────────│       │──────────────────│
│ PK id       │       │ PK id        │       │ PK id            │
│ name        │       │ FK dataset_id│       │ FK class_id      │
│ type        │       │ name         │       │ FK session_id    │
│ description │       │ model_name   │       │ rgb_path         │
│ num_classes │       │ part_type    │       │ depth_path       │
│ created_at  │       │ display_name │       │ width, height    │
└─────────────┘       └──────────────┘       │ blur_score       │
                                              │ data_source      │
                                              │ is_valid         │
┌─────────────────┐                           │ captured_at      │
│ capture_sessions│──1:N──────────────────────└──────────────────┘
│─────────────────│
│ PK id           │
│ FK dataset_id   │       ┌──────────────────┐
│ camera_type     │       │ training_sessions│
│ started_at      │       │──────────────────│       ┌────────────────┐
│ ended_at        │       │ PK id            │──1:N──│training_metrics│
│ frame_interval  │       │ FK dataset_id    │       │────────────────│
│ blur_threshold  │       │ FK model_id      │       │ PK id          │
└─────────────────┘       │ optimizer        │       │ FK session_id  │
                          │ learning_rate    │       │ epoch          │
┌──────────────┐          │ batch_size       │       │ train_loss     │
│    models    │──1:N─────│ epochs           │       │ val_loss       │
│──────────────│          │ best_accuracy    │       │ val_accuracy   │
│ PK id        │          │ started_at       │       │ learning_rate  │
│ name         │          └──────────────────┘       └────────────────┘
│ architecture │
│ in_channels  │          ┌────────────────────┐
│ weights_path │          │ evaluation_results │
│ onnx_path    │          │────────────────────│
│ created_at   │          │ PK id              │
└──────────────┘          │ FK session_id      │
                          │ FK model_id        │
┌─────────────────────┐   │ accuracy           │
│ augmentation_configs│   │ precision_macro    │
│─────────────────────│   │ recall_macro       │
│ PK id               │   │ f1_macro           │
│ FK session_id       │   │ confusion_matrix   │
│ name                │   │ evaluated_at       │
│ parameters (JSON)   │   └────────────────────┘
│ applied_to          │
└─────────────────────┘
```

---

## 3. 테이블 상세 정의

### 3.1 datasets (데이터셋)

데이터셋 단위를 관리한다. 실물 촬영/CAD 합성 등 데이터셋 유형별로 구분한다.

```sql
CREATE TABLE datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            VARCHAR(200) NOT NULL UNIQUE,
    type            VARCHAR(20)  NOT NULL CHECK (type IN ('real', 'synthetic', 'mixed')),
    description     TEXT,
    num_classes     INTEGER NOT NULL DEFAULT 0,
    total_images    INTEGER NOT NULL DEFAULT 0,
    base_path       VARCHAR(500) NOT NULL,
    background_mode VARCHAR(20),
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER PK | 데이터셋 고유 ID |
| name | VARCHAR(200) | 데이터셋 이름 (예: "Excavator Door Classification Dataset") |
| type | VARCHAR(20) | 데이터 유형: real(실물), synthetic(합성), mixed(혼합) |
| description | TEXT | 데이터셋 설명 |
| num_classes | INTEGER | 클래스 수 |
| total_images | INTEGER | 전체 이미지 수 |
| base_path | VARCHAR(500) | 파일시스템 기본 경로 (예: datasets/, datasets_cad/) |
| background_mode | VARCHAR(20) | 배경 생성 모드 (random, fixed 등) |
| created_at | TIMESTAMP | 생성 일시 |
| updated_at | TIMESTAMP | 수정 일시 |

**초기 데이터 예시:**
```sql
INSERT INTO datasets (name, type, description, num_classes, total_images, base_path)
VALUES
('Excavator Door Real RGBD', 'real',
 'ZED X Mini로 촬영한 실물 도어 RGBD 데이터', 3, 337, 'datasets/'),
('Excavator Door CAD RGBD', 'synthetic',
 'Isaac Sim으로 생성한 CAD 합성 RGBD 데이터', 8, 4000, 'datasets_cad/');
```

---

### 3.2 classes (클래스)

부품 클래스 정보를 관리한다. 굴착기 모델명, 부품 유형 등을 구조화하여 검색·필터링을 지원한다.

```sql
CREATE TABLE classes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name            VARCHAR(100) NOT NULL,
    display_name    VARCHAR(100),
    model_name      VARCHAR(10),
    part_type       VARCHAR(30),
    image_count     INTEGER NOT NULL DEFAULT 0,
    cad_available   BOOLEAN NOT NULL DEFAULT FALSE,
    notes           TEXT,

    UNIQUE(dataset_id, name)
);
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER PK | 클래스 고유 ID |
| dataset_id | INTEGER FK | 소속 데이터셋 |
| name | VARCHAR(100) | 클래스 식별자 (예: E25_door_LH_FRT) |
| display_name | VARCHAR(100) | 표시명 (예: E25 Door LH Front) |
| model_name | VARCHAR(10) | 굴착기 모델 (E25, E30, E38) |
| part_type | VARCHAR(30) | 부품 유형 (door_RH, door_LH_FRT, door_LH_RR) |
| image_count | INTEGER | 해당 클래스 이미지 수 |
| cad_available | BOOLEAN | CAD 모델 확보 여부 |
| notes | TEXT | 비고 (예: "E30/E38 RH 동일 부품으로 통합") |

**초기 데이터 예시:**
```sql
INSERT INTO classes (dataset_id, name, display_name, model_name, part_type, image_count, cad_available)
VALUES
(1, 'E25_door_LH_FRT', 'E25 Door LH Front', 'E25', 'door_LH_FRT', 110, TRUE),
(1, 'E30_door_LH_FRT', 'E30 Door LH Front', 'E30', 'door_LH_FRT', 108, TRUE),
(1, 'E38_door_LH_FRT', 'E38 Door LH Front', 'E38', 'door_LH_FRT', 119, TRUE),
(2, 'E25_door_LH_FRT', 'E25 Door LH Front', 'E25', 'door_LH_FRT', 500, TRUE),
(2, 'E25_door_LH_RR',  'E25 Door LH Rear',  'E25', 'door_LH_RR',  500, TRUE),
(2, 'E25_door_RH',      'E25 Door RH',       'E25', 'door_RH',     500, TRUE),
(2, 'E30_door_LH_FRT', 'E30 Door LH Front', 'E30', 'door_LH_FRT', 500, TRUE),
(2, 'E30_door_LH_RR',  'E30 Door LH Rear',  'E30', 'door_LH_RR',  500, TRUE),
(2, 'E30_E38_door_RH',  'E30/E38 Door RH',   'E30', 'door_RH',     500, TRUE),
(2, 'E38_door_LH_FRT', 'E38 Door LH Front', 'E38', 'door_LH_FRT', 500, TRUE),
(2, 'E38_door_LH_RR',  'E38 Door LH Rear',  'E38', 'door_LH_RR',  500, TRUE);
```

---

### 3.3 images (이미지)

개별 RGB+Depth 이미지 쌍의 메타데이터를 관리한다.

```sql
CREATE TABLE images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id        INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    session_id      INTEGER REFERENCES capture_sessions(id),
    rgb_filename    VARCHAR(50)  NOT NULL,
    depth_filename  VARCHAR(50),
    rgb_path        VARCHAR(500) NOT NULL,
    depth_path      VARCHAR(500),
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    channels        INTEGER NOT NULL DEFAULT 4,
    depth_unit      VARCHAR(10) DEFAULT 'mm',
    depth_bit       INTEGER DEFAULT 16,
    blur_score      REAL,
    data_source     VARCHAR(20) NOT NULL CHECK (data_source IN ('camera', 'isaac_sim', 'augmented')),
    is_valid        BOOLEAN NOT NULL DEFAULT TRUE,
    split           VARCHAR(10) CHECK (split IN ('train', 'test', 'val')),
    captured_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(class_id, rgb_filename)
);

CREATE INDEX idx_images_class    ON images(class_id);
CREATE INDEX idx_images_source   ON images(data_source);
CREATE INDEX idx_images_split    ON images(split);
CREATE INDEX idx_images_valid    ON images(is_valid);
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER PK | 이미지 고유 ID |
| class_id | INTEGER FK | 소속 클래스 |
| session_id | INTEGER FK | 촬영 세션 (실물만 해당) |
| rgb_filename | VARCHAR(50) | RGB 파일명 (예: rgb_0000.png) |
| depth_filename | VARCHAR(50) | Depth 파일명 (예: depth_0000.png) |
| rgb_path | VARCHAR(500) | RGB 전체 경로 |
| depth_path | VARCHAR(500) | Depth 전체 경로 |
| width, height | INTEGER | 이미지 해상도 |
| channels | INTEGER | 채널 수 (기본 4: RGBD) |
| depth_unit | VARCHAR(10) | Depth 단위 (mm) |
| depth_bit | INTEGER | Depth 비트 깊이 (16) |
| blur_score | REAL | Laplacian Variance 블러 점수 |
| data_source | VARCHAR(20) | 데이터 소스 (camera/isaac_sim/augmented) |
| is_valid | BOOLEAN | 유효 이미지 여부 (블러·잘림 제외) |
| split | VARCHAR(10) | 학습 분할 (train/test/val) |
| captured_at | TIMESTAMP | 촬영/생성 일시 |

---

### 3.4 capture_sessions (촬영 세션)

실물 데이터 촬영 세션을 기록하여 수집 이력을 추적한다.

```sql
CREATE TABLE capture_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id),
    camera_type     VARCHAR(50)  NOT NULL,
    camera_mode     VARCHAR(30),
    resolution      VARCHAR(20),
    fps             INTEGER,
    depth_mode      VARCHAR(30),
    capture_method  VARCHAR(30) CHECK (capture_method IN ('streaming', 'snapshot', 'video_extraction')),
    frame_interval  INTEGER DEFAULT 5,
    blur_threshold  REAL    DEFAULT 100.0,
    total_frames    INTEGER DEFAULT 0,
    extracted_frames INTEGER DEFAULT 0,
    valid_frames    INTEGER DEFAULT 0,
    started_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at        TIMESTAMP,
    notes           TEXT
);
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| camera_type | VARCHAR(50) | 카메라 종류 (예: ZED X Mini) |
| camera_mode | VARCHAR(30) | 카메라 모드 (예: HD1080) |
| depth_mode | VARCHAR(30) | Depth 모드 (예: NEURAL) |
| capture_method | VARCHAR(30) | 촬영 방식 (스트리밍/스냅샷/비디오 추출) |
| frame_interval | INTEGER | 프레임 추출 간격 (N 프레임당 1장) |
| blur_threshold | REAL | 블러 필터링 임계값 |
| total_frames | INTEGER | 총 촬영 프레임 수 |
| extracted_frames | INTEGER | 추출된 프레임 수 |
| valid_frames | INTEGER | 유효(저장) 프레임 수 |

---

### 3.5 models (학습 모델)

학습된 AI 모델의 정보를 관리한다.

```sql
CREATE TABLE models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            VARCHAR(100) NOT NULL,
    architecture    VARCHAR(50)  NOT NULL,
    in_channels     INTEGER NOT NULL DEFAULT 4,
    num_classes     INTEGER NOT NULL,
    pretrained_base VARCHAR(50),
    weights_path    VARCHAR(500),
    onnx_path       VARCHAR(500),
    trt_path        VARCHAR(500),
    input_size      VARCHAR(20) DEFAULT '224x224',
    description     TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| architecture | VARCHAR(50) | 모델 아키텍처 (예: ResNet18) |
| in_channels | INTEGER | 입력 채널 수 (4: RGBD) |
| pretrained_base | VARCHAR(50) | 사전학습 가중치 (예: IMAGENET1K_V1) |
| weights_path | VARCHAR(500) | PyTorch 가중치 파일 경로 (.pth) |
| onnx_path | VARCHAR(500) | ONNX 변환 모델 경로 |
| trt_path | VARCHAR(500) | TensorRT 엔진 경로 |

---

### 3.6 training_sessions (학습 세션)

모델 학습 실행 이력을 관리한다. 하이퍼파라미터, 데이터 분할 정보 등을 기록한다.

```sql
CREATE TABLE training_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id),
    model_id        INTEGER NOT NULL REFERENCES models(id),
    optimizer       VARCHAR(20)  NOT NULL DEFAULT 'Adam',
    learning_rate   REAL         NOT NULL DEFAULT 0.001,
    batch_size      INTEGER      NOT NULL DEFAULT 64,
    max_epochs      INTEGER      NOT NULL DEFAULT 60,
    actual_epochs   INTEGER,
    early_stop_patience INTEGER DEFAULT 10,
    train_ratio     REAL    NOT NULL DEFAULT 0.8,
    train_count     INTEGER,
    test_count      INTEGER,
    best_val_accuracy   REAL,
    best_val_loss       REAL,
    best_epoch          INTEGER,
    total_time_sec      REAL,
    gpu_device          VARCHAR(50),
    loss_function       VARCHAR(50) DEFAULT 'CrossEntropyLoss',
    class_weights       BOOLEAN DEFAULT TRUE,
    split_indices_path  VARCHAR(500),
    started_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at        TIMESTAMP,
    status          VARCHAR(20) DEFAULT 'running'
                    CHECK (status IN ('running', 'completed', 'failed', 'stopped'))
);
```

---

### 3.7 training_metrics (학습 지표)

에포크별 학습 지표를 기록하여 학습 곡선 분석을 지원한다.

```sql
CREATE TABLE training_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES training_sessions(id) ON DELETE CASCADE,
    epoch           INTEGER NOT NULL,
    train_loss      REAL NOT NULL,
    val_loss        REAL,
    val_accuracy    REAL,
    learning_rate   REAL,
    elapsed_sec     REAL,

    UNIQUE(session_id, epoch)
);

CREATE INDEX idx_metrics_session ON training_metrics(session_id);
```

---

### 3.8 evaluation_results (평가 결과)

모델 평가 결과를 저장한다. 혼동 행렬은 JSON 형식으로 저장한다.

```sql
CREATE TABLE evaluation_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES training_sessions(id),
    model_id        INTEGER NOT NULL REFERENCES models(id),
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id),
    eval_type       VARCHAR(30) NOT NULL
                    CHECK (eval_type IN ('in_domain', 'cross_domain', 'inference_pipeline')),
    total_samples   INTEGER NOT NULL,
    correct         INTEGER NOT NULL,
    accuracy        REAL    NOT NULL,
    precision_macro REAL,
    recall_macro    REAL,
    f1_macro        REAL,
    confusion_matrix TEXT,
    per_class_results TEXT,
    inference_time_ms REAL,
    inference_device  VARCHAR(50),
    report_path     VARCHAR(500),
    evaluated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| eval_type | VARCHAR(30) | 평가 유형: in_domain(동일 도메인), cross_domain(교차 도메인), inference_pipeline(추론 파이프라인) |
| confusion_matrix | TEXT (JSON) | 혼동 행렬 (JSON 배열) |
| per_class_results | TEXT (JSON) | 클래스별 정확도/재현율/F1 (JSON) |
| inference_time_ms | REAL | 평균 추론 시간 (ms/장) |

**평가 결과 JSON 형식 예시:**
```json
{
  "confusion_matrix": [[22,0,0],[0,21,1],[0,0,24]],
  "per_class": {
    "E25_door_LH_FRT": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 22},
    "E30_door_LH_FRT": {"precision": 0.955, "recall": 0.955, "f1": 0.955, "support": 22},
    "E38_door_LH_FRT": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 24}
  }
}
```

---

### 3.9 augmentation_configs (데이터 증강 설정)

학습 세션에 적용된 데이터 증강 설정을 JSON으로 기록한다.

```sql
CREATE TABLE augmentation_configs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES training_sessions(id) ON DELETE CASCADE,
    name            VARCHAR(50) NOT NULL,
    applied_to      VARCHAR(10) NOT NULL CHECK (applied_to IN ('rgb', 'depth', 'both')),
    parameters      TEXT NOT NULL,

    UNIQUE(session_id, name)
);
```

**증강 설정 JSON 예시:**
```sql
INSERT INTO augmentation_configs (session_id, name, applied_to, parameters) VALUES
(1, 'RandomRotation', 'both', '{"degrees": [-5, 5]}'),
(1, 'AdjustBrightness', 'rgb', '{"factor_range": [0.7, 1.3]}'),
(1, 'AdjustContrast', 'rgb', '{"factor_range": [0.7, 1.3]}'),
(1, 'AdjustSaturation', 'rgb', '{"factor_range": [0.8, 1.2]}'),
(1, 'DomainRandomization', 'both', '{"background": {"none": 0.2, "solid": 0.3, "factory": 0.5}}');
```

---

## 4. 인덱스 설계

성능 최적화를 위한 주요 인덱스를 아래와 같이 설계하였다.

```sql
-- 이미지 조회 최적화
CREATE INDEX idx_images_class        ON images(class_id);
CREATE INDEX idx_images_source       ON images(data_source);
CREATE INDEX idx_images_split        ON images(split);
CREATE INDEX idx_images_valid        ON images(is_valid);
CREATE INDEX idx_images_blur         ON images(blur_score);

-- 클래스 조회 최적화
CREATE INDEX idx_classes_dataset     ON classes(dataset_id);
CREATE INDEX idx_classes_model       ON classes(model_name);
CREATE INDEX idx_classes_part        ON classes(part_type);

-- 학습 이력 조회 최적화
CREATE INDEX idx_training_dataset    ON training_sessions(dataset_id);
CREATE INDEX idx_training_model      ON training_sessions(model_id);
CREATE INDEX idx_training_status     ON training_sessions(status);

-- 평가 결과 조회 최적화
CREATE INDEX idx_eval_model          ON evaluation_results(model_id);
CREATE INDEX idx_eval_type           ON evaluation_results(eval_type);

-- 학습 지표 조회 최적화
CREATE INDEX idx_metrics_session     ON training_metrics(session_id);
```

---

## 5. 주요 활용 쿼리 예시

### 5.1 데이터셋 현황 조회

```sql
-- 클래스별 이미지 수 현황
SELECT d.name AS dataset_name,
       c.name AS class_name,
       c.model_name,
       c.part_type,
       c.image_count,
       c.cad_available
FROM classes c
JOIN datasets d ON c.dataset_id = d.id
ORDER BY d.name, c.name;
```

### 5.2 학습 이력 조회

```sql
-- 최근 학습 세션과 결과
SELECT ts.id,
       d.name AS dataset,
       m.architecture,
       ts.optimizer,
       ts.learning_rate,
       ts.batch_size,
       ts.actual_epochs,
       ts.best_val_accuracy,
       ts.total_time_sec,
       ts.gpu_device,
       ts.started_at
FROM training_sessions ts
JOIN datasets d ON ts.dataset_id = d.id
JOIN models m ON ts.model_id = m.id
WHERE ts.status = 'completed'
ORDER BY ts.started_at DESC;
```

### 5.3 모델 성능 비교

```sql
-- 동일 데이터셋에 대한 모델별 최고 성능 비교
SELECT m.name AS model_name,
       m.architecture,
       er.eval_type,
       er.accuracy,
       er.precision_macro,
       er.recall_macro,
       er.f1_macro,
       er.inference_time_ms
FROM evaluation_results er
JOIN models m ON er.model_id = m.id
WHERE er.dataset_id = 1
ORDER BY er.accuracy DESC;
```

### 5.4 촬영 세션별 수집 현황

```sql
-- 촬영 세션별 프레임 수집 효율
SELECT cs.id,
       cs.camera_type,
       cs.capture_method,
       cs.total_frames,
       cs.extracted_frames,
       cs.valid_frames,
       ROUND(cs.valid_frames * 100.0 / NULLIF(cs.total_frames, 0), 1) AS valid_ratio_pct,
       cs.started_at
FROM capture_sessions cs
ORDER BY cs.started_at DESC;
```

### 5.5 데이터 소스별 이미지 통계

```sql
-- 데이터 소스(실물/합성)별 이미지 통계
SELECT i.data_source,
       COUNT(*) AS total_images,
       COUNT(i.depth_path) AS with_depth,
       ROUND(AVG(i.blur_score), 1) AS avg_blur_score,
       SUM(CASE WHEN i.is_valid THEN 1 ELSE 0 END) AS valid_images
FROM images i
GROUP BY i.data_source;
```

### 5.6 학습 곡선 데이터 조회

```sql
-- 특정 학습 세션의 에포크별 학습 곡선 데이터
SELECT epoch,
       train_loss,
       val_loss,
       val_accuracy,
       learning_rate
FROM training_metrics
WHERE session_id = 1
ORDER BY epoch;
```

---

## 6. 현행 파일시스템 구조와의 매핑

현재 파일시스템 기반으로 운영 중인 데이터 관리 체계를 DBMS로 마이그레이션하기 위한 매핑 관계이다.

| 현행 (파일시스템) | DBMS 테이블 | 비고 |
|------------------|------------|------|
| `datasets/dataset_info.json` | `datasets` + `classes` | JSON → 정규화된 테이블 |
| `datasets/<class>/metadata.json` | `classes` | 클래스별 메타데이터 |
| `datasets/<class>/rgb_NNNN.png` | `images.rgb_path` | 파일 경로만 DB 저장 |
| `datasets/<class>/depth_NNNN.png` | `images.depth_path` | 파일 경로만 DB 저장 |
| `artifacts/best_*_model.pth` | `models.weights_path` | 모델 파일 경로 |
| `artifacts/evaluation_results.json` | `evaluation_results` | JSON → 정규화 테이블 |
| `artifacts/training_indices.json` | `training_sessions.split_indices_path` | 분할 정보 참조 |
| `artifacts/class_names.json` | `classes.name` | 클래스 목록 |

---

## 7. 마이그레이션 전략

### 7.1 단계별 전환 계획

| 단계 | 내용 | 시기 |
|------|------|------|
| 1단계 | SQLite 기반 메타데이터 DB 구축, 기존 JSON → DB 마이그레이션 | 2차년도 Q1 |
| 2단계 | Flask 웹 서버에 DB 연동 (촬영 세션·이미지 자동 등록) | 2차년도 Q2 |
| 3단계 | 학습·평가 스크립트에 DB 로깅 통합 | 2차년도 Q3 |
| 4단계 | PostgreSQL 전환 및 다중 사용자 환경 지원 | 2차년도 Q4 |

### 7.2 기존 JSON 데이터 마이그레이션 스크립트 구조

```
기존 파일시스템                           DBMS
──────────────                           ─────
dataset_info.json  ──parse──→  INSERT INTO datasets, classes
metadata.json      ──parse──→  UPDATE classes
rgb_*.png 목록     ──scan───→  INSERT INTO images
depth_*.png 목록   ──scan───→  UPDATE images SET depth_path
evaluation_results.json ────→  INSERT INTO evaluation_results
training_indices.json ──────→  INSERT INTO training_sessions
```
