-- 부품 인식 AI 학습 데이터 관리 DB 스키마 (SQLite 3)
-- 원 설계: report/DBMS_SCHEMA_DESIGN.md
-- 시각은 모두 한국 시간(KST, localtime)으로 기록 — CURRENT_TIMESTAMP(UTC) 사용 금지 (2026-08-28, migrate_kst.py)
-- 설계 대비 변경점:
--   * images.width/height NULL 허용 — NAS에만 있고 아직 동기화 안 된 이미지는 해상도를 모름
--   * images.synced_local 추가 — 학습 PC 로컬 존재 여부
--   * capture_sessions에 06_factory_capture.py의 meta.json 필드 반영
--     (session_dir, class_name, capture_interval_s, saved_pairs, finished, stop_reason)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            VARCHAR(200) NOT NULL UNIQUE,
    type            VARCHAR(20)  NOT NULL CHECK (type IN ('real', 'synthetic', 'mixed')),
    description     TEXT,
    num_classes     INTEGER NOT NULL DEFAULT 0,
    total_images    INTEGER NOT NULL DEFAULT 0,
    base_path       VARCHAR(500) NOT NULL,
    background_mode VARCHAR(20),
    created_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS classes (
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

CREATE TABLE IF NOT EXISTS capture_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id),
    session_dir     VARCHAR(500) UNIQUE,          -- 예: 20260707/E30_E38_door_RH/s_103015
    class_name      VARCHAR(100),                 -- meta.json의 class_name (라벨 정정 전 원본)
    camera_type     VARCHAR(50)  NOT NULL,
    camera_mode     VARCHAR(30),
    resolution      VARCHAR(20),
    fps             INTEGER,
    depth_mode      VARCHAR(30),
    capture_method  VARCHAR(30) CHECK (capture_method IN ('streaming', 'snapshot', 'video_extraction')),
    frame_interval  INTEGER,
    capture_interval_s REAL,
    session_duration_s REAL,
    blur_threshold  REAL,
    total_frames    INTEGER DEFAULT 0,
    extracted_frames INTEGER DEFAULT 0,
    valid_frames    INTEGER DEFAULT 0,
    saved_pairs     INTEGER DEFAULT 0,
    finished        BOOLEAN,
    stop_reason     VARCHAR(30),
    started_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime')),
    ended_at        TIMESTAMP,
    notes           TEXT,
    -- 2026-09-23 옵션(RADAR) — 세션 단위 속성. 품번 = classes.name(형상군) × option_radar (partno/part_numbers.json)
    --   option_radar 1=레이더 사양, 0=미장착, NULL=미정/해당 없음(FRT). option_source 'auto'(partno/radar_check.py) | 'user'(webapp /options 확정)
    option_radar    INTEGER CHECK (option_radar IN (0, 1)),
    option_source   VARCHAR(10) CHECK (option_source IN ('auto', 'user')),
    option_note     TEXT,
    option_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id        INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    session_id      INTEGER REFERENCES capture_sessions(id),
    rgb_filename    VARCHAR(50)  NOT NULL,
    depth_filename  VARCHAR(50),
    rgb_path        VARCHAR(500) NOT NULL,        -- datasets.base_path 기준 상대경로
    depth_path      VARCHAR(500),
    width           INTEGER,
    height          INTEGER,
    channels        INTEGER NOT NULL DEFAULT 4,
    depth_unit      VARCHAR(10) DEFAULT 'mm',
    depth_bit       INTEGER DEFAULT 16,
    blur_score      REAL,
    data_source     VARCHAR(20) NOT NULL CHECK (data_source IN ('camera', 'isaac_sim', 'augmented')),
    is_valid        BOOLEAN NOT NULL DEFAULT TRUE,
    synced_local    BOOLEAN NOT NULL DEFAULT TRUE,
    split           VARCHAR(10) CHECK (split IN ('train', 'test', 'val')),
    captured_at     TIMESTAMP,
    UNIQUE(class_id, rgb_path)
);

CREATE TABLE IF NOT EXISTS models (
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
    created_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS training_sessions (
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
    started_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime')),
    ended_at        TIMESTAMP,
    status          VARCHAR(20) DEFAULT 'running'
                    CHECK (status IN ('running', 'completed', 'failed', 'stopped'))
);

CREATE TABLE IF NOT EXISTS training_metrics (
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

CREATE TABLE IF NOT EXISTS evaluation_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES training_sessions(id),
    model_id        INTEGER NOT NULL REFERENCES models(id),
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id),
    eval_type       VARCHAR(30) NOT NULL
                    -- pose_pipeline 추가: migrate_eval_types.py (user_version=2, 2026-08-31)
                    CHECK (eval_type IN ('in_domain', 'cross_domain', 'inference_pipeline', 'pose_pipeline')),
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
    evaluated_at    TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS augmentation_configs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES training_sessions(id) ON DELETE CASCADE,
    name            VARCHAR(50) NOT NULL,
    applied_to      VARCHAR(10) NOT NULL CHECK (applied_to IN ('rgb', 'depth', 'both')),
    parameters      TEXT NOT NULL,
    UNIQUE(session_id, name)
);

CREATE INDEX IF NOT EXISTS idx_images_class     ON images(class_id);
CREATE INDEX IF NOT EXISTS idx_images_source    ON images(data_source);
CREATE INDEX IF NOT EXISTS idx_images_split     ON images(split);
CREATE INDEX IF NOT EXISTS idx_images_valid     ON images(is_valid);
CREATE INDEX IF NOT EXISTS idx_images_blur      ON images(blur_score);
CREATE INDEX IF NOT EXISTS idx_images_synced    ON images(synced_local);
CREATE INDEX IF NOT EXISTS idx_classes_dataset  ON classes(dataset_id);
CREATE INDEX IF NOT EXISTS idx_classes_model    ON classes(model_name);
CREATE INDEX IF NOT EXISTS idx_classes_part     ON classes(part_type);
CREATE INDEX IF NOT EXISTS idx_training_dataset ON training_sessions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_training_model   ON training_sessions(model_id);
CREATE INDEX IF NOT EXISTS idx_training_status  ON training_sessions(status);
CREATE INDEX IF NOT EXISTS idx_eval_model       ON evaluation_results(model_id);
CREATE INDEX IF NOT EXISTS idx_eval_type        ON evaluation_results(eval_type);
CREATE INDEX IF NOT EXISTS idx_metrics_session  ON training_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_dataset ON capture_sessions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_sessions_class   ON capture_sessions(class_name);

-- 2026-09-08: 같은 세션·파일의 이미지 행 중복 방지 (relabel 후 --refresh 재삽입 사고 재발 방지)
CREATE UNIQUE INDEX IF NOT EXISTS ux_images_session_file ON images(session_id, rgb_filename);

-- 2026-09-16: 세션별 홀 판별기 드리프트 지표 (20_session_drift.py 기록, webapp /drift 열람)
--   k_session = CAD_D / median(D_raw): 카메라 depth 스케일 드리프트 지표 (클래스 무관, 기준 k_applied ±1% 밖이면 경보)
--   dev_mm    = median(D_mm) − CAD_D: 판정 마진 소모량 (|dev| > 15mm 경보)
CREATE TABLE IF NOT EXISTS session_hole_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES capture_sessions(id) ON DELETE CASCADE,
    model_id        INTEGER NOT NULL REFERENCES models(id),
    class_name      VARCHAR(100) NOT NULL,        -- 정정 라벨(classes.name)
    serial          INTEGER,                      -- 카메라 시리얼 (intrinsics 출처)
    k_src           VARCHAR(10),                  -- camera | metric | depth
    k_applied       REAL,                         -- 판정에 쓴 잔차 K
    n_frames        INTEGER NOT NULL,
    n_judged        INTEGER NOT NULL,
    n_wrong         INTEGER,
    n_unknown       INTEGER,
    d_raw_med       REAL,                         -- 역투영 원거리 중앙값(mm, K 미적용)
    d_med           REAL,                         -- 판정 D 중앙값(mm)
    cad_d           REAL,
    dev_mm          REAL,
    k_session       REAL,
    z_med           REAL,
    tilt_med        REAL,
    margin_min      REAL,
    span_med        REAL,                         -- 힌지↔래치 픽셀 폭 중앙값(px, 2026-09-24)
    s_session       REAL,                         -- CAD_D·fx/span_med (mm) — 픽셀 폭 일관성, S_PIXEL 대비 ±0.5% 경보
    s_ref           REAL,                         -- 적용 S_PIXEL[serial]
    d_pix_med       REAL,                         -- 픽셀 폭 기준 D 중앙값(mm) = span_med·s_ref/fx
    dev_pix_mm      REAL,                         -- d_pix_med − CAD_D (판정 마진 소모, 현행 판정 기준)
    margin_pix_min  REAL,
    pred_major      VARCHAR(100),                 -- 2026-09-23 세션 홀 판정 다수결 클래스 (build_dataset.py auto-relabel 근거)
    n_pred_major    INTEGER,                      -- 다수결 클래스 프레임 수 (n_judged 중)
    evaluated_at    TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime')),
    UNIQUE(session_id, model_id)
);
CREATE INDEX IF NOT EXISTS idx_shm_session ON session_hole_metrics(session_id);

-- 2026-09-23: 세션별 RADAR 옵션 규칙 검사 결과 (partno/radar_check.py 기록, webapp /options 열람·확정)
--   6점 도어 좌표계로 CAD 브래킷 홀 2개 위치를 투영해 어두운 홀 유무를 프레임별 판정 → 세션 집계.
--   확정값은 capture_sessions.option_radar/option_source 에 두고, 이 표는 자동 검사 근거(표·점수·크롭)만 보관.
CREATE TABLE IF NOT EXISTS session_radar_check (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL UNIQUE REFERENCES capture_sessions(id) ON DELETE CASCADE,
    class_name      VARCHAR(100) NOT NULL,        -- 검사 시점 DB 라벨(브래킷 좌표 출처)
    n_frames        INTEGER NOT NULL,
    n_judged        INTEGER NOT NULL,             -- 6점 검출 + 투영 영역이 프레임 안인 프레임
    n_radar         INTEGER NOT NULL,             -- 홀 2개 모두 어두움(브래킷 있음) 표
    n_none          INTEGER NOT NULL,             -- 둘 다 없음 표
    score_med       REAL,                         -- 프레임 점수 중앙값 (min(홀1,홀2) 대비 — 양수=있음)
    auto_flag       INTEGER,                      -- 1/0, NULL=미정(표 부족·갈림)
    agree           REAL,                         -- 다수 표 비율
    crop_path       VARCHAR(500),                 -- 확인용 몽타주 jpg (door_pipeline 기준 상대경로)
    params          TEXT,                         -- 임계값 등 json
    checked_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime'))
);

