-- 부품 인식 AI 학습 데이터 관리 DB 스키마 (SQLite 3)
-- 원 설계: report/DBMS_SCHEMA_DESIGN.md
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
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
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
    started_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at        TIMESTAMP,
    notes           TEXT
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
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
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
    started_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
