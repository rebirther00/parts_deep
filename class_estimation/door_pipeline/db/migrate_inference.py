"""추론 서비스(07_factory_infer.py) 결과 테이블 inference_results 생성 — 멱등 (2026-11-02, user_version=3).

    python db/migrate_inference.py [--db db/door_pipeline.db]

세션 자체는 capture_sessions(dataset 'door_factory_infer', class_name = 판정 클래스 또는 Unknown) 에 그대로 두고,
판정값·확정값은 이 표에 둔다. 확정(confirmed_*)은 webapp /inference 에서 사용자가 넣는다 — ingest 재실행으로 덮어쓰지 않는다.
"""
import argparse
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DDL = """
CREATE TABLE IF NOT EXISTS inference_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL UNIQUE REFERENCES capture_sessions(id) ON DELETE CASCADE,
    status          VARCHAR(16) NOT NULL CHECK (status IN ('judged', 'unknown', 'unrecognized', 'abstain')),
    pred_class      VARCHAR(100),                 -- 홀/CNN 판정 클래스 (unknown 포함), 없으면 NULL
    pred_group      VARCHAR(8),
    part_no         VARCHAR(30),
    part_name       VARCHAR(100),
    radar           INTEGER CHECK (radar IN (0, 1)),   -- 판정 레이더 (FRT·미정 NULL)
    radar_txt       VARCHAR(30),
    source          VARCHAR(8),                   -- hole | cnn
    conflict        BOOLEAN,
    confidence      REAL,
    d_mm            REAL,
    margin_mm       REAL,
    nearest_mm      REAL,
    n_judged        INTEGER,
    n_window        INTEGER,
    n_processed     INTEGER,
    n_clean         INTEGER,
    gate_counts     TEXT,                         -- json
    candidates      TEXT,                         -- json [{class, cad_D_mm, diff_mm}]
    cnn             TEXT,                         -- json 투표
    model_hole      VARCHAR(200),
    model_hole_sha  VARCHAR(16),
    model_cnn       VARCHAR(200),
    service_version VARCHAR(16),
    dwell_s         REAL,
    judged_at       TIMESTAMP,
    started_at      TIMESTAMP,
    ended_at        TIMESTAMP,
    thumb_path      VARCHAR(500),                 -- 로컬 미러 상대경로 (datasets_factory_infer/<day>/<sess>/thumb.jpg)
    result_json     TEXT,                         -- result.json 원문
    confirmed_class VARCHAR(100),                 -- 사용자 확정 (재학습 GT)
    confirmed_radar INTEGER CHECK (confirmed_radar IN (0, 1)),
    confirmed_by    VARCHAR(30),
    confirmed_at    TIMESTAMP,
    note            TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_infer_status ON inference_results(status);
CREATE INDEX IF NOT EXISTS idx_infer_started ON inference_results(started_at);
"""


def migrate(con):
    con.executescript(DDL)
    v = con.execute("PRAGMA user_version").fetchone()[0]
    if v < 3:
        con.execute("PRAGMA user_version = 3")
    con.commit()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(BASE_DIR / "db" / "door_pipeline.db"))
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    v = migrate(con)
    n = con.execute("SELECT COUNT(*) FROM inference_results").fetchone()[0]
    print(f"inference_results 준비됨 (user_version {v} → {max(v, 3)}), 행 {n}")


if __name__ == "__main__":
    main()
