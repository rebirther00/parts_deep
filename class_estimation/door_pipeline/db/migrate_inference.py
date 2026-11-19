"""추론 서비스(07_factory_infer.py) 결과 테이블 inference_results 생성 — 멱등 (2026-11-02, user_version=3).
2026-11-16 (user_version=4): 제외 표시 열 excluded/excluded_reason/excluded_by/excluded_at 추가 — 기존 DB 에는 ALTER TABLE 로 붙인다.
2026-11-19 (user_version=5): 합쳐진 세션 분할 열 split_parent/split_part/split_info 추가 (db/split_infer_sessions.py).

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
    excluded        INTEGER NOT NULL DEFAULT 0,   -- 1 = 잘못 등록된 세션(추론 오류·조각·도어 아님) — 목록·집계·학습에서 제외 (webapp /inference, 2026-11-16)
    excluded_reason TEXT,
    excluded_by     VARCHAR(30),
    excluded_at     TIMESTAMP,
    split_parent    INTEGER,                      -- 복원 세션: 합쳐져 있던 원 세션의 session_id (split_infer_sessions.py, 2026-11-19). NAS 에는 없는 DB 전용 레코드
    split_part      INTEGER,                      -- 복원 세션: 원 세션 안에서 몇 번째 도어인가 (1부터)
    split_info      TEXT,                         -- json. 복원 세션 = 구간 판정 근거(통과 프레임·득표·D·앞뒤 빈 지그 초), 원 세션 = 복원 세션 목록
    created_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime')),
    updated_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_infer_status ON inference_results(status);
CREATE INDEX IF NOT EXISTS idx_infer_started ON inference_results(started_at);
"""


# 제외 표시 (2026-11-16): 삭제 대신 표시만 한다 — NAS 가 정본이라 행을 지워도 ingest_infer 가 다시 등록하고,
# 왜 뺐는지(사유·주체·시각)가 남아야 인식율 분모에서 뺀 근거가 된다. ingest_infer 의 갱신은 열을 명시하므로 이 열을 건드리지 않는다.
EXCLUDE_COLS = [("excluded", "INTEGER NOT NULL DEFAULT 0"), ("excluded_reason", "TEXT"),
                ("excluded_by", "VARCHAR(30)"), ("excluded_at", "TIMESTAMP")]
# 분할 (2026-11-19): 빈 지그를 못 알아봐 도어 여러 대가 합쳐진 세션을 초 단위 지표 로그로 도어별 세션으로 복원한다.
# 원 세션은 excluded=1·excluded_by='split' 로 남기고, 복원 세션은 split_parent 로 원 세션을 가리킨다.
SPLIT_COLS = [("split_parent", "INTEGER"), ("split_part", "INTEGER"), ("split_info", "TEXT")]
VERSION = 5


def migrate(con):
    con.executescript(DDL)
    have = {r[1] for r in con.execute("PRAGMA table_info(inference_results)")}
    for name, typ in EXCLUDE_COLS + SPLIT_COLS:
        if name not in have:
            con.execute(f"ALTER TABLE inference_results ADD COLUMN {name} {typ}")
    v = con.execute("PRAGMA user_version").fetchone()[0]
    if v < VERSION:
        con.execute(f"PRAGMA user_version = {VERSION}")
    con.commit()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(BASE_DIR / "db" / "door_pipeline.db"))
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    v = migrate(con)
    n = con.execute("SELECT COUNT(*) FROM inference_results").fetchone()[0]
    ex = con.execute("SELECT COUNT(*) FROM inference_results WHERE excluded").fetchone()[0]
    print(f"inference_results 준비됨 (user_version {v} → {max(v, VERSION)}), 행 {n}, 제외 {ex}")


if __name__ == "__main__":
    main()
