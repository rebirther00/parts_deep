"""evaluation_results.eval_type CHECK에 'pose_pipeline' 추가 (PRAGMA user_version=2).

pos_pipeline/03_evaluate_field.py가 eval_type='pose_pipeline'으로 기록을 시도하지만
기존 CHECK(in_domain/cross_domain/inference_pipeline)가 거부해 조용히 유실되던 문제의 수정.
SQLite는 CHECK 변경을 지원하지 않아 테이블을 재생성한다. 실행 전 자동 백업(.bak_evaltype_*).

  python db/migrate_eval_types.py [--db 경로]
"""
import argparse
import shutil
import sqlite3
import time
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent / "door_pipeline.db"

NEW_DDL = """
CREATE TABLE evaluation_results_new (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES training_sessions(id),
    model_id        INTEGER NOT NULL REFERENCES models(id),
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id),
    eval_type       VARCHAR(30) NOT NULL
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
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    a = ap.parse_args()
    db = Path(a.db)
    con = sqlite3.connect(db)
    ver = con.execute("PRAGMA user_version").fetchone()[0]
    if ver >= 2:
        print(f"user_version={ver} — 이미 적용됨, 종료")
        return
    con.close()
    bak = db.with_name(db.name + f".bak_evaltype_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(db, bak)
    print(f"백업: {bak.name}")

    con = sqlite3.connect(db)
    con.execute("PRAGMA foreign_keys = OFF")
    con.executescript(NEW_DDL + """
    INSERT INTO evaluation_results_new SELECT * FROM evaluation_results;
    DROP TABLE evaluation_results;
    ALTER TABLE evaluation_results_new RENAME TO evaluation_results;
    CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluation_results(model_id);
    CREATE INDEX IF NOT EXISTS idx_eval_type  ON evaluation_results(eval_type);
    PRAGMA user_version = 2;
    """)
    con.commit()
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    n = con.execute("SELECT COUNT(*) FROM evaluation_results").fetchone()[0]
    ic = con.execute("PRAGMA integrity_check").fetchone()[0]
    print(f"적용 완료: 행 {n}개 보존, FK 위반 {len(fk)}건, integrity={ic}, user_version=2")
    print("eval_type 허용값: in_domain, cross_domain, inference_pipeline, pose_pipeline")


if __name__ == "__main__":
    main()
