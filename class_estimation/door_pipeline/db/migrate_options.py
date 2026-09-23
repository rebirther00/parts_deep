"""세션 옵션(RADAR) 컬럼·검사 결과 표·홀 지표 다수결 컬럼 추가 (2026-09-23, 멱등). schema.sql 의 DDL 과 동일.

  python db/migrate_options.py [--db 경로]

추가되는 것:
  capture_sessions.option_radar / option_source / option_note / option_at   — 세션 옵션 확정값(auto|user)
  session_radar_check                                                       — partno/radar_check.py 자동 검사 근거
  session_hole_metrics.pred_major / n_pred_major                            — 20_session_drift.py 다수결(자동 재배정 근거)
"""
import argparse, re, sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "door_pipeline.db"
SESSION_COLS = (("option_radar", "INTEGER"), ("option_source", "VARCHAR(10)"), ("option_note", "TEXT"), ("option_at", "TIMESTAMP"))
METRIC_COLS = (("pred_major", "VARCHAR(100)"), ("n_pred_major", "INTEGER"))


def ensure(con):
    have = {r[1] for r in con.execute("PRAGMA table_info(capture_sessions)")}
    for col, typ in SESSION_COLS:
        if col not in have:
            con.execute(f"ALTER TABLE capture_sessions ADD COLUMN {col} {typ}")
    if con.execute("SELECT 1 FROM sqlite_master WHERE name='session_hole_metrics'").fetchone():
        have = {r[1] for r in con.execute("PRAGMA table_info(session_hole_metrics)")}
        for col, typ in METRIC_COLS:
            if col not in have:
                con.execute(f"ALTER TABLE session_hole_metrics ADD COLUMN {col} {typ}")
    sql = (HERE / "schema.sql").read_text()
    m = re.search(r"CREATE TABLE IF NOT EXISTS session_radar_check.*?;", sql, re.S)
    if not m:
        raise SystemExit("schema.sql 에 session_radar_check DDL 이 없습니다")
    con.executescript(m.group(0))
    con.commit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default=str(DEFAULT_DB)); a = ap.parse_args()
    con = sqlite3.connect(a.db); ensure(con)
    n = con.execute("SELECT COUNT(*) FROM capture_sessions WHERE option_radar IS NOT NULL").fetchone()[0]
    m = con.execute("SELECT COUNT(*) FROM session_radar_check").fetchone()[0]
    print(f"옵션 확정 세션 {n}, 자동 검사 기록 {m}")
