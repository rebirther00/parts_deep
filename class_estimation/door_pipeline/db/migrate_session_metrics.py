"""session_hole_metrics 테이블 생성 (2026-09-16, 멱등). schema.sql 의 DDL 과 동일.

  python db/migrate_session_metrics.py [--db 경로]
"""
import argparse, re, sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "door_pipeline.db"


def ensure(con):
    sql = (HERE / "schema.sql").read_text()
    m = re.search(r"CREATE TABLE IF NOT EXISTS session_hole_metrics.*?;\s*CREATE INDEX[^;]*;", sql, re.S)
    if not m:
        raise SystemExit("schema.sql 에 session_hole_metrics DDL 이 없습니다")
    con.executescript(m.group(0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default=str(DEFAULT_DB)); a = ap.parse_args()
    con = sqlite3.connect(a.db); ensure(con); con.commit()
    print("session_hole_metrics:", con.execute("SELECT COUNT(*) FROM session_hole_metrics").fetchone()[0], "행")
