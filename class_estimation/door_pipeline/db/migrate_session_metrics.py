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
    # 2026-09-24 픽셀 폭 지표 컬럼 추가(멱등)
    have = {r[1] for r in con.execute("PRAGMA table_info(session_hole_metrics)")}
    for col in ("span_med", "s_session", "s_ref", "d_pix_med", "dev_pix_mm", "margin_pix_min"):
        if col not in have:
            con.execute(f"ALTER TABLE session_hole_metrics ADD COLUMN {col} REAL")
    # 2026-09-23 세션 다수결 클래스(자동 재배정 근거) — migrate_options.py 와 동일
    for col, typ in (("pred_major", "VARCHAR(100)"), ("n_pred_major", "INTEGER")):
        if col not in have:
            con.execute(f"ALTER TABLE session_hole_metrics ADD COLUMN {col} {typ}")
    con.commit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default=str(DEFAULT_DB)); a = ap.parse_args()
    con = sqlite3.connect(a.db); ensure(con); con.commit()
    print("session_hole_metrics:", con.execute("SELECT COUNT(*) FROM session_hole_metrics").fetchone()[0], "행")
