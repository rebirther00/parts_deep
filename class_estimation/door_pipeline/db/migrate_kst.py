"""DB 시각을 한국 시간(KST)으로 통일하는 1회성 마이그레이션 (2026-08-28).

배경: schema.sql의 DEFAULT CURRENT_TIMESTAMP 와 ingest의 `updated_at = CURRENT_TIMESTAMP`는
SQLite가 UTC로 기록한다. 반면 db_log.py(datetime.now())와 현장 meta.json 시각은 KST라
datasets.created_at/updated_at, models.created_at 만 9시간 빠른 UTC로 남아 있었다.

처리: 새 schema.sql(DEFAULT datetime('now','localtime'))로 DB를 재생성하고 전 테이블을 복사하되
위 3개 컬럼만 +9시간 보정한다. 원본은 door_pipeline.db.bak_utc_<시각> 으로 보존.

실행: python db/migrate_kst.py            (door_pipeline.db 대상)
      python db/migrate_kst.py --db <경로>
"""
import argparse
import shutil
import sqlite3
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
UTC_COLS = {"datasets": ["created_at", "updated_at"], "models": ["created_at"]}
TABLES = ["datasets", "classes", "capture_sessions", "images", "models",
          "training_sessions", "training_metrics", "evaluation_results", "augmentation_configs"]

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=str(BASE / "door_pipeline.db"))
args = ap.parse_args()

src = Path(args.db)
if not src.exists():
    raise SystemExit(f"DB 없음: {src}")
if sqlite3.connect(src).execute("PRAGMA user_version").fetchone()[0] >= 1:
    raise SystemExit("이미 KST 마이그레이션 완료된 DB (user_version>=1)")

stamp = time.strftime("%Y%m%d_%H%M%S")
bak = src.with_name(src.name + f".bak_utc_{stamp}")
# WAL 체크포인트 후 백업 (-wal/-shm 내용 포함)
sqlite3.connect(src).execute("PRAGMA wal_checkpoint(TRUNCATE)")
shutil.copy2(src, bak)
print(f"백업: {bak}")

new = src.with_name(src.name + ".new")
if new.exists():
    new.unlink()
dst = sqlite3.connect(new)
dst.executescript((BASE / "schema.sql").read_text())
dst.execute("ATTACH DATABASE ? AS old", (str(src),))

for t in TABLES:
    cols = [r[1] for r in dst.execute(f"PRAGMA old.table_info({t})")]
    shift = UTC_COLS.get(t, [])
    sel = ", ".join(f"datetime({c}, '+9 hours')" if c in shift else c for c in cols)
    dst.execute(f"INSERT INTO main.{t} ({', '.join(cols)}) SELECT {sel} FROM old.{t}")
    n = dst.execute(f"SELECT count(*) FROM main.{t}").fetchone()[0]
    print(f"  {t:22s} {n:6d}행" + (f"  (+9h: {', '.join(shift)})" if shift else ""))

dst.execute("PRAGMA user_version = 1")
dst.commit()
dst.execute("DETACH DATABASE old")
dst.close()

for suf in ("-wal", "-shm"):
    p = src.with_name(src.name + suf)
    if p.exists():
        p.unlink()
new.replace(src)
print(f"완료: {src} (KST 통일, user_version=1)")
