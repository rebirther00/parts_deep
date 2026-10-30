"""door_pipeline.db 야간 백업 + 사람 판단 내보내기 (2026-10-30).

이미지 정본은 NAS 지만 라벨 정정·무효화 사유·split(공식 벤치마크 정의)·레이더 옵션 확정은 이 DB 에만 있다.
그래서 매일 DB 를 통째로 떠서 NAS 에 보관하고, 사람 판단만 세션 단위 CSV 로 뽑아 git 으로 추적한다.

  python db/backup_db.py                 # 백업 → 무결성 검사 → gzip → NAS 업로드 → decisions.csv → 보관 정리
  python db/backup_db.py --no-remote     # NAS 업로드 생략 (로컬 백업·CSV 만)
  python db/backup_db.py --restore-test  # 가장 최근 로컬 백업을 풀어 열어 보고 행 수를 현재 DB 와 비교

  cron (학습 PC, 매일 02:40):
  40 2 * * * cd <door_pipeline> && PATH=$HOME/.local/bin:$PATH <python> db/backup_db.py >> logs/backup_db.log 2>&1

산출:
  db/backups/door_pipeline_YYYYMMDD_HHMM.db.gz   로컬 최근 --keep-local 개 (git 미추적)
  nas:Guest/weld_db_backup/                      같은 파일, --keep-days 일 보관 + 매월 1일분은 계속 보관
  db/decisions.csv                               세션별 사람 판단 (git 추적 — diff 로 변경 이력 확인, DB 유실 시 복원 근거)
  db/backups/last_backup.json                    마지막 실행 결과 (webapp 대시보드 '수집·백업 상태' 카드가 읽음)

SQLite 온라인 백업 API 를 쓰므로 webapp·cron(ingest_nas)이 DB 를 쓰는 중에도 일관된 사본이 만들어진다(WAL).
"""
import argparse
import csv
import gzip
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB = BASE_DIR / "db" / "door_pipeline.db"
OUT_DIR = BASE_DIR / "db" / "backups"
DECISIONS = BASE_DIR / "db" / "decisions.csv"
PARTNO_JSON = BASE_DIR / "partno" / "part_numbers.json"
RCLONE = os.environ.get("RCLONE", str(Path.home() / ".local/bin/rclone"))
PREFIX = "door_pipeline_"
COUNT_TABLES = ("capture_sessions", "images", "models", "training_sessions", "evaluation_results",
                "session_hole_metrics", "session_radar_check")


def table_counts(con):
    out = {}
    for t in COUNT_TABLES:
        try:
            out[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            out[t] = None
    return out


def make_backup(db_path, out_dir, stamp):
    """온라인 백업 → quick_check → gzip. (gz 경로, 원본 크기, quick_check, 행 수) 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gz = out_dir / f"{PREFIX}{stamp}.db.gz"
    with tempfile.TemporaryDirectory(dir=out_dir) as td:
        tmp = Path(td) / "snapshot.db"
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        dst = sqlite3.connect(tmp)
        with dst:
            src.backup(dst)
        src.close()
        check = dst.execute("PRAGMA quick_check").fetchone()[0]
        counts = table_counts(dst)
        dst.execute("PRAGMA journal_mode=DELETE")       # 사본은 단일 파일로 (-wal 없이 열리게)
        dst.close()
        size = tmp.stat().st_size
        with open(tmp, "rb") as f, gzip.open(gz, "wb", compresslevel=6) as g:
            shutil.copyfileobj(f, g)
    return gz, size, check, counts


def load_partno():
    try:
        d = json.loads(PARTNO_JSON.read_text(encoding="utf8"))
    except OSError:
        return {}
    return {(p["class_name"], p["radar"]): f"{p['part_no']}-{p['rev']}" for p in d["parts"]}


def export_decisions(con, path):
    """세션 단위 사람 판단을 CSV 로. 세션 순 정렬·줄바꿈 제거로 git diff 가 판단 변경만 보이게 한다."""
    pn = load_partno()
    rows = con.execute(
        """SELECT s.session_dir, s.class_name AS class_input, MIN(c.name) AS class_db,
                  MAX(i.split) AS split, COALESCE(MIN(i.is_valid), 1) AS valid,
                  COUNT(i.id) AS n_images, COALESCE(SUM(i.synced_local), 0) AS n_local,
                  s.option_radar, s.option_source, s.option_at, s.notes
           FROM capture_sessions s LEFT JOIN images i ON i.session_id = s.id
           LEFT JOIN classes c ON c.id = i.class_id
           GROUP BY s.id ORDER BY s.session_dir""").fetchall()
    cols = ["session_dir", "class_input", "class_db", "split", "valid", "n_images", "n_local",
            "option_radar", "option_source", "option_at", "part_no", "notes"]
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            cls, radar = r["class_db"], r["option_radar"]
            part = pn.get((cls, None)) or (pn.get((cls, radar)) if radar is not None else None)
            w.writerow([r["session_dir"], r["class_input"], cls, r["split"] or "", r["valid"],
                        r["n_images"], r["n_local"], "" if radar is None else radar, r["option_source"] or "",
                        r["option_at"] or "", part or "", (r["notes"] or "").replace("\n", " | ")])
    tmp.replace(path)
    return len(rows)


def rclone(*args, timeout=600):
    p = subprocess.run([RCLONE, *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout) else f"rc={p.returncode}")


def prune_local(out_dir, keep):
    files = sorted(out_dir.glob(f"{PREFIX}*.db.gz"))
    for f in files[:-keep] if keep > 0 else []:
        f.unlink()
    return len(files[:-keep]) if keep > 0 and len(files) > keep else 0


def restore_test(db_path, out_dir):
    files = sorted(out_dir.glob(f"{PREFIX}*.db.gz"))
    if not files:
        sys.exit("로컬 백업 없음")
    with tempfile.TemporaryDirectory(dir=out_dir) as td:
        tmp = Path(td) / "restore.db"
        with gzip.open(files[-1], "rb") as g, open(tmp, "wb") as f:
            shutil.copyfileobj(g, f)
        b = sqlite3.connect(tmp)
        check, cb = b.execute("PRAGMA integrity_check").fetchone()[0], table_counts(b)
        b.close()
    cur = table_counts(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True))
    print(f"복원 시험: {files[-1].name} → integrity_check={check}")
    for t in COUNT_TABLES:
        print(f"  {t:22s} 백업 {cb[t]!s:>7s}  현재 {cur[t]!s:>7s}{'' if cb[t] == cur[t] else '  (백업 이후 변경)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--remote", default="nas:Guest/weld_db_backup")
    ap.add_argument("--no-remote", action="store_true", help="NAS 업로드 생략")
    ap.add_argument("--keep-local", type=int, default=14, help="로컬 보관 개수")
    ap.add_argument("--keep-days", type=int, default=30, help="NAS 보관 일수 (매월 1일분은 계속 보관)")
    ap.add_argument("--decisions", default=str(DECISIONS))
    ap.add_argument("--restore-test", action="store_true", help="최근 로컬 백업 복원 시험만")
    a = ap.parse_args()
    out_dir = Path(a.out)
    if a.restore_test:
        restore_test(a.db, out_dir)
        return

    t0 = time.time()
    stamp = time.strftime("%Y%m%d_%H%M")
    status = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "ok": False}
    try:
        gz, size, check, counts = make_backup(a.db, out_dir, stamp)
        status.update(file=gz.name, db_bytes=size, gz_bytes=gz.stat().st_size, quick_check=check, counts=counts)
        con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        status["decisions_rows"] = export_decisions(con, Path(a.decisions))
        con.close()
        if a.no_remote:
            status["remote"] = "skipped"
        else:
            try:
                rclone("copy", str(gz), a.remote)
                # 보관 정리: keep-days 지난 파일 삭제, 단 매월 1일 백업(…YYYYMM01_…)은 남김
                rclone("delete", a.remote, "--min-age", f"{a.keep_days}d",
                       "--filter", f"- {PREFIX}??????01_*.db.gz", "--filter", f"+ {PREFIX}*.db.gz", "--filter", "- *")
                status["remote"] = f"ok ({a.remote})"
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
                status["remote"] = f"실패: {e}"
        status["pruned_local"] = prune_local(out_dir, a.keep_local)
        status["ok"] = check == "ok" and not str(status["remote"]).startswith("실패")
    except (sqlite3.Error, OSError) as e:
        status["error"] = str(e)
    status["elapsed_s"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "last_backup.json").write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{status['at']}] {'OK' if status['ok'] else '실패'} {status.get('file', '')} "
          f"{status.get('db_bytes', 0) / 1e6:.1f}MB→{status.get('gz_bytes', 0) / 1e6:.1f}MB "
          f"check={status.get('quick_check')} 원격={status.get('remote')} decisions={status.get('decisions_rows')} "
          f"{status.get('error', '')}")
    sys.exit(0 if status["ok"] else 1)


if __name__ == "__main__":
    main()
