"""NAS 정본 → 학습 PC 미러(datasets_factory_collect/) 샘플 pull.

DB(capture_sessions/images)에 등록된 세션을 기준으로 세션당 N장(기본 20)을 시간축으로 고르게 골라
rclone으로 내려받는다(+meta.json). 미러는 NAS와 같은 <날짜>/<클래스>/s_HHMMSS/ 구조이며
샘플만 존재하는 부분집합이다. 내려받은 뒤 images.synced_local 을 갱신한다.

  python db/pull_nas.py                    # 미등록/미동기화 세션 전부, 세션당 20장
  python db/pull_nas.py --date 20260828    # 특정 날짜만
  python db/pull_nas.py --session 20260828/E25_door_RH/s_073451 --all   # 한 세션 전량
  python db/pull_nas.py --per-session 40 --dry-run

선행: python db/ingest_nas.py (세션 등록). 이미 로컬에 있는 파일은 다시 받지 않는다.
"""
import argparse
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MIRROR = BASE_DIR / "datasets_factory_collect"
DATASET_NAME = "door_factory_collect"
RCLONE = os.environ.get("RCLONE", str(Path.home() / ".local/bin/rclone"))


def pick_even(items, n):
    """정렬된 items에서 n개를 시간축으로 고르게 선택 (양 끝 포함)."""
    if n <= 0 or len(items) <= n:
        return list(items)
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(BASE_DIR / "db" / "door_pipeline.db"))
    ap.add_argument("--remote", default="nas:Guest/weld")
    ap.add_argument("--per-session", type=int, default=20)
    ap.add_argument("--all", action="store_true", help="세션 전량 (per-session 무시)")
    ap.add_argument("--date", help="YYYYMMDD — 해당 날짜 세션만")
    ap.add_argument("--session", help="session_dir 하나만 (예: 20260828/E25_door_RH/s_073451)")
    ap.add_argument("--include-unknown", action="store_true", help="Unknown 라벨 세션도 pull")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    q = """SELECT s.id, s.session_dir, s.class_name FROM capture_sessions s
           JOIN datasets d ON d.id = s.dataset_id WHERE d.name = ?"""
    params = [DATASET_NAME]
    if args.session:
        q += " AND s.session_dir = ?"; params.append(args.session)
    if args.date:
        q += " AND s.session_dir LIKE ?"; params.append(f"{args.date}/%")
    sessions = con.execute(q + " ORDER BY s.session_dir", params).fetchall()
    if not sessions:
        sys.exit("대상 세션 없음 — 먼저 python db/ingest_nas.py 실행")

    want = []          # NAS 상대경로
    per_sess = {}
    for s in sessions:
        if s["class_name"] == "Unknown" and not args.include_unknown:
            continue
        rows = con.execute(
            """SELECT rgb_path, depth_path FROM images
               WHERE session_id = ? ORDER BY rgb_filename""", (s["id"],)).fetchall()
        chosen = rows if args.all else pick_even(rows, args.per_session)
        files = [s["session_dir"] + "/meta.json"]
        for r in chosen:
            files.append(r["rgb_path"])
            if r["depth_path"]:
                files.append(r["depth_path"])
        missing = [f for f in files if not (MIRROR / f).exists()]
        per_sess[s["session_dir"]] = (len(rows), len(chosen), len(missing))
        want += missing

    print(f"{'세션':40s} NAS  선택  신규다운")
    for k, (n, c, m) in per_sess.items():
        print(f"{k:40s} {n:4d} {c:5d} {m:6d}")
    if not want:
        print("내려받을 파일 없음")
    elif args.dry_run:
        print(f"[dry-run] {len(want)}개 파일")
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fp:
            fp.write("\n".join(want)); lst = fp.name
        cmd = [RCLONE, "copy", args.remote, str(MIRROR), "--files-from", lst,
               "--transfers", "8", "--retries", "3", "-q"]
        print(f"rclone: {len(want)}개 파일 → {MIRROR}")
        subprocess.run(cmd, check=True)
        os.unlink(lst)

    # synced_local 갱신 (미러 존재 여부 기준)
    ds_id = con.execute("SELECT id FROM datasets WHERE name = ?", (DATASET_NAME,)).fetchone()[0]
    rows = con.execute(
        """SELECT i.id, i.rgb_path FROM images i JOIN classes c ON c.id = i.class_id
           WHERE c.dataset_id = ?""", (ds_id,)).fetchall()
    n_local = 0
    for r in rows:
        local = (MIRROR / r["rgb_path"]).exists()
        con.execute("UPDATE images SET synced_local = ? WHERE id = ?", (local, r["id"]))
        n_local += local
    con.commit()
    print(f"synced_local: {n_local}/{len(rows)} 장이 학습 PC 미러에 존재")


if __name__ == "__main__":
    main()
