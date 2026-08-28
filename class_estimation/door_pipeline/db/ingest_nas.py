"""NAS(세종, nas:Guest/weld)의 현장 수집 세션 → SQLite 인제스트.

06_factory_capture.py가 올린 <day>/<class>/s_HHMMSS/ 세션 트리를 스캔해
capture_sessions + images 테이블에 등록한다. 이미지 파일은 내려받지 않고
메타데이터만 읽으므로 (synced_local=FALSE), "NAS에 뭐가 있는지"를 DB로
조회하는 용도다. 실제 파일 동기화는 별도로 한다.

사용:
  python db/ingest_nas.py                          # rclone 리모트 (기본 nas:Guest/weld)
  python db/ingest_nas.py --remote nas:Guest/weld
  python db/ingest_nas.py --local-dir /mnt/nas/weld  # NAS 마운트/복사본 스캔
"""

import argparse
import json
import re
import sqlite3
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_NAME = "door_factory_collect"
SESSION_RE = re.compile(r"^(?P<day>[^/]+)/(?P<cls>[^/]+)/(?P<sess>s_\d{6})$")


def list_remote(remote):
    """rclone lsjson -R 로 전체 파일 목록(상대경로)을 얻는다."""
    out = subprocess.run(
        ["rclone", "lsjson", "-R", "--files-only", remote],
        capture_output=True, text=True, check=True,
    ).stdout
    return [e["Path"] for e in json.loads(out)]


def read_remote(remote, relpath):
    out = subprocess.run(
        ["rclone", "cat", f"{remote}/{relpath}"],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def list_local(root):
    root = Path(root)
    return [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]


def read_local(root, relpath):
    return json.loads((Path(root) / relpath).read_text())


LOCAL_COPY = BASE_DIR / "datasets_factory_collect"


def mark_synced_local(cur, dataset_id):
    """학습 PC에 복사된 세션(datasets_factory_collect/<day>/<class>/s_*)은 synced_local=TRUE."""
    if not LOCAL_COPY.is_dir():
        return 0
    rows = cur.execute(
        """SELECT i.id, i.rgb_path FROM images i JOIN classes c ON i.class_id = c.id
           WHERE c.dataset_id = ?""", (dataset_id,)).fetchall()
    n = 0
    for iid, rel in rows:
        local = (LOCAL_COPY / rel).exists()
        cur.execute("UPDATE images SET synced_local = ? WHERE id = ?", (local, iid))
        n += local
    return n


def parse_class_name(name):
    if "_door_" not in name:
        return None, None
    model, part = name.split("_door_", 1)
    return model, f"door_{part}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(BASE_DIR / "db" / "door_pipeline.db"))
    ap.add_argument("--remote", default="nas:Guest/weld")
    ap.add_argument("--local-dir", help="NAS 마운트 경로/복사본 (지정 시 rclone 대신 사용)")
    ap.add_argument("--refresh", action="store_true",
                    help="이미 등록된 세션도 메타를 다시 읽어 갱신")
    args = ap.parse_args()

    if args.local_dir:
        files = list_local(args.local_dir)
        read_meta = lambda rel: read_local(args.local_dir, rel)
        source_label = args.local_dir
    else:
        files = list_remote(args.remote)
        read_meta = lambda rel: read_remote(args.remote, rel)
        source_label = args.remote

    # 세션 디렉터리별 파일 그룹핑
    sessions = {}  # sess_dir -> {"meta": bool, "files": [names]}
    for rel in files:
        parent, _, name = rel.rpartition("/")
        m = SESSION_RE.match(parent)
        if not m:
            continue
        s = sessions.setdefault(parent, {"meta": False, "files": []})
        if name == "meta.json":
            s["meta"] = True
        else:
            s["files"].append(name)
    print(f"소스 {source_label}: 세션 {len(sessions)}개 발견")

    con = sqlite3.connect(args.db)
    con.executescript((BASE_DIR / "db" / "schema.sql").read_text())
    cur = con.cursor()

    cur.execute(
        """INSERT INTO datasets (name, type, description, base_path)
           VALUES (?, 'real', ?, ?)
           ON CONFLICT(name) DO UPDATE SET
             base_path = excluded.base_path, updated_at = datetime('now','localtime')""",
        (DATASET_NAME, "경주 공장 현장 수집 원본 (NAS 정본, 세션 단위)", source_label),
    )
    dataset_id = cur.execute(
        "SELECT id FROM datasets WHERE name = ?", (DATASET_NAME,)
    ).fetchone()[0]

    n_new = n_skip = 0
    for sess_dir in sorted(sessions):
        info = sessions[sess_dir]
        row = cur.execute(
            "SELECT id FROM capture_sessions WHERE session_dir = ?", (sess_dir,)
        ).fetchone()
        if row and not args.refresh:
            n_skip += 1
            continue

        if not info["meta"]:
            print(f"  경고: {sess_dir} meta.json 없음 — 건너뜀")
            continue
        meta = read_meta(f"{sess_dir}/meta.json")
        cls = meta.get("class_name") or SESSION_RE.match(sess_dir)["cls"]

        cur.execute(
            """INSERT INTO capture_sessions
                 (dataset_id, session_dir, class_name, camera_type,
                  capture_method, capture_interval_s, session_duration_s,
                  saved_pairs, valid_frames, finished, stop_reason,
                  started_at, ended_at)
               VALUES (?, ?, ?, ?, 'snapshot', ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_dir) DO UPDATE SET
                 saved_pairs = excluded.saved_pairs,
                 valid_frames = excluded.valid_frames,
                 finished = excluded.finished,
                 stop_reason = excluded.stop_reason,
                 ended_at = excluded.ended_at""",
            (dataset_id, sess_dir, cls, meta.get("camera", "unknown"),
             meta.get("capture_interval_s"), meta.get("session_duration_s"),
             meta.get("saved_pairs", 0), meta.get("saved_pairs", 0),
             bool(meta.get("finished")), meta.get("stop_reason") or None,
             meta.get("started_at"), meta.get("ended_at")),
        )
        session_id = cur.execute(
            "SELECT id FROM capture_sessions WHERE session_dir = ?", (sess_dir,)
        ).fetchone()[0]

        model_name, part_type = parse_class_name(cls)
        cur.execute(
            """INSERT OR IGNORE INTO classes
                 (dataset_id, name, display_name, model_name, part_type)
               VALUES (?, ?, ?, ?, ?)""",
            (dataset_id, cls, cls.replace("_", " "), model_name, part_type),
        )
        class_id = cur.execute(
            "SELECT id FROM classes WHERE dataset_id = ? AND name = ?",
            (dataset_id, cls),
        ).fetchone()[0]

        depth_files = {f for f in info["files"] if f.startswith("depth_")}
        for rgb in sorted(f for f in info["files"] if f.startswith("rgb_")):
            depth = rgb.replace("rgb_", "depth_")
            has_depth = depth in depth_files
            cur.execute(
                """INSERT OR IGNORE INTO images
                     (class_id, session_id, rgb_filename, depth_filename,
                      rgb_path, depth_path, channels, data_source,
                      synced_local, captured_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'camera', FALSE, ?)""",
                (class_id, session_id, rgb, depth if has_depth else None,
                 f"{sess_dir}/{rgb}",
                 f"{sess_dir}/{depth}" if has_depth else None,
                 4 if has_depth else 3, meta.get("started_at")),
            )
        cur.execute(
            "UPDATE classes SET image_count = "
            "(SELECT COUNT(*) FROM images WHERE class_id = ?) WHERE id = ?",
            (class_id, class_id),
        )
        n_new += 1

    cur.execute(
        """UPDATE datasets SET
             num_classes = (SELECT COUNT(*) FROM classes WHERE dataset_id = ?),
             total_images = (SELECT COUNT(*) FROM images i
                             JOIN classes c ON i.class_id = c.id
                             WHERE c.dataset_id = ?),
             updated_at = datetime('now','localtime')
           WHERE id = ?""",
        (dataset_id, dataset_id, dataset_id),
    )
    n_local = mark_synced_local(cur, dataset_id)
    con.commit()

    total, finished = cur.execute(
        """SELECT COUNT(*), SUM(finished) FROM capture_sessions
           WHERE dataset_id = ?""", (dataset_id,)
    ).fetchone()
    imgs = cur.execute(
        """SELECT COUNT(*) FROM images i JOIN classes c ON i.class_id = c.id
           WHERE c.dataset_id = ?""", (dataset_id,)
    ).fetchone()[0]
    print(f"등록 {n_new}개 / 기존 {n_skip}개 — DB에 세션 {total}개"
          f"(완료 {finished}), 이미지 {imgs}장, 학습 PC 로컬 보유 {n_local}장")
    con.close()


if __name__ == "__main__":
    main()
