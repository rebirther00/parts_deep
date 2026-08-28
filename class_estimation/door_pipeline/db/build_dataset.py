"""현장 수집 데이터의 학습·평가용 뷰(datasets_factory_v2/) 생성 + 세션 단위 split/라벨 관리.

datasets_factory_v2/ 는 파일 복사가 아니라 미러(datasets_factory_collect/)를 가리키는 심볼릭 링크 뷰다.
DB 기준(synced_local=1, is_valid=1, 클래스 확정, split 지정)으로만 채우며, 재실행 시 전부 다시 만든다.

  datasets_factory_v2/
    all/<class>/rgb_<날짜>_<세션>_<idx>.png (+depth_)   split 무관 전체
    train|val|test/<class>/...                          images.split 별
    manifest.json                                       이미지별 세션·클래스·split (학습 세션 DB 연결용)

명령:
  python db/build_dataset.py build                       # 뷰 생성
  python db/build_dataset.py status                      # 클래스·세션별 split 현황
  python db/build_dataset.py split <session_dir> <train|val|test|none>
  python db/build_dataset.py auto-split                  # 클래스별 첫 세션=test, 나머지=train (미지정 세션만)
  python db/build_dataset.py relabel <session_dir> <class>   # 라벨 정정(Unknown 포함) — 파일 이동 없이 DB만

규칙: split은 항상 세션 단위(프레임 단위 분할 금지). 라벨 정정은 capture_sessions.class_name(현장 입력 원본)은
그대로 두고 images.class_id 만 재배정, notes 에 이력을 남긴다.
"""
import argparse
import json
import os
import sqlite3
import shutil
import time
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MIRROR = BASE_DIR / "datasets_factory_collect"
VIEW = BASE_DIR / "datasets_factory_v2"
DATASET_NAME = "door_factory_collect"
SPLITS = ("train", "val", "test")


def connect(db):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def dataset_id(con):
    return con.execute("SELECT id FROM datasets WHERE name = ?", (DATASET_NAME,)).fetchone()[0]


def session_row(con, session_dir):
    r = con.execute("SELECT * FROM capture_sessions WHERE session_dir = ?", (session_dir,)).fetchone()
    if not r:
        raise SystemExit(f"세션 없음: {session_dir}")
    return r


def parse_class_name(name):
    if "_door_" not in name:
        return None, None
    model, part = name.split("_door_", 1)
    return model, f"door_{part}"


def cmd_split(con, session_dir, split):
    s = session_row(con, session_dir)
    val = None if split == "none" else split
    con.execute("UPDATE images SET split = ? WHERE session_id = ?", (val, s["id"]))
    con.commit()
    print(f"{session_dir}: split={val}")


def cmd_auto_split(con):
    ds = dataset_id(con)
    rows = con.execute(
        """SELECT s.id, s.session_dir, c.name AS cls, s.started_at,
                  (SELECT split FROM images WHERE session_id = s.id LIMIT 1) AS split,
                  (SELECT COUNT(*) FROM images WHERE session_id = s.id AND synced_local) AS n_local
           FROM capture_sessions s
           JOIN images i ON i.session_id = s.id JOIN classes c ON c.id = i.class_id
           WHERE s.dataset_id = ? AND c.name != 'Unknown'
           GROUP BY s.id ORDER BY c.name, s.started_at""", (ds,)).fetchall()
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    for cls, ss in by_cls.items():
        has_test = any(r["split"] == "test" for r in ss)
        for r in ss:
            if r["split"] or r["n_local"] == 0:
                continue
            new = "train" if has_test else "test"
            has_test = has_test or new == "test"
            con.execute("UPDATE images SET split = ? WHERE session_id = ?", (new, r["id"]))
            print(f"  {r['session_dir']:40s} → {new}")
    con.commit()


def cmd_relabel(con, session_dir, new_cls):
    s = session_row(con, session_dir)
    ds = dataset_id(con)
    model, part = parse_class_name(new_cls)
    con.execute(
        """INSERT OR IGNORE INTO classes (dataset_id, name, display_name, model_name, part_type)
           VALUES (?, ?, ?, ?, ?)""", (ds, new_cls, new_cls.replace("_", " "), model, part))
    cid = con.execute("SELECT id FROM classes WHERE dataset_id = ? AND name = ?", (ds, new_cls)).fetchone()[0]
    old = con.execute(
        """SELECT DISTINCT c.name FROM images i JOIN classes c ON c.id = i.class_id
           WHERE i.session_id = ?""", (s["id"],)).fetchall()
    old = ",".join(r[0] for r in old) or "?"
    n = con.execute("UPDATE images SET class_id = ? WHERE session_id = ?", (cid, s["id"])).rowcount
    note = f"[{time.strftime('%Y-%m-%d %H:%M')}] relabel {old} → {new_cls} ({n}장)"
    con.execute("UPDATE capture_sessions SET notes = COALESCE(notes || '\n', '') || ? WHERE id = ?",
                (note, s["id"]))
    for r in con.execute("SELECT id FROM classes WHERE dataset_id = ?", (ds,)):
        con.execute("UPDATE classes SET image_count = (SELECT COUNT(*) FROM images WHERE class_id = ?) WHERE id = ?",
                    (r[0], r[0]))
    con.commit()
    print(f"{session_dir}: {note}  (현장 입력 원본 class_name={s['class_name']} 유지)")


def cmd_status(con):
    ds = dataset_id(con)
    rows = con.execute(
        """SELECT c.name AS cls, s.session_dir, s.class_name AS field_label,
                  COUNT(*) AS n, SUM(i.synced_local) AS n_local,
                  MAX(i.split) AS split, s.notes
           FROM images i JOIN classes c ON c.id = i.class_id
           JOIN capture_sessions s ON s.id = i.session_id
           WHERE c.dataset_id = ? GROUP BY s.id ORDER BY c.name, s.session_dir""", (ds,)).fetchall()
    print(f"{'클래스(DB)':18s} {'세션':38s} {'현장입력':16s}  NAS 로컬  split")
    for r in rows:
        flag = "" if r["field_label"] == r["cls"] else " *정정"
        print(f"{r['cls']:18s} {r['session_dir']:38s} {r['field_label']:16s} {r['n']:4d} {r['n_local']:4d}  "
              f"{r['split'] or '-'}{flag}")
    tot = con.execute(
        """SELECT c.name, i.split, COUNT(*) FROM images i JOIN classes c ON c.id = i.class_id
           WHERE c.dataset_id = ? AND i.synced_local AND i.is_valid AND c.name != 'Unknown'
           GROUP BY 1, 2 ORDER BY 1, 2""", (ds,)).fetchall()
    print("\n뷰 포함 대상(로컬·유효·라벨확정) 클래스×split:")
    for name, split, n in tot:
        print(f"  {name:18s} {split or '(미지정)':8s} {n:4d}")


def cmd_build(con):
    ds = dataset_id(con)
    rows = con.execute(
        """SELECT i.rgb_path, i.depth_path, i.split, c.name AS cls, s.session_dir
           FROM images i JOIN classes c ON c.id = i.class_id
           JOIN capture_sessions s ON s.id = i.session_id
           WHERE c.dataset_id = ? AND i.synced_local AND i.is_valid AND c.name != 'Unknown'
           ORDER BY c.name, i.rgb_path""", (ds,)).fetchall()
    if VIEW.exists():
        shutil.rmtree(VIEW)
    manifest = []
    counts = defaultdict(int)
    for r in rows:
        date, cls_dir, sess = r["session_dir"].split("/")
        idx = Path(r["rgb_path"]).stem.split("_")[-1]
        stem = f"{date}_{sess}_{idx}"
        targets = ["all"] + ([r["split"]] if r["split"] else [])
        for t in targets:
            d = VIEW / t / r["cls"]
            d.mkdir(parents=True, exist_ok=True)
            for kind, rel in (("rgb", r["rgb_path"]), ("depth", r["depth_path"])):
                if not rel:
                    continue
                src = MIRROR / rel
                if not src.exists():
                    continue
                link = d / f"{kind}_{stem}.png"
                os.symlink(os.path.relpath(src, d), link)
            counts[(t, r["cls"])] += 1
        manifest.append({"rgb": f"{r['cls']}/rgb_{stem}.png", "session_dir": r["session_dir"],
                         "class": r["cls"], "split": r["split"], "src": r["rgb_path"]})
    (VIEW / "manifest.json").write_text(json.dumps(
        {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "mirror": str(MIRROR),
         "n_images": len(manifest), "images": manifest}, ensure_ascii=False, indent=1))
    print(f"{VIEW} 생성: {len(manifest)}장")
    for (t, cls), n in sorted(counts.items()):
        print(f"  {t:6s} {cls:18s} {n:4d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(BASE_DIR / "db" / "door_pipeline.db"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build"); sub.add_parser("status"); sub.add_parser("auto-split")
    p = sub.add_parser("split"); p.add_argument("session_dir"); p.add_argument("split", choices=SPLITS + ("none",))
    p = sub.add_parser("relabel"); p.add_argument("session_dir"); p.add_argument("new_class")
    a = ap.parse_args()
    con = connect(a.db)
    if a.cmd == "build":
        cmd_build(con)
    elif a.cmd == "status":
        cmd_status(con)
    elif a.cmd == "auto-split":
        cmd_auto_split(con)
    elif a.cmd == "split":
        cmd_split(con, a.session_dir, a.split)
    elif a.cmd == "relabel":
        cmd_relabel(con, a.session_dir, a.new_class)


if __name__ == "__main__":
    main()
