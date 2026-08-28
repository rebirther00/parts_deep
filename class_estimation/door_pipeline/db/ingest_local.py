"""로컬 데이터셋(datasets*, door_pipeline 기준) → SQLite 인제스트.

report/DBMS_SCHEMA_DESIGN.md 7.2절의 1단계 마이그레이션:
  dataset_info.json / metadata.json / rgb_*.png 스캔 → datasets, classes, images

사용:
  python db/ingest_local.py            # door_pipeline 루트에서 실행
  python db/ingest_local.py --db db/door_pipeline.db
"""

import argparse
import json
import sqlite3
import struct
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# (DB dataset name, 디렉터리, dataset type, 이미지 data_source, 설명)
LOCAL_DATASETS = [
    ("door_real", "datasets", "real", "camera",
     "ZED X Mini 실물 촬영 원본 (사무실 수집)"),
    ("door_factory", "datasets_factory", "real", "camera",
     "경주 공장 현장 수집분 중 학습 PC로 가져온 선별본"),
    ("door_aug", "datasets_aug", "real", "augmented",
     "증강 데이터 — 강건성 평가 전용, 학습 사용 금지"),
    ("door_aug2", "datasets_aug2", "real", "augmented",
     "증강 데이터 2차 — 강건성 평가 전용, 학습 사용 금지"),
    ("door_field", "datasets_field", "real", "camera",
     "경주 공장 현장 세션 로컬 복사본 (<class>_s_HHMMSS 폴더, 홀 판별기 평가용)"),
]


def png_size(path: Path):
    """PNG IHDR에서 (width, height)를 읽는다. 실패 시 (None, None)."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None, None
        return struct.unpack(">II", head[16:24])
    except OSError:
        return None, None


def parse_class_name(name: str):
    """'E30_E38_door_RH' → (model_name='E30_E38', part_type='door_RH')"""
    if "_door_" not in name:
        return None, None
    model, part = name.split("_door_", 1)
    return model, f"door_{part}"


def ingest_dataset(cur, db_name, dir_name, ds_type, data_source, desc):
    root = BASE_DIR / dir_name
    if not root.is_dir():
        print(f"  건너뜀: {dir_name}/ 없음")
        return

    info_path = root / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    cur.execute(
        """INSERT INTO datasets (name, type, description, base_path)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
             description = excluded.description,
             updated_at = datetime('now','localtime')""",
        (db_name, ds_type, desc, dir_name + "/"),
    )
    dataset_id = cur.execute(
        "SELECT id FROM datasets WHERE name = ?", (db_name,)
    ).fetchone()[0]

    n_images = 0
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    for cdir in class_dirs:
        dname = cdir.name
        cname = dname.split("_s_")[0]   # 'E25_door_RH_s_091317' → 'E25_door_RH'
        model_name, part_type = parse_class_name(cname)

        meta_path = cdir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        cur.execute(
            """INSERT OR IGNORE INTO classes
               (dataset_id, name, display_name, model_name, part_type, cad_available)
               VALUES (?, ?, ?, ?, ?, TRUE)""",
            (dataset_id, cname, meta.get("display_name", cname.replace("_", " ")),
             model_name, part_type),
        )
        class_id = cur.execute(
            "SELECT id FROM classes WHERE dataset_id = ? AND name = ?",
            (dataset_id, cname),
        ).fetchone()[0]

        count = 0
        for rgb in sorted(cdir.glob("rgb_*.png")):
            depth = cdir / rgb.name.replace("rgb_", "depth_")
            has_depth = depth.exists()
            w, h = png_size(rgb)
            captured = datetime.fromtimestamp(rgb.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            cur.execute(
                """INSERT OR IGNORE INTO images
                   (class_id, rgb_filename, depth_filename, rgb_path, depth_path,
                    width, height, channels, data_source, synced_local, captured_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE, ?)""",
                (class_id, rgb.name, depth.name if has_depth else None,
                 f"{dname}/{rgb.name}",
                 f"{dname}/{depth.name}" if has_depth else None,
                 w, h, 4 if has_depth else 3, data_source, captured),
            )
            count += 1

        cur.execute(
            "UPDATE classes SET image_count = "
            "(SELECT COUNT(*) FROM images WHERE class_id = ?) WHERE id = ?",
            (class_id, class_id),
        )
        n_images += count

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
    print(f"  {db_name}: {len(class_dirs)}클래스, {n_images}장 스캔")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(BASE_DIR / "db" / "door_pipeline.db"))
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.executescript((BASE_DIR / "db" / "schema.sql").read_text())
    cur = con.cursor()
    for spec in LOCAL_DATASETS:
        ingest_dataset(cur, *spec)
    con.commit()

    for name, total, nc in cur.execute(
        "SELECT name, total_images, num_classes FROM datasets ORDER BY id"
    ):
        print(f"  DB: {name:20s} classes={nc:2d} images={total}")
    con.close()


if __name__ == "__main__":
    main()
