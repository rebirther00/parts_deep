"""추론 서비스 세션(NAS nas:Guest/weld_infer) → SQLite 인제스트 + 썸네일 미러 (2026-11-02).

07_factory_infer.py 가 올린 <day>/s_HHMMSS/{result.json, meta.json, thumb.jpg, rgb_*.png} 를 스캔해
  - datasets 'door_factory_infer' / capture_sessions(class_name = 판정 클래스 또는 Unknown, option_radar = 판정 레이더(auto))
  - images (rgb/depth 메타, synced_local=FALSE)
  - inference_results (판정값 전체; 사용자 확정 confirmed_* 는 건드리지 않음)
에 등록하고, result.json·thumb.jpg 만 로컬 미러 datasets_factory_infer/<day>/<sess>/ 로 받아 webapp 이 보여 준다.
heartbeat.json 도 미러해 대시보드가 서비스 생존을 표시한다.

사용 (학습 PC, cron 30분 권장 — ingest_nas 와 같은 줄에):
  python db/ingest_infer.py                                  # rclone 리모트 (기본 nas:Guest/weld_infer)
  python db/ingest_infer.py --local-dir /path/to/datasets_factory_infer   # 로컬/마운트 (테스트)
  python db/ingest_infer.py --refresh                        # 등록된 세션도 result.json 다시 읽어 갱신
"""
import argparse
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "db"))
from migrate_inference import migrate                      # noqa: E402

DATASET_NAME = "door_factory_infer"
MIRROR = BASE_DIR / "datasets_factory_infer"
SESSION_RE = re.compile(r"^(?P<day>(unsynced_)?\d{8})/(?P<sess>s_\d{6})$")
KNOWN_CLASSES = ("E23_door_LH_FRT", "E25_door_LH_FRT", "E25_door_LH_RR", "E25_door_RH", "E30_door_LH_FRT",
                 "E30_door_LH_RR", "E30_E38_door_RH", "E38_door_LH_FRT", "E38_door_LH_RR")
RCLONE = "rclone"


def list_remote(remote):
    out = subprocess.run([RCLONE, "lsjson", "-R", "--files-only", remote], capture_output=True, text=True, check=True).stdout
    return [e["Path"] for e in json.loads(out)]


def read_remote(remote, rel):
    return json.loads(subprocess.run([RCLONE, "cat", f"{remote}/{rel}"], capture_output=True, text=True, check=True).stdout)


def copy_remote(remote, rel, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([RCLONE, "copyto", f"{remote}/{rel}", str(dst)], capture_output=True, text=True, check=False)


def list_local(root):
    root = Path(root)
    return [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()]


def read_local(root, rel):
    return json.loads((Path(root) / rel).read_text(encoding="utf-8"))


def copy_local(root, rel, dst):
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(root) / rel, dst)


def parse_class_name(name):
    if "_door_" not in name:
        return None, None
    model, part = name.split("_door_", 1)
    return model, f"door_{part}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(BASE_DIR / "db" / "door_pipeline.db"))
    ap.add_argument("--remote", default="nas:Guest/weld_infer")
    ap.add_argument("--local-dir", help="NAS 마운트/복사본 경로 (지정 시 rclone 대신)")
    ap.add_argument("--refresh", action="store_true", help="등록된 세션도 result.json 다시 읽어 판정값 갱신")
    ap.add_argument("--no-mirror", action="store_true", help="result.json·thumb.jpg 로컬 미러 생략")
    ap.add_argument("--mirror-dir", default=str(MIRROR), help="미러 위치 (기본 datasets_factory_infer/; 테스트용)")
    ap.add_argument("--include-open", action="store_true", help="final=false(진행 중) 세션도 등록")
    args = ap.parse_args()
    mirror = Path(args.mirror_dir).resolve()

    if args.local_dir:
        files = list_local(args.local_dir)
        read_json = lambda rel: read_local(args.local_dir, rel)
        copy = lambda rel, dst: copy_local(args.local_dir, rel, dst)
        source_label = args.local_dir
    else:
        files = list_remote(args.remote)
        read_json = lambda rel: read_remote(args.remote, rel)
        copy = lambda rel, dst: copy_remote(args.remote, rel, dst)
        source_label = args.remote

    sessions = {}
    for rel in files:
        parent, _, name = rel.rpartition("/")
        if not SESSION_RE.match(parent):
            continue
        s = sessions.setdefault(parent, {"result": False, "files": []})
        if name == "result.json":
            s["result"] = True
        else:
            s["files"].append(name)
    print(f"소스 {source_label}: 세션 {len(sessions)}개 발견")

    con = sqlite3.connect(args.db)
    con.executescript((BASE_DIR / "db" / "schema.sql").read_text(encoding="utf-8"))
    migrate(con)
    cur = con.cursor()
    cur.execute(
        """INSERT INTO datasets (name, type, description, base_path) VALUES (?, 'real', ?, ?)
           ON CONFLICT(name) DO UPDATE SET base_path = excluded.base_path, updated_at = datetime('now','localtime')""",
        (DATASET_NAME, "경주 공장 실시간 판정 세션 (07_factory_infer, 판정 프레임만·NAS 정본, 라벨 = 모델 판정 → 사용자 확정)", source_label))
    dataset_id = cur.execute("SELECT id FROM datasets WHERE name = ?", (DATASET_NAME,)).fetchone()[0]

    n_new = n_skip = n_open = 0
    for sess_dir in sorted(sessions):
        info = sessions[sess_dir]
        if not info["result"]:
            continue
        row = cur.execute("SELECT s.id, r.id FROM capture_sessions s LEFT JOIN inference_results r ON r.session_id = s.id "
                          "WHERE s.session_dir = ?", (sess_dir,)).fetchone()
        if row and row[1] and not args.refresh:
            n_skip += 1
            continue
        try:
            r = read_json(f"{sess_dir}/result.json")
        except (ValueError, subprocess.CalledProcessError, OSError) as e:
            print(f"  경고: {sess_dir} result.json 읽기 실패 — {e}")
            continue
        if not r.get("final") and not args.include_open:
            n_open += 1
            continue
        status = r.get("status") or "abstain"
        pred = r.get("pred_class")
        cls = pred if pred in KNOWN_CLASSES else "Unknown"
        radar = r.get("radar") if r.get("radar") in (0, 1) else None

        cur.execute(
            """INSERT INTO capture_sessions
                 (dataset_id, session_dir, class_name, camera_type, capture_method, saved_pairs, valid_frames,
                  finished, stop_reason, started_at, ended_at, option_radar, option_source, option_at)
               VALUES (?, ?, ?, ?, 'snapshot', ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_dir) DO UPDATE SET
                 class_name = excluded.class_name, saved_pairs = excluded.saved_pairs, valid_frames = excluded.valid_frames,
                 finished = excluded.finished, stop_reason = excluded.stop_reason, ended_at = excluded.ended_at,
                 option_radar = CASE WHEN capture_sessions.option_source = 'user' THEN capture_sessions.option_radar ELSE excluded.option_radar END,
                 option_source = CASE WHEN capture_sessions.option_source = 'user' THEN 'user' ELSE excluded.option_source END,
                 option_at = CASE WHEN capture_sessions.option_source = 'user' THEN capture_sessions.option_at ELSE excluded.option_at END""",
            (dataset_id, sess_dir, cls, r.get("camera") or "unknown", r.get("saved_pairs", 0), r.get("saved_pairs", 0),
             bool(r.get("final")), status, r.get("started_at"), r.get("ended_at"),
             radar, "auto" if radar is not None else None, r.get("judged_at") if radar is not None else None))
        session_id = cur.execute("SELECT id FROM capture_sessions WHERE session_dir = ?", (sess_dir,)).fetchone()[0]

        # 이미지 행: 사용자가 확정(relabel)한 세션은 기존 class_id 유지
        existing = cur.execute("SELECT class_id FROM images WHERE session_id = ? LIMIT 1", (session_id,)).fetchone()
        if existing:
            class_id = existing[0]
        else:
            model_name, part_type = parse_class_name(cls)
            cur.execute("INSERT OR IGNORE INTO classes (dataset_id, name, display_name, model_name, part_type) VALUES (?, ?, ?, ?, ?)",
                        (dataset_id, cls, cls.replace("_", " "), model_name, part_type))
            class_id = cur.execute("SELECT id FROM classes WHERE dataset_id = ? AND name = ?", (dataset_id, cls)).fetchone()[0]
        depth_files = {f for f in info["files"] if f.startswith("depth_")}
        for rgb in sorted(f for f in info["files"] if f.startswith("rgb_") and f.endswith(".png")):
            depth = rgb.replace("rgb_", "depth_")
            has_depth = depth in depth_files
            cur.execute(
                """INSERT OR IGNORE INTO images (class_id, session_id, rgb_filename, depth_filename, rgb_path, depth_path,
                                                 channels, data_source, synced_local, captured_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'camera', FALSE, ?)""",
                (class_id, session_id, rgb, depth if has_depth else None, f"{sess_dir}/{rgb}",
                 f"{sess_dir}/{depth}" if has_depth else None, 4 if has_depth else 3, r.get("started_at")))
        cur.execute("UPDATE classes SET image_count = (SELECT COUNT(*) FROM images WHERE class_id = ?) WHERE id = ?", (class_id, class_id))

        thumb_rel = None
        if not args.no_mirror:
            for name in ("result.json", "thumb.jpg"):
                if name == "thumb.jpg" and name not in info["files"]:
                    continue
                dst = mirror / sess_dir / name
                if not dst.exists() or args.refresh:
                    copy(f"{sess_dir}/{name}", dst)
                if name == "thumb.jpg" and dst.exists():
                    thumb_rel = str(dst.relative_to(BASE_DIR)) if dst.is_relative_to(BASE_DIR) else str(dst)
        models = r.get("models") or {}
        cur.execute(
            """INSERT INTO inference_results
                 (session_id, status, pred_class, pred_group, part_no, part_name, radar, radar_txt, source, conflict, confidence,
                  d_mm, margin_mm, nearest_mm, n_judged, n_window, n_processed, n_clean, gate_counts, candidates, cnn,
                  model_hole, model_hole_sha, model_cnn, service_version, dwell_s, judged_at, started_at, ended_at, thumb_path, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 status = excluded.status, pred_class = excluded.pred_class, pred_group = excluded.pred_group,
                 part_no = excluded.part_no, part_name = excluded.part_name, radar = excluded.radar, radar_txt = excluded.radar_txt,
                 source = excluded.source, conflict = excluded.conflict, confidence = excluded.confidence,
                 d_mm = excluded.d_mm, margin_mm = excluded.margin_mm, nearest_mm = excluded.nearest_mm,
                 n_judged = excluded.n_judged, n_window = excluded.n_window, n_processed = excluded.n_processed, n_clean = excluded.n_clean,
                 gate_counts = excluded.gate_counts, candidates = excluded.candidates, cnn = excluded.cnn,
                 model_hole = excluded.model_hole, model_hole_sha = excluded.model_hole_sha, model_cnn = excluded.model_cnn,
                 service_version = excluded.service_version, dwell_s = excluded.dwell_s, judged_at = excluded.judged_at,
                 started_at = excluded.started_at, ended_at = excluded.ended_at,
                 thumb_path = COALESCE(excluded.thumb_path, inference_results.thumb_path), result_json = excluded.result_json,
                 updated_at = datetime('now','localtime')""",
            (session_id, status, pred, r.get("group") if r.get("group") != "-" else None, r.get("part_no"), r.get("part_name"),
             radar, r.get("radar_txt"), r.get("source"), bool(r.get("conflict")), r.get("confidence"),
             r.get("D_mm"), r.get("margin_mm"), r.get("nearest_mm"), r.get("n_judged"), r.get("window"),
             r.get("n_processed"), r.get("n_clean"), json.dumps(r.get("gate_counts") or {}, ensure_ascii=False),
             json.dumps(r.get("candidates") or [], ensure_ascii=False), json.dumps(r.get("cnn"), ensure_ascii=False),
             models.get("hole"), models.get("hole_sha"), models.get("cnn"), r.get("service_version"), r.get("dwell_s"),
             r.get("judged_at"), r.get("started_at"), r.get("ended_at"), thumb_rel, json.dumps(r, ensure_ascii=False)))
        n_new += 1

    cur.execute(
        """UPDATE datasets SET num_classes = (SELECT COUNT(*) FROM classes WHERE dataset_id = ?),
             total_images = (SELECT COUNT(*) FROM images i JOIN classes c ON i.class_id = c.id WHERE c.dataset_id = ?),
             updated_at = datetime('now','localtime') WHERE id = ?""", (dataset_id, dataset_id, dataset_id))
    con.commit()
    if not args.no_mirror and "heartbeat.json" in files:
        try:
            copy("heartbeat.json", mirror / "heartbeat.json")
        except Exception as e:                      # noqa: BLE001
            print(f"  경고: heartbeat.json 미러 실패 — {e}")
    total = cur.execute("SELECT COUNT(*) FROM inference_results").fetchone()[0]
    by = dict(cur.execute("SELECT status, COUNT(*) FROM inference_results GROUP BY status").fetchall())
    print(f"등록·갱신 {n_new}개 / 기존 {n_skip}개 / 진행 중 {n_open}개 — inference_results {total}행 {by}")
    con.close()


if __name__ == "__main__":
    main()
