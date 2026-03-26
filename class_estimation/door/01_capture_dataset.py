"""굴착기 Door 분류 데이터셋 구축 — Flask 캡처 서버

실행:
    python 01_capture_dataset.py
    브라우저에서 http://0.0.0.0:5000 접속
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import (
    Flask, Response, jsonify, render_template, request, send_from_directory,
)

from camera_utils import CameraManager

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
TEMP_DIR = BASE_DIR / "temp_frames"

CLASSES = [
    "E25_door_LH_FRT", "E25_door_LH_RR", "E25_door_RH",
    "E30_door_LH_FRT", "E30_door_LH_RR",
    "E30_E38_door_RH",
    "E38_door_LH_FRT", "E38_door_LH_RR",
]

app = Flask(__name__)
camera: CameraManager = None  # type: ignore[assignment]


# ── 초기화 ──────────────────────────────────────────────

def init_directories():
    for d in [DATASETS_DIR, TEMP_DIR, BASE_DIR / "artifacts"]:
        d.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (DATASETS_DIR / cls).mkdir(exist_ok=True)

    info_path = DATASETS_DIR / "dataset_info.json"
    if not info_path.exists():
        info = {
            "dataset_name": "Excavator Door Classification Dataset",
            "num_classes": len(CLASSES),
            "classes": {c: c for c in CLASSES},
            "data_source": "real_camera",
            "camera": "ZED X Mini",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))


# ── 유틸리티 ────────────────────────────────────────────

def _next_pair_index(class_name: str) -> int:
    """rgb_XXXX.png / depth_XXXX.png 쌍의 다음 인덱스 반환"""
    existing = sorted((DATASETS_DIR / class_name).glob("rgb_*.png"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[1]) + 1


def _dataset_status() -> dict:
    return {
        cls: len(list((DATASETS_DIR / cls).glob("rgb_*.png")))
        for cls in CLASSES
    }


def _update_metadata(class_name: str):
    class_dir = DATASETS_DIR / class_name
    meta = {
        "class_name": class_name,
        "display_name": class_name.replace("_", " "),
        "data_source": "real_camera",
        "camera": camera.camera_type if camera else "unknown",
        "num_images": len(list(class_dir.glob("rgb_*.png"))),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (class_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )


def _generate_mjpeg():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = cv2.resize(frame, (960, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )


# ── 라우트 ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES, cache_bust=int(time.time()))


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/camera_info")
def api_camera_info():
    return jsonify({"camera_type": camera.camera_type, "connected": camera.running})


@app.route("/api/classes")
def api_classes():
    return jsonify(CLASSES)


@app.route("/api/dataset_status")
def api_dataset_status():
    return jsonify(_dataset_status())


@app.route("/api/start_recording", methods=["POST"])
def api_start_recording():
    data = request.json or {}
    cls = data.get("class_name")
    if cls not in CLASSES:
        return jsonify({"error": "잘못된 클래스명"}), 400
    camera.start_recording(
        str(TEMP_DIR),
        interval=data.get("frame_interval", 5),
        blur_threshold=data.get("blur_threshold", 100),
    )
    return jsonify({"status": "recording", "class_name": cls})


@app.route("/api/stop_recording", methods=["POST"])
def api_stop_recording():
    result = camera.stop_recording()
    return jsonify({"status": "stopped", **result})


@app.route("/api/recording_status")
def api_recording_status():
    return jsonify({
        "recording": camera.recording,
        "extracted_count": camera._extracted_count,
        "frame_count": camera._frame_count,
    })


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    entry = camera.snapshot(str(TEMP_DIR))
    if entry is None:
        return jsonify({"error": "프레임 캡처 실패"}), 500
    return jsonify(entry)


@app.route("/api/extracted_frames")
def api_extracted_frames():
    frames = sorted(
        f for f in TEMP_DIR.glob("frame_*.png") if "depth" not in f.name
    )
    return jsonify([{"filename": f.name} for f in frames])


@app.route("/api/save_selected", methods=["POST"])
def api_save_selected():
    data = request.json or {}
    cls = data.get("class_name")
    filenames = data.get("filenames", [])
    if cls not in CLASSES:
        return jsonify({"error": "잘못된 클래스명"}), 400
    if not filenames:
        return jsonify({"error": "선택된 프레임이 없습니다"}), 400

    idx = _next_pair_index(cls)
    saved = []
    for fname in sorted(filenames):
        src = TEMP_DIR / fname
        if not src.exists():
            continue
        dst_rgb = DATASETS_DIR / cls / f"rgb_{idx:04d}.png"
        shutil.copy2(str(src), str(dst_rgb))
        saved.append(dst_rgb.name)

        # Depth 파일이 있으면 함께 복사
        depth_src = TEMP_DIR / fname.replace("frame_", "frame_depth_")
        if depth_src.exists():
            dst_depth = DATASETS_DIR / cls / f"depth_{idx:04d}.png"
            shutil.copy2(str(depth_src), str(dst_depth))

        idx += 1

    _update_metadata(cls)

    # 제외된(선택되지 않은) 프레임과 매칭 Depth를 temp에서 삭제
    selected_set = set(filenames)
    for f in sorted(TEMP_DIR.glob("frame_*.png")):
        if "depth" in f.name:
            continue
        if f.name not in selected_set:
            f.unlink(missing_ok=True)
            depth_f = TEMP_DIR / f.name.replace("frame_", "frame_depth_")
            if depth_f.exists():
                depth_f.unlink()

    return jsonify({"saved": saved, "count": len(saved), "class_name": cls})


@app.route("/api/clear_temp", methods=["POST"])
def api_clear_temp():
    for f in TEMP_DIR.glob("frame_*.png"):
        f.unlink()
    for f in TEMP_DIR.glob("frame_depth_*.png"):
        f.unlink()
    camera.reset_counter()
    return jsonify({"status": "cleared"})


@app.route("/temp_frames/<filename>")
def serve_temp_frame(filename):
    return send_from_directory(str(TEMP_DIR), filename)


# ── 엔트리포인트 ────────────────────────────────────────

if __name__ == "__main__":
    init_directories()
    camera = CameraManager()
    camera.start()
    print(f"카메라 타입: {camera.camera_type}")
    print("서버 시작: http://0.0.0.0:5000")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        camera.stop()
