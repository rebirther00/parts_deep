"""실시간 부품 치수 측정 서버 (SAM + ZED RGBD)

ZED X Mini RGBD 영상에서 MobileSAM으로 물체를 세그멘테이션한 뒤
두 가지 방법(minAreaRect / PCA)으로 물리 치수(mm)를 측정하고
웹 UI에 실시간으로 표시한다.

노이즈 저감: depth bilateral filter, 마스크 morphology, 이동 평균

실행:
    python 99_dimension_measurement.py
    브라우저에서 http://0.0.0.0:5002 접속
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from camera_utils import CameraManager
from dimension_utils import DimensionEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOBILE_SAM_PATH = os.path.join(BASE_DIR, "sam_models", "mobile_sam.pt")

parser = argparse.ArgumentParser(description="실시간 부품 치수 측정 서버")
parser.add_argument("--no_sam", action="store_true",
                    help="MobileSAM 비활성화 (depth Otsu 폴백)")
parser.add_argument("--port", type=int, default=5002)
parser.add_argument("--interval", type=float, default=1.0,
                    help="측정 주기 (초, 기본: 1.0)")
parser.add_argument("--avg_window", type=int, default=10,
                    help="이동 평균 윈도우 크기 (기본: 10)")
args = parser.parse_args()

app = Flask(__name__)
camera: CameraManager = None  # type: ignore[assignment]
engine: DimensionEngine = None  # type: ignore[assignment]

# ── 전역 상태 ────────────────────────────────────────────

measurement_result: dict = {
    "rect": {}, "pca": {},
    "rect_avg": {}, "pca_avg": {},
    "rect_stats": {}, "pca_stats": {},
    "elapsed_ms": 0, "sample_count": 0, "timestamp": 0,
}
result_lock = threading.Lock()
latest_mask: np.ndarray | None = None
mask_lock = threading.Lock()


# ── 측정 루프 ────────────────────────────────────────────

def measurement_loop():
    global measurement_result, latest_mask
    while True:
        frame = camera.get_frame()
        depth = camera.get_depth()
        if frame is None or engine is None:
            time.sleep(0.1)
            continue

        t0 = time.time()
        result = engine.measure(frame, depth)
        elapsed_ms = (time.time() - t0) * 1000

        with mask_lock:
            latest_mask = result.get("mask")

        with result_lock:
            measurement_result = {
                "rect": result["rect"],
                "pca": result["pca"],
                "rect_avg": result["rect_avg"],
                "pca_avg": result["pca_avg"],
                "rect_stats": result["rect_stats"],
                "pca_stats": result["pca_stats"],
                "elapsed_ms": round(elapsed_ms, 1),
                "sample_count": result["sample_count"],
                "timestamp": time.time(),
            }

        time.sleep(max(0, args.interval - elapsed_ms / 1000))


# ── MJPEG 스트리밍 ──────────────────────────────────────

def generate_mjpeg():
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = cv2.resize(frame, (960, int(h * scale)))
            h, w = frame.shape[:2]

        with mask_lock:
            mask = latest_mask

        if mask is not None:
            mask_resized = cv2.resize(mask, (w, h))
            overlay = np.zeros_like(frame)
            overlay[:, :, 1] = mask_resized
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            contours, _ = cv2.findContours(
                mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)

        with result_lock:
            r = measurement_result.copy()

        rect_data = r.get("rect", {})
        rw = rect_data.get("width_mm", 0)
        rh = rect_data.get("height_mm", 0)
        rar = rect_data.get("ar", 0)

        # OSD 배경
        osd = frame.copy()
        cv2.rectangle(osd, (0, 0), (320, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(osd, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"W:{rw:.0f} H:{rh:.0f} AR:{rar:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.putText(frame,
                    f"{r.get('elapsed_ms', 0):.0f}ms | N={r.get('sample_count', 0)}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)


# ── Flask 라우트 ──────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dimension.html", cache_bust=int(time.time()))


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/measurement")
def api_measurement():
    with result_lock:
        data = measurement_result.copy()
    for key in ("rect", "pca"):
        if "box" in data.get(key, {}):
            data[key] = {k: v for k, v in data[key].items() if k != "box"}
    return jsonify(data)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    engine.reset_history()
    return jsonify({"status": "ok"})


@app.route("/api/camera_info")
def api_camera_info():
    return jsonify({
        "camera_type": camera.camera_type,
        "connected": camera.running,
    })


# ── 엔트리포인트 ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("실시간 부품 치수 측정 서버")
    print("=" * 60)

    engine = DimensionEngine(
        sam_path=MOBILE_SAM_PATH,
        use_sam=not args.no_sam,
        avg_window=args.avg_window,
    )
    sam_status = "MobileSAM" if engine.sam_predictor else "depth 폴백"
    print(f"측정 엔진 준비 완료 (전경분리: {sam_status})")
    print(f"측정 주기: {args.interval}초, 이동 평균 윈도우: {args.avg_window}")

    camera = CameraManager()
    camera.start()
    print(f"카메라: {camera.camera_type}")

    try:
        import pyzed.sl as sl
        cam_info = camera._zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
        engine.update_intrinsics(calib.fx, calib.fy, calib.cx, calib.cy)
        print(f"ZED 캘리브레이션 적용: fx={calib.fx:.1f}, fy={calib.fy:.1f}")
    except Exception:
        print("ZED 캘리브레이션 사용 불가 → 기본값 사용")

    measure_thread = threading.Thread(target=measurement_loop, daemon=True)
    measure_thread.start()
    print("측정 스레드 시작")

    print(f"\n서버 시작: http://0.0.0.0:{args.port}")
    try:
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    finally:
        camera.stop()
