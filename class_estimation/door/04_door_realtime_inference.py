"""굴착기 Door 실시간 분류 추론 서버

ZED X Mini 카메라 + PyTorch 모델을 사용하여
3종 도어(E25/E30/E38)를 실시간 분류하고 결과를 웹에 표시한다.

실행:
    python 04_door_realtime_inference.py
    브라우저에서 http://0.0.0.0:5001 접속
"""

import json
import os
import threading
import time

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template
from PIL import Image as PILImage

from camera_utils import CameraManager
from depth_utils import (
    RGBDAuxResNet18, RGBDTransform, MAX_DEPTH_MM, IN_CHANNELS,
    compute_aux_features, ISAAC_SIM_INTRINSICS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_door_model_5090.pth")
CLASS_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "class_names_door_5090.json")

IMAGE_SIZE = 448
INFERENCE_INTERVAL = 0.5

app = Flask(__name__)
camera: CameraManager = None  # type: ignore[assignment]


class PyTorchInferenceEngine:
    """PyTorch RGBD + 보조 피처 추론 엔진"""

    def __init__(self, model_path: str, class_names: list):
        self.class_names = class_names
        self.device = torch.device("cpu")

        self.model = RGBDAuxResNet18(len(class_names), pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = RGBDTransform(IMAGE_SIZE, is_train=False)

    def infer(self, frame: np.ndarray, depth: np.ndarray = None) -> tuple:
        """(클래스명, 확률, 전체확률리스트) 반환"""
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        depth_raw_mm = None
        if depth is not None:
            depth_raw_mm = depth.astype(np.float32)
            depth_norm = np.clip(depth_raw_mm / MAX_DEPTH_MM, 0.0, 1.0)
        else:
            h, w = frame.shape[:2]
            depth_norm = np.zeros((h, w), dtype=np.float32)

        inp = self.transform(pil_img, depth_norm).unsqueeze(0).to(self.device)

        aux = compute_aux_features(
            depth_raw_mm if depth_raw_mm is not None
            else np.zeros(frame.shape[:2], dtype=np.float32),
            ISAAC_SIM_INTRINSICS,
        )
        aux_t = torch.tensor([aux], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(inp, aux_t)

        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return (
            self.class_names[pred_idx],
            float(probs[pred_idx]),
            [{"class": self.class_names[i], "prob": float(probs[i])}
             for i in range(len(self.class_names))],
        )


# ── 추론 상태 ───────────────────────────────────────────

inference_result = {
    "class": "대기 중",
    "confidence": 0.0,
    "all_probs": [],
    "inference_ms": 0.0,
    "timestamp": 0,
}
result_lock = threading.Lock()
engine: PyTorchInferenceEngine = None  # type: ignore[assignment]


def inference_loop():
    global inference_result
    while True:
        frame = camera.get_frame()
        if frame is None or engine is None:
            time.sleep(0.1)
            continue

        depth = camera.get_depth()

        t0 = time.time()
        pred_class, pred_conf, all_probs = engine.infer(frame, depth)
        elapsed_ms = (time.time() - t0) * 1000

        with result_lock:
            inference_result = {
                "class": pred_class,
                "confidence": round(pred_conf * 100, 1),
                "all_probs": all_probs,
                "inference_ms": round(elapsed_ms, 1),
                "timestamp": time.time(),
            }

        time.sleep(max(0, INFERENCE_INTERVAL - elapsed_ms / 1000))


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

        with result_lock:
            r = inference_result.copy()

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (420, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        color = (0, 255, 0) if r["confidence"] > 70 else (0, 200, 255)
        cv2.putText(frame, r["class"], (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, f"{r['confidence']:.1f}%  |  {r['inference_ms']:.0f}ms",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        time.sleep(0.03)


# ── Flask 라우트 ────────────────────────────────────────

@app.route("/")
def index():
    return render_template("inference.html", cache_bust=int(time.time()))


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/inference_result")
def api_inference_result():
    with result_lock:
        return jsonify(inference_result)


@app.route("/api/camera_info")
def api_camera_info():
    return jsonify({
        "camera_type": camera.camera_type,
        "connected": camera.running,
    })


# ── 엔트리포인트 ────────────────────────────────────────

if __name__ == "__main__":
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    print(f"클래스: {class_names}")

    engine = PyTorchInferenceEngine(MODEL_PATH, class_names)
    print(f"RGBD PyTorch 엔진 준비 완료 (디바이스: {engine.device}, 입력: {IN_CHANNELS}ch)")

    camera = CameraManager()
    camera.start()
    print(f"카메라: {camera.camera_type}")

    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    infer_thread.start()
    print("추론 스레드 시작")

    print("서버 시작: http://0.0.0.0:5001")
    try:
        app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
    finally:
        camera.stop()
