"""RGBE NoAux 실시간 분류 추론 서버

ZED 카메라(X Mini / 2i 등) + RGBE NoAux PyTorch 모델을 사용하여
8종 도어를 실시간 분류하고 결과를 웹에 표시한다.

RGBE = RGB + Canny Edge (4채널). Depth/Intrinsics/보조피처 불필요.
→ 카메라 기종 교체 없이 바로 사용 가능.

실행:
    python realtime_inference.py
    python realtime_inference.py --model artifacts/rgbe_noaux_448_seed42/model.pth
    브라우저에서 http://0.0.0.0:5001 접속
"""

import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, Response, jsonify, render_template
from PIL import Image as PILImage
from torchvision import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOOR_DIR = os.path.join(os.path.dirname(BASE_DIR), "door")
sys.path.insert(0, DOOR_DIR)

from camera_utils import CameraManager
from rgbe_utils import RGBETransform, RGBE_IN_CHANNELS, CANNY_LOW, CANNY_HIGH

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DEFAULT_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "rgbe_noaux_448_seed42")

IMAGE_SIZE = 448
INFERENCE_INTERVAL = 0.0


# ── NoAuxResNet18 (train_paper.py와 동일 구조) ──────────

class NoAuxResNet18(nn.Module):
    """Aux MLP 없이 이미지만 사용하는 분류 모델.

    forward()는 aux_features를 인자로 받되 무시하여 API 호환성 유지.
    """

    def __init__(self, num_classes, in_channels=RGBE_IN_CHANNELS,
                 pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(in_channels, 64,
                                 kernel_size=7, stride=2, padding=3,
                                 bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                for c in range(3, in_channels):
                    new_conv.weight[:, c:c + 1] = old_conv.weight.mean(
                        dim=1, keepdim=True)
            backbone.conv1 = new_conv

        self.backbone_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, aux_features=None):
        img_feat = self.backbone(images)
        return self.classifier(img_feat)


# ── 명령줄 인자 ───────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="RGBE NoAux 실시간 분류 추론 서버",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--model", type=str,
    default=os.path.join(DEFAULT_MODEL_DIR, "model.pth"),
    help="모델 파일 경로",
)
parser.add_argument(
    "--class_names", type=str, default=None,
    help="클래스명 JSON 파일 경로 (미지정 시 split_info.json에서 추출)",
)
parser.add_argument(
    "--port", type=int, default=5001,
    help="웹 서버 포트 (기본: 5001)",
)
args = parser.parse_args()


def _load_class_names(model_path, explicit_path=None):
    """클래스명을 로드한다.

    1) --class_names로 직접 지정된 경우
    2) 모델과 같은 폴더의 split_info.json에서 추출
    """
    if explicit_path and os.path.exists(explicit_path):
        with open(explicit_path, encoding="utf-8") as f:
            return json.load(f)

    split_info_path = os.path.join(os.path.dirname(model_path),
                                   "split_info.json")
    if os.path.exists(split_info_path):
        with open(split_info_path, encoding="utf-8") as f:
            info = json.load(f)
        if "class_names" in info:
            return info["class_names"]

    raise FileNotFoundError(
        f"클래스명을 찾을 수 없습니다. --class_names 옵션을 지정하세요.\n"
        f"  확인 경로: {split_info_path}"
    )


if not os.path.exists(args.model):
    print(f"모델 파일을 찾을 수 없습니다: {args.model}")
    raise SystemExit(1)

CLASS_NAMES = _load_class_names(args.model, args.class_names)

app = Flask(__name__)
camera: CameraManager = None  # type: ignore[assignment]


# ── 추론 엔진 ──────────────────────────────────────────────

class RGBEInferenceEngine:
    """RGBE NoAux PyTorch 추론 엔진

    RGB 프레임에서 Canny Edge를 실시간 계산하여 4채널 입력을 구성한다.
    Depth, 보조 피처, MobileSAM 등 일체 불필요.
    """

    def __init__(self, model_path: str, class_names: list):
        self.class_names = class_names
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = NoAuxResNet18(
            len(class_names), in_channels=RGBE_IN_CHANNELS, pretrained=False)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
            print("TF32 + cuDNN benchmark 활성화")

        self.transform = RGBETransform(IMAGE_SIZE, is_train=False)

        if self.device.type == "cuda":
            print("CUDA 워밍업 중...")
            dummy = torch.randn(1, RGBE_IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
                                device=self.device)
            with torch.no_grad():
                self.model(dummy)
            torch.cuda.synchronize()
            print("CUDA 워밍업 완료")

    def infer(self, frame: np.ndarray) -> tuple:
        """(클래스명, 확률, 전체확률리스트) 반환"""
        t_start = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(frame_rgb)

        inp = self.transform(pil_img).unsqueeze(0).to(self.device)
        t_preprocess = time.time()

        with torch.no_grad():
            output = self.model(inp)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_model = time.time()

        probs = torch.softmax(output.float(), dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

        self._log_timing(t_start, t_preprocess, t_model)

        return (
            self.class_names[pred_idx],
            float(probs[pred_idx]),
            [{"class": self.class_names[i], "prob": float(probs[i])}
             for i in range(len(self.class_names))],
        )

    _log_count = 0

    def _log_timing(self, t_start, t_preprocess, t_model):
        self._log_count += 1
        if self._log_count % 10 != 1:
            return
        pre_ms = (t_preprocess - t_start) * 1000
        model_ms = (t_model - t_preprocess) * 1000
        total_ms = (t_model - t_start) * 1000
        print(f"[추론 #{self._log_count}] "
              f"전처리(+Edge): {pre_ms:.0f}ms | "
              f"모델: {model_ms:.0f}ms | 합계: {total_ms:.0f}ms")


# ── 추론 상태 ───────────────────────────────────────────

inference_result = {
    "class": "대기 중",
    "confidence": 0.0,
    "all_probs": [],
    "inference_ms": 0.0,
    "timestamp": 0,
}
result_lock = threading.Lock()
engine: RGBEInferenceEngine = None  # type: ignore[assignment]


def inference_loop():
    global inference_result
    while True:
        frame = camera.get_frame()
        if frame is None or engine is None:
            time.sleep(0.1)
            continue

        t0 = time.time()
        pred_class, pred_conf, all_probs = engine.infer(frame)
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
        cv2.putText(
            frame,
            f"{r['confidence']:.1f}%  |  {r['inference_ms']:.0f}ms",
            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
        )

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
        "engine": f"RGBE NoAux PyTorch FP32 ({engine.device})",
    })


# ── 엔트리포인트 ────────────────────────────────────────

if __name__ == "__main__":
    print(f"모델: {args.model}")
    print(f"클래스: {CLASS_NAMES} ({len(CLASS_NAMES)}종)")
    print(f"입력: RGBE {RGBE_IN_CHANNELS}ch (RGB + Canny Edge), "
          f"해상도: {IMAGE_SIZE}x{IMAGE_SIZE}")

    engine = RGBEInferenceEngine(args.model, CLASS_NAMES)
    print(f"RGBE NoAux 엔진 준비 완료 (디바이스: {engine.device})")

    camera = CameraManager()
    camera.start()
    print(f"카메라: {camera.camera_type}")

    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    infer_thread.start()
    print("추론 스레드 시작")

    print(f"서버 시작: http://0.0.0.0:{args.port}")
    try:
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    finally:
        camera.stop()
