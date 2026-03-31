"""굴착기 Door 실시간 분류 추론 서버

ZED X Mini 카메라 + PyTorch 모델을 사용하여
3종 도어(E25/E30/E38)를 실시간 분류하고 결과를 웹에 표시한다.

실행:
    python 04_door_realtime_inference.py
    python 04_door_realtime_inference.py --model artifacts/best_door_cad_model_5090.pth
    브라우저에서 http://0.0.0.0:5001 접속
"""

import argparse
import glob as glob_mod
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
    compute_aux_features, DEFAULT_INTRINSICS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ── 명령줄 인자 ───────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="굴착기 Door 실시간 분류 추론 서버",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--model", type=str,
    default=os.path.join(ARTIFACTS_DIR, "best_door_model_5090.pth"),
    help="모델 파일 경로 (기본: artifacts/best_door_model_5090.pth)",
)
parser.add_argument(
    "--class_names", type=str, default=None,
    help="클래스명 JSON 파일 경로 (미지정 시 모델명에서 자동 추론)",
)
parser.add_argument(
    "--list_models", action="store_true",
    help="사용 가능한 모델 목록 출력 후 종료",
)
parser.add_argument(
    "--no_sam", action="store_true",
    help="MobileSAM 비활성화 (depth 기반 전경 분리로 폴백)",
)
args = parser.parse_args()


def _infer_class_names_path(model_path):
    """모델 파일명에서 대응하는 class_names JSON 경로를 추론한다.

    best_door_cad_model_5090.pth → class_names_door_cad_5090.json
    """
    fname = os.path.basename(model_path)
    cn_fname = fname.replace("best_", "class_names_").replace("_model", "")
    cn_fname = os.path.splitext(cn_fname)[0] + ".json"
    return os.path.join(os.path.dirname(model_path), cn_fname)


def _list_available_models():
    models = sorted(glob_mod.glob(os.path.join(ARTIFACTS_DIR, "*.pth")))
    if not models:
        print("사용 가능한 모델이 없습니다.")
        return
    print(f"사용 가능한 모델 ({len(models)}개):")
    for m in models:
        cn = _infer_class_names_path(m)
        cn_status = "✅" if os.path.exists(cn) else "⚠️  클래스명 파일 없음"
        print(f"  {os.path.basename(m)}  [{cn_status}]")


if args.list_models:
    _list_available_models()
    raise SystemExit(0)

MODEL_PATH = args.model
CLASS_NAMES_PATH = args.class_names or _infer_class_names_path(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    _list_available_models()
    raise SystemExit(1)

if not os.path.exists(CLASS_NAMES_PATH):
    print(f"❌ 클래스명 파일을 찾을 수 없습니다: {CLASS_NAMES_PATH}")
    print(f"   --class_names 옵션으로 직접 지정해주세요.")
    raise SystemExit(1)

IMAGE_SIZE = 448
INFERENCE_INTERVAL = 0.5

app = Flask(__name__)
camera: CameraManager = None  # type: ignore[assignment]


MOBILE_SAM_PATH = os.path.join(BASE_DIR, "sam_models", "mobile_sam.pt")


def _load_mobile_sam(device="cpu"):
    """MobileSAM 모델과 predictor를 로드한다."""
    from mobile_sam import sam_model_registry, SamPredictor

    model = sam_model_registry["vit_t"](checkpoint=MOBILE_SAM_PATH)
    model.to(device)
    model.eval()
    return SamPredictor(model)


class PyTorchInferenceEngine:
    """PyTorch RGBD + 보조 피처 추론 엔진 (MobileSAM 통합)"""

    def __init__(self, model_path: str, class_names: list,
                 use_sam: bool = True):
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = (self.device.type == "cuda")

        self.model = RGBDAuxResNet18(len(class_names), pretrained=False)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()
        self.model.eval()

        self.transform = RGBDTransform(IMAGE_SIZE, is_train=False)

        # CUDA 워밍업 (첫 추론 지연 방지)
        if self.device.type == "cuda":
            dummy_img = torch.randn(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
                                    device=self.device)
            dummy_aux = torch.randn(1, 3, device=self.device)
            if self.use_fp16:
                dummy_img, dummy_aux = dummy_img.half(), dummy_aux.half()
            with torch.no_grad():
                self.model(dummy_img, dummy_aux)
            torch.cuda.synchronize()
            print("CUDA 워밍업 완료")

        self.sam_predictor = None
        if use_sam and os.path.exists(MOBILE_SAM_PATH):
            try:
                self.sam_predictor = _load_mobile_sam(str(self.device))
                print(f"MobileSAM 로드 완료 ({self.device})")
            except Exception as e:
                print(f"MobileSAM 로드 실패, depth 폴백: {e}")

    def _generate_sam_mask(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """MobileSAM으로 전경 마스크를 생성한다."""
        if self.sam_predictor is None:
            return None

        self.sam_predictor.set_image(frame_rgb)
        h, w = frame_rgb.shape[:2]
        center = np.array([[w // 2, h // 2]], dtype=np.float32)
        label = np.array([1], dtype=np.int32)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=center,
            point_labels=label,
            multimask_output=True,
        )
        best_idx = max(range(len(masks)), key=lambda i: masks[i].sum())
        return masks[best_idx].astype(np.uint8) * 255

    def infer(self, frame: np.ndarray, depth: np.ndarray = None) -> tuple:
        """(클래스명, 확률, 전체확률리스트) 반환"""
        t_start = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(frame_rgb)

        depth_raw_mm = None
        if depth is not None:
            depth_raw_mm = depth.astype(np.float32)
            depth_norm = np.clip(depth_raw_mm / MAX_DEPTH_MM, 0.0, 1.0)
        else:
            h, w = frame.shape[:2]
            depth_norm = np.zeros((h, w), dtype=np.float32)

        inp = self.transform(pil_img, depth_norm).unsqueeze(0).to(self.device)
        if self.use_fp16:
            inp = inp.half()
        t_preprocess = time.time()

        fg_mask = self._generate_sam_mask(frame_rgb)
        t_sam = time.time()

        aux = compute_aux_features(
            depth_raw_mm if depth_raw_mm is not None
            else np.zeros(frame.shape[:2], dtype=np.float32),
            DEFAULT_INTRINSICS,
            fg_mask=fg_mask,
        )
        aux_t = torch.tensor([aux], dtype=torch.float32).to(self.device)
        if self.use_fp16:
            aux_t = aux_t.half()

        with torch.no_grad():
            output = self.model(inp, aux_t)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_model = time.time()

        probs = torch.softmax(output.float(), dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

        self._log_timing(t_start, t_preprocess, t_sam, t_model)

        return (
            self.class_names[pred_idx],
            float(probs[pred_idx]),
            [{"class": self.class_names[i], "prob": float(probs[i])}
             for i in range(len(self.class_names))],
        )

    _log_count = 0

    def _log_timing(self, t_start, t_preprocess, t_sam, t_model):
        """10회마다 단계별 소요 시간을 로그로 출력한다."""
        self._log_count += 1
        if self._log_count % 10 != 1:
            return
        pre_ms = (t_preprocess - t_start) * 1000
        sam_ms = (t_sam - t_preprocess) * 1000
        model_ms = (t_model - t_sam) * 1000
        total_ms = (t_model - t_start) * 1000
        print(f"[추론 #{self._log_count}] "
              f"전처리: {pre_ms:.0f}ms | SAM: {sam_ms:.0f}ms | "
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
    device_str = str(engine.device) if engine else "N/A"
    fp16 = engine.use_fp16 if engine else False
    return jsonify({
        "camera_type": camera.camera_type,
        "connected": camera.running,
        "engine": f"PyTorch {'FP16' if fp16 else 'FP32'} ({device_str})",
    })


# ── 엔트리포인트 ────────────────────────────────────────

if __name__ == "__main__":
    print(f"모델: {os.path.basename(MODEL_PATH)}")
    print(f"클래스명: {os.path.basename(CLASS_NAMES_PATH)}")

    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    print(f"클래스: {class_names}")

    engine = PyTorchInferenceEngine(
        MODEL_PATH, class_names, use_sam=not args.no_sam)
    sam_status = "MobileSAM" if engine.sam_predictor else "depth 폴백"
    print(f"RGBD PyTorch 엔진 준비 완료 "
          f"(디바이스: {engine.device}, 입력: {IN_CHANNELS}ch, "
          f"전경분리: {sam_status})")

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
