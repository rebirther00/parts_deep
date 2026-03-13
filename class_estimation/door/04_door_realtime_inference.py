"""굴착기 Door 실시간 분류 추론 서버

ZED X Mini 카메라 + TensorRT ONNX 모델을 사용하여
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
import tensorrt as trt
from cuda import cudart
from flask import Flask, Response, jsonify, render_template

from camera_utils import CameraManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "best_door_model_5090.onnx")
ENGINE_PATH = os.path.join(ARTIFACTS_DIR, "best_door_model_5090.trt")
CLASS_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "class_names_door_5090.json")

IMAGE_SIZE = 224
INFERENCE_INTERVAL = 0.5

app = Flask(__name__)
camera: CameraManager = None  # type: ignore[assignment]


class TRTInferenceEngine:
    """TensorRT 기반 ONNX 모델 추론 엔진 (cuda-python 사용)"""

    def __init__(self, onnx_path: str, engine_path: str, class_names: list):
        self.class_names = class_names
        self.logger = trt.Logger(trt.Logger.WARNING)

        cudart.cudaSetDevice(0)

        self.engine = self._load_or_build_engine(onnx_path, engine_path)
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("input", (1, 3, IMAGE_SIZE, IMAGE_SIZE))

        # CUDA 스트림
        _, self.stream = cudart.cudaStreamCreate()

        # GPU 메모리 사전 할당
        input_size = 1 * 3 * IMAGE_SIZE * IMAGE_SIZE * 4  # float32
        output_size = 1 * len(class_names) * 4
        _, self.d_input = cudart.cudaMalloc(input_size)
        _, self.d_output = cudart.cudaMalloc(output_size)
        self.output_buf = np.zeros((1, len(class_names)), dtype=np.float32)

        # ImageNet 정규화 파라미터
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def _load_or_build_engine(self, onnx_path, engine_path):
        if os.path.exists(engine_path):
            print(f"캐시된 TRT 엔진 로드: {engine_path}")
            runtime = trt.Runtime(self.logger)
            with open(engine_path, "rb") as f:
                return runtime.deserialize_cuda_engine(f.read())

        print("TRT 엔진 빌드 중 (최초 1회, 약 25초 소요)...")
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        original_cwd = os.getcwd()
        os.chdir(os.path.dirname(onnx_path))
        with open(os.path.basename(onnx_path), "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  파싱 에러: {parser.get_error(i)}")
                raise RuntimeError("ONNX 파싱 실패")
        os.chdir(original_cwd)

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            (1, 3, IMAGE_SIZE, IMAGE_SIZE),
            (1, 3, IMAGE_SIZE, IMAGE_SIZE),
            (1, 3, IMAGE_SIZE, IMAGE_SIZE),
        )
        config.add_optimization_profile(profile)

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT 엔진 빌드 실패")

        with open(engine_path, "wb") as f:
            f.write(bytes(serialized))
        print(f"TRT 엔진 저장 완료: {engine_path}")

        runtime = trt.Runtime(self.logger)
        return runtime.deserialize_cuda_engine(serialized)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self.mean) / self.std
        return np.expand_dims(img, axis=0).astype(np.float32)

    def infer(self, frame: np.ndarray) -> tuple:
        """(클래스명, 확률, 전체확률리스트) 반환"""
        input_tensor = self.preprocess(frame)

        cudart.cudaMemcpyAsync(
            self.d_input, input_tensor.ctypes.data, input_tensor.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        )

        self.context.set_tensor_address("input", self.d_input)
        self.context.set_tensor_address("output", self.d_output)
        self.context.execute_async_v3(self.stream)

        cudart.cudaMemcpyAsync(
            self.output_buf.ctypes.data, self.d_output, self.output_buf.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
        )
        cudart.cudaStreamSynchronize(self.stream)

        logits = self.output_buf[0]
        exp_vals = np.exp(logits - np.max(logits))
        probs = exp_vals / exp_vals.sum()

        pred_idx = int(np.argmax(probs))
        return (
            self.class_names[pred_idx],
            float(probs[pred_idx]),
            [{"class": self.class_names[i], "prob": float(probs[i])}
             for i in range(len(self.class_names))],
        )

    def cleanup(self):
        cudart.cudaFree(self.d_input)
        cudart.cudaFree(self.d_output)
        cudart.cudaStreamDestroy(self.stream)


# ── 추론 상태 ───────────────────────────────────────────

inference_result = {
    "class": "대기 중",
    "confidence": 0.0,
    "all_probs": [],
    "inference_ms": 0.0,
    "timestamp": 0,
}
result_lock = threading.Lock()
engine: TRTInferenceEngine = None  # type: ignore[assignment]


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

        # 반투명 배경
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

    engine = TRTInferenceEngine(ONNX_PATH, ENGINE_PATH, class_names)
    print("TensorRT 엔진 준비 완료")

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
        engine.cleanup()
        camera.stop()
