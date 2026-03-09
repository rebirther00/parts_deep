"""ZED X Mini 카메라 관리 모듈 (OpenCV 폴백 지원)"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyzed.sl as sl
    HAS_ZED = True
except ImportError:
    HAS_ZED = False


class CameraManager:
    """카메라 캡처, 녹화, 프레임 추출, 블러 필터링을 관리한다."""

    def __init__(self, fps=30):
        self.fps = fps
        self.latest_frame = None
        self.camera_type = "unknown"
        self.running = False
        self.recording = False
        self._lock = threading.Lock()
        self._frame_count = 0
        self._extracted_count = 0
        self._record_interval = 5
        self._blur_threshold = 100
        self._temp_dir = None
        self._thread = None

        if HAS_ZED:
            self._init_zed()
        else:
            self._init_opencv()

    def _init_zed(self):
        self._zed = sl.Camera()
        params = sl.InitParameters()
        params.camera_resolution = sl.RESOLUTION.HD1080
        params.camera_fps = self.fps
        err = self._zed.open(params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED 카메라 초기화 실패: {err}")
        self._zed_image = sl.Mat()
        self._zed_runtime = sl.RuntimeParameters()
        self.camera_type = "ZED X Mini"

    def _init_opencv(self):
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera_type = "OpenCV (Fallback)"

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        if HAS_ZED:
            self._zed.close()
        else:
            self._cap.release()

    def get_frame(self):
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def start_recording(self, temp_dir, interval=5, blur_threshold=100):
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._record_interval = interval
        self._blur_threshold = blur_threshold
        self._frame_count = 0
        self._sync_counter()
        self.recording = True

    def stop_recording(self):
        self.recording = False
        return {
            "extracted_count": self._extracted_count,
            "frame_count": self._frame_count,
        }

    def snapshot(self, temp_dir):
        frame = self.get_frame()
        if frame is None:
            return None
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._sync_counter()
        score = compute_blur_score(frame)
        return self._save_temp(frame, score)

    def reset_counter(self):
        self._extracted_count = 0

    # -- internal --

    def _grab_frame(self):
        if HAS_ZED:
            if self._zed.grab(self._zed_runtime) == sl.ERROR_CODE.SUCCESS:
                self._zed.retrieve_image(self._zed_image, sl.VIEW.LEFT)
                return cv2.cvtColor(
                    self._zed_image.get_data(), cv2.COLOR_BGRA2BGR
                )
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def _capture_loop(self):
        while self.running:
            frame = self._grab_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            with self._lock:
                self.latest_frame = frame.copy()

            if self.recording and self._temp_dir:
                self._frame_count += 1
                if self._frame_count % self._record_interval == 0:
                    score = compute_blur_score(frame)
                    if score >= self._blur_threshold:
                        self._save_temp(frame, score)

            time.sleep(max(0.001, 1.0 / self.fps - 0.005))

    def _sync_counter(self):
        """temp 폴더의 기존 파일 기반으로 카운터를 동기화한다."""
        existing = list(self._temp_dir.glob("frame_*.png"))
        if existing:
            indices = [int(f.stem.split("_")[1]) for f in existing]
            self._extracted_count = max(indices) + 1

    def _save_temp(self, frame, blur_score):
        filename = f"frame_{self._extracted_count:04d}.png"
        cv2.imwrite(str(self._temp_dir / filename), frame)
        self._extracted_count += 1
        return {"filename": filename, "blur_score": round(blur_score, 1)}


def compute_blur_score(image):
    """Laplacian variance 기반 블러 점수. 높을수록 선명하다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
