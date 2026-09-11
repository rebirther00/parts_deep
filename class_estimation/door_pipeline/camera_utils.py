"""ZED X Mini 카메라 관리 모듈 (OpenCV 폴백 지원, RGBD)

RGB + Depth 프레임을 동시에 캡처·저장한다.
ZED 카메라가 없으면 OpenCV 폴백으로 RGB만 반환(Depth=None).
"""

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

    # 06_factory_capture.py(현장 수집)와 동일한 해상도 우선순위.
    # 추론 카메라는 수집 카메라와 같은 설정이어야 프레임 기하(화각·세로
    # 해상도 → hole_classifier K_DEPTH 키)가 일치한다.
    DEFAULT_RESOLUTIONS = ("HD1200", "HD1080", "AUTO")

    def __init__(self, fps=30, resolutions=None):
        self.fps = fps
        self.resolutions = tuple(resolutions or self.DEFAULT_RESOLUTIONS)
        self.resolution = None          # (width, height) — open 후 채워짐
        self.calib = None               # 좌안 rectified intrinsics (ZED만)
        self.latest_rgb = None
        self.latest_depth = None
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
        last_err = None
        for name in self.resolutions:
            params = sl.InitParameters()
            params.camera_resolution = getattr(sl.RESOLUTION, name)
            params.camera_fps = self.fps
            params.depth_mode = sl.DEPTH_MODE.NEURAL
            params.coordinate_units = sl.UNIT.MILLIMETER
            err = self._zed.open(params)
            if err == sl.ERROR_CODE.SUCCESS:
                break
            last_err = err
        else:
            raise RuntimeError(f"ZED 카메라 초기화 실패: {last_err}")
        self._zed_image = sl.Mat()
        self._zed_depth = sl.Mat()
        self._zed_runtime = sl.RuntimeParameters()
        info = self._zed.get_camera_information()
        cc = info.camera_configuration
        self.resolution = (int(cc.resolution.width), int(cc.resolution.height))
        self.camera_type = (
            f"ZED {str(info.camera_model).split('.')[-1]} "
            f"{self.resolution[0]}x{self.resolution[1]}@{int(cc.fps)}fps "
            f"SN{info.serial_number}"
        )
        self.calib = self._read_calib(info)

    @staticmethod
    def _read_calib(info):
        """좌안(rectified, retrieve_image LEFT 와 동일 기하) intrinsics.

        렌즈(광각/협각)별로 fx 가 크게 다르므로 depth 역투영에는 반드시
        이 값을 써야 한다. 실패해도 캡처에는 영향 없음(None)."""
        try:
            cc = info.camera_configuration
            cam = cc.calibration_parameters.left_cam
            return {
                "serial": int(info.serial_number),
                "width": int(cc.resolution.width),
                "height": int(cc.resolution.height),
                "fx": float(cam.fx), "fy": float(cam.fy),
                "cx": float(cam.cx), "cy": float(cam.cy),
                "h_fov": float(cam.h_fov), "v_fov": float(cam.v_fov),
            }
        except Exception as e:                      # noqa: BLE001
            print(f"intrinsics 읽기 실패(캡처 영향 없음): {e}")
            return None

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
        """RGB 프레임 반환 (하위 호환성 유지)."""
        with self._lock:
            return self.latest_rgb.copy() if self.latest_rgb is not None else None

    def get_depth(self):
        """Depth 프레임 반환 (float32, mm). ZED 미사용 시 None."""
        with self._lock:
            if self.latest_depth is not None:
                return self.latest_depth.copy()
            return None

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
        with self._lock:
            rgb = self.latest_rgb.copy() if self.latest_rgb is not None else None
            depth = self.latest_depth.copy() if self.latest_depth is not None else None
        if rgb is None:
            return None
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._sync_counter()
        score = compute_blur_score(rgb)
        return self._save_temp(rgb, depth, score)

    def reset_counter(self):
        self._extracted_count = 0

    # -- internal --

    def _grab_frame(self):
        """(rgb, depth) 튜플 반환. 실패 시 (None, None)."""
        if HAS_ZED:
            if self._zed.grab(self._zed_runtime) == sl.ERROR_CODE.SUCCESS:
                self._zed.retrieve_image(self._zed_image, sl.VIEW.LEFT)
                rgb = cv2.cvtColor(
                    self._zed_image.get_data(), cv2.COLOR_BGRA2BGR
                )
                self._zed.retrieve_measure(self._zed_depth, sl.MEASURE.DEPTH)
                depth_raw = self._zed_depth.get_data().copy()
                depth_raw = np.nan_to_num(
                    depth_raw, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
                return rgb, depth_raw
            return None, None
        ret, frame = self._cap.read()
        return (frame, None) if ret else (None, None)

    def _capture_loop(self):
        while self.running:
            rgb, depth = self._grab_frame()
            if rgb is None:
                time.sleep(0.01)
                continue

            with self._lock:
                self.latest_rgb = rgb.copy()
                self.latest_depth = depth.copy() if depth is not None else None

            if self.recording and self._temp_dir:
                self._frame_count += 1
                if self._frame_count % self._record_interval == 0:
                    score = compute_blur_score(rgb)
                    if score >= self._blur_threshold:
                        self._save_temp(rgb, depth, score)

            time.sleep(max(0.001, 1.0 / self.fps - 0.005))

    def _sync_counter(self):
        """temp 폴더의 기존 파일 기반으로 카운터를 동기화한다."""
        existing = list(self._temp_dir.glob("frame_*.png"))
        # frame_depth_ 패턴은 제외
        existing = [f for f in existing if "depth" not in f.stem]
        if existing:
            indices = [int(f.stem.split("_")[1]) for f in existing]
            self._extracted_count = max(indices) + 1

    def _save_temp(self, rgb, depth, blur_score):
        idx = self._extracted_count
        rgb_name = f"frame_{idx:04d}.png"
        cv2.imwrite(str(self._temp_dir / rgb_name), rgb)

        if depth is not None:
            depth_name = f"frame_depth_{idx:04d}.png"
            depth_uint16 = np.clip(depth, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(self._temp_dir / depth_name), depth_uint16)

        self._extracted_count += 1
        return {"filename": rgb_name, "blur_score": round(blur_score, 1)}


def compute_blur_score(image):
    """Laplacian variance 기반 블러 점수. 높을수록 선명하다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
