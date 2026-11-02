"""ZED X mini 상시 구동 카메라 (현장 서비스 공용).

06_factory_capture.py 의 FactoryCamera 를 추론 서비스(07_factory_infer.py)에서도 쓰도록 분리했다.
06 은 수집 서비스 이력 보존용으로 그대로 두고(2026-11-02 중지), 신규 서비스는 이 모듈을 쓴다.

- grab 연속 실패 시 자동 재연결, open 반복 실패 시 프로세스 종료(systemd 재시작) — 06 과 동일
- get_pair() 가 (rgb, depth, ts, seq) 를 돌려줘 같은 프레임을 두 번 처리하지 않게 한다
- calib: 좌안 rectified intrinsics + 시리얼 + 화각 (hole_classifier 의 S_PIXEL/K_CAMERA 키 = serial)
"""

import logging
import os
import threading
import time

import cv2
import numpy as np

try:
    import pyzed.sl as sl
    HAS_ZED = True
except ImportError:
    HAS_ZED = False

log = logging.getLogger("factory_camera")

CAMERA_FPS = 15
RECONNECT_AFTER_FAILS = 60      # 연속 grab 실패 횟수 → 카메라 재연결
OPEN_FAILS_BEFORE_EXIT = 6      # open 반복 실패 → os._exit(3) (CUDA 801 은 프로세스 안에서 복구 불가)


class FactoryCamera:
    """ZED X mini 상시 구동. 최신 프레임만 보관한다."""

    def __init__(self, fps=CAMERA_FPS):
        self.fps = fps
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_ts = 0.0
        self.seq = 0
        self.camera_type = "none"
        self.calib = None
        self.ok = False
        self.last_error = ""
        self.running = False
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_pair(self):
        """(rgb BGR, depth float32 mm, ts, seq). 프레임 없으면 (None, None, 0, seq)."""
        with self._lock:
            if self.latest_rgb is None:
                return None, None, 0.0, self.seq
            return self.latest_rgb.copy(), (None if self.latest_depth is None else self.latest_depth.copy()), \
                self.latest_ts, self.seq

    def get_rgb(self):
        with self._lock:
            return None if self.latest_rgb is None else self.latest_rgb.copy()

    # -- internal --

    def _open(self):
        if not HAS_ZED:
            raise RuntimeError("pyzed 미설치")
        self._zed = sl.Camera()
        last_err = None
        for res in (sl.RESOLUTION.HD1200, sl.RESOLUTION.HD1080, sl.RESOLUTION.AUTO):
            params = sl.InitParameters()
            params.camera_resolution = res
            params.camera_fps = self.fps
            params.depth_mode = sl.DEPTH_MODE.NEURAL
            params.coordinate_units = sl.UNIT.MILLIMETER
            err = self._zed.open(params)
            if err == sl.ERROR_CODE.SUCCESS:
                info = self._zed.get_camera_information()
                cam_res = info.camera_configuration.resolution
                self.camera_type = (f"ZED {str(info.camera_model).split('.')[-1]} "
                                    f"{cam_res.width}x{cam_res.height}")
                self._img, self._dep = sl.Mat(), sl.Mat()
                self._runtime = sl.RuntimeParameters()
                self.calib = self._read_calib(info)
                log.info("카메라 연결: %s intrinsics=%s", self.camera_type, self.calib)
                return
            last_err = err
        raise RuntimeError(f"ZED open 실패: {last_err}")

    @staticmethod
    def _read_calib(info):
        """좌안(rectified, 저장 영상과 동일 기하) intrinsics + 시리얼 + 화각. 실패해도 서비스는 계속(None)."""
        try:
            cc = getattr(info, "camera_configuration", info)
            cam = cc.calibration_parameters.left_cam
            res = cc.resolution
            return {
                "serial": int(info.serial_number),
                "width": int(res.width), "height": int(res.height),
                "fx": float(cam.fx), "fy": float(cam.fy),
                "cx": float(cam.cx), "cy": float(cam.cy),
                "h_fov": float(cam.h_fov), "v_fov": float(cam.v_fov),
                "rectified": True,
            }
        except Exception as e:                      # noqa: BLE001
            log.warning("intrinsics 읽기 실패: %s", e)
            return None

    def _close(self):
        try:
            self._zed.close()
        except Exception:
            pass

    def _loop(self):
        fails = 0
        open_fails = 0
        while self.running:
            if not self.ok:
                try:
                    self._open()
                    self.ok = True
                    self.last_error = ""
                    fails = 0
                    open_fails = 0
                except Exception as e:
                    self.last_error = str(e)
                    open_fails += 1
                    log.error("카메라 연결 실패(%d/%d), 10초 후 재시도: %s",
                              open_fails, OPEN_FAILS_BEFORE_EXIT, e)
                    if open_fails >= OPEN_FAILS_BEFORE_EXIT:
                        log.error("카메라 연결 반복 실패 — 프로세스 재시작")
                        os._exit(3)
                    time.sleep(10)
                    continue

            if self._zed.grab(self._runtime) == sl.ERROR_CODE.SUCCESS:
                fails = 0
                self._zed.retrieve_image(self._img, sl.VIEW.LEFT)
                rgb = cv2.cvtColor(self._img.get_data(), cv2.COLOR_BGRA2BGR)
                self._zed.retrieve_measure(self._dep, sl.MEASURE.DEPTH)
                depth = np.nan_to_num(self._dep.get_data().copy(), nan=0.0, posinf=0.0, neginf=0.0
                                      ).astype(np.float32)
                with self._lock:
                    self.latest_rgb = rgb
                    self.latest_depth = depth
                    self.latest_ts = time.time()
                    self.seq += 1
            else:
                fails += 1
                if fails >= RECONNECT_AFTER_FAILS:
                    log.error("grab %d회 연속 실패 → 카메라 재연결", fails)
                    self._close()
                    self.ok = False
                time.sleep(0.05)
            time.sleep(max(0.001, 1.0 / self.fps - 0.005))
        self._close()


def require_cuda_ready(exit_code=3):
    """부팅 직후 CUDA 미초기화(801) 상태면 즉시 종료해 systemd 가 새 프로세스로 재시도하게 한다.
    (801 을 한 번 맞은 프로세스는 이후 ZED open 이 영원히 실패 — 06 과 동일)"""
    import ctypes
    import sys
    try:
        rt = ctypes.CDLL("libcudart.so")
        n = ctypes.c_int(-1)
        r1 = rt.cudaGetDeviceCount(ctypes.byref(n))
        ptr = ctypes.c_void_p()
        r2 = rt.cudaMalloc(ctypes.byref(ptr), 1024 * 1024)
        if r2 == 0:
            rt.cudaFree(ptr)
        log.info("CUDA probe: getDeviceCount=%d n=%d cudaMalloc=%d", r1, n.value, r2)
        if r1 != 0 or n.value < 1 or r2 != 0:
            log.error("CUDA 미준비 — 프로세스 종료 후 systemd 재시작 대기")
            sys.exit(exit_code)
    except OSError as e:
        log.error("libcudart 로드 실패 — 프로세스 종료: %r", e)
        sys.exit(exit_code)
