"""경주 공장 도어 이미지 수집 서비스 (키오스크 + 원격 모니터링 겸용)

운영 흐름:
    1. 작업자가 터치모니터에서 도어 종류 버튼 선택 (기본값 Unknown)
    2. [취득 시작] → 워밍업 후 2초에 1장씩 rgb/depth 쌍 저장
    3. 300초 경과 시 자동 종료 (보통은 [취득 종료] 버튼으로 먼저 종료)
    4. 종료 시 선택 클래스는 Unknown으로 자동 복귀 (라벨링 실수 방지)

원격(세종)에서는 tailscale IP로 같은 페이지를 열어 상태 확인.

실행:
    ~/workspace/zed_env/bin/python 06_factory_capture.py
    (systemd 등록: install_factory.sh 참고)

환경변수 오버라이드 (테스트용):
    DOOR_SESSION_DURATION / DOOR_CAPTURE_INTERVAL / DOOR_WARMUP / DOOR_PORT
"""

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

try:
    import pyzed.sl as sl
    HAS_ZED = True
except ImportError:
    HAS_ZED = False

BASE_DIR = Path(__file__).parent
OUTPUT_ROOT = BASE_DIR / "datasets_factory_collect"
LOG_DIR = BASE_DIR / "logs"

CLASSES = [
    "E25_door_LH_FRT", "E25_door_LH_RR", "E25_door_RH",
    "E30_door_LH_FRT", "E30_door_LH_RR",
    "E30_E38_door_RH",
    "E38_door_LH_FRT", "E38_door_LH_RR",
]
UNKNOWN = "Unknown"
ALL_CLASSES = CLASSES + [UNKNOWN]

SESSION_DURATION = float(os.environ.get("DOOR_SESSION_DURATION", 300))
CAPTURE_INTERVAL = float(os.environ.get("DOOR_CAPTURE_INTERVAL", 2.0))
WARMUP_SEC = float(os.environ.get("DOOR_WARMUP", 2.0))
PORT = int(os.environ.get("DOOR_PORT", 5000))
CAMERA_FPS = 15
RECONNECT_AFTER_FAILS = 60      # 연속 grab 실패 횟수 → 카메라 재연결

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("factory_capture")


# ── 카메라 (자동 재연결 지원) ───────────────────────────

class FactoryCamera:
    """ZED X mini 상시 구동 카메라. grab 연속 실패 시 자동 재연결."""

    def __init__(self):
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_type = "none"
        self.ok = False
        self.last_error = ""
        self.running = False
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_pair(self):
        with self._lock:
            rgb = self.latest_rgb.copy() if self.latest_rgb is not None else None
            depth = (
                self.latest_depth.copy() if self.latest_depth is not None else None
            )
        return rgb, depth

    # -- internal --

    def _open(self):
        if not HAS_ZED:
            raise RuntimeError("pyzed 미설치")
        self._zed = sl.Camera()
        last_err = None
        for res in (sl.RESOLUTION.HD1200, sl.RESOLUTION.HD1080,
                    sl.RESOLUTION.AUTO):
            params = sl.InitParameters()
            params.camera_resolution = res
            params.camera_fps = CAMERA_FPS
            params.depth_mode = sl.DEPTH_MODE.NEURAL
            params.coordinate_units = sl.UNIT.MILLIMETER
            err = self._zed.open(params)
            if err == sl.ERROR_CODE.SUCCESS:
                info = self._zed.get_camera_information()
                cam_res = info.camera_configuration.resolution
                self.camera_type = (
                    f"ZED {str(info.camera_model).split('.')[-1]} "
                    f"{cam_res.width}x{cam_res.height}"
                )
                self._img, self._dep = sl.Mat(), sl.Mat()
                self._runtime = sl.RuntimeParameters()
                log.info("카메라 연결: %s", self.camera_type)
                return
            last_err = err
        raise RuntimeError(f"ZED open 실패: {last_err}")

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
                    log.error("카메라 연결 실패(%d/6), 10초 후 재시도: %s",
                              open_fails, e)
                    # CUDA가 801을 한 번 내면 프로세스 안에서는 복구 불가
                    # → 반복 실패 시 프로세스를 죽여 systemd로 새로 시작
                    if open_fails >= 6:
                        log.error("카메라 연결 반복 실패 — 프로세스 재시작")
                        os._exit(3)
                    time.sleep(10)
                    continue

            if self._zed.grab(self._runtime) == sl.ERROR_CODE.SUCCESS:
                fails = 0
                self._zed.retrieve_image(self._img, sl.VIEW.LEFT)
                rgb = cv2.cvtColor(self._img.get_data(), cv2.COLOR_BGRA2BGR)
                self._zed.retrieve_measure(self._dep, sl.MEASURE.DEPTH)
                depth = np.nan_to_num(
                    self._dep.get_data().copy(), nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
                with self._lock:
                    self.latest_rgb = rgb
                    self.latest_depth = depth
            else:
                fails += 1
                if fails >= RECONNECT_AFTER_FAILS:
                    log.error("grab %d회 연속 실패 → 카메라 재연결", fails)
                    self._close()
                    self.ok = False
                time.sleep(0.05)
            time.sleep(max(0.001, 1.0 / CAMERA_FPS - 0.005))
        self._close()


# ── 세션 (1회 취득) ─────────────────────────────────────

class CaptureSession(threading.Thread):
    def __init__(self, camera, class_name, on_finish):
        super().__init__(daemon=True)
        self.camera = camera
        self.class_name = class_name
        self.on_finish = on_finish
        self.stop_event = threading.Event()
        self.saved = 0
        self.started_at = datetime.now()
        self.t0 = time.monotonic()

        day = self.started_at.strftime("%Y%m%d")
        stamp = self.started_at.strftime("%H%M%S")
        self.session_dir = OUTPUT_ROOT / day / class_name / f"s_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @property
    def elapsed(self):
        return time.monotonic() - self.t0

    def run(self):
        self._write_meta(final=False)
        next_shot = WARMUP_SEC
        while not self.stop_event.is_set() and self.elapsed < SESSION_DURATION:
            if self.elapsed >= next_shot:
                self._save_pair()
                next_shot += CAPTURE_INTERVAL
            time.sleep(0.05)
        reason = "manual_stop" if self.stop_event.is_set() else "auto_stop"
        self._write_meta(final=True, reason=reason)
        log.info(
            "세션 종료(%s): %s, %d쌍 저장", reason, self.session_dir.name, self.saved
        )
        self.on_finish(self, reason)

    def _save_pair(self):
        rgb, depth = self.camera.get_pair()
        if rgb is None:
            log.warning("프레임 없음 — 저장 건너뜀 (idx %d)", self.saved)
            return
        cv2.imwrite(str(self.session_dir / f"rgb_{self.saved:04d}.png"), rgb)
        if depth is not None:
            depth16 = np.clip(depth, 0, 65535).astype(np.uint16)
            cv2.imwrite(
                str(self.session_dir / f"depth_{self.saved:04d}.png"), depth16
            )
        self.saved += 1

    def _write_meta(self, final, reason=""):
        meta = {
            "class_name": self.class_name,
            "started_at": self.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "camera": self.camera.camera_type,
            "capture_interval_s": CAPTURE_INTERVAL,
            "session_duration_s": SESSION_DURATION,
            "saved_pairs": self.saved,
            "finished": final,
            "stop_reason": reason,
        }
        if final:
            meta["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (self.session_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )


# ── Flask 앱 ────────────────────────────────────────────

app = Flask(__name__)
camera = FactoryCamera()
state_lock = threading.Lock()
selected_class = UNKNOWN
session: CaptureSession | None = None


def log_event(event, **kw):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "event": event}
    entry.update(kw)
    with (OUTPUT_ROOT / "events.jsonl").open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def today_stats():
    day_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d")
    per_class = {c: {"sessions": 0, "images": 0} for c in ALL_CLASSES}
    total_sessions = 0
    if day_dir.exists():
        for cls_dir in day_dir.iterdir():
            if cls_dir.name not in per_class:
                continue
            for s in cls_dir.glob("s_*"):
                n = len(list(s.glob("rgb_*.png")))
                per_class[cls_dir.name]["sessions"] += 1
                per_class[cls_dir.name]["images"] += n
                total_sessions += 1
    return total_sessions, per_class


def _on_session_finish(finished_session, reason):
    global session, selected_class
    with state_lock:
        if session is finished_session:
            session = None
        selected_class = UNKNOWN          # 라벨링 실수 방지: 항상 복귀
    log_event(
        reason,
        class_name=finished_session.class_name,
        session_dir=str(finished_session.session_dir.relative_to(OUTPUT_ROOT)),
        saved=finished_session.saved,
    )


@app.route("/")
def index():
    return render_template("factory.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            rgb, _ = camera.get_pair()
            if rgb is not None:
                h, w = rgb.shape[:2]
                preview = cv2.resize(rgb, (640, int(640 * h / w)))
                ok, jpg = cv2.imencode(
                    ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ok:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + jpg.tobytes() + b"\r\n"
                    )
            time.sleep(0.1)
    return Response(
        gen(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def api_status():
    total_sessions, per_class = today_stats()
    disk = shutil.disk_usage(str(BASE_DIR))
    with state_lock:
        capturing = session is not None
        return jsonify({
            "state": "capturing" if capturing else "idle",
            "selected_class": session.class_name if capturing else selected_class,
            "elapsed": round(session.elapsed, 1) if capturing else 0,
            "duration": SESSION_DURATION,
            "saved": session.saved if capturing else 0,
            "camera_ok": camera.ok,
            "camera_type": camera.camera_type,
            "camera_error": camera.last_error,
            "today_sessions": total_sessions,
            "per_class": per_class,
            "disk_free_gb": round(disk.free / 1e9, 1),
            "classes": ALL_CLASSES,
            "unknown": UNKNOWN,
            "now": datetime.now().strftime("%m/%d %H:%M"),
        })


@app.route("/api/select", methods=["POST"])
def api_select():
    global selected_class
    cls = request.json.get("class_name", "")
    if cls not in ALL_CLASSES:
        return jsonify({"error": "unknown class"}), 400
    with state_lock:
        if session is not None:
            return jsonify({"error": "취득 중에는 변경할 수 없습니다"}), 409
        selected_class = cls
    log_event("select", class_name=cls)
    return jsonify({"selected_class": cls})


@app.route("/api/start", methods=["POST"])
def api_start():
    global session
    with state_lock:
        if session is not None:
            return jsonify({"error": "이미 취득 중입니다"}), 409
        if not camera.ok:
            return jsonify({"error": "카메라가 연결되지 않았습니다"}), 503
        session = CaptureSession(camera, selected_class, _on_session_finish)
        session.start()
    log_event("start", class_name=session.class_name)
    log.info("세션 시작: %s (%s)", session.session_dir, session.class_name)
    return jsonify({"ok": True, "class_name": session.class_name})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        if session is None:
            return jsonify({"error": "취득 중이 아닙니다"}), 409
        session.stop_event.set()
    return jsonify({"ok": True})


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    with state_lock:
        if session is not None:
            return jsonify({"error": "취득 중에는 종료할 수 없습니다"}), 409
    log_event("shutdown")
    log.info("시스템 종료 요청")
    try:
        r = subprocess.run(
            ["sudo", "-n", "/usr/sbin/shutdown", "-h", "now"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return jsonify({"error": f"종료 실패: {r.stderr.strip()}"}), 500
    except Exception as e:
        return jsonify({"error": f"종료 실패: {e}"}), 500
    return jsonify({"ok": True})


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    fh = logging.FileHandler(
        LOG_DIR / f"factory_{datetime.now():%Y%m%d}.log"
    )
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logging.getLogger().addHandler(fh)

    log.info(
        "수집 서비스 시작 (duration=%ss, interval=%ss, warmup=%ss, port=%s)",
        SESSION_DURATION, CAPTURE_INTERVAL, WARMUP_SEC, PORT,
    )
    log_event("service_start")
    camera.start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)


def _require_cuda_ready():
    # 부팅 직후에는 CUDA가 아직 초기화 전이라 801(cudaErrorNotSupported)이
    # 나오고, 한 번 801을 맞은 프로세스는 이후에도 영원히 ZED open이
    # 실패한다(런타임이 초기화 실패를 캐시). 준비 안 됐으면 즉시 종료해서
    # systemd(Restart=always, 5초 간격)가 새 프로세스로 재시도하게 한다.
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
        log.info("CUDA probe: getDeviceCount=%d n=%d cudaMalloc=%d",
                 r1, n.value, r2)
        if r1 != 0 or n.value < 1 or r2 != 0:
            log.error("CUDA 미준비 — 프로세스 종료 후 systemd 재시작 대기")
            sys.exit(3)
    except OSError as e:
        log.error("libcudart 로드 실패 — 프로세스 종료: %r", e)
        sys.exit(3)


if __name__ == "__main__":
    _require_cuda_ready()
    main()
