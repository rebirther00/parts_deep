"""홀 랜드마크 판별기 전용 실시간 추론 서버 (ZED 호환, 웹 UI).

16_train_hole_landmarks.py 로 학습한 attribute_models/hole_landmarks/model.pth 만
사용한다. MobileSAM·U-Net·CAD 템플릿(속성 파이프라인)은 로드하지 않는다.
속성 폴백까지 함께 쓰는 통합 서버는 14_realtime_inference_attribute.py.

파이프라인(프레임마다):
    RGB(+Depth) → ResNet18-FPN 히트맵 → 볼트홀 4 + 모서리 홀 2 검출
    → 기하 게이트 → depth 평면상 모서리 홀 거리 D(mm)
    → 슬라이딩 윈도(기본 10프레임) D 중앙값 → CAD D 최근접 클래스

depth 가 없으면(리플레이 폴더에 depth_*.png 없음) 볼트 피치 스케일로 D 를
추정한다(정밀도 낮음, 검증용).

실행:
    python 18_realtime_inference_hole.py                     # ZED 카메라, :5004
    python 18_realtime_inference_hole.py --port 5004
    python 18_realtime_inference_hole.py \
        --replay datasets_field/E25_door_RH_s_091317         # 리플레이 검증
    python 18_realtime_inference_hole.py --fp16              # Jetson 가속
"""
import argparse
import collections
import glob
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

import hole_classifier
from hole_classifier import CAD_D, GROUP

DOOR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(DOOR_DIR))
sys.path.insert(0, REPO_DIR)

parser = argparse.ArgumentParser(
    description='홀 랜드마크 판별기 전용 실시간 추론 서버',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model', type=str, default=hole_classifier.MODEL_PATH,
                    help='홀 랜드마크 모델 경로 '
                         '(기본: attribute_models/hole_landmarks/model.pth)')
parser.add_argument('--n_frames', type=int, default=10,
                    help='판정 집계 윈도 크기 (기본 10)')
parser.add_argument('--min_judged', type=int, default=3,
                    help='판정 확정에 필요한 윈도 내 게이트 통과 프레임 수 (기본 3)')
parser.add_argument('--port', type=int, default=5004,
                    help='웹 UI 포트 (기본 5004; 14번 통합 서버는 5003)')
parser.add_argument('--replay', type=str, default=None,
                    help='카메라 대신 rgb_*.png(+depth_*.png) 폴더 재생 (검증용)')
parser.add_argument('--fp16', action='store_true',
                    help='FP16 autocast (Jetson GPU 가속)')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

app = Flask(__name__)
result_lock = threading.Lock()
inference_result = {
    'class': '대기 중', 'group': '-', 'confidence': 0.0,
    'D_mm': None, 'margin_mm': None, 'n_judged': 0, 'window': 0,
    'gate': None, 'gate_counts': {}, 'candidates': [],
    'frame': None, 'inference_ms': 0.0, 'timestamp': 0.0,
}
latest_frame = None
frame_lock = threading.Lock()
reset_event = threading.Event()
stop_event = threading.Event()


# ── 프레임 소스 ──────────────────────────────────────────

class ReplaySource:
    """저장된 rgb(+depth) 폴더를 순환 재생 (카메라 없는 검증용)."""

    def __init__(self, folder):
        if not os.path.isabs(folder):
            folder = os.path.join(DOOR_DIR, folder)
        self.pairs = []
        for rp in sorted(glob.glob(f'{folder}/rgb_*.png')):
            dp = rp.replace('rgb_', 'depth_')
            self.pairs.append((rp, dp if os.path.exists(dp) else None))
        if not self.pairs:
            raise SystemExit(f'리플레이 폴더에 rgb_*.png 가 없습니다: {folder}')
        self.n_depth = sum(1 for _, dp in self.pairs if dp)
        self.i = 0
        print(f'리플레이 모드: {folder} ({len(self.pairs)}장 순환, '
              f'depth {self.n_depth}장)')

    def get(self):
        rp, dp = self.pairs[self.i % len(self.pairs)]
        self.i += 1
        rgb = cv2.imread(rp)
        depth = None
        if dp:
            depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32)
        return rgb, depth

    def info(self):
        return {'camera': f'Replay ({len(self.pairs)} frames, '
                          f'depth {self.n_depth})',
                'connected': True, 'depth': self.n_depth > 0}

    def close(self):
        pass


class ZedSource:
    """CameraManager 래퍼 (depth 있으면 depth D, 없으면 볼트 스케일 D)."""

    def __init__(self):
        from camera_utils import CameraManager
        self.cam = CameraManager()
        self.cam.start()

    def get(self):
        rgb = self.cam.get_frame()
        if rgb is None:
            return None, None
        return rgb, self.cam.get_depth()

    def info(self):
        has_depth = self.cam.get_depth() is not None
        return {'camera': 'ZED' if has_depth else '카메라(depth 없음!)',
                'connected': self.cam.get_frame() is not None,
                'depth': has_depth}

    def close(self):
        # 캡처 스레드 join + zed.close(). 이걸 빼먹으면 Ctrl+C 시 ZED SDK의
        # C++ 스레드가 열린 채 인터프리터가 내려가 "terminate called without
        # an active exception" 으로 abort 되고 드라이버 해제에서 수 분간 멈춘다.
        self.cam.stop()


# ── 추론 루프 ────────────────────────────────────────────

def candidates_from_D(D):
    """집계 D 에 대한 클래스별 CAD 거리 (가까운 순)."""
    if D is None:
        return []
    return sorted(({'class': c, 'group': GROUP[c], 'cad_D_mm': CAD_D[c],
                    'diff_mm': round(abs(CAD_D[c] - D), 1)} for c in CAD_D),
                  key=lambda x: x['diff_mm'])


def inference_loop(source, net, dev):
    global inference_result, latest_frame
    import torch
    window = collections.deque(maxlen=args.n_frames)
    while not stop_event.is_set():
        if reset_event.is_set():
            window.clear()
            reset_event.clear()
        rgb, depth = source.get()
        if rgb is None:
            time.sleep(0.1)
            continue
        with frame_lock:
            latest_frame = rgb.copy()
        t0 = time.time()
        try:
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=args.fp16):
                hr = hole_classifier.classify(net, dev, rgb, depth)
            window.append(hr)
            agg = hole_classifier.aggregate(list(window))
            gate_counts = dict(collections.Counter(
                r['gate'] for r in window))
            if agg['pred'] and agg['n_judged'] >= args.min_judged:
                pred, grp = agg['pred'], agg['group']
                conf = min(100.0, 100.0 * agg['n_judged'] / args.n_frames)
            elif agg['pred']:
                pred, grp = f'수집 중 ({agg["n_judged"]}/{args.min_judged})', agg['group']
                conf = 0.0
            else:
                pred, grp, conf = '보류', '-', 0.0
            frame_info = {
                'gate': hr['gate'],
                'pred': hr['pred'],
                'D_mm': round(hr['D_mm'], 1) if hr['D_mm'] else None,
                'D_src': hr['D_src'],
                'points': {c: [[round(float(p[0]), 1), round(float(p[1]), 1),
                                round(float(p[2]), 2)] for p in v]
                           for c, v in hr['points'].items()},
            }
            result = {
                'class': pred, 'group': grp, 'confidence': round(conf, 1),
                'D_mm': round(agg['D_mm'], 1) if agg['D_mm'] else None,
                'margin_mm': (round(agg['margin_mm'], 1)
                              if agg.get('margin_mm') is not None else None),
                'n_judged': agg['n_judged'], 'window': len(window),
                'gate': hr['gate'], 'gate_counts': gate_counts,
                'candidates': candidates_from_D(agg['D_mm'])[:4],
                'frame': frame_info,
            }
        except Exception as e:
            result = {'class': f'오류: {e}', 'group': '-', 'confidence': 0.0,
                      'D_mm': None, 'margin_mm': None, 'n_judged': 0,
                      'window': len(window), 'gate': None, 'gate_counts': {},
                      'candidates': [], 'frame': None}
        result['inference_ms'] = round((time.time() - t0) * 1000, 1)
        result['timestamp'] = time.time()
        with result_lock:
            inference_result = result


# ── MJPEG 스트리밍 ──────────────────────────────────────

POINT_COLORS = (('bolt', (255, 0, 0)), ('corner_hinge', (0, 0, 255)),
                ('corner_latch', (0, 140, 255)))


def generate_mjpeg():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        h, w = frame.shape[:2]
        sc = 960.0 / w if w > 960 else 1.0
        if sc != 1.0:
            frame = cv2.resize(frame, (960, int(h * sc)))
        with result_lock:
            r = inference_result.copy()
        fi = r.get('frame') or {}
        for c, col in POINT_COLORS:
            for p in (fi.get('points') or {}).get(c, []):
                cv2.circle(frame, (int(p[0] * sc), int(p[1] * sc)), 6, col, 2)
        pts = fi.get('points') or {}
        if pts.get('corner_hinge') and pts.get('corner_latch'):
            a, b = pts['corner_hinge'][0], pts['corner_latch'][0]
            cv2.line(frame, (int(a[0] * sc), int(a[1] * sc)),
                     (int(b[0] * sc), int(b[1] * sc)), (0, 200, 255), 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (520, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        color = (0, 255, 0) if r['confidence'] > 60 else (0, 200, 255)
        cv2.putText(frame, f"{r['class']}", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        d_txt = f"D={r['D_mm']:.0f}mm" if r.get('D_mm') else 'D=-'
        m_txt = (f" margin {r['margin_mm']:.0f}mm"
                 if r.get('margin_mm') is not None else '')
        cv2.putText(frame,
                    f"group {r['group']} | {d_txt}{m_txt} | "
                    f"judged {r['n_judged']}/{r['window']}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        f_d = f"{fi['D_mm']:.0f}mm/{fi.get('D_src')}" if fi.get('D_mm') else '-'
        cv2.putText(frame,
                    f"frame gate [{fi.get('gate')}] D {f_d} | "
                    f"{r['inference_ms']:.0f}ms",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
        time.sleep(0.03)


# ── Flask 라우트 (14와 동일 구조) ───────────────────────

@app.route('/')
def index():
    return render_template('inference_hole.html',
                           cache_bust=int(time.time()),
                           n_frames=args.n_frames, min_judged=args.min_judged)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/inference_result')
def api_inference_result():
    with result_lock:
        return jsonify(inference_result)


@app.route('/api/camera_info')
def api_camera_info():
    info = source.info()
    info['engine'] = 'PyTorch' + (' FP16' if args.fp16 else '')
    info['model'] = os.path.relpath(args.model, DOOR_DIR)
    return jsonify(info)


@app.route('/api/reset')
def api_reset():
    """도어 교체 시 판정 윈도 초기화."""
    reset_event.set()
    return jsonify({'ok': True})


if __name__ == '__main__':
    if not os.path.exists(args.model):
        raise SystemExit(f'홀 랜드마크 모델이 없습니다: {args.model}\n'
                         f'  → python 16_train_hole_landmarks.py 로 학습하거나 '
                         f'scripts/model_sync.sh 로 받아오세요')
    print(f'홀 랜드마크 모델: {args.model}')
    net, dev = hole_classifier.load_model(args.model, device=args.device)
    source = ReplaySource(args.replay) if args.replay else ZedSource()
    infer_thread = threading.Thread(target=inference_loop,
                                    args=(source, net, dev), daemon=True)
    infer_thread.start()

    def _on_signal(signum, frame):
        # SIGTERM(kill/systemd)도 Ctrl+C 와 같은 경로로 정리. 두 번째 신호는 강제 종료.
        if stop_event.is_set():
            print('\n강제 종료', flush=True)
            os._exit(1)
        stop_event.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f'서버 시작: http://0.0.0.0:{args.port}  (종료: Ctrl+C)')
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        print('\n종료 중: 추론 스레드 정지 → 카메라 닫기', flush=True)
        stop_event.set()
        infer_thread.join(timeout=5)
        source.close()
        print('종료 완료', flush=True)
