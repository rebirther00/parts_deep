"""속성 파이프라인 실시간 추론 서버 (ZED 호환, 웹 UI).

파이프라인(프레임마다):
    RGB+Depth → MobileSAM 도어 마스크 → 렉티파이 → U-Net 통풍구
    → 템플릿 매칭 + 종횡비 → 슬라이딩 윈도(기본 10프레임) 집계 판정

레거시 05_realtime_inference.py와 같은 웹 UI 구조. depth가 필수라서
ZED 카메라가 필요하며, 카메라 없는 PC에서는 --replay로 저장된
rgb/depth 폴더를 재생해 전체 경로를 검증할 수 있다.

실행:
    python 14_realtime_inference_attribute.py                  # ZED 카메라
    python 14_realtime_inference_attribute.py --port 5003
    python 14_realtime_inference_attribute.py \
        --replay datasets_factory/E30_E38_door_RH             # 리플레이 검증
    python 14_realtime_inference_attribute.py --seed 29186    # 특정 run 모델
"""
import argparse
import collections
import glob
import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from attribute_utils import (
    CLASSES, GROUP, decide, frame_scores, load_templates, load_vent_unet)
from dimension_utils import _load_mobile_sam, refine_mask

DOOR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(DOOR_DIR))
sys.path.insert(0, REPO_DIR)
SAM_CKPT = os.path.join(DOOR_DIR, 'sam_models', 'mobile_sam.pt')

parser = argparse.ArgumentParser(
    description='속성 파이프라인 실시간 추론 서버',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model', type=str, default=None,
                    help='모델 경로 (기본: attribute_models/vent_unet.pth)')
parser.add_argument('--seed', type=int, default=None,
                    help='특정 run 사용: attribute_models/runs/'
                         'vent_unet_seed<시드>/model.pth')
parser.add_argument('--n_frames', type=int, default=10,
                    help='판정 집계 윈도 크기 (기본 10)')
parser.add_argument('--port', type=int, default=5003)
parser.add_argument('--replay', type=str, default=None,
                    help='카메라 대신 rgb_*/depth_*.png 폴더 재생 (검증용)')
parser.add_argument('--sam_every', type=int, default=3,
                    help='MobileSAM 마스크 갱신 주기 (N프레임마다 1회, '
                         '기본 3 — 도어가 정적이므로 재사용 안전. 1=매 프레임)')
parser.add_argument('--fp16', action='store_true',
                    help='FP16 autocast (Jetson GPU 단계 가속). 동일 프레임 '
                         'A/B에서 판정 100% 일치 검증됨')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

if args.model is None:
    if args.seed is not None:
        args.model = (f'attribute_models/runs/vent_unet_seed{args.seed}'
                      f'/model.pth')
    else:
        args.model = 'attribute_models/vent_unet.pth'

if args.replay and args.sam_every != 1:
    # 리플레이는 프레임마다 촬영 포즈가 달라 마스크 재사용이 무효
    # (실카메라의 정적 도어에서만 유효한 최적화)
    print(f'리플레이 모드 → --sam_every {args.sam_every} 무시, 1로 강제')
    args.sam_every = 1

app = Flask(__name__)
result_lock = threading.Lock()
inference_result = {
    'class': '대기 중', 'group': '-', 'confidence': 0.0,
    'all_probs': [], 'inference_ms': 0.0, 'window': 0, 'timestamp': 0.0,
}
latest_frame = None
frame_lock = threading.Lock()
reset_event = threading.Event()


# ── 프레임 소스 ──────────────────────────────────────────

class ReplaySource:
    """저장된 rgb/depth 폴더를 순환 재생 (카메라 없는 검증용)."""

    def __init__(self, folder):
        folder = os.path.join(DOOR_DIR, folder)
        self.pairs = []
        for rp in sorted(glob.glob(f'{folder}/rgb_*.png')):
            dp = rp.replace('rgb_', 'depth_')
            if os.path.exists(dp):
                self.pairs.append((rp, dp))
        if not self.pairs:
            raise SystemExit(f'리플레이 폴더에 rgb/depth 쌍이 없습니다: '
                             f'{folder}')
        self.i = 0
        print(f'리플레이 모드: {folder} ({len(self.pairs)}쌍 순환)')

    def get(self):
        rp, dp = self.pairs[self.i % len(self.pairs)]
        self.i += 1
        rgb = cv2.imread(rp)
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32)
        return rgb, depth

    def info(self):
        return {'camera': f'Replay ({len(self.pairs)} frames)',
                'connected': True}


class ZedSource:
    """CameraManager 래퍼 (depth 필수 → ZED 전용)."""

    def __init__(self):
        from camera_utils import CameraManager
        self.cam = CameraManager()
        self.cam.start()

    def get(self):
        rgb = self.cam.get_frame()
        depth = self.cam.get_depth()
        if rgb is None or depth is None:
            return None, None
        return rgb, depth

    def info(self):
        has_depth = self.cam.get_depth() is not None
        return {'camera': 'ZED' if has_depth else '카메라(depth 없음!)',
                'connected': self.cam.get_frame() is not None}


# ── 추론 루프 ────────────────────────────────────────────

def make_mask(sam, rgb):
    sam.set_image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    masks, _, _ = sam.predict(
        point_coords=np.array([[w // 2, h // 2]], np.float32),
        point_labels=np.array([1], np.int32), multimask_output=True)
    best = max(range(len(masks)), key=lambda i: masks[i].sum())
    return masks[best].astype(np.uint8) * 255


def inference_loop(source, net, templates, sam):
    global inference_result, latest_frame
    window = collections.deque(maxlen=args.n_frames)
    cached_mask = None
    frame_i = 0
    while True:
        if reset_event.is_set():
            window.clear()
            cached_mask = None
            reset_event.clear()
        rgb, depth = source.get()
        if rgb is None:
            time.sleep(0.1)
            continue
        with frame_lock:
            latest_frame = rgb.copy()
        t0 = time.time()
        try:
            import torch
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=args.fp16):
                # 도어는 정적 → SAM 마스크는 주기적으로만 갱신
                if cached_mask is None or frame_i % args.sam_every == 0:
                    cached_mask = refine_mask(make_mask(sam, rgb))
                sam_ms = (time.time() - t0) * 1000
                fs = frame_scores(rgb, depth, cached_mask, net, templates,
                                  device=args.device, profile=True)
            stage_ms = fs.pop('stage_ms', {})
            window.append(fs)
            pred, grp, scores = decide(list(window))
            total = sum(scores.values()) or 1.0
            probs = sorted(({'class': c, 'prob': s / total}
                            for c, s in scores.items()),
                           key=lambda x: -x['prob'])
            conf = probs[0]['prob'] * 100 if probs else 0.0
        except Exception as e:
            pred, grp, probs, conf = f'오류: {e}', '-', [], 0.0
            sam_ms, stage_ms = 0.0, {}
        frame_i += 1
        elapsed_ms = (time.time() - t0) * 1000
        with result_lock:
            inference_result = {
                'class': pred, 'group': grp,
                'confidence': round(conf, 1),
                'all_probs': [{'class': p['class'],
                               'prob': round(p['prob'], 4)}
                              for p in probs],
                'inference_ms': round(elapsed_ms, 1),
                'sam_ms': round(sam_ms, 1),
                'stage_ms': stage_ms,
                'window': len(window),
                'timestamp': time.time(),
            }


# ── MJPEG 스트리밍 ──────────────────────────────────────

def generate_mjpeg():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))
        with result_lock:
            r = inference_result.copy()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (480, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        color = (0, 255, 0) if r['confidence'] > 60 else (0, 200, 255)
        cv2.putText(frame, str(r['class']), (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.putText(frame,
                    f"group {r['group']} | {r['confidence']:.1f}% | "
                    f"{r['inference_ms']:.0f}ms | "
                    f"window {r['window']}/{args.n_frames}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)
        ok, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
        time.sleep(0.03)


# ── Flask 라우트 (05와 동일 구조) ───────────────────────

@app.route('/')
def index():
    return render_template('inference_attr.html',
                           cache_bust=int(time.time()))


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
    return jsonify(source.info())


@app.route('/api/reset')
def api_reset():
    """도어 교체 시 판정 윈도 초기화."""
    reset_event.set()
    return jsonify({'ok': True})


if __name__ == '__main__':
    print(f'모델: {args.model}')
    net = load_vent_unet(os.path.join(DOOR_DIR, args.model),
                         device=args.device)
    templates = load_templates()
    sam = _load_mobile_sam(SAM_CKPT, args.device)
    source = ReplaySource(args.replay) if args.replay else ZedSource()
    threading.Thread(target=inference_loop,
                     args=(source, net, templates, sam),
                     daemon=True).start()
    print(f'서버 시작: http://0.0.0.0:{args.port}')
    app.run(host='0.0.0.0', port=args.port, threaded=True)
