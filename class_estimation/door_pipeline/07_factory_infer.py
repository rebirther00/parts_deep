"""경주 공장 도어 실시간 추론 서비스 (무인, 터치모니터 없음) — 06_factory_capture.py 대체 (2026-11-02).

도어가 지그에 올라오면 자동으로 클래스·레이더·품번을 판정하고 판정 프레임만 저장한다.
설계: DOC_infer_service_plan.md

상태기계:
    EMPTY ─점유(ENTER_S)─▶ JUDGING ─잠금─▶ JUDGED ─이탈(LEAVE_S)─▶ EMPTY (세션 마감)
                              └─JUDGE_TIMEOUT_S 내 미확정─▶ UNRESOLVED ─이탈─▶ EMPTY
점유 신호 (빈 지그 = 카메라 정면 평판 ~1475mm, 도어는 그 위 ~30mm — 절대 z 로는 구분 불가):
    diff  = 빈 지그 배경 기준맵(depth 1/4, 시드 reference_images/empty_jig_depth_small.png → EMPTY 에서 EMA 갱신,
            OUTPUT_ROOT/background.npz 로 보존) 대비 BG_MM 넘게 다른 픽셀 비율. **OCC_ROI(지그 중앙 빈 부분)** 안에서만 잰다 —
            양쪽 세로 바가 도어 기종마다 레일을 따라 가로로 움직여 도어 ROI 전체로 재면 빈 지그도 변화로 잡힘(2026-11-03).
            E23(최소 도어)도 중앙은 덮으므로 빈 지그 ≈0, 도어 ≈1. 도어 ROI 전체 값은 diff_door 로 진단만.
    pts   = 홀 검출기 점수 ≥ PTS_SCORE 인 점 수 (볼트 4 + 코너 2)
    점유 = pts ≥ OCC_PTS 또는 diff ≥ OCC_ON / 비움 = pts < 2 이고 diff ≤ OCC_OFF (ENTER_S·LEAVE_S 디바운스)
게이트 (모두 통과한 프레임만 판정 윈도·저장):
    정지(motion < MOTION_THR, STATIC_FRAMES 연속) · 침입(intr < INTR_FRAC: 중앙값보다 INTR_MM 이상 가까운 픽셀) · 점유
    + hole_classifier 기하 게이트 · pose_warn. CNN 폴백은 "도어는 보이는데(pts≥OCC_PTS) 기하 실패"인 프레임만.
저장: datasets_factory_infer/<YYYYMMDD>/s_HHMMSS/{rgb,depth}_NNNN.png(판정 프레임 ≤MAX_SAVE, 미확정은 후보 CAND_KEEP)
      + result.json(원자적) + meta.json(06 호환) + thumb.jpg. 하루 events.jsonl(append) + metrics/metrics_YYYYMMDD.jsonl(1Hz, 임계값 튜닝용).
업로드: DOOR_INFER_UPLOAD=1 이면 세션 마감 때 deploy/nas_sync_infer.sh 킥 (그림자 모드 = 0, 기본).

실행:
    ~/workspace/zed_env/bin/python 07_factory_infer.py                       # ZED 카메라, :5000
    ~/workspace/zed_env/bin/python 07_factory_infer.py --replay "A:30,B:60,A:30"   # 폴더 재생(초) 으로 상태 전이 검증
환경변수: DOOR_INFER_* (아래 CFG 참조). systemd: deploy/door-infer.service
"""
import argparse
import collections
import glob
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

DOOR_DIR = Path(__file__).resolve().parent
REPO_DIR = DOOR_DIR.parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(DOOR_DIR / 'partno'))

import hole_classifier                                    # noqa: E402
from hole_classifier import CAD_D, GROUP, UNKNOWN         # noqa: E402
import cnn_classifier                                     # noqa: E402
import radar_check as rc                                  # noqa: E402
from factory_camera import FactoryCamera, require_cuda_ready   # noqa: E402
import depth_view                                         # noqa: E402

VERSION = '0.1.0'
UNKNOWN_LABEL = '미등록 도어 (unknown)'
STATUS_LABEL = {'judged': '판정', 'unknown': '미등록 도어', 'unrecognized': '미인식(기종 외)', 'abstain': '보류'}
# CNN 폴백을 돌리는 홀 게이트 실패 종류 — 6점은 잡혔는데 기하가 안 맞는 경우만. 홀 자체가 없는(no_corner/no_frame) 맨 밑판·
# 조립 전 상태에서 CNN 이 E23(외판 단일 솔리드 클래스)으로 오판한 사례(2026-11-02 s_145702) 차단
CNN_GATES = {'near_border', 'latch_offset', 'not_collinear', 'same_side', 'D_range', 'pose_warn'}


def _env(name, default):
    v = os.environ.get('DOOR_INFER_' + name)
    if v is None:
        return default
    if isinstance(default, bool):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    if isinstance(default, int):
        return int(float(v))
    if isinstance(default, float):
        return float(v)
    return v


CFG = dict(
    PORT=_env('PORT', 5000),
    OUTPUT_ROOT=_env('OUTPUT_ROOT', str(DOOR_DIR / 'datasets_factory_infer')),
    UPLOAD=_env('UPLOAD', False),            # NAS 업로드 킥 (그림자 모드 = False)
    FP16=_env('FP16', True),
    N_FRAMES=_env('N_FRAMES', 10),           # 판정 윈도
    MIN_JUDGED=_env('MIN_JUDGED', 3),
    CNN_MIN_PROB=_env('CNN_MIN_PROB', 0.5),
    OPTION_MIN_JUDGED=_env('OPTION_MIN_JUDGED', 3),
    OPTION_WAIT_S=_env('OPTION_WAIT_S', 60.0),      # 클래스 확정 후 레이더 미정이면 이만큼 더 기다렸다 잠금
    JUDGE_TIMEOUT_S=_env('JUDGE_TIMEOUT_S', 600.0),  # 점유 후 이 시간 안에 못 잠그면 미확정
    OCC_ON=_env('OCC_ON', 0.15),             # 배경 기준맵 대비 차이 픽셀 비율(diff, OCC_ROI 안) 이상이면 점유 (빈 지그 ≈0, 도어 ≈1)
    OCC_OFF=_env('OCC_OFF', 0.08),           # diff 이하(+pts<2)면 비움
    OCC_PTS=_env('OCC_PTS', 4),              # 홀 점 수 이상이면 점유(도어 확실)
    PTS_SCORE=_env('PTS_SCORE', 0.5),
    BG_MM=_env('BG_MM', 25.0),               # 배경과 이만큼 다르면 '변화' (도어 면은 평판보다 ≈33mm 앞)
    BG_TAU_EMPTY_S=_env('BG_TAU_EMPTY_S', 300.0),     # EMPTY 에서 배경 갱신 시정수 (카메라 드리프트 추종)
    BG_UPDATE_MM=_env('BG_UPDATE_MM', 10.0),  # 배경 갱신은 기존 배경과 이 안쪽으로 같은 픽셀만(보수적). 밑판(≈20~33mm 앞)이 흡수되던 사고(11/3→11/4) 방지
    BG_SAVE_S=_env('BG_SAVE_S', 600.0),
    Z_PLATE_MM=_env('Z_PLATE_MM', 1468.0),    # 빈 지그 평판 깊이(OCC_ROI 중앙값, 11/3 등록 시 1468). 배경 등록 때 갱신·npz 에 보존
    Z_EMPTY_TOL_MM=_env('Z_EMPTY_TOL_MM', 15.0),  # OCC_ROI 중앙값이 평판 − 이 값 이상(= 평판 높이)이면 배경맵과 무관하게 '비움' (도어 면은 ≈33mm 앞)
    TL_EVERY_S=_env('TL_EVERY_S', 60.0),      # 타임랩스: 1분 1장 (육안 판정 근거·세션 경계 검증), timelapse/<날짜>/HHMMSS.jpg + index.jsonl
    TL_BG_EVERY_S=_env('TL_BG_EVERY_S', 600.0),   # 배경 비교 그림(/bg.jpg 와 동일) 10분 1장 → bg_HHMMSS.jpg
    TL_WIDTH=_env('TL_WIDTH', 960),
    TL_RETAIN_DAYS=_env('TL_RETAIN_DAYS', 30),
    PROTR_MM=_env('PROTR_MM', 40.0),         # 진단 지표용 (판정에 안 씀 — 지그 레일·클램프가 평판보다 40mm+ 돌출해 구분 안 됨)
    INTR_MM=_env('INTR_MM', 250.0),          # 배경(없으면 프레임 중앙값)보다 이만큼 가까우면 침입(사람·물체). 도어 ROI 안에서만
    INTR_FRAC=_env('INTR_FRAC', 0.025),      # 실측(T=250, ROI): 도어만 0.007~0.016, 작업자 유입 0.033~0.062 → 중간값
    LOCK_WAIT_S=_env('LOCK_WAIT_S', 10.0),   # 클래스 확정 후 윈도가 찰 때까지 최대 대기 (신뢰도 ↑)
    MOTION_THR=_env('MOTION_THR', 4.0),      # 480x300 gray 평균 절대차 (현장 정지 장면 1.9~3.2, 작업자 3~15 — 윈도 중앙값·기하 게이트가 뒤를 받침)
    STATIC_FRAMES=_env('STATIC_FRAMES', 3),
    ENTER_S=_env('ENTER_S', 3.0),
    LEAVE_S=_env('LEAVE_S', 20.0),
    MIN_DWELL_S=_env('MIN_DWELL_S', 10.0),   # 이보다 짧고 도어도 못 본 세션은 폐기
    ROI=_env('ROI', '0.12,0.05,0.88,0.95'),  # x0,y0,x1,y1 (비율) — 침입 게이트·진단 지표용 도어 ROI
    OCC_ROI=_env('OCC_ROI', '0.42,0.10,0.66,0.90'),  # 점유(diff) 전용: 세로 바가 못 닿는 지그 중앙 빈 부분 (2026-11-03 사용자 확정, E23 기준)
    MAX_SAVE=_env('MAX_SAVE', 10),
    CAND_EVERY_S=_env('CAND_EVERY_S', 60.0),
    CAND_MAX=_env('CAND_MAX', 10),
    CAND_KEEP=_env('CAND_KEEP', 5),
    RETAIN_PNG_DAYS=_env('RETAIN_PNG_DAYS', 3),
    RETAIN_SESSION_DAYS=_env('RETAIN_SESSION_DAYS', 90),
    RETAIN_METRICS_DAYS=_env('RETAIN_METRICS_DAYS', 14),
    IDLE_HZ=_env('IDLE_HZ', 2.0),            # 판정 중이 아닐 때 검출기 실행 빈도
    LOCKED_HZ=_env('LOCKED_HZ', 2.0),        # 잠정 판정 뒤 계속 재판정하는 빈도 (브라켓 추가 조립·레이더 브라켓 반영, 이탈 시 최종)
    CHANGE_CONFIRM_N=_env('CHANGE_CONFIRM_N', 3),   # 잠정 판정 변경은 같은 (클래스, 레이더) 가 연속 이 횟수 나와야 반영 (히스테리시스)
    LOOP_HZ=_env('LOOP_HZ', 10.0),
    METRICS_HZ=_env('METRICS_HZ', 1.0),
    HEARTBEAT_S=_env('HEARTBEAT_S', 600.0),
    DISK_WARN_GB=_env('DISK_WARN_GB', 20.0),
)
OUTPUT_ROOT = Path(CFG['OUTPUT_ROOT'])
LOG_DIR = DOOR_DIR / 'logs'
REF_DIR = DOOR_DIR / 'reference_images'
STATES = ('EMPTY', 'JUDGING', 'JUDGED', 'UNRESOLVED')
import re as _re                                          # noqa: E402
re_tl = _re.compile(r'(bg_)?\d{6}\.jpg')
POINT_COLORS = (('bolt', (255, 0, 0)), ('corner_hinge', (0, 0, 255)),
                ('corner_latch', (0, 140, 255)), ('bracket', (255, 0, 255)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.getLogger('werkzeug').setLevel(logging.WARNING)
log = logging.getLogger('factory_infer')


# ── 유틸 ─────────────────────────────────────────────────

def now_str(ts=None):
    return datetime.fromtimestamp(ts if ts else time.time()).strftime('%Y-%m-%d %H:%M:%S')


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default), encoding='utf-8')
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + '\n')


def log_event(event, **kw):
    entry = {'ts': now_str(), 'event': event}
    entry.update(kw)
    append_jsonl(OUTPUT_ROOT / 'events.jsonl', entry)


def file_sha(path, n=12):
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()[:n]
    except OSError:
        return None


def count_pts(det, thr):
    if not det:
        return 0
    return sum(1 for c in ('bolt', 'corner_hinge', 'corner_latch')
               for p in det.get(c, []) if p[2] >= thr)


def cnn_vote(cnn_window):
    votes = [v for v in cnn_window if v]
    if not votes:
        return None
    cnt = collections.Counter(p for p, _ in votes)
    pred, n = cnt.most_common(1)[0]
    prob = float(np.mean([q for p, q in votes if p == pred]))
    top = [{'class': c, 'n': k} for c, k in cnt.most_common(3)]
    return {'pred': pred, 'prob': round(prob, 3), 'n': n, 'n_total': len(votes), 'top': top}


def candidates_from_D(D):
    if D is None:
        return []
    return sorted(({'class': c, 'group': GROUP[c], 'cad_D_mm': CAD_D[c],
                    'diff_mm': round(abs(CAD_D[c] - D), 1)} for c in CAD_D),
                  key=lambda x: x['diff_mm'])


def draw_points(img, det, scale=1.0):
    for c, col in POINT_COLORS:
        for p in (det or {}).get(c, []):
            cv2.circle(img, (int(p[0] * scale), int(p[1] * scale)), max(3, int(6 * scale)), col, 2)
    pts = det or {}
    if pts.get('corner_hinge') and pts.get('corner_latch'):
        a, b = pts['corner_hinge'][0], pts['corner_latch'][0]
        cv2.line(img, (int(a[0] * scale), int(a[1] * scale)), (int(b[0] * scale), int(b[1] * scale)),
                 (0, 200, 255), 1)
    return img


def ref_image_path(cls, radar=None):
    """클래스(+레이더) 대표 이미지 경로. 없으면 None."""
    if not cls or cls not in CAD_D:
        return None
    names = []
    if radar in (0, 1):
        names.append(f'{cls}__radar{radar}.jpg')
    names += [f'{cls}.jpg', f'{cls}__radar1.jpg', f'{cls}__radar0.jpg']
    for n in names:
        p = REF_DIR / n
        if p.exists():
            return p
    return None


# ── 프레임 소스 ──────────────────────────────────────────

class ReplaySource:
    """'dir:sec,dir:sec,...' 스크립트를 벽시계 기준으로 재생 (상태 전이 검증용). 끝나면 처음부터 반복."""

    def __init__(self, script, fps=4.0):
        self.segments = []
        for part in script.split(','):
            d, _, sec = part.partition(':')
            d = d if os.path.isabs(d) else str(DOOR_DIR / d)
            rgbs = sorted(glob.glob(f'{d}/rgb_*.png'))
            if not rgbs:
                raise SystemExit(f'리플레이 폴더에 rgb_*.png 없음: {d}')
            self.segments.append((d, rgbs, float(sec or 30)))
        self.fps = fps
        self.total = sum(s[2] for s in self.segments)
        self.t0 = time.time()
        self.calib = None
        for d, _, _ in self.segments:
            m = Path(d) / 'meta.json'
            if m.exists():
                intr = json.loads(m.read_text()).get('intrinsics')
                if intr and intr.get('fx'):
                    self.calib = intr
                    break
        self.camera_type = f'Replay ({len(self.segments)} segments, {self.total:.0f}s loop)'
        self.ok, self.last_error, self.seq = True, '', 0
        self._cache = {}
        self._last_key = None
        self._cur = (None, None)

    def start(self):
        pass

    def stop(self):
        pass

    def _load(self, rp):
        if rp not in self._cache:
            if len(self._cache) > 40:
                self._cache.clear()
            dp = rp.replace('rgb_', 'depth_')
            depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.exists(dp) else None
            self._cache[rp] = (cv2.imread(rp), None if depth is None else depth.astype(np.float32))
        return self._cache[rp]

    def get_pair(self):
        t = (time.time() - self.t0) % self.total
        acc = 0.0
        for d, rgbs, sec in self.segments:
            if t < acc + sec:
                idx = int((t - acc) * self.fps) % len(rgbs)
                key = (d, idx)
                if key != self._last_key:
                    self._last_key = key
                    self.seq += 1
                    self._cur = self._load(rgbs[idx])
                rgb, depth = self._cur
                return rgb.copy(), (None if depth is None else depth.copy()), time.time(), self.seq
            acc += sec
        return None, None, 0.0, self.seq

    def get_rgb(self):
        return self.get_pair()[0]


# ── 장면 게이트 ──────────────────────────────────────────

class SceneGates:
    """정지·침입·점유 지표. 점유는 빈 지그 배경 기준맵(depth, 1/4 해상도) 대비 차이 픽셀 비율 diff 로 잰다.

    지그(평판+레일+클램프)는 도어 종류·날짜와 무관하게 고정이라(2026-11-02 확인: 8/27 빈 지그 기준맵 대비
    10/29 도어 프레임 diff 0.68~0.74, 빈 지그 0.015) 배경 비교가 가장 확실하다. 기준맵은 EMPTY 상태에서 천천히
    갱신(카메라 드리프트 추종)하고, UNRESOLVED(기종 외 부품) 에서는 더 느리게 흡수해 세션이 끝날 수 있게 한다."""

    def __init__(self, cfg):
        x0, y0, x1, y1 = (float(v) for v in cfg['ROI'].split(','))
        self.roi = (x0, y0, x1, y1)
        self.occ_roi = tuple(float(v) for v in cfg['OCC_ROI'].split(','))   # 점유(diff) 전용 중앙 영역
        self.cfg = cfg
        self.prev_gray = None
        self.bg = None               # (H/4, W/4) float32 mm, 0 = 무효
        self.bg_src = None
        self.bg_dirty = False
        self.last_d4 = None

    def _roi(self, a, roi=None):
        h, w = a.shape[:2]
        x0, y0, x1, y1 = roi or self.roi
        return a[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]

    def load_background(self):
        """학습된 OUTPUT_ROOT/background.npz → 없으면 시드 reference_images/empty_jig_depth_small.png (8/27 빈 지그)."""
        p = OUTPUT_ROOT / 'background.npz'
        self.z_plate = float(self.cfg['Z_PLATE_MM'])
        try:
            if p.exists():
                z = np.load(p)
                self.bg = z['bg'].astype(np.float32)
                self.bg_src = f"learned {now_str(float(z['ts']))}"
                if 'z_plate' in z.files and float(z['z_plate']) > 0:
                    self.z_plate = float(z['z_plate'])
                return
        except Exception as e:                      # noqa: BLE001
            log.warning('background.npz 로드 실패: %s', e)
        seed = REF_DIR / 'empty_jig_depth_small.png'
        if seed.exists():
            self.bg = cv2.imread(str(seed), cv2.IMREAD_UNCHANGED).astype(np.float32)
            self.bg_src = 'seed(empty_jig_depth_small.png)'

    def save_background(self):
        if self.bg is None or not self.bg_dirty:
            return
        tmp = OUTPUT_ROOT / 'background.tmp.npz'      # savez 는 .npz 로 끝나지 않으면 확장자를 덧붙인다
        np.savez_compressed(tmp, bg=self.bg.astype(np.float32), ts=time.time(), z_plate=float(self.z_plate or 0))
        os.replace(tmp, OUTPUT_ROOT / 'background.npz')
        self.bg_dirty = False

    def occ_median(self, d4):
        """OCC_ROI(지그 중앙 빈 부분) 유효 depth 중앙값(mm). 없으면 None."""
        if d4 is None:
            return None
        do = self._roi(d4, self.occ_roi)
        v = do > 0
        return float(np.median(do[v])) if int(v.sum()) >= 100 else None

    def update_background(self, d4, tau, dt):
        """1/4 해상도 depth 로 EMA 갱신 — **기존 배경과 BG_UPDATE_MM 안쪽으로 같은 픽셀만**(카메라 드리프트·온도 변화 추종).
        밑판·도어처럼 수십 mm 다른 것은 절대 흡수하지 않는다(2026-11-03 오후 EMA 가 밑판을 흡수해 11/4 하루 종일 세션이 안 끝남).
        bg 가 없거나 크기가 다르면 즉시 채택."""
        if d4 is None:
            return
        if self.bg is None or self.bg.shape != d4.shape:
            self.bg = d4.copy()
            self.bg_src, self.bg_dirty = 'init', True
            self.z_plate = self.occ_median(d4) or self.z_plate
            return
        a = min(1.0, dt / max(tau, 1e-3))
        v = (d4 > 0) & (self.bg > 0) & (np.abs(d4 - self.bg) < self.cfg['BG_UPDATE_MM'])
        self.bg[v] = (1 - a) * self.bg[v] + a * d4[v]
        self.bg_dirty = True

    def set_background(self, d4, src):
        """현재 depth(1/4) 를 빈 지그 배경으로 즉시 채택 (웹 [빈 지그 배경 등록]). depth 가 비는 픽셀은 기존 값을 남긴다.
        평판 깊이 z_plate(OCC_ROI 중앙값)도 함께 갱신 — 배경맵과 독립된 '비움' 기준."""
        if d4 is None:
            return False
        bg = d4.astype(np.float32).copy()
        if self.bg is not None and self.bg.shape == bg.shape:
            hole = bg <= 0
            bg[hole] = self.bg[hole]
        self.bg, self.bg_src, self.bg_dirty = bg, src, True
        zp = self.occ_median(d4)
        if zp:
            self.z_plate = zp
        return True

    def measure(self, rgb, depth):
        small = cv2.resize(rgb, (480, 300), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = self._roi(gray)
        motion = float(np.abs(g - self.prev_gray).mean()) if self.prev_gray is not None and self.prev_gray.shape == g.shape else 0.0
        self.prev_gray = g
        out = dict(z_med=None, z_occ=None, protr=0.0, intr=0.0, diff=None, diff_door=None, motion=round(motion, 2), valid=0.0)
        self.last_d4 = None
        if depth is None:
            return out
        d4 = depth[::4, ::4]
        self.last_d4 = d4
        zo = self.occ_median(d4)
        out['z_occ'] = round(zo) if zo else None
        d = self._roi(d4)
        valid = d > 0
        n = int(valid.sum())
        out['valid'] = round(n / d.size, 3)
        if n < 200:
            return out
        z = float(np.median(d[valid]))
        out['z_med'] = round(z)
        out['protr'] = round(float(((d < z - self.cfg['PROTR_MM']) & valid).sum() / n), 4)
        # 침입(판정 게이트)은 도어 ROI 안에서만 잰다 — 홀이 가려지는 것이 문제이지, 지그 옆에 선 작업자·매달린 토치 케이블(현장 2026-11-02:
        # 전체 프레임 intr 0.10 이 ROI 밖 우측 장비)은 판정을 막을 이유가 없다. 전체 프레임 값은 intr_full 로 진단만.
        vf = d4 > 0
        out['intr'] = round(float(((d < z - self.cfg['INTR_MM']) & valid).sum() / n), 4)
        out['intr_full'] = round(float(((d4 < z - self.cfg['INTR_MM']) & vf).sum() / max(1, int(vf.sum()))), 4)
        if self.bg is not None and self.bg.shape == d4.shape:
            b = self._roi(self.bg)
            both = valid & (b > 0)
            nb = int(both.sum())
            if nb >= 200:
                out['diff_door'] = round(float((np.abs(d - b) > self.cfg['BG_MM'])[both].sum() / nb), 4)
                out['intr'] = round(float(((b - d) > self.cfg['INTR_MM'])[both].sum() / nb), 4)
            # 점유 판정용 diff 는 중앙 빈 부분(OCC_ROI)만 — 세로 바 이동에 영향받지 않는다
            do, bo = self._roi(d4, self.occ_roi), self._roi(self.bg, self.occ_roi)
            botho = (do > 0) & (bo > 0)
            no = int(botho.sum())
            if no >= 100:
                out['diff'] = round(float((np.abs(do - bo) > self.cfg['BG_MM'])[botho].sum() / no), 4)
            bothf = vf & (self.bg > 0)
            nf = int(bothf.sum())
            if nf >= 200:
                out['intr_full'] = round(float(((self.bg - d4) > self.cfg['INTR_MM'])[bothf].sum() / nf), 4)
        return out


# ── 세션 ─────────────────────────────────────────────────

class DoorSession:
    """도어 1개의 체류. 판정은 **잠정**(lock) 으로 먼저 내고 체류 내내 깨끗한 프레임으로 계속 재평가하다가
    이탈(finalize) 때 마지막 윈도로 **최종** 확정한다 — 현장은 밑판을 먼저 올리고 브라켓을 차례로 올린 뒤 용접하므로
    초기 잠정 판정은 브라켓이 덜 올라간 상태일 수 있다(2026-11-02 사용자 지적)."""

    def __init__(self, cfg, camera_type, calib, models, geom_on=True):
        self.cfg = cfg
        self.geom_on = geom_on
        self.provisional = False
        self.pred_history = []          # [{ts, status, pred_class, radar, source}] 잠정 판정 변경 이력
        self.final_at = None
        self.saved_at_clean = -1        # 프레임을 저장한 시점의 n_clean (이탈 시 윈도가 바뀌었으면 다시 저장)
        self.hole_judged_once = False   # 홀 판정이 한 번이라도 났으면 CNN 폴백이 덮어쓰지 못한다 (작업 중 기하 실패 프레임의 CNN 표가 흔들림)
        self._pending = None            # [key, 연속 횟수] — 변경 후보
        self.started_at = datetime.now()
        self.t0 = time.monotonic()
        day = self.started_at.strftime('%Y%m%d')
        if self.started_at.year < 2020:
            day = f'unsynced_{day}'
        self.dir = OUTPUT_ROOT / day / f's_{self.started_at:%H%M%S}'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rel = str(self.dir.relative_to(OUTPUT_ROOT))
        self.camera_type, self.calib, self.models = camera_type, calib, models
        self.frames = collections.deque(maxlen=cfg['N_FRAMES'])      # 게이트 통과 프레임 (rgb/depth 보관)
        self.cnn_votes = collections.deque(maxlen=cfg['N_FRAMES'])
        self.opts = collections.deque(maxlen=cfg['N_FRAMES'])
        self.status = None
        self.locked = False
        self.class_resolved_at = None
        self.judged_at = None
        self.n_processed = 0
        self.n_clean = 0
        self.door_seen = False
        self.max_pts = 0
        self.cands = []                 # (ts, rgb_path, depth_path)
        self.last_cand_ts = None
        self.saved = []                 # result['frames']
        self.result = {}
        self.ended_at = None
        self.last_live = None

    @property
    def dwell(self):
        return time.monotonic() - self.t0

    # -- 판정 --

    def push(self, rgb, depth, hr, cv, opt, ts):
        self.frames.append(dict(rgb=rgb, depth=depth, hr=hr, ts=ts))
        self.cnn_votes.append(cv)
        self.opts.append(opt)
        self.n_clean += 1

    def evaluate(self, geom_on=None):
        """18 과 같은 집계 규칙. 반환 (result dict, lock 여부). lock 은 아직 잠정 판정이 없을 때만 True."""
        cfg = self.cfg
        geom_on = self.geom_on if geom_on is None else geom_on
        hrs = [f['hr'] for f in self.frames]
        agg = hole_classifier.aggregate(hrs)
        vote = cnn_vote(self.cnn_votes)
        gate_counts = dict(collections.Counter(h['gate'] for h in hrs))
        pred = src = None
        conf, conflict = 0.0, False
        if agg['pred'] and agg['n_judged'] >= cfg['MIN_JUDGED']:
            pred, src = agg['pred'], 'hole'
            conf = min(100.0, 100.0 * agg['n_judged'] / cfg['N_FRAMES'])
            if pred != UNKNOWN and vote and vote['pred'] != pred:
                conflict = True
            txt = UNKNOWN_LABEL if pred == UNKNOWN else pred
        elif agg['pred']:
            txt = f"수집 중 ({agg['n_judged']}/{cfg['MIN_JUDGED']})"
        elif vote and vote['n'] >= cfg['MIN_JUDGED'] and vote['prob'] >= cfg['CNN_MIN_PROB']:
            pred, src = vote['pred'], 'cnn'
            conf = round(100.0 * vote['prob'], 1)
            txt = pred
        elif vote:
            txt = f"CNN 수집 중 ({vote['n']}/{cfg['MIN_JUDGED']}, p={vote['prob']:.2f})"
        else:
            txt = '보류'
        grp = GROUP.get(pred, '-') if pred else '-'
        ov = rc.option_vote([o['status'] for o in self.opts if o], min_judged=cfg['OPTION_MIN_JUDGED']) if geom_on else None
        radar_flag = radar_txt = None
        if pred in CAD_D and GROUP[pred] == 'FRT':
            radar_txt = '해당 없음'
        elif ov:
            radar_flag = ov['flag']
            radar_txt = {1: 'O', 0: 'X'}.get(radar_flag, f"미정 ({ov['n_judged']}/{cfg['OPTION_MIN_JUDGED']})")
        part = rc.part_lookup(pred, radar_flag) if pred in CAD_D else None
        last_opt = next((o for o in reversed(self.opts) if o), None)
        last = self.frames[-1]['hr'] if self.frames else None
        frame_info = None
        if last:
            frame_info = {'gate': last['gate'], 'pred': last['pred'],
                          'D_mm': round(last['D_mm'], 1) if last['D_mm'] else None, 'D_src': last['D_src'],
                          'z_mm': round(last['z_mm']) if last.get('z_mm') else None,
                          'tilt_deg': round(last['tilt_deg'], 1) if last.get('tilt_deg') else None,
                          'points': {c: [[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 2)] for p in v]
                                     for c, v in last['points'].items()}}
        now = time.monotonic()
        lock = False
        if pred and not self.locked:
            if self.class_resolved_at is None:
                self.class_resolved_at = now
            radar_done = (pred == UNKNOWN or (pred in CAD_D and GROUP[pred] == 'FRT') or radar_flag is not None
                          or now - self.class_resolved_at >= cfg['OPTION_WAIT_S'])
            # 윈도가 찰 때까지(최대 LOCK_WAIT_S) 더 모아 신뢰도를 올린 뒤 잠근다
            window_ok = len(self.frames) >= cfg['N_FRAMES'] or now - self.class_resolved_at >= cfg['LOCK_WAIT_S']
            lock = bool(radar_done and window_ok)
        elif not pred:
            self.class_resolved_at = None
        result = {
            'class': txt, 'pred_class': pred, 'group': grp, 'source': src, 'conflict': conflict,
            'confidence': round(conf, 1),
            'D_mm': round(agg['D_mm'], 1) if agg['D_mm'] else None,
            'margin_mm': round(agg['margin_mm'], 1) if agg.get('margin_mm') is not None else None,
            'nearest_mm': round(agg['nearest_mm'], 1) if agg.get('nearest_mm') is not None else None,
            'n_judged': agg['n_judged'], 'window': len(self.frames), 'gate_counts': gate_counts,
            'candidates': candidates_from_D(agg['D_mm'])[:4], 'cnn': vote,
            'radar': radar_flag, 'radar_txt': radar_txt, 'radar_votes': ov,
            'radar_src': last_opt['src'] if last_opt else None,
            'part_no': f"{part['part_no']}-{part['rev']}" if part else None,
            'part_name': part['name'] if part else None,
            'frame': frame_info,
        }
        self.last_live = result
        return result, lock

    # -- 저장 --

    def _save_pair(self, idx, rgb, depth):
        cv2.imwrite(str(self.dir / f'rgb_{idx:04d}.png'), rgb)
        if depth is not None:
            cv2.imwrite(str(self.dir / f'depth_{idx:04d}.png'), np.clip(depth, 0, 65535).astype(np.uint16))

    def _thumb(self, rgb, det):
        h, w = rgb.shape[:2]
        sc = 640.0 / w
        th = cv2.resize(rgb, (640, int(h * sc)))
        draw_points(th, det, sc)
        cv2.imwrite(str(self.dir / 'thumb.jpg'), th, [cv2.IMWRITE_JPEG_QUALITY, 80])

    def add_candidate(self, rgb, depth, ts, hr=None):
        """미확정 대비 후보 프레임 (CAND_EVERY_S 간격, 최대 CAND_MAX)."""
        if len(self.cands) >= self.cfg['CAND_MAX']:
            return
        if self.last_cand_ts is not None and ts - self.last_cand_ts < self.cfg['CAND_EVERY_S']:
            return
        i = len(self.cands)
        rp, dp = self.dir / f'cand_{i:02d}_rgb.png', self.dir / f'cand_{i:02d}_depth.png'
        cv2.imwrite(str(rp), rgb)
        if depth is not None:
            cv2.imwrite(str(dp), np.clip(depth, 0, 65535).astype(np.uint16))
        else:
            dp = None
        self.cands.append((ts, rp, dp, hr['gate'] if hr else None, hr['points'] if hr else None))
        self.last_cand_ts = ts

    @staticmethod
    def _key(result):
        return (result.get('pred_class'), result.get('radar'))

    @staticmethod
    def _apply_radar(result, radar):
        """레이더 값을 바꿔 끼우고 표시 문자열·품번을 다시 맞춘다 (윈도에서 표가 빠져 '미정'이 된 경우 이전 확정값 유지용)."""
        cls = result.get('pred_class')
        result = dict(result)
        result['radar'] = radar
        if cls in CAD_D and GROUP[cls] == 'FRT':
            result['radar_txt'] = '해당 없음'
        elif radar in (0, 1):
            result['radar_txt'] = {1: 'O', 0: 'X'}[radar]
        part = rc.part_lookup(cls, radar) if cls in CAD_D else None
        result['part_no'] = f"{part['part_no']}-{part['rev']}" if part else None
        result['part_name'] = part['name'] if part else None
        return result

    def _merge(self, result):
        """재평가 결과를 현재 잠정값에 비춰 '유효한 변경'인지 판단. 반환 (적용할 result 또는 None=유지, changed 여부).
        규칙: ① 윈도가 판정을 못 내면 유지 ② 홀 판정이 한 번 났으면 CNN 결과는 무시 ③ 레이더 미정(None)은 이전 값 유지
             ④ (클래스, 레이더) 가 같으면 수치만 갱신(변경 아님)"""
        if not result.get('pred_class'):
            return None, False
        if result.get('source') == 'cnn' and self.hole_judged_once:
            return None, False
        cur_cls, cur_radar = self._key(self.result)
        radar = result.get('radar') if result.get('radar') in (0, 1) else cur_radar
        if radar != result.get('radar'):
            result = self._apply_radar(result, radar)
        return result, (result['pred_class'], radar) != (cur_cls, cur_radar)

    def _save_window(self, result):
        """현재 윈도의 판정 프레임을 rgb/depth_NNNN 으로 저장(기존 저장분 교체) + thumb."""
        src = result['source']
        if src == 'hole':
            sel = [f for f in self.frames if f['hr']['gate'] == 'ok']
        else:
            sel = [f for f, v in zip(self.frames, self.cnn_votes) if v]
        sel = sel[-self.cfg['MAX_SAVE']:]
        if not sel:                                   # 현재 윈도에 저장할 판정 프레임이 없으면 기존 저장분 유지
            return
        for p in list(self.dir.glob('rgb_*.png')) + list(self.dir.glob('depth_*.png')):
            p.unlink()
        self.saved = []
        for i, f in enumerate(sel):
            self._save_pair(i, f['rgb'], f['depth'])
            hr = f['hr']
            self.saved.append({'file': f'rgb_{i:04d}.png', 'ts': now_str(f['ts']), 'gate': hr['gate'], 'pred': hr['pred'],
                               'D_mm': round(hr['D_mm'], 1) if hr['D_mm'] else None,
                               'margin_mm': round(hr['margin_mm'], 1) if hr.get('margin_mm') is not None else None,
                               'points': {c: [[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 2)] for p in v]
                                          for c, v in hr['points'].items()}})
        if sel:
            best = max(sel, key=lambda f: (f['hr'].get('margin_mm') or 0))
            self._thumb(best['rgb'], best['hr']['points'])
        self.saved_at_clean = self.n_clean

    def lock(self, result):
        """잠정 판정: 클래스(+레이더) 가 처음 확정됨 → 판정 프레임 저장, 상태 judged/unknown. 이후 update() 로 계속 재평가."""
        self.locked = True
        self.provisional = True
        self.judged_at = now_str()
        self.status = 'unknown' if result['pred_class'] == UNKNOWN else 'judged'
        self.pred_history.append({'ts': now_str(), 'status': self.status, 'pred_class': result['pred_class'],
                                  'radar': result['radar'], 'source': result['source'], 'confidence': result['confidence']})
        self.hole_judged_once = self.hole_judged_once or result['source'] == 'hole'
        self._save_window(result)
        for _, rp, dp, _, _ in self.cands:            # 후보는 불필요
            for p in (rp, dp):
                if p and p.exists():
                    p.unlink()
        self.cands = []
        self.result = result
        self.write(final=False)

    def update(self, result):
        """잠정 판정 뒤 재평가 결과 반영. (클래스, 레이더) 변경이 CHANGE_CONFIRM_N 회 연속이면 이력에 남기고 프레임·result.json 갱신.
        반환 = 변경 여부."""
        merged, changed = self._merge(result)
        if merged is None:
            return False
        if not changed:
            self._pending = None
            merged['part_no'] = merged.get('part_no') or self.result.get('part_no')   # 수치(신뢰도·D 등)만 조용히 갱신
            merged['part_name'] = merged.get('part_name') or self.result.get('part_name')
            self.result.update({k: merged[k] for k in ('confidence', 'D_mm', 'margin_mm', 'nearest_mm', 'n_judged', 'window',
                                                       'gate_counts', 'candidates', 'cnn', 'radar_votes', 'conflict')})
            self.hole_judged_once = self.hole_judged_once or merged.get('source') == 'hole'
            return False
        key = self._key(merged)
        if self._pending and self._pending[0] == key:
            self._pending[1] += 1
        else:
            self._pending = [key, 1]
        if self._pending[1] < self.cfg['CHANGE_CONFIRM_N']:
            return False
        self._pending = None
        result = merged
        self.hole_judged_once = self.hole_judged_once or result['source'] == 'hole'
        self.status = 'unknown' if result['pred_class'] == UNKNOWN else 'judged'
        self.pred_history.append({'ts': now_str(), 'status': self.status, 'pred_class': result['pred_class'],
                                  'radar': result['radar'], 'source': result['source'], 'confidence': result['confidence']})
        log.info('잠정 판정 변경 %s → %s (radar %s→%s, %s, %d번째)', self.result.get('pred_class'), result['pred_class'],
                 self.result.get('radar'), result['radar'], result['source'], len(self.pred_history))
        self._save_window(result)
        self.result = result
        self.write(final=False)
        return True

    def unresolved(self):
        """JUDGE_TIMEOUT_S 경과 — 도어를 봤으면 보류, 아니면 미인식(기종 외 부품)."""
        self.status = 'abstain' if self.door_seen else 'unrecognized'
        self._keep_candidates()
        self.write(final=False)

    def _keep_candidates(self):
        keep = self.cfg['CAND_KEEP']
        c = self.cands
        if len(c) > keep:
            idx = sorted({round(i * (len(c) - 1) / (keep - 1)) for i in range(keep)})
            chosen = [c[i] for i in idx]
        else:
            chosen = list(c)
        self.saved = []
        for i, (ts, rp, dp, gate, pts) in enumerate(chosen):
            os.replace(rp, self.dir / f'rgb_{i:04d}.png')
            if dp and dp.exists():
                os.replace(dp, self.dir / f'depth_{i:04d}.png')
            self.saved.append({'file': f'rgb_{i:04d}.png', 'ts': now_str(ts), 'gate': gate, 'pred': None,
                               'D_mm': None, 'margin_mm': None,
                               'points': {k: [[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 2)] for p in v]
                                          for k, v in (pts or {}).items()}})
        for ts, rp, dp, _, _ in c:
            for p in (rp, dp):
                if p and p.exists():
                    p.unlink()
        self.cands = []
        if self.saved:
            rgb = cv2.imread(str(self.dir / self.saved[0]['file']))
            if rgb is not None:
                self._thumb(rgb, chosen[0][4])

    def finalize(self, reason):
        """이탈/서비스 종료 → 최종 result.json·meta.json. 폐기한 세션은 None 반환.
        잠정 판정이 있으면 **마지막 윈도(이탈 전 마지막 깨끗한 프레임들)** 로 최종 판정·프레임을 확정한다."""
        self.ended_at = now_str()
        if self.locked:
            result, _ = self.evaluate()
            result, changed = self._merge(result)      # 최종도 같은 규칙(CNN 은 홀 판정을 못 덮음, 레이더 미정은 유지)
            if result is not None:
                self.status = 'unknown' if result['pred_class'] == UNKNOWN else 'judged'
                if changed:
                    self.pred_history.append({'ts': now_str(), 'status': self.status, 'pred_class': result['pred_class'],
                                              'radar': result['radar'], 'source': result['source'], 'confidence': result['confidence'],
                                              'final': True})
                    log.info('최종 판정이 잠정과 다름 %s → %s', self.result.get('pred_class'), result['pred_class'])
                if changed or self.saved_at_clean != self.n_clean:
                    self._save_window(result)
                self.result = result
            self.provisional = False
            self.final_at = now_str()
            for _, rp, dp, _, _ in self.cands:
                for p in (rp, dp):
                    if p and p.exists():
                        p.unlink()
            self.cands = []
        elif self.status is None:
            if self.dwell < self.cfg['MIN_DWELL_S'] and not self.door_seen and not self.cands:
                shutil.rmtree(self.dir, ignore_errors=True)
                return None
            self.status = 'abstain' if self.door_seen else 'unrecognized'
            if not self.cands and self.frames:            # 후보가 없으면 윈도 프레임이라도 남긴다
                for f in list(self.frames)[-self.cfg['CAND_KEEP']:]:
                    self.cands.append((f['ts'], None, None, f['hr']['gate'], f['hr']['points']))
                    i = len(self.cands) - 1
                    rp, dp = self.dir / f'cand_{i:02d}_rgb.png', self.dir / f'cand_{i:02d}_depth.png'
                    self._save_pair_to(rp, dp, f['rgb'], f['depth'])
                    self.cands[-1] = (f['ts'], rp, dp, f['hr']['gate'], f['hr']['points'])
            self._keep_candidates()
        self.write(final=True, reason=reason)
        return self.result

    def _save_pair_to(self, rp, dp, rgb, depth):
        cv2.imwrite(str(rp), rgb)
        if depth is not None:
            cv2.imwrite(str(dp), np.clip(depth, 0, 65535).astype(np.uint16))

    def write(self, final, reason=''):
        # 잠금 뒤에는 확정된 self.result(잠정/최종 판정)를 쓴다. last_live 는 마지막 윈도의 부분 집계라 판정 프레임이 2장뿐이면
        # pred_class 가 None 인데, 그걸 그대로 써서 service_stop 마감 때 판정이 지워졌다(2026-11-03 s_132558·s_142022, 20시 자동 종료도 같은 경로).
        base = self.result if (self.locked and self.result and self.result.get('pred_class')) else self.last_live
        r = dict(base or {})
        r.pop('frame', None)
        if self.status in ('abstain', 'unrecognized'):     # 잠금 전 윈도의 부분 집계값이 판정처럼 남지 않게
            for k in ('pred_class', 'group', 'part_no', 'part_name', 'radar', 'radar_txt', 'source', 'confidence'):
                r[k] = None
            r['class'] = STATUS_LABEL[self.status]
        r.update({
            'session_dir': self.rel, 'status': self.status, 'status_label': STATUS_LABEL.get(self.status, self.status),
            'final': final, 'stop_reason': reason if final else '',
            'provisional': self.provisional and not final, 'pred_history': self.pred_history,
            'n_changes': max(0, len(self.pred_history) - 1), 'final_at': self.final_at,
            'started_at': self.started_at.strftime('%Y-%m-%d %H:%M:%S'), 'ended_at': self.ended_at,
            'judged_at': self.judged_at, 'dwell_s': round(self.dwell, 1),
            'n_processed': self.n_processed, 'n_clean': self.n_clean, 'max_pts': self.max_pts,
            'frames': self.saved, 'saved_pairs': len(self.saved),
            'camera': self.camera_type, 'intrinsics': self.calib, 'models': self.models,
            'service_version': VERSION,
            'config': {k: self.cfg[k] for k in ('N_FRAMES', 'MIN_JUDGED', 'CNN_MIN_PROB', 'OPTION_MIN_JUDGED', 'OPTION_WAIT_S',
                                                 'JUDGE_TIMEOUT_S', 'OCC_ON', 'OCC_OFF', 'OCC_PTS', 'PTS_SCORE', 'PROTR_MM',
                                                 'INTR_MM', 'INTR_FRAC', 'MOTION_THR', 'STATIC_FRAMES', 'ENTER_S', 'LEAVE_S', 'ROI', 'OCC_ROI')},
        })
        self.result = r
        write_json_atomic(self.dir / 'result.json', r)
        # 06 meta.json 호환 (ingest 가 class_name/finished 를 읽음) — 라벨이 아니라 판정값임을 infer 로 표시
        write_json_atomic(self.dir / 'meta.json', {
            'class_name': r.get('pred_class') if r.get('pred_class') in CAD_D else 'Unknown',
            'infer': True, 'status': self.status,
            'started_at': r['started_at'], 'ended_at': self.ended_at, 'camera': self.camera_type,
            'intrinsics': self.calib, 'saved_pairs': len(self.saved), 'finished': final, 'stop_reason': reason,
        })


# ── 모니터 (상태기계 + 추론 루프) ───────────────────────

class Monitor(threading.Thread):
    def __init__(self, source, cfg, net, dev, cnn, geom):
        super().__init__(daemon=True, name='monitor')
        self.source, self.cfg, self.net, self.dev, self.cnn, self.geom = source, cfg, net, dev, cnn, geom
        self.gates = SceneGates(cfg)
        self.gates.load_background()
        log.info('배경 기준맵: %s', self.gates.bg_src or '없음 (첫 정지 프레임으로 초기화)')
        self.last_step_t = time.monotonic()
        self.state = 'EMPTY'
        self.state_since = time.monotonic()
        self.session = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.occ_since = self.empty_since = None
        self.static_n = 0
        self.last_det_t = 0.0
        self.last_pts = 0
        self.last_hr = None
        self.last_hr_t = 0.0
        self.last_metrics = {}
        self.last_metrics_log = 0.0
        self.last_infer_ms = 0.0
        self.last_seq = -1
        self.last_result = None            # 마지막으로 마감된 세션 결과
        self.started = time.time()
        self.models = {
            'hole': os.path.relpath(hole_classifier.MODEL_PATH, DOOR_DIR), 'hole_sha': file_sha(hole_classifier.MODEL_PATH),
            'hole_channels': list(getattr(net, 'channels', []) or []),
            'cnn': (os.path.relpath(cnn_classifier.MODEL_PATH, DOOR_DIR) if cnn else None),
            'cnn_sha': (file_sha(cnn_classifier.MODEL_PATH) if cnn else None),
            'option': 'hybrid(peaks→rule)' if geom else 'off', 'fp16': cfg['FP16'],
            'scale': hole_classifier.SCALE_DEFAULT,
        }

    # -- 외부 조작 --

    def reset(self):
        self.reset_event.set()

    def learn_background(self, force=False):
        """웹 [빈 지그 배경 등록]: 지금 depth 화면을 빈 지그 기준맵으로 채택해 background.npz 에 저장(재시작 후에도 유지).
        시드(8/27)가 현재 카메라·지그와 안 맞으면 빈 지그도 diff≈0.6 으로 읽혀 EMPTY 로 못 들어가고, 배경은 EMPTY 에서만
        학습하므로 스스로 못 빠져나온다(2026-11-03: 도어 5대가 세션 하나로 합쳐짐). 홀이 보이거나 움직이면 force 없이는 거부.
        등록 뒤 diff 가 떨어지면 LEAVE_S 뒤 세션이 자연히 마감된다."""
        m = dict(self.last_metrics)
        d4 = self.gates.last_d4
        if d4 is None or not m:
            return {'ok': False, 'error': 'depth 프레임이 아직 없습니다'}
        pts = int(m.get('pts') or 0)
        if pts >= 2 and not force:
            return {'ok': False, 'need_force': True, 'error': f'홀 {pts}점이 보입니다 — 도어가 올라가 있는 것 같습니다'}
        if not m.get('static') and not force:
            return {'ok': False, 'need_force': True, 'error': f"움직임 {m.get('motion')} — 정지 상태가 아닙니다"}
        src = f'manual {now_str()}'
        self.gates.set_background(d4, src)
        try:
            self.gates.save_background()
        except Exception as e:                      # noqa: BLE001
            log.warning('배경 기준맵 저장 실패: %s', e)
        log.info('빈 지그 배경 등록(웹): 등록 전 diff %s, pts %d, z %s, force=%s', m.get('diff'), pts, m.get('z_med'), force)
        log_event('bg_learn', diff_before=m.get('diff'), pts=pts, z=m.get('z_med'), force=bool(force))
        return {'ok': True, 'src': src, 'diff_before': m.get('diff'), 'pts': pts, 'z': m.get('z_med')}

    def _set_state(self, s):
        if s != self.state:
            log.info('상태 %s → %s', self.state, s)
            self.state = s
            self.state_since = time.monotonic()

    # -- 루프 --

    def run(self):
        import torch
        cfg = self.cfg
        period = 1.0 / cfg['LOOP_HZ']
        while not self.stop_event.is_set():
            t_loop = time.monotonic()
            rgb, depth, ts, seq = self.source.get_pair()
            if rgb is None or seq == self.last_seq:
                time.sleep(0.02)
                continue
            self.last_seq = seq
            try:
                self._step(rgb, depth, ts, torch)
            except Exception as e:                      # noqa: BLE001
                log.exception('루프 오류: %s', e)
                time.sleep(0.5)
            dt = time.monotonic() - t_loop
            if dt < period:
                time.sleep(period - dt)
        # 서비스 종료: 진행 중 세션 마감
        if self.session is not None:
            self._finish('service_stop')

    def _classify(self, rgb, depth, torch):
        t0 = time.monotonic()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.cfg['FP16']):
            hr = hole_classifier.classify(self.net, self.dev, rgb, depth, intrinsics=self.source.calib,
                                          scale=hole_classifier.SCALE_DEFAULT)
        self.last_infer_ms = (time.monotonic() - t0) * 1000
        self.last_hr, self.last_hr_t = hr, time.monotonic()
        self.last_det_t = self.last_hr_t
        self.last_pts = count_pts(hr['points'], self.cfg['PTS_SCORE'])
        return hr

    def _step(self, rgb, depth, ts, torch):
        cfg = self.cfg
        now = time.monotonic()
        m = self.gates.measure(rgb, depth)
        static = m['motion'] < cfg['MOTION_THR']
        self.static_n = self.static_n + 1 if static else 0
        is_static = self.static_n >= cfg['STATIC_FRAMES']
        intruded = m['intr'] >= cfg['INTR_FRAC']

        if self.reset_event.is_set():
            self.reset_event.clear()
            if self.session is not None:
                log.info('재판정 요청 — 윈도 초기화')
                log_event('reset', session_dir=self.session.rel)
                self.session.frames.clear(); self.session.cnn_votes.clear(); self.session.opts.clear()
                self.session.locked = False; self.session.provisional = False; self.session.status = None
                self.session.class_resolved_at = None; self.session.saved = []
                self._set_state('JUDGING')

        # 검출기: 판정 중이면 깨끗한 프레임마다, 아니면 IDLE_HZ 로 (점유 신호 pts 용)
        hr = None
        locked = bool(self.session and self.session.locked)
        judging = self.state in ('JUDGING', 'UNRESOLVED') and not locked
        # 잠정 판정 전: 깨끗한 프레임마다 / 잠정 판정 후: LOCKED_HZ 로 계속 재판정 / 빈 지그: IDLE_HZ (점유 신호 pts 용)
        run_det = (judging and is_static and not intruded) or \
            (now - self.last_det_t >= 1.0 / (cfg['LOCKED_HZ'] if locked else cfg['IDLE_HZ']))
        if run_det:
            hr = self._classify(rgb, depth, torch)
        pts = self.last_pts if now - self.last_hr_t < 2.0 else 0

        diff = m.get('diff')
        # 평판 높이 기준 비움: OCC_ROI 중앙값이 평판(z_plate) − 허용치 이상이면 배경맵이 틀려도(도어 흡수 등) 비움으로 본다
        zp, zo = self.gates.z_plate, m.get('z_occ')
        z_empty = bool(zp and zo and zo >= zp - cfg['Z_EMPTY_TOL_MM'])
        occupied_raw = pts >= cfg['OCC_PTS'] or (diff is not None and diff >= cfg['OCC_ON'] and not z_empty)
        empty_raw = pts < 2 and (diff is None or diff <= cfg['OCC_OFF'] or z_empty)
        m['z_empty'] = z_empty
        self.occ_since = (self.occ_since or now) if occupied_raw else None
        self.empty_since = (self.empty_since or now) if empty_raw else None
        occ_for = now - self.occ_since if self.occ_since else 0.0
        empty_for = now - self.empty_since if self.empty_since else 0.0
        clean = occupied_raw and is_static and not intruded

        # -- 상태 전이 --
        if self.state == 'EMPTY':
            if occ_for >= cfg['ENTER_S']:
                self._enter()
        else:
            s = self.session
            s.n_processed += 1
            if hr is not None:
                s.max_pts = max(s.max_pts, self.last_pts)
                if self.last_pts >= cfg['OCC_PTS'] or hr['gate'] == 'ok':
                    s.door_seen = True
            if self.state in ('JUDGING', 'JUDGED', 'UNRESOLVED'):
                if hr is not None and clean:
                    cv = opt = None
                    # CNN 폴백: 6점은 잡혔는데 기하가 안 맞는 프레임만 (홀이 없는 맨 밑판·조립 전 상태는 제외)
                    if self.cnn is not None and hr['gate'] in CNN_GATES and self.last_pts >= cfg['OCC_PTS']:
                        p, q, _ = self.cnn.predict(rgb)
                        cv = (p, q)
                    if self.geom is not None and hr['gate'] == 'ok' and hr['pred'] in CAD_D and GROUP[hr['pred']] != 'FRT':
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg['FP16']):
                            opt = rc.decide_option(self.net, self.dev, rgb, depth, hr['pred'], self.source.calib,
                                                   self.geom, det=hr['points'])
                    s.push(rgb, depth, hr, cv, opt, ts)
                    if not s.locked:
                        s.add_candidate(rgb, depth, ts, hr)
                    result, lock = s.evaluate()
                    if lock and not s.locked:
                        s.lock(result)
                        log.info('잠정 판정 [%s] %s part=%s radar=%s src=%s conf=%.0f D=%s margin=%s (%d/%d, %.0fs) — 이탈 시 최종',
                                 s.status, result['pred_class'], result['part_no'], result['radar_txt'], result['source'],
                                 result['confidence'], result['D_mm'], result['margin_mm'], result['n_judged'],
                                 result['window'], s.dwell)
                        log_event('judged', session_dir=s.rel, status=s.status, pred_class=result['pred_class'],
                                  part_no=result['part_no'], radar=result['radar'], source=result['source'],
                                  confidence=result['confidence'], D_mm=result['D_mm'], dwell_s=round(s.dwell, 1), provisional=True)
                        self._set_state('JUDGED')
                    elif s.locked and s.update(result):
                        log_event('judged_update', session_dir=s.rel, status=s.status, pred_class=result['pred_class'],
                                  part_no=result['part_no'], radar=result['radar'], source=result['source'],
                                  confidence=result['confidence'], dwell_s=round(s.dwell, 1), n_changes=len(s.pred_history) - 1)
                        if self.state != 'JUDGED':
                            self._set_state('JUDGED')
                elif not s.locked and (is_static or (not s.cands and s.dwell >= cfg['CAND_EVERY_S'])):
                    s.add_candidate(rgb, depth, ts, hr)        # 깨끗하지 않아도 정지 프레임은 후보로(미확정 세션 증거)
                if self.state == 'JUDGING' and s.dwell >= cfg['JUDGE_TIMEOUT_S']:
                    s.unresolved()
                    log.info('판정 시한 초과 → %s (pts max %d, clean %d/%d)', s.status, s.max_pts, s.n_clean, s.n_processed)
                    log_event('timeout', session_dir=s.rel, status=s.status, max_pts=s.max_pts, n_clean=s.n_clean)
                    self._set_state('UNRESOLVED')
            if empty_for >= cfg['LEAVE_S']:
                self._finish('leave')

        # 배경 기준맵 갱신 — EMPTY(정지·도어 없음·비움 판정) 에서만, 그것도 기존 배경과 거의 같은 픽셀만(update_background 참조).
        # UNRESOLVED 흡수는 폐지: 기종 외 부품은 치워질 때 비움으로 끝나며, 흡수는 밑판·도어를 배경에 넣는 사고만 냈다.
        dt = now - self.last_step_t
        self.last_step_t = now
        d4 = self.gates.last_d4
        if d4 is not None and is_static and pts < 2 and self.state == 'EMPTY' and \
                (self.gates.bg is None or (diff is not None and diff <= cfg['OCC_OFF'])):
            self.gates.update_background(d4, cfg['BG_TAU_EMPTY_S'], dt)
        # 자가 복구: 평판 높이로는 분명히 비었는데(z_empty, 홀 0, 정지 60초) 배경맵은 '점유'라고 하면 배경맵이 틀린 것(도어 흡수 등)
        # → 지금 화면을 빈 지그 배경으로 다시 등록 (사용자가 [빈 지그 배경 등록] 을 누르는 것과 같음)
        if d4 is not None and z_empty and is_static and pts == 0 and diff is not None and diff >= cfg['OCC_ON'] \
                and empty_for >= 60.0 and now - getattr(self, 'last_bg_auto_t', 0.0) >= 600.0:
            self.last_bg_auto_t = now
            self.gates.set_background(d4, f'auto {now_str()}')
            try:
                self.gates.save_background()
            except Exception as e:                      # noqa: BLE001
                log.warning('배경 기준맵 저장 실패: %s', e)
            log.info('빈 지그 배경 자동 재등록: 평판 높이 비움(z_occ %s ≥ z_plate %s−%s)인데 diff %.2f — 배경맵이 틀려 있었음',
                     zo, round(zp), cfg['Z_EMPTY_TOL_MM'], diff)
            log_event('bg_auto', z_occ=zo, z_plate=round(zp), diff_before=diff)
        self._timelapse(rgb, dict(m, pts=pts), now)

        self.last_metrics = dict(m, pts=pts, occ=occupied_raw, static=is_static, intr_flag=intruded, clean=clean,
                                 occ_for=round(occ_for, 1), empty_for=round(empty_for, 1))
        if now - self.last_metrics_log >= 1.0 / cfg['METRICS_HZ']:
            self.last_metrics_log = now
            lh = self.last_hr if now - self.last_hr_t < 2.0 else None
            append_jsonl(OUTPUT_ROOT / 'metrics' / f'metrics_{datetime.now():%Y%m%d}.jsonl', {
                'ts': now_str(), 'state': self.state, 'z': m['z_med'], 'z_occ': m.get('z_occ'), 'z_empty': int(bool(m.get('z_empty'))),
                'diff': m.get('diff'), 'diff_door': m.get('diff_door'), 'protr': m['protr'],
                'intr': m['intr'], 'intr_full': m.get('intr_full'),
                'motion': m['motion'], 'pts': pts, 'occ': int(occupied_raw), 'static': int(is_static),
                'gate': lh['gate'] if lh else None, 'pred': lh['pred'] if lh else None,
                'D': round(lh['D_mm'], 1) if lh and lh['D_mm'] else None,
                'ms': round(self.last_infer_ms) if lh else None,
            })

    # -- 타임랩스 (1분 1장, 육안 판정 근거·세션 경계 검증) --

    def _timelapse(self, rgb, m, now):
        cfg = self.cfg
        if now - getattr(self, 'last_tl_t', 0.0) < cfg['TL_EVERY_S']:
            return
        self.last_tl_t = now
        try:
            day = datetime.now().strftime('%Y%m%d')
            tdir = OUTPUT_ROOT / 'timelapse' / day
            tdir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime('%H%M%S')
            h, w = rgb.shape[:2]
            sc = cfg['TL_WIDTH'] / w
            img = cv2.resize(rgb, (int(cfg['TL_WIDTH']), int(h * sc)), interpolation=cv2.INTER_AREA)
            hr = self.last_hr if now - self.last_hr_t < 2.0 else None
            if hr:
                draw_points(img, hr['points'], sc)
            cv2.imwrite(str(tdir / f'{stamp}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            s = self.session
            append_jsonl(tdir / 'index.jsonl', {
                'ts': now_str(), 'file': f'{stamp}.jpg', 'state': self.state,
                'session_dir': s.rel if s else None, 'status': s.status if s else None,
                'pred_class': (s.result or {}).get('pred_class') if s else None,
                'radar': (s.result or {}).get('radar') if s else None,
                'z_occ': m.get('z_occ'), 'diff': m.get('diff'), 'pts': m.get('pts'), 'intr': m.get('intr'),
                'z_empty': bool(m.get('z_empty')), 'gate': hr['gate'] if hr else None,
            })
            if now - getattr(self, 'last_tlbg_t', 0.0) >= cfg['TL_BG_EVERY_S']:
                self.last_tlbg_t = now
                g = self.gates
                if g.bg is not None or g.last_d4 is not None:
                    bimg, _ = depth_view.render_compare(g.bg, g.last_d4, bg_mm=cfg['BG_MM'], roi=g.roi, occ_roi=g.occ_roi,
                                                        titles=(f"BG: {g.bg_src or 'none'} zp {g.z_plate and round(g.z_plate)}",
                                                                f"NOW {now_str()[11:]} zo {m.get('z_occ')}", 'DIFF'))
                    cv2.imwrite(str(tdir / f'bg_{stamp}.jpg'), bimg, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as e:                          # noqa: BLE001
            log.warning('타임랩스 저장 실패: %s', e)

    def _enter(self):
        with self.lock:
            self.session = DoorSession(self.cfg, self.source.camera_type, self.source.calib, self.models,
                                       geom_on=self.geom is not None)
        log.info('점유 감지 → 세션 시작 %s', self.session.rel)
        log_event('enter', session_dir=self.session.rel)
        self._set_state('JUDGING')

    def _finish(self, reason):
        s = self.session
        r = s.finalize(reason)
        with self.lock:
            self.session = None
            if r is not None:
                self.last_result = r
        if r is None:
            log.info('세션 폐기(짧은 점유, 도어 없음): %s', s.rel)
            log_event('discard', session_dir=s.rel, reason=reason)
        else:
            log.info('세션 마감(%s) %s [%s] %s saved=%d dwell=%.0fs', reason, s.rel, s.status,
                     r.get('pred_class'), len(s.saved), s.dwell)
            log_event('leave', session_dir=s.rel, status=s.status, pred_class=r.get('pred_class'),
                      part_no=r.get('part_no'), saved=len(s.saved), dwell_s=round(s.dwell, 1), reason=reason)
            kick_upload()
        self._set_state('EMPTY')

    # -- 조회 --

    def snapshot(self):
        with self.lock:
            s = self.session
            cur = None
            if s is not None:
                cur = dict(s.last_live or {})
                cur.update(session_dir=s.rel, status=s.status, status_label=STATUS_LABEL.get(s.status),
                           started_at=s.started_at.strftime('%H:%M:%S'), dwell_s=round(s.dwell, 1),
                           locked=s.locked, provisional=s.provisional, pred_history=s.pred_history,
                           n_processed=s.n_processed, n_clean=s.n_clean, max_pts=s.max_pts,
                           saved_pairs=len(s.saved), judged_at=s.judged_at)
                if s.locked and s.result:               # 화면의 클래스·품번은 잠정 판정값(윈도가 비어도 유지)
                    for k in ('class', 'pred_class', 'group', 'part_no', 'part_name', 'radar', 'radar_txt', 'source',
                              'confidence', 'D_mm', 'margin_mm', 'n_judged', 'window', 'gate_counts', 'candidates', 'conflict'):
                        if cur.get(k) in (None, 0, '보류', '-') and s.result.get(k) not in (None, ''):
                            cur[k] = s.result[k]
            last = self.last_result
        return {'state': self.state, 'state_for_s': round(time.monotonic() - self.state_since, 1),
                'session': cur, 'last_result': last, 'metrics': self.last_metrics,
                'infer_ms': round(self.last_infer_ms, 1)}


# ── 업로드·보존·하트비트 ─────────────────────────────────

def kick_upload():
    if not CFG['UPLOAD']:
        return
    try:
        subprocess.Popen(['bash', str(DOOR_DIR / 'deploy' / 'nas_sync_infer.sh')],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:                              # noqa: BLE001
        log.warning('NAS 동기화 실행 실패: %s', e)


def session_dirs(day=None):
    pat = f'{day}/s_*' if day else '*/s_*'
    return sorted(p for p in OUTPUT_ROOT.glob(pat) if p.is_dir())


def retention_once():
    """업로드 완료(.uploaded) 세션만 정리: png 는 RETAIN_PNG_DAYS, 세션 전체는 RETAIN_SESSION_DAYS, metrics 는 RETAIN_METRICS_DAYS."""
    now = time.time()
    n_png = n_sess = 0
    for d in session_dirs():
        up = d / '.uploaded'
        if not up.exists():
            continue
        age_d = (now - up.stat().st_mtime) / 86400
        if age_d >= CFG['RETAIN_SESSION_DAYS']:
            shutil.rmtree(d, ignore_errors=True)
            n_sess += 1
        elif age_d >= CFG['RETAIN_PNG_DAYS'] and not (d / '.png_purged').exists():
            for p in d.glob('*.png'):
                p.unlink()
                n_png += 1
            (d / '.png_purged').touch()
    for p in (OUTPUT_ROOT / 'metrics').glob('metrics_*.jsonl'):
        if (now - p.stat().st_mtime) / 86400 >= CFG['RETAIN_METRICS_DAYS']:
            p.unlink()
    for d in (OUTPUT_ROOT / 'timelapse').glob('*'):
        if d.is_dir() and (now - d.stat().st_mtime) / 86400 >= CFG['TL_RETAIN_DAYS']:
            shutil.rmtree(d, ignore_errors=True)
    if n_png or n_sess:
        log.info('보존 정리: png %d개 삭제, 세션 %d개 삭제', n_png, n_sess)


def housekeeping_loop(monitor):
    last_ret = 0.0
    last_hb = 0.0
    last_bg = time.time()
    while not monitor.stop_event.is_set():
        now = time.time()
        if now - last_bg >= CFG['BG_SAVE_S']:
            last_bg = now
            try:
                monitor.gates.save_background()
            except Exception as e:                      # noqa: BLE001
                log.warning('배경 기준맵 저장 실패: %s', e)
        if now - last_hb >= CFG['HEARTBEAT_S']:
            last_hb = now
            try:
                snap = monitor.snapshot()
                disk = shutil.disk_usage(str(DOOR_DIR))
                write_json_atomic(OUTPUT_ROOT / 'heartbeat.json', {
                    'ts': now_str(), 'host': os.uname().nodename, 'state': snap['state'], 'version': VERSION,
                    'camera_ok': monitor.source.ok, 'disk_free_gb': round(disk.free / 1e9, 1),
                    'today': today_counts(), 'upload': CFG['UPLOAD'], 'uptime_s': round(now - monitor.started)})
            except Exception as e:                      # noqa: BLE001
                log.warning('하트비트 실패: %s', e)
        if now - last_ret >= 3600:
            last_ret = now
            try:
                retention_once()
            except Exception as e:                      # noqa: BLE001
                log.warning('보존 정리 실패: %s', e)
        time.sleep(5)


def load_result(d):
    try:
        return json.loads((d / 'result.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def today_counts(day=None):
    day = day or datetime.now().strftime('%Y%m%d')
    c = collections.Counter()
    for d in session_dirs(day):
        r = load_result(d)
        if r:
            c[r.get('status') or 'open'] += 1
    c['total'] = sum(c.values())
    return dict(c)


def last_upload_info():
    logp = LOG_DIR / 'nas_sync_infer.log'
    try:
        lines = logp.read_text(encoding='utf-8', errors='ignore').strip().splitlines()
        ok = [l for l in lines if ' OK ' in l]
        fail = [l for l in lines if ' FAIL ' in l]
        pending = sum(1 for d in session_dirs() if (d / 'result.json').exists() and not (d / '.uploaded').exists()
                      and (load_result(d) or {}).get('final'))
        return {'last_ok': ok[-1][:19] if ok else None, 'last_fail': fail[-1][:19] if fail else None, 'pending': pending}
    except OSError:
        return {'last_ok': None, 'last_fail': None, 'pending': None}


# ── Flask ────────────────────────────────────────────────

app = Flask(__name__)
monitor: Monitor | None = None
source = None


def _preview_jpeg(width=800):
    rgb = source.get_rgb() if source else None
    if rgb is None:
        return None
    h, w = rgb.shape[:2]
    sc = width / w
    img = cv2.resize(rgb, (width, int(h * sc)))
    snap = monitor.snapshot() if monitor else {}
    hr = monitor.last_hr if monitor and time.monotonic() - monitor.last_hr_t < 2.0 else None
    if hr:
        draw_points(img, hr['points'], sc)
    x0, y0, x1, y1 = monitor.gates.roi if monitor else (0, 0, 1, 1)
    cv2.rectangle(img, (int(width * x0), int(img.shape[0] * y0)), (int(width * x1), int(img.shape[0] * y1)), (80, 80, 80), 1)
    if monitor:                                        # 초록 = 점유(diff) 판정 영역 OCC_ROI (지그 중앙 빈 부분)
        ox0, oy0, ox1, oy1 = monitor.gates.occ_roi
        cv2.rectangle(img, (int(width * ox0), int(img.shape[0] * oy0)), (int(width * ox1), int(img.shape[0] * oy1)), (60, 200, 60), 1)
    band = np.zeros((44, width, 3), np.uint8); band[:] = (24, 24, 24)
    m = snap.get('metrics') or {}
    st = snap.get('state', '-')
    col = {'EMPTY': (160, 160, 160), 'JUDGING': (0, 200, 255), 'JUDGED': (0, 220, 0), 'UNRESOLVED': (0, 120, 255)}.get(st, (200, 200, 200))
    cv2.putText(band, f'{st}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    dtxt = '-' if m.get('diff') is None else f"{m['diff']:.3f}"
    zp = monitor.gates.z_plate if monitor else None
    txt = (f"z {m.get('z_med') or '-'} zo {m.get('z_occ') or '-'}/{round(zp) if zp else '-'}{' Z-EMPTY' if m.get('z_empty') else ''} "
           f"diff {dtxt} intr {m.get('intr', 0):.3f} "
           f"motion {m.get('motion', 0):.1f} pts {m.get('pts', 0)} "
           f"{'OCC' if m.get('occ') else 'empty'} {'static' if m.get('static') else 'moving'}"
           f"{' INTR' if m.get('intr_flag') else ''} | {snap.get('infer_ms', 0):.0f}ms")
    cv2.putText(band, txt, (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    img = np.vstack([img, band])
    ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return jpg.tobytes() if ok else None


@app.route('/')
def index():
    return render_template('factory_infer.html', cache_bust=int(time.time()), version=VERSION)


@app.route('/frame.jpg')
def frame_jpg():
    b = _preview_jpeg()
    if b is None:
        return Response(status=204)
    return Response(b, mimetype='image/jpeg')


@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            b = _preview_jpeg()
            if b:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b + b'\r\n'
            time.sleep(0.15)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    snap = monitor.snapshot()
    disk = shutil.disk_usage(str(DOOR_DIR))
    snap.update({
        'camera_ok': source.ok, 'camera_type': source.camera_type, 'camera_error': source.last_error,
        'serial': (source.calib or {}).get('serial'),
        'disk_free_gb': round(disk.free / 1e9, 1), 'disk_warn': disk.free / 1e9 < CFG['DISK_WARN_GB'],
        'today': today_counts(), 'upload': {'enabled': CFG['UPLOAD'], **last_upload_info()},
        'background': {'src': monitor.gates.bg_src, 'loaded': monitor.gates.bg is not None,
                       'z_plate': round(monitor.gates.z_plate) if monitor.gates.z_plate else None},
        'service': {'version': VERSION, 'started': now_str(monitor.started), 'models': monitor.models,
                    'shadow': not CFG['UPLOAD']},
        'now': datetime.now().strftime('%m/%d %H:%M:%S'),
    })
    return jsonify(snap)


@app.route('/api/config')
def api_config():
    return jsonify(CFG)


def _session_entry(d, r):
    day, sess = d.parent.name, d.name
    pred, radar = r.get('pred_class'), r.get('radar')
    cands = r.get('candidates') or []
    second = next((c['class'] for c in cands if c['class'] != pred), None)
    if second is None and r.get('cnn') and len(r['cnn'].get('top', [])) > 1:
        second = r['cnn']['top'][1]['class']
    # 잠정 판정이 갱신되면 thumb/프레임 파일이 같은 이름으로 바뀌므로 수정 시각을 버전으로 붙여 브라우저 캐시를 무효화한다
    thumb = d / 'thumb.jpg'
    tv = int(thumb.stat().st_mtime) if thumb.exists() else 0
    return dict(r, day=day, sess=sess,
                thumb_url=f'/thumb/{day}/{sess}?v={tv}' if tv else None,
                thumb_at=time.strftime('%H:%M:%S', time.localtime(tv)) if tv else None,
                frame_urls=[f'/view/{day}/{sess}/{f["file"]}?v={int((d / f["file"]).stat().st_mtime)}'
                            for f in r.get('frames', []) if (d / f['file']).exists()],
                ref_url=f'/ref/{pred}?radar={radar if radar is not None else ""}' if pred in CAD_D else None,
                second_class=second,
                ref2_url=f'/ref/{second}?radar={radar if radar is not None else ""}' if second in CAD_D else None,
                uploaded=(d / '.uploaded').exists())


@app.route('/api/sessions')
def api_sessions():
    day = request.args.get('date') or datetime.now().strftime('%Y%m%d')
    out = []
    for d in session_dirs(day):
        r = load_result(d)
        if r:
            out.append(_session_entry(d, r))
    out.sort(key=lambda x: x['sess'], reverse=True)
    return jsonify({'date': day, 'sessions': out, 'counts': today_counts(day)})


@app.route('/api/dates')
def api_dates():
    days = sorted({p.parent.name for p in OUTPUT_ROOT.glob('*/s_*') if p.is_dir()}, reverse=True)
    return jsonify([{'date': d, **today_counts(d)} for d in days[:60]])


@app.route('/api/metrics')
def api_metrics():
    day = request.args.get('date') or datetime.now().strftime('%Y%m%d')
    maxn = int(request.args.get('max', 1500))
    p = OUTPUT_ROOT / 'metrics' / f'metrics_{day}.jsonl'
    if not p.exists():
        return jsonify([])
    lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
    step = max(1, len(lines) // maxn)
    out = []
    for l in lines[::step]:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return jsonify(out)


def _safe_session_dir(day, sess):
    if not (day.replace('unsynced_', '').isdigit() and sess.startswith('s_') and sess[2:].isdigit()):
        abort(404)
    d = OUTPUT_ROOT / day / sess
    if not d.is_dir():
        abort(404)
    return d


@app.route('/thumb/<day>/<sess>')
def thumb(day, sess):
    p = _safe_session_dir(day, sess) / 'thumb.jpg'
    if not p.exists():
        abort(404)
    return send_file(p, mimetype='image/jpeg', max_age=3600)


@app.route('/view/<day>/<sess>/<name>')
def view_frame(day, sess, name):
    d = _safe_session_dir(day, sess)
    if not (name.startswith('rgb_') and name.endswith('.png')) or '/' in name:
        abort(404)
    p = d / name
    if not p.exists():
        abort(404)
    img = cv2.imread(str(p))
    if img is None:
        abort(404)
    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)))
    r = load_result(d) or {}
    f = next((f for f in r.get('frames', []) if f['file'] == name), None)
    if f and f.get('points'):
        draw_points(img, f['points'], img.shape[1] / w)
    ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpg.tobytes(), mimetype='image/jpeg')


@app.route('/ref/<cls>')
def ref_image(cls):
    radar = request.args.get('radar', '')
    radar = int(radar) if radar in ('0', '1') else None
    p = ref_image_path(cls, radar)
    if p is None:
        img = np.full((400, 640, 3), 40, np.uint8)
        cv2.putText(img, 'no reference image', (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
        cv2.putText(img, cls, (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)
        ok, jpg = cv2.imencode('.jpg', img)
        return Response(jpg.tobytes(), mimetype='image/jpeg')
    return send_file(p, mimetype='image/jpeg', max_age=3600)


@app.route('/api/timelapse')
def api_timelapse():
    """날짜별 타임랩스 목록 (index.jsonl). ?date=YYYYMMDD, ?from=HH:MM:SS&to=HH:MM:SS 로 세션 구간만."""
    day = request.args.get('date') or datetime.now().strftime('%Y%m%d')
    if not day.replace('unsynced_', '').isdigit():
        abort(404)
    p = OUTPUT_ROOT / 'timelapse' / day / 'index.jsonl'
    if not p.exists():
        return jsonify({'date': day, 'frames': []})
    t0, t1 = request.args.get('from'), request.args.get('to')
    out = []
    for l in p.read_text(encoding='utf-8', errors='ignore').splitlines():
        try:
            r = json.loads(l)
        except ValueError:
            continue
        hms = r['ts'][11:19]
        if (t0 and hms < t0) or (t1 and hms > t1):
            continue
        r['url'] = f"/tl/{day}/{r['file']}"
        bgf = f"bg_{r['file']}"
        if (p.parent / bgf).exists():
            r['bg_url'] = f'/tl/{day}/{bgf}'
        out.append(r)
    return jsonify({'date': day, 'frames': out})


@app.route('/tl/<day>/<name>')
def tl_image(day, name):
    if not day.replace('unsynced_', '').isdigit() or not re_tl.fullmatch(name):
        abort(404)
    p = OUTPUT_ROOT / 'timelapse' / day / name
    if not p.exists():
        abort(404)
    return send_file(p, mimetype='image/jpeg', max_age=86400)


@app.route('/api/reset', methods=['POST', 'GET'])
def api_reset():
    monitor.reset()
    return jsonify({'ok': True})


@app.route('/bg.jpg')
def bg_jpg():
    """[등록된 빈 지그 배경 | 지금 depth | 차이] 확인용 컬러 이미지 — 16비트 depth 는 뷰어에서 검게 보여 눈으로 못 본다.
    차이: 빨강=배경보다 가까움(도어·돌출), 파랑=멀어짐, 회색=같음(±BG_MM). 빈 지그인데 넓게 빨강/파랑이면 배경이 안 맞는 것."""
    g = monitor.gates
    bg, d4 = g.bg, g.last_d4
    if bg is None and d4 is None:
        return Response(status=204)
    img, _ = depth_view.render_compare(bg, d4, bg_mm=CFG['BG_MM'], roi=g.roi, occ_roi=g.occ_roi,
                                       titles=(f"BG: {g.bg_src or 'none'}", f"NOW {now_str()[11:]}", 'DIFF'))
    ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(jpg.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})


@app.route('/api/bg_learn', methods=['POST'])
def api_bg_learn():
    """지금 화면(빈 지그)을 배경 기준맵으로 등록. ?force=1 이면 홀·움직임 경고를 무시."""
    force = request.args.get('force') in ('1', 'true')
    r = monitor.learn_background(force=force)
    return jsonify(r), (200 if r.get('ok') else 409)


@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    log_event('shutdown')
    log.info('시스템 종료 요청(웹)')
    try:
        r = subprocess.run(['sudo', '-n', '/usr/sbin/shutdown', '-h', '+1'], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return jsonify({'error': f'종료 실패: {r.stderr.strip()}'}), 500
    except Exception as e:                              # noqa: BLE001
        return jsonify({'error': f'종료 실패: {e}'}), 500
    return jsonify({'ok': True, 'note': '1분 후 종료 (systemd 가 서비스를 먼저 내려 세션을 마감)'})


# ── main ─────────────────────────────────────────────────

def main():
    global monitor, source
    ap = argparse.ArgumentParser(description='경주 공장 도어 실시간 추론 서비스')
    ap.add_argument('--port', type=int, default=CFG['PORT'])
    ap.add_argument('--replay', type=str, default=None, help='"dir:sec,dir:sec" 폴더 재생 (카메라 대신)')
    ap.add_argument('--replay_fps', type=float, default=4.0)
    ap.add_argument('--no_cnn', action='store_true')
    ap.add_argument('--no_option', action='store_true')
    ap.add_argument('--no_cuda_probe', action='store_true')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    fh = logging.FileHandler(LOG_DIR / f'infer_{datetime.now():%Y%m%d}.log')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(fh)
    if not args.no_cuda_probe:
        require_cuda_ready()

    log.info('추론 서비스 시작 v%s port=%d upload=%s fp16=%s cfg=%s', VERSION, args.port, CFG['UPLOAD'], CFG['FP16'],
             {k: v for k, v in CFG.items() if k not in ('OUTPUT_ROOT',)})
    net, dev = hole_classifier.load_model(hole_classifier.MODEL_PATH, device=args.device)
    log.info('홀 판별기: %s 채널 %s', os.path.relpath(hole_classifier.MODEL_PATH, DOOR_DIR), getattr(net, 'channels', None))
    cnn = None
    if not args.no_cnn and os.path.exists(cnn_classifier.MODEL_PATH):
        cnn = cnn_classifier.CNNClassifier(cnn_classifier.MODEL_PATH, device=args.device)
        log.info('CNN 폴백: %s (%d클래스)', os.path.relpath(cnn_classifier.MODEL_PATH, DOOR_DIR), len(cnn.class_names))
    geom = None if args.no_option else rc.load_geometry()

    source = ReplaySource(args.replay, args.replay_fps) if args.replay else FactoryCamera()
    source.start()
    if isinstance(source, FactoryCamera):
        for _ in range(600):                           # 카메라 첫 프레임 대기(최대 60s) — 실패해도 서비스는 뜸
            if source.ok and source.get_rgb() is not None:
                break
            time.sleep(0.1)
        sn = (source.calib or {}).get('serial')
        if sn not in hole_classifier.S_PIXEL:
            log.warning('카메라 시리얼 %s 의 S_PIXEL 미등록 → depth 스케일로 D 계산 (정밀도 저하)', sn)
    log.info('카메라: %s calib=%s', source.camera_type, source.calib)

    monitor = Monitor(source, CFG, net, dev, cnn, geom)
    monitor.start()
    threading.Thread(target=housekeeping_loop, args=(monitor,), daemon=True, name='housekeeping').start()
    log_event('service_start', version=VERSION, upload=CFG['UPLOAD'], replay=bool(args.replay))

    def _on_signal(signum, frame):
        if monitor.stop_event.is_set():
            os._exit(1)
        monitor.stop_event.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        log.info('종료 중: 모니터 정지(세션 마감) → 카메라 닫기')
        monitor.stop_event.set()
        monitor.join(timeout=15)
        try:
            monitor.gates.save_background()
        except Exception as e:                          # noqa: BLE001
            log.warning('배경 기준맵 저장 실패: %s', e)
        source.stop()
        log_event('service_stop')
        log.info('종료 완료')


if __name__ == '__main__':
    main()
