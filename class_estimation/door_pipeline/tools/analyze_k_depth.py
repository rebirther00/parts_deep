"""K_DEPTH 재유도 분석 — 현장 전 프레임에서 legacy(fx=1065×K_DEPTH) vs 실측 intrinsics 경로의 D 를 비교하고
잔차 스케일 K 를 세션 풀에서 다시 맞춘다.

    python tools/analyze_k_depth.py                       # datasets_factory_v2/all → report/hole_analysis/k_depth_20260916/
    python tools/analyze_k_depth.py --base datasets_factory_v2/test

기록(프레임별): cls, session, gate, D_legacy(mm, 기존 판정값), D_real_raw(mm, 실측 intrinsics 역투영·K=1), z(mm, 코너 홀 중점 평면 깊이),
                tilt(deg, 평면 법선 vs 광축)
분석: 클래스·세션별 편차, 전역 K 적합(세션 중앙값 기준), z·기울기 의존성, FRT 최소 마진.
"""
import argparse, glob, json, os, sys, time, collections
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hole_classifier as hc
from hole_classifier import CAD_D, GROUP

DOOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from camera_utils import KNOWN_CAMERAS
FIELD_CAM = KNOWN_CAMERAS[54910212]
ap = argparse.ArgumentParser()
ap.add_argument('--base', default='datasets_factory_v2/all')
ap.add_argument('--out', default='report/hole_analysis/k_depth_20260916')
args = ap.parse_args()
out = os.path.join(DOOR, args.out); os.makedirs(out, exist_ok=True)


def session_of(f):
    n = os.path.basename(f)          # rgb_20260827_s_123607_0000.png
    p = n.split('_'); return p[1] + '_' + p[2] + '_' + p[3]


def measure(depth, pts_all, hinge, latch, intrinsics):
    pf = hc.plane_from_depth(depth, pts_all, intrinsics)
    if pf is None: return None
    to3d, k = pf
    H, L = to3d(hinge), to3d(latch)
    mid = (H + L) / 2
    # 평면 법선: to3d 는 클로저라 법선을 직접 못 받음 → 세 점으로 재계산
    B = to3d(pts_all[0]); n = np.cross(L - H, B - H); n /= (np.linalg.norm(n) + 1e-9)
    tilt = float(np.degrees(np.arccos(abs(n[2]))))
    return float(np.linalg.norm(H - L)), float(mid[2]), tilt, k


rows = []
if __name__ == '__main__':
    net, dev = hc.load_model()
    files = sorted(glob.glob(os.path.join(DOOR, args.base, '*', 'rgb_*.png')))
    t0 = time.time()
    for i, f in enumerate(files):
        cls = os.path.basename(os.path.dirname(f))
        rgb = cv2.imread(f); depth = cv2.imread(f.replace('rgb_', 'depth_'), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None: continue
        det = hc.detect(net, dev, rgb)
        hinge = det['corner_hinge'][0] if det['corner_hinge'] else None
        latch = det['corner_latch'][0] if det['corner_latch'] else None
        gate = hc.geometry_gate(hc.bolt_frame(det['bolt']), hinge, latch, rgb.shape)
        r = dict(image=os.path.relpath(f, DOOR), cls=cls, session=session_of(f), gate=gate)
        if gate == 'ok':
            pts = det['bolt'] + [hinge, latch]
            a = measure(depth, pts, hinge, latch, None)
            b = measure(depth, pts, hinge, latch, FIELD_CAM)
            if a and b:
                r.update(D_legacy=a[0] * a[3], D_real_raw=b[0], z=b[1], tilt=b[2], k_legacy=a[3], k_metric=b[3])
        rows.append(r)
        if i % 200 == 0: print(f'{i}/{len(files)} {time.time() - t0:.0f}s', flush=True)
    json.dump(rows, open(os.path.join(out, 'frames.json'), 'w'), indent=0)
    print('저장', os.path.join(out, 'frames.json'), len(rows))
