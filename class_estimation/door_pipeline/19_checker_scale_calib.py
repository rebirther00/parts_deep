"""체커보드로 depth 스케일(잔차 K)·intrinsics 교차 검증 / 카메라별 캘리브레이션.

원리: 체커 코너(간격 --square mm 기지)를 검출 → 코너 영역 depth 로 평면 피팅 → 실제 intrinsics 로
코너를 평면에 역투영 → 인접 코너 3D 거리를 잰다. hole_classifier 의 볼트 피치 정규화·D 계산과
동일한 스케일 체인(depth 평면 + intrinsics)이므로, 여기서 나온 배율이 곧 이 카메라의 잔차 K 다.

    python 19_checker_scale_calib.py --square 25            # 카메라에서 20프레임 측정
    python 19_checker_scale_calib.py --square 25 --pattern 9x6 --frames 30 --out /tmp/calib

출력: 축별 코너 간격(mm), 배율 K=square/측정, 보드 거리·기울기, 픽셀 피치로 본 fx 교차검증,
      코너를 그린 스냅샷 png. 카메라는 06/18 과 같은 설정(HD1200, 15fps)으로 연다.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hole_classifier as hc                                   # noqa: E402

PATTERNS = [(9, 6), (8, 6), (7, 6), (10, 7), (11, 8), (7, 5), (8, 5), (6, 5),
            (12, 9), (6, 4), (5, 4), (4, 3), (9, 7), (10, 6)]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--square', type=float, required=True, help='체커 한 칸 크기(mm)')
ap.add_argument('--pattern', type=str, default=None, help='내부 코너 수 CxR (예 9x6). 생략 시 자동 탐색')
ap.add_argument('--frames', type=int, default=20, help='측정 프레임 수 (기본 20)')
ap.add_argument('--out', type=str, default=None, help='스냅샷·결과 json 저장 폴더')
ap.add_argument('--replay', type=str, default=None,
                help='카메라 대신 rgb png 경로(+ 같은 이름의 depth_*.png, intrinsics json 필요)')
args = ap.parse_args()


def find_board(gray, pattern=None):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    for pat in ([pattern] if pattern else PATTERNS):
        ok, corners = cv2.findChessboardCornersSB(gray, pat, flags=flags)
        if ok and corners is not None and len(corners) == pat[0] * pat[1]:
            return pat, corners.reshape(pat[1], pat[0], 2)       # [row, col, xy]
    return None, None


def fit_plane(depth, corners_xy, K):
    """코너 볼록껍질 내부 depth → 평면. hole_classifier.plane_from_depth 와 같은 절차(잔차 K=1)."""
    h, w = depth.shape
    m = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(corners_xy.reshape(-1, 2).astype(np.float32)).astype(np.int32), 255)
    m = cv2.erode(m, np.ones((7, 7), np.uint8))
    rows, cols = np.where((m > 0) & (depth > 0))
    if len(rows) < 300:
        return None
    if len(rows) > 40000:
        sel = np.random.default_rng(0).choice(len(rows), 40000, replace=False)
        rows, cols = rows[sel], cols[sel]
    z = depth[rows, cols].astype(np.float64)
    Q = np.stack([(cols - K['cx']) * z / K['fx'], (rows - K['cy']) * z / K['fy'], z], 1)
    c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c, full_matrices=False); n = Vt[2]
    keep = np.abs((Q - c) @ n) < 10
    if keep.sum() > 200:
        c = Q[keep].mean(0); _, _, Vt = np.linalg.svd(Q[keep] - c, full_matrices=False); n = Vt[2]
    resid = float(np.std((Q[keep] - c) @ n)) if keep.sum() > 2 else float('nan')
    if n[2] < 0:
        n = -n

    def to3d(p):
        r = np.array([(p[0] - K['cx']) / K['fx'], (p[1] - K['cy']) / K['fy'], 1.0])
        return r * (np.dot(c, n) / np.dot(r, n))
    return dict(c=c, n=n, to3d=to3d, resid_mm=resid, inlier=float(keep.mean()), z_med=float(np.median(z)))


def measure(rgb, depth, K):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    pat, C = find_board(gray, PATTERN)
    if C is None:
        return None
    pl = fit_plane(depth, C, K)
    if pl is None:
        return None
    R, Cn = C.shape[:2]
    P = np.array([[pl['to3d'](C[r, c]) for c in range(Cn)] for r in range(R)])   # [R, Cn, 3]
    d_col = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=2).ravel()      # 가로 인접 간격
    d_row = np.linalg.norm(P[1:] - P[:-1], axis=2).ravel()            # 세로 인접 간격
    span_c = np.linalg.norm(P[:, -1] - P[:, 0], axis=1) / (Cn - 1)    # 행 전체 스팬 / 칸 수 (노이즈 작음)
    span_r = np.linalg.norm(P[-1] - P[0], axis=1) / (R - 1)
    px_c = np.linalg.norm(C[:, 1:] - C[:, :-1], axis=2).mean()
    px_r = np.linalg.norm(C[1:] - C[:-1], axis=2).mean()
    tilt = float(np.degrees(np.arccos(min(1.0, abs(pl['n'][2])))))
    z_c = float(pl['to3d'](C.reshape(-1, 2).mean(0))[2])
    return dict(pattern=pat, corners=C,
                col_mm=float(np.mean(d_col)), row_mm=float(np.mean(d_row)),
                col_sd=float(np.std(d_col)), row_sd=float(np.std(d_row)),
                span_col_mm=float(np.mean(span_c)), span_row_mm=float(np.mean(span_r)),
                px_col=float(px_c), px_row=float(px_r), z_center=z_c, tilt_deg=tilt,
                plane_resid_mm=pl['resid_mm'], plane_inlier=pl['inlier'])


PATTERN = None
if args.pattern:
    c, r = args.pattern.lower().split('x'); PATTERN = (int(c), int(r))

# ── 프레임 소스 ──────────────────────────────────────────
if args.replay:
    rgb0 = cv2.imread(args.replay)
    dp = args.replay.replace('rgb_', 'depth_')
    depth0 = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32)
    with open(os.path.splitext(args.replay)[0] + '_intrinsics.json') as f:
        K = json.load(f)
    frames = [(rgb0, depth0)] * 1
    cam = None
else:
    from camera_utils import CameraManager
    cam = CameraManager(fps=15)
    cam.start()
    K = cam.calib
    print(f'카메라: {cam.camera_type}')
    if not K:
        cam.stop(); raise SystemExit('intrinsics 를 읽지 못했습니다')
    print(f"intrinsics fx={K['fx']:.2f} fy={K['fy']:.2f} cx={K['cx']:.1f} cy={K['cy']:.1f} hfov={K['h_fov']:.1f}°")
    t0 = time.time()
    while cam.get_depth() is None and time.time() - t0 < 15:
        time.sleep(0.2)
    frames = None

results, snap = [], None
try:
    i = 0
    n_fail = 0
    while len(results) < args.frames:
        if frames is not None:
            if i >= len(frames):
                break
            rgb, depth = frames[i]
        else:
            rgb, depth = cam.get_frame(), cam.get_depth()
            if rgb is None or depth is None:
                time.sleep(0.1); continue
        i += 1
        m = measure(rgb, depth, K)
        if m is None:
            n_fail += 1
            if n_fail in (5, 20, 50):
                print(f'  체커 검출/평면 실패 {n_fail}회 — 보드가 화면 안에, 적당한 거리로 있는지 확인')
            if n_fail >= 100:
                break
            time.sleep(0.1); continue
        results.append(m)
        if snap is None:
            snap = rgb.copy()
            cv2.drawChessboardCorners(snap, m['pattern'], m['corners'].reshape(-1, 1, 2).astype(np.float32), True)
        print(f"  [{len(results):2d}] {m['pattern'][0]}x{m['pattern'][1]} z={m['z_center']:.0f}mm tilt={m['tilt_deg']:.1f}° "
              f"가로 {m['col_mm']:.3f}±{m['col_sd']:.3f} 세로 {m['row_mm']:.3f}±{m['row_sd']:.3f} mm "
              f"(스팬 {m['span_col_mm']:.3f}/{m['span_row_mm']:.3f}) 평면잔차 {m['plane_resid_mm']:.1f}mm")
        if frames is None:
            time.sleep(1.0 / 15)
finally:
    if cam is not None:
        cam.stop()

if not results:
    raise SystemExit('측정 실패: 체커보드를 찾지 못했습니다 (--pattern 으로 코너 수 지정 가능)')

S = args.square
med = {k: float(np.median([r[k] for r in results])) for k in
       ('col_mm', 'row_mm', 'span_col_mm', 'span_row_mm', 'px_col', 'px_row', 'z_center', 'tilt_deg', 'plane_resid_mm')}
k_col, k_row = S / med['span_col_mm'], S / med['span_row_mm']
k_all = S / ((med['span_col_mm'] + med['span_row_mm']) / 2)
# 픽셀 피치 교차검증: 정면(tilt≈0)이면 fx ≈ px_pitch × z / S. depth 의 z 를 믿을 때의 fx 추정치.
fx_from_px = med['px_col'] * med['z_center'] / S
fy_from_px = med['px_row'] * med['z_center'] / S
print()
print('══ 결과 (프레임 중앙값, n=%d, 패턴 %dx%d, 칸 %.1fmm) ══' % (len(results), *results[0]['pattern'], S))
print(f"  보드 거리 {med['z_center']:.0f}mm, 기울기 {med['tilt_deg']:.1f}°, 평면 잔차 {med['plane_resid_mm']:.2f}mm")
print(f"  코너 간격  가로 {med['span_col_mm']:.3f}mm ({100*(med['span_col_mm']/S-1):+.2f}%)  "
      f"세로 {med['span_row_mm']:.3f}mm ({100*(med['span_row_mm']/S-1):+.2f}%)")
print(f"  → 이 카메라 잔차 K = {k_all:.4f}  (가로 {k_col:.4f} / 세로 {k_row:.4f})")
print(f"     현재 hole_classifier K_METRIC[{K['height']}] = {hc.K_METRIC.get(K['height'], float('nan')):.4f}  "
      f"(현장 카메라 역산값; 볼트 정규화 ON 이면 D 에 미영향)")
print(f"  픽셀 피치 교차검증(정면 가정): fx≈{fx_from_px:.1f} fy≈{fy_from_px:.1f}  vs 캘리브 fx={K['fx']:.1f} fy={K['fy']:.1f} "
      f"→ depth z 스케일 {100*(fx_from_px/K['fx']-1):+.2f}% / {100*(fy_from_px/K['fy']-1):+.2f}%")
print('  해석: 잔차 K 가 1 에서 벗어난 만큼이 depth 스케일 편향(NEURAL depth 절대값). 볼트 정규화는 이를 프레임마다 상쇄한다.')

out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib_out')
os.makedirs(out, exist_ok=True)
tag = time.strftime('%Y%m%d_%H%M%S')
if snap is not None:
    cv2.putText(snap, f"K={k_all:.4f} col {med['span_col_mm']:.2f} row {med['span_row_mm']:.2f} mm z={med['z_center']:.0f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out, f'checker_{tag}.png'), snap)
with open(os.path.join(out, f'checker_{tag}.json'), 'w') as f:
    json.dump(dict(square_mm=S, n=len(results), pattern=results[0]['pattern'], intrinsics=K,
                   median=med, K_col=k_col, K_row=k_row, K=k_all,
                   fx_from_px=fx_from_px, fy_from_px=fy_from_px,
                   per_frame=[{k: v for k, v in r.items() if k != 'corners'} for r in results]), f,
              ensure_ascii=False, indent=1)
print(f'  저장: {out}/checker_{tag}.png, .json')
