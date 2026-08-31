"""도어 6DoF 포즈 솔버 — 홀 랜드마크 + depth 평면 + CAD 강체 정합 (추론 모듈).

출력 T_cam→door: 카메라 광학 좌표계(OpenCV: x우/y하/z전방, mm) 기준 도어 로컬 프레임
(01_extract_cad_holes 정의: 원점=코너 홀 중점, X=힌지→래치, Z=법선(카메라 쪽), Y=Z×X).

프레임당 절차:
  1) hole_classifier.detect → 랜드마크 픽셀 (볼트≤4 + 코너 힌지/래치)
  2) depth 평면 피팅 → 코너는 평면상, 볼트는 CAD z_door 오프셋(리세스 ≈ -6mm)만큼
     평면을 밀어 레이 교차 (01 공면성 실측 반영)
  3) 볼트 대응: 코너 2점 유사변환으로 CAD 볼트를 이미지에 사상 → 상호 최근접
  4) Umeyama 강체 정합(도어 좌표 → 카메라 3D) → T + 홀별 잔차(mm)
분해 리포트: x/y/z(mm) = 도어 원점 위치, theta_deg = 이미지 면내 회전(도어 X축 vs 카메라 x축),
tilt_deg = 도어 법선 vs 카메라 광축 정대면 이탈각.
집계: aggregate_poses — 병진 중앙값 + 회전 쿼터니언 평균 + 세션 분산(반복 정밀도).

지표 체계(정의 2026-08-31): 잔차 RMS = 정합 정확도(우선), 세션 std = 반복 정밀도,
절대 정확도는 합성/실측(레이저 트래커) 검증으로만 주장.
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.normpath(os.path.join(BASE, '..', '..', 'class_estimation', 'door_pipeline'))
for p in (BASE, DOOR):
    if p not in sys.path:
        sys.path.insert(0, p)
import hole_classifier as hc
import pose_utils as pu

CAD_JSON = os.path.join(BASE, 'cad_holes.json')
BOLT_NAMES = ['bolt_tl', 'bolt_tr', 'bolt_bl', 'bolt_br']
LM6 = BOLT_NAMES + ['corner_hinge', 'corner_latch']
RMS_GATE = 15.0   # 정합 잔차 게이트(mm) — 초과 시 보류 권고


def load_cad(path=CAD_JSON):
    """cad_holes.json → {클래스: {'pts': {이름: 도어좌표 3D}, 'T_door_to_cad': 4x4}}."""
    d = json.load(open(path))
    out = {}
    for cls, e in d['classes'].items():
        out[cls] = dict(pts={k: np.array(v, float) for k, v in e['holes_door'].items()},
                        T_door_to_cad=np.array(e['T_door_to_cad'], float))
    return out


def correspond_bolts(det_bolts, hinge, latch, cad_pts, max_mm=25.0):
    """코너 2점 유사변환으로 CAD 볼트를 이미지에 사상 → 상호 최근접 대응.

    도어 프레임은 Y가 위(카메라 시점), 이미지 y는 아래 → (x, -y)로 사상.
    반환: {볼트이름: (x, y)} (대응 성공분만)"""
    C = lambda p: complex(p[0], -p[1])                   # 도어 (x,y) → 손지향 일치 2D
    hC, lC = C(cad_pts['corner_hinge']), C(cad_pts['corner_latch'])
    hI, lI = complex(*hinge[:2]), complex(*latch[:2])
    s = (lI - hI) / (lC - hC)
    mapped = {k: hI + s * (C(cad_pts[k]) - hC) for k in BOLT_NAMES}
    det = [complex(b[0], b[1]) for b in det_bolts]
    lim = max_mm * abs(s)
    pairs = sorted(((abs(mapped[k] - z), k, i) for k in mapped for i, z in enumerate(det)),
                   key=lambda t: t[0])
    used_k, used_i, out = set(), set(), {}
    for dist, k, i in pairs:
        if dist > lim or k in used_k or i in used_i:
            continue
        used_k.add(k); used_i.add(i)
        out[k] = (det_bolts[i][0], det_bolts[i][1])
    return out


def solve_frame(det, depth, cls, cad, K=None, min_holes=4):
    """검출 결과 + depth → T_cam→door. 실패 시 dict(ok=False, reason=...).

    det: hole_classifier.detect 출력. cad: load_cad()[cls] 자리엔 전체 dict을 주고 cls로 선택."""
    e = cad.get(cls)
    if e is None:
        return dict(ok=False, reason='no_cad')
    hinge = det['corner_hinge'][0] if det['corner_hinge'] else None
    latch = det['corner_latch'][0] if det['corner_latch'] else None
    if hinge is None or latch is None:
        return dict(ok=False, reason='no_corner')
    K = K or pu.intrinsics_for(depth.shape)
    named_px = dict(corner_hinge=hinge[:2], corner_latch=latch[:2])
    named_px.update(correspond_bolts(det['bolt'], hinge, latch, e['pts']))
    if len(named_px) < min_holes:
        return dict(ok=False, reason='few_holes', n_holes=len(named_px))
    pl = pu.fit_plane(depth, [det_pt for det_pt in named_px.values()], K)
    if pl is None:
        return dict(ok=False, reason='no_plane', n_holes=len(named_px))
    c, n, pstats = pl
    offsets = {k: float(e['pts'][k][2]) for k in named_px}       # 도어 z = 평면 오프셋(카메라 쪽 +)
    P3 = pu.landmarks_3d(named_px, c, n, K, offsets, depth=depth)
    if P3 is None:
        return dict(ok=False, reason='backproject', n_holes=len(named_px))
    names = sorted(named_px)
    A = np.stack([e['pts'][k] for k in names])                   # 도어 좌표
    B = np.stack([P3[k] for k in names])                         # 카메라 좌표(mm)
    R, t, rms, res = pu.umeyama_rigid(A, B)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    theta = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    tilt = float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))
    return dict(ok=True, T=T, R=R, t=t, rms=float(rms),
                res={k: float(v) for k, v in zip(names, res)},
                n_holes=len(names), theta_deg=theta, tilt_deg=tilt,
                normal_ok=bool(R[2, 2] < 0), plane=pstats, gate=(rms <= RMS_GATE))


def solve_frame_pnp(det, shape, cls, cad, K=None, min_holes=4):
    """PnP 대조군: depth 미사용, 2D 검출점 + CAD 3D만으로 포즈 (cv2.solvePnP SQPNP).

    depth 경로와 오차 원인이 독립(렌즈 파라미터만 의존) — 교차 검증·depth 편향 진단용."""
    import cv2
    e = cad.get(cls)
    hinge = det['corner_hinge'][0] if det['corner_hinge'] else None
    latch = det['corner_latch'][0] if det['corner_latch'] else None
    if e is None or hinge is None or latch is None:
        return dict(ok=False, reason='no_corner')
    K = K or pu.intrinsics_for(shape)
    named_px = dict(corner_hinge=hinge[:2], corner_latch=latch[:2])
    named_px.update(correspond_bolts(det['bolt'], hinge, latch, e['pts']))
    if len(named_px) < min_holes:
        return dict(ok=False, reason='few_holes', n_holes=len(named_px))
    names = sorted(named_px)
    obj = np.stack([e['pts'][k] for k in names]).astype(np.float64)
    img = np.array([named_px[k] for k in names], np.float64)
    Km = np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1.0]])
    ok, rvec, tvec = cv2.solvePnP(obj, img, Km, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return dict(ok=False, reason='pnp_fail', n_holes=len(names))
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, Km, None)
    rms_px = float(np.sqrt(np.mean(np.sum((proj[:, 0] - img) ** 2, 1))))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return dict(ok=True, T=T, R=R, t=t, rms_px=rms_px, n_holes=len(names),
                theta_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                tilt_deg=float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))),
                normal_ok=bool(R[2, 2] < 0), gate=(rms_px <= 4.0))


def solve(net, dev, rgb, depth, cls, cad, K=None, pnp=False):
    """편의 함수: 검출부터 포즈까지. hole_classifier.classify 게이트와 독립."""
    det = hc.detect(net, dev, rgb)
    out = solve_frame(det, depth, cls, cad, K)
    out['det'] = det
    if pnp:
        out['pnp'] = solve_frame_pnp(det, rgb.shape, cls, cad, K)
    return out


def aggregate_poses(frames):
    """게이트 통과 프레임 집계: 병진 중앙값 + 쿼터니언 평균, 세션 분산(반복 정밀도).

    반환 stats 단위: mm/deg. 유효 프레임 없으면 ok=False."""
    ok = [f for f in frames if f.get('ok') and f.get('gate')]
    if not ok:
        return dict(ok=False, n=len(frames), n_used=0)
    from scipy.spatial.transform import Rotation
    ts = np.stack([f['t'] for f in ok])
    Rm = Rotation.from_matrix(np.stack([f['R'] for f in ok]))
    R_avg = Rm.mean().as_matrix()
    t_avg = np.median(ts, 0)
    T = np.eye(4); T[:3, :3] = R_avg; T[:3, 3] = t_avg
    th = np.array([f['theta_deg'] for f in ok]); ti = np.array([f['tilt_deg'] for f in ok])
    rot_dev = Rm * Rm.mean().inv()
    ang = np.degrees(rot_dev.magnitude())
    return dict(ok=True, n=len(frames), n_used=len(ok), T=T,
                t=t_avg.tolist(), theta_deg=float(np.median(th)), tilt_deg=float(np.median(ti)),
                rms_med=float(np.median([f['rms'] for f in ok])),
                std=dict(x=float(ts[:, 0].std()), y=float(ts[:, 1].std()), z=float(ts[:, 2].std()),
                         theta=float(th.std()), tilt=float(ti.std()),
                         rot_deg=float(ang.std()) if len(ang) > 1 else 0.0))
