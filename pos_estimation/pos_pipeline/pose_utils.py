"""포즈 파이프라인 공용 기하 유틸 (01 CAD 검증 · 02 솔버 공용).

- intrinsics: 실측 캘리브레이션 우선(KNOWN_CAMERAS 시리얼 키 또는 세션 meta 'intrinsics';
  06_factory_capture가 2026-08-31부터 기록) + depth_scale로 depth 절대 편향 보정.
  실측이 없으면 fx=1065 근사 + K_DEPTH 폴백(hole_classifier 규약) — 이 경로의 절대 위치는
  K_DEPTH배 압축이므로 상대 기하 전용.
- fit_plane: 검출점 볼록껍질 내부 depth SVD 평면(10mm inlier 재피팅), 법선은 카메라 쪽(-Z).
- backproject: 픽셀 레이 × (법선 방향 offset 적용) 평면 교점. 홀별 리세스 보정용.
- landmarks_3d: 랜드마크 픽셀 → 카메라 좌표 3D(mm, K_DEPTH 보정 포함).
- umeyama_rigid: 3D-3D 강체 정합(스케일 고정) + 잔차.
- uw_features: 볼트 4점 장축 기준 코너 홀 (ul, wl, uh, wh, D) — 부호 정준화(ul>0, wl>0),
  관측면 무관. CAD/실측 비교용.
"""
import os
import sys

import cv2
import numpy as np

DOOR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..', 'class_estimation', 'door_pipeline'))
if DOOR not in sys.path:
    sys.path.insert(0, DOOR)
from hole_classifier import K_DEPTH  # 해상도(세로)별 근사 intrinsics 편향 보정


# 실측 캘리브레이션 (SDK rectified, 세션 meta.json 'intrinsics'와 동일 형식).
# depth_scale 유도: K_DEPTH는 '근사 fx=1065 역투영 × K_DEPTH = CAD 실거리'로 현장 캘리브레이션
# 되었으므로 z_true = z_meas × K_DEPTH[h]·fx_real/1065. 좌우(상대) 거리는 legacy 경로와 동일,
# 절대 위치 t만 물리 스케일로 복원된다 (legacy 경로의 t는 K_DEPTH배 압축 — 상대 기하 전용).
KNOWN_CAMERAS = {
    54910212: dict(width=1920, height=1200, fx=1269.746, fy=1269.746,
                   cx=961.669, cy=598.594,
                   note='현장 ZED X Mini, SDK rectified, 2026-08-31 확보'),
}


def intrinsics_for(shape, calib=None):
    """K dict 구성. calib(세션 meta 'intrinsics' dict) 지정 시 그 값을, 없으면 해상도가
    KNOWN_CAMERAS와 일치할 때 실측 값을 사용(k=1, depth_scale 보정). 그 외 근사 폴백."""
    h, w = shape[:2]
    if calib is None:
        calib = next((c for c in KNOWN_CAMERAS.values()
                      if c['width'] == w and c['height'] == h), None)
    if calib:
        ds = calib.get('depth_scale')                 # 합성 등 무편향 depth는 1.0 명시
        if ds is None:
            ds = K_DEPTH[h] * float(calib['fx']) / 1065.0 if h in K_DEPTH else 1.0
        return dict(fx=float(calib['fx']), fy=float(calib['fy']),
                    cx=float(calib['cx']), cy=float(calib['cy']),
                    k=1.0, depth_scale=ds)
    return dict(fx=1065.0, fy=1065.0, cx=w / 2.0, cy=h / 2.0,
                k=K_DEPTH.get(h, K_DEPTH[1080]), depth_scale=1.0)


def _ring_z(depth, x, y, r_in=6, r_out=16, min_px=20):
    """(x,y) 주변 링(홀 내부 제외)의 유효 depth 중앙값 (무보정 mm) — 홀을 품은 국소 면."""
    h, w = depth.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r_out), min(w, xi + r_out + 1)
    y0, y1 = max(0, yi - r_out), min(h, yi + r_out + 1)
    sub = depth[y0:y1, x0:x1].astype(np.float64)
    gy, gx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(gx - x, gy - y)
    vals = sub[(rr >= r_in) & (rr <= r_out) & (sub > 0)]
    return float(np.median(vals)) if len(vals) >= min_px else None


def _ring_seed(depth, pts, K):
    """랜드마크 링 depth들 → 시드 평면 (c, n).

    벤트 관통 배경·돌출 부품이 섞인 다봉 depth에서 '홀을 품은 면'으로 초기화."""
    ds = K.get('depth_scale', 1.0)
    P3 = []
    for x, y in pts:
        z = _ring_z(depth, x, y)
        if z is None:
            continue
        z *= ds
        P3.append([(x - K['cx']) * z / K['fx'], (y - K['cy']) * z / K['fy'], z])
    if len(P3) < 3:
        return None
    P3 = np.array(P3)
    c = P3.mean(0)
    return c, np.linalg.svd(P3 - c, full_matrices=False)[2][2]


def fit_plane(depth, pts, K):
    """검출점 볼록껍질 내부 유효 depth로 평면 피팅 (링 시드 → ±15 → ±10 inlier 재피팅).

    반환: (c, n, stats) — 좌표는 depth_scale 보정된 카메라 mm, n은 카메라 쪽(nz<0)."""
    P = np.array([(p[0], p[1]) for p in pts], np.float32)
    if len(P) < 3:
        return None
    m = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(P).astype(np.int32), 255)
    m = cv2.erode(m, np.ones((15, 15), np.uint8))
    rows, cols = np.where((m > 0) & (depth > 0))
    if len(rows) < 500:
        return None
    if len(rows) > 40000:
        sel = np.random.default_rng(0).choice(len(rows), 40000, replace=False)
        rows, cols = rows[sel], cols[sel]
    z = depth[rows, cols].astype(np.float64) * K.get('depth_scale', 1.0)
    Q = np.stack([(cols - K['cx']) * z / K['fx'], (rows - K['cy']) * z / K['fy'], z], 1)
    seed = _ring_seed(depth, [(p[0], p[1]) for p in pts], K)
    if seed is not None:
        c, n = seed
        keep = np.abs((Q - c) @ n) < 15
        if keep.sum() > 300:
            c = Q[keep].mean(0)
            n = np.linalg.svd(Q[keep] - c, full_matrices=False)[2][2]
    else:
        c = Q.mean(0)
        n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
    keep = np.abs((Q - c) @ n) < 10
    if keep.sum() > 200:
        c = Q[keep].mean(0)
        n = np.linalg.svd(Q[keep] - c, full_matrices=False)[2][2]
    if n[2] > 0:
        n = -n
    d = (Q - c) @ n
    inl = np.abs(d) < 10
    stats = dict(inlier_rms=float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else None,
                 n_px=int(len(rows)), inlier_ratio=float(inl.mean()), seeded=seed is not None)
    return c, n, stats


def backproject(p, c, n, K, offset=0.0):
    """픽셀 p 레이와 (c + offset·n, n) 평면 교점 (offset: 카메라 쪽 +, 무보정 mm)."""
    r = np.array([(p[0] - K['cx']) / K['fx'], (p[1] - K['cy']) / K['fy'], 1.0])
    c2 = c + offset * n
    denom = np.dot(r, n)
    if abs(denom) < 1e-9:
        return None
    return r * (np.dot(c2, n) / denom)


def landmarks_3d(pts_px, c, n, K, offsets=None, depth=None, gate_mm=15.0):
    """{이름: (x,y)px} → {이름: 3D mm(스케일 보정)}.

    depth 지정 시 홀별 링 국소 depth(홀을 품은 실제 면)를 우선 사용 — 전역 평면과의
    z 차이가 gate_mm 이내일 때만. 아니면 평면+offsets(CAD z_door, 카메라 쪽 +) 폴백.
    국소 depth가 전역 평면의 오프셋·기울기 편향(합성 검증에서 dz~11mm/3° 확인)을 제거한다."""
    ds, k = K.get('depth_scale', 1.0), K['k']
    out = {}
    for name, p in pts_px.items():
        off = (offsets.get(name, 0.0) if offsets else 0.0) / k
        q = backproject(p, c, n, K, off)
        if q is None:
            return None
        if depth is not None:
            zl = _ring_z(depth, p[0], p[1])
            if zl is not None:
                zl *= ds
                if abs(zl - q[2]) < gate_mm:
                    q = np.array([(p[0] - K['cx']) / K['fx'],
                                  (p[1] - K['cy']) / K['fy'], 1.0]) * zl
        out[name] = q * k
    return out


def umeyama_rigid(A, B):
    """A(N,3)→B(N,3) 강체 정합(스케일 고정). 반환: R, t, rms, res(N)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    res = np.linalg.norm(A @ R.T + t - B, axis=1)
    return R, t, float(np.sqrt(np.mean(res ** 2))), res


def uw_features(hinge, latch, bolts, n_cam):
    """볼트 4점(4,3) 장축 기준 코너 홀 상대 기하 [ul, wl, uh, wh, D] (mm).

    n_cam = 카메라를 향하는 대략적 법선. 손지향 보존(ey = n×ex, ul>0 정준화만) —
    관측면이 틀리면 wl/wh 부호가 뒤집혀 큰 오차로 드러난다. CAD/실측 공용."""
    hinge = np.asarray(hinge, float); latch = np.asarray(latch, float)
    bolts = np.asarray(bolts, float)
    cq = bolts.mean(0)
    P = np.vstack([bolts, hinge[None], latch[None]])
    c = P.mean(0)
    nrm = np.linalg.svd(P - c, full_matrices=False)[2][2]
    if np.dot(nrm, np.asarray(n_cam, float)) < 0:
        nrm = -nrm
    B = bolts - cq
    Bp = B - np.outer(B @ nrm, nrm)
    ex = np.linalg.svd(Bp, full_matrices=False)[2][0]     # 평면 내 장축(157 방향)
    if (latch - cq) @ ex < 0:
        ex = -ex
    ey = np.cross(nrm, ex)
    ul, wl = float((latch - cq) @ ex), float((latch - cq) @ ey)
    uh, wh = float((hinge - cq) @ ex), float((hinge - cq) @ ey)
    return np.array([ul, wl, uh, wh, float(np.linalg.norm(latch - hinge))])
