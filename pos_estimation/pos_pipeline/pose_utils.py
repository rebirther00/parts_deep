"""포즈 파이프라인 공용 기하 유틸 (01 CAD 검증 · 02 솔버 공용).

- intrinsics: fx=1065 근사 + K_DEPTH 스칼라 보정 — hole_classifier(현장 검증)와 동일 규약.
  실측 캘리브레이션이 세션 메타로 제공되면 그쪽을 우선 사용(수집측 TODO).
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


def intrinsics_for(shape):
    h, w = shape[:2]
    return dict(fx=1065.0, fy=1065.0, cx=w / 2.0, cy=h / 2.0,
                k=K_DEPTH.get(h, K_DEPTH[1080]))


def fit_plane(depth, pts, K):
    """검출점 볼록껍질 내부 유효 depth로 평면 피팅.

    반환: (c, n, stats) — c/n 무보정 카메라 좌표(mm), n은 카메라 쪽(nz<0), 실패 시 None."""
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
    z = depth[rows, cols].astype(np.float64)
    Q = np.stack([(cols - K['cx']) * z / K['fx'], (rows - K['cy']) * z / K['fy'], z], 1)
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
                 n_px=int(len(rows)), inlier_ratio=float(inl.mean()))
    return c, n, stats


def backproject(p, c, n, K, offset=0.0):
    """픽셀 p 레이와 (c + offset·n, n) 평면 교점 (offset: 카메라 쪽 +, 무보정 mm)."""
    r = np.array([(p[0] - K['cx']) / K['fx'], (p[1] - K['cy']) / K['fy'], 1.0])
    c2 = c + offset * n
    denom = np.dot(r, n)
    if abs(denom) < 1e-9:
        return None
    return r * (np.dot(c2, n) / denom)


def landmarks_3d(pts_px, c, n, K, offsets=None):
    """{이름: (x,y)px} → {이름: 3D mm(K_DEPTH 보정)}. offsets: 홀별 평면 오프셋(진짜 mm, 카메라 쪽 +)."""
    out = {}
    for name, p in pts_px.items():
        off = (offsets.get(name, 0.0) if offsets else 0.0) / K['k']
        q = backproject(p, c, n, K, off)
        if q is None:
            return None
        out[name] = q * K['k']
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
