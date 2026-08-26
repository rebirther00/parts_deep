"""통합 홀-거리 측정기 (사무실 datasets / 현장 공용).

원본 RGB + depth → 도어 평면 호모그래피 warp(1mm/px, 보간) → 작은 홀 검출
→ 래치 볼트홀 4개(157×96mm)로 회전·스케일·방향 확정 → 상단 모서리 홀 2개 → D(mm).
스케일은 볼트홀 피치에서 얻으므로 depth 절대치수·intrinsics 오차에 무관.

사용:
  python hole_pipeline.py datasets            # datasets/ 8종 전체 (mask_ 사용)
  python hole_pipeline.py datasets 5          # 클래스당 5장
  python hole_pipeline.py file rgb.png depth.png [mask.png]
"""
import glob, json, os, sys, time
import cv2, numpy as np

DOOR = '/home/koceti/parts_deep/class_estimation/door_pipeline'
SCR = os.path.dirname(os.path.abspath(__file__))
CAD = json.load(open(f'{SCR}/holes/corner_small_hole_distances.json'))
CAD_D = {k: v['D'] for k, v in CAD.items()}
RES = 1.0                      # warp 해상도 mm/px
BOLT_W, BOLT_H = 157.0, 96.0   # 래치 볼트홀 피치 (mm)
# 볼트홀 사각형 중심 → 우측 모서리 홀 오프셋 (mm, 래치가 우상단인 프레임)
# CAD: 볼트 중심 (W-220, 288), 모서리 홀 (W-54, 105) → (+166, -183)
OFF_R = (158.0, -184.0)   # LH 147 / RH 168 → 중간값, 탐색반경으로 흡수
CORNER_EDGE = 53.0             # 모서리 홀 ↔ 가장자리 (mm)


def intrinsics(h):
    return dict(fx=1065.0, fy=1065.0, cx=960.0, cy=h / 2.0)


def backproject(depth, K):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    X = (xs - K['cx']) * z / K['fx']
    Y = (ys - K['cy']) * z / K['fy']
    return np.dstack([X, Y, z])


def plane_mask(depth, K, seed_mask, tol=8.0):
    """seed 영역으로 평면 피팅 → 전체 이미지에서 평면 인라이어(±tol mm) 중
    seed 중심을 포함하는 연결 성분 = 도어 마스크."""
    P = backproject(depth, K)
    m = (seed_mask > 0) & (depth > 0)
    pts = P[m]
    if len(pts) > 60000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]
    c = pts.mean(0); _, _, Vt = np.linalg.svd(pts - c, full_matrices=False); n = Vt[2]
    d = np.abs((pts - c) @ n); keep = d < tol
    if keep.sum() > 100:
        c = pts[keep].mean(0); _, _, Vt = np.linalg.svd(pts[keep] - c, full_matrices=False); n = Vt[2]
    dist = np.abs((P - c) @ n)
    inl = ((dist < tol) & (depth > 0)).astype(np.uint8)
    inl = cv2.morphologyEx(inl, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nlab, lab = cv2.connectedComponents(inl)
    ys, xs = np.where(seed_mask > 0)
    seed_lab = lab[int(ys.mean()), int(xs.mean())]
    if seed_lab == 0:
        cnt = np.bincount(lab[seed_mask > 0]); cnt[0] = 0; seed_lab = int(cnt.argmax())
    out = (lab == seed_lab).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return out


def plane_homography(depth, mask, K):
    """마스크 내 depth로 평면 피팅 → 이미지→평면(mm) 호모그래피."""
    m = (mask > 0) & (depth > 0)
    rows, cols = np.where(m)
    if len(rows) > 60000:
        sel = np.random.default_rng(0).choice(len(rows), 60000, replace=False)
        rows, cols = rows[sel], cols[sel]
    z = depth[rows, cols].astype(np.float64)
    X = (cols - K['cx']) * z / K['fx']
    Y = (rows - K['cy']) * z / K['fy']
    P = np.stack([X, Y, z], 1)
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    # 인라이어 재피팅 (10mm)
    d = np.abs((P - c) @ n)
    keep = d < 10
    if keep.sum() > 100:
        c = P[keep].mean(0)
        _, _, Vt = np.linalg.svd(P[keep] - c, full_matrices=False)
    u_ax, v_ax = Vt[0], Vt[1]
    uv = np.stack([(P - c) @ u_ax, (P - c) @ v_ax], 1)
    src = np.stack([cols, rows], 1).astype(np.float32)
    H, _ = cv2.findHomography(src, uv.astype(np.float32), cv2.RANSAC, 3.0)
    return H


def warp_plane(rgb, mask, H, margin=80):
    """이미지 전체를 평면 좌표(1mm/px)로 warp. 마스크 bbox+margin 영역만."""
    ys, xs = np.where(mask > 0)
    pts = np.stack([xs, ys], 1).astype(np.float32)[None]
    uv = cv2.perspectiveTransform(pts, H)[0]
    u0, v0 = uv.min(0) - margin
    u1, v1 = uv.max(0) + margin
    T = np.array([[1 / RES, 0, -u0 / RES], [0, 1 / RES, -v0 / RES], [0, 0, 1]])
    Hw = T @ H
    W, Hh = int((u1 - u0) / RES), int((v1 - v0) / RES)
    wr = cv2.warpPerspective(rgb, Hw, (W, Hh), flags=cv2.INTER_CUBIC)
    wm = cv2.warpPerspective(mask, Hw, (W, Hh), flags=cv2.INTER_NEAREST)
    return wr, wm


def small_dark_blobs(gray, k=15, thr=18, amin=8, amax=200, dmax=18):
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = (bh > thr).astype(np.uint8)
    n, l, st, c = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < amin or a > amax or w > dmax or h > dmax:
            continue
        if max(w, h) / max(1, min(w, h)) > 1.7:
            continue
        out.append((float(c[i][0]), float(c[i][1]), int(a)))
    return out


def local_hole(gray, pt, r, s):
    """예측점 주변에서 작은 홀(밝은 링+어두운 중심 포함)을 완화된 파라미터로 탐색."""
    x, y = int(pt[0]), int(pt[1]); r = int(r)
    x0, y0 = max(0, x - r), max(0, y - r)
    roi = gray[y0:y + r, x0:x + r]
    if roi.size == 0:
        return None
    bh = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(9 * s) | 1, int(9 * s) | 1)))
    m = (bh > 10).astype(np.uint8)
    n, l, st, c = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        xx, yy, w, h, a = st[i]
        if a < 2 or a > 90 or w > 12 * s or h > 12 * s:
            continue
        cxy = np.array([c[i][0] + x0, c[i][1] + y0])
        d = np.linalg.norm(cxy - pt)
        peak = float(bh[l == i].max())          # 가장 어두운(뚜렷한) 작은 홀 우선
        score = d / (25.0 * s) - peak / 25.0
        if best is None or score < best[0]:
            best = (score, cxy)
    return best[1] if best else None


def find_bolt_quad(blobs, tol=0.08):
    """157×96 직사각형(4점, 또는 3점 L자)을 이루는 볼트홀 → (center, angle, s).
    슬롯 격자 오검출 방지: 사각형 내부·주변(60mm 반경) blob 밀도가 낮아야 함."""
    ALL = np.array([(b[0], b[1]) for b in blobs]) if blobs else np.zeros((0, 2))
    big = [b for b in blobs if b[2] >= 6]
    P = np.array([(b[0], b[1]) for b in big])
    if len(P) < 3:
        return None
    DA = np.linalg.norm(P[:, None] - ALL[None], axis=2)
    iso = (DA < 25).sum(1) - 1
    n = len(P)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    best = None
    for i in range(n):
        if iso[i] > 3:
            continue
        for j in range(i + 1, n):
            if iso[j] > 3:
                continue
            d = D[i, j]
            s = d / BOLT_W
            if not (0.95 <= s <= 1.32):
                continue
            e = (P[j] - P[i]) / d
            perp = np.array([-e[1], e[0]])
            for sign in (1, -1):
                q0 = P[i] + perp * BOLT_H * s * sign
                q1 = P[j] + perp * BOLT_H * s * sign
                r = max(8.0, 0.13 * BOLT_H * s)
                k0 = np.where((np.linalg.norm(P - q0, axis=1) < r) & (iso <= 3))[0]
                k1 = np.where((np.linalg.norm(P - q1, axis=1) < r) & (iso <= 3))[0]
                if not (len(k0) or len(k1)):
                    continue
                pts = [P[i], P[j]]
                err = 0.0
                if len(k0):
                    pts.append(P[k0[0]]); err += np.linalg.norm(P[k0[0]] - q0)
                else:
                    pts.append(q0); err += 6.0
                if len(k1):
                    pts.append(P[k1[0]]); err += np.linalg.norm(P[k1[0]] - q1)
                else:
                    pts.append(q1); err += 6.0
                quad = np.array([pts[0], pts[1], pts[3], pts[2]], np.float32)
                cen = quad.mean(0)
                dist = np.linalg.norm(ALL - cen, axis=1)
                inside = sum(cv2.pointPolygonTest(quad, (float(a[0]), float(a[1])), False) > 0
                             for a in ALL[dist < 120 * s])
                ring = int(((dist > 95 * s) & (dist < 200 * s)).sum())   # 주변 링 밀도
                if inside > 8 or ring > 14:
                    continue
                n4 = 2 + int(len(k0) > 0) + int(len(k1) > 0)
                score = err + 0.5 * inside + 0.3 * ring + (0 if n4 == 4 else 25.0)
                dv = np.linalg.norm(pts[2] - pts[0]) if len(k0) else np.linalg.norm(pts[3] - pts[1])
                s2 = (s + dv / BOLT_H) / 2
                ang = np.degrees(np.arctan2(e[1], e[0]))
                if best is None or score < best[0]:
                    best = (score, cen, ang, s2, n4, err, inside, ring)
    if best is None:
        return None
    return dict(center=best[1], angle=best[2], s=best[3], n_pts=best[4], err=best[5], inside=best[6], ring=best[7])


def rotate_to(img, mask, center, angle):
    """볼트홀 장변이 수평이 되도록 회전 (center 고정)."""
    Hh, W = img.shape[:2]
    diag = int(np.hypot(Hh, W)) + 4
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    M[0, 2] += diag / 2 - center[0]
    M[1, 2] += diag / 2 - center[1]
    ri = cv2.warpAffine(img, M, (diag, diag), flags=cv2.INTER_CUBIC)
    rm = cv2.warpAffine(mask, M, (diag, diag), flags=cv2.INTER_NEAREST)
    return ri, rm, M


def measure(rgb, depth, mask, debug=None):
    K = intrinsics(rgb.shape[0])
    H = plane_homography(depth, mask, K)
    wr, wm = warp_plane(rgb, mask, H)
    gray = cv2.cvtColor(wr, cv2.COLOR_BGR2GRAY)
    blobs = small_dark_blobs(gray)
    quad = find_bolt_quad(blobs)
    if quad is None:
        return dict(ok=False, reason='no_bolt_quad', n_blobs=len(blobs))
    s = quad['s']
    # 회전: 장변 수평
    ri, rm, M = rotate_to(wr, wm, quad['center'], quad['angle'])
    cen = np.array([ri.shape[1] / 2, ri.shape[0] / 2])  # 회전 후 볼트 중심
    g2 = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)
    b2 = small_dark_blobs(g2)
    P2 = np.array([(b[0], b[1]) for b in b2]) if b2 else np.zeros((0, 2))
    ys, xs = np.where(rm > 0)
    if len(xs) == 0:
        return dict(ok=False, reason='empty_mask')
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()

    def nearest(pt, r):
        if len(P2) == 0:
            return None
        d = np.linalg.norm(P2 - pt, axis=1)
        k = int(np.argmin(d))
        return P2[k] if d[k] <= r else None

    # 방향: 각공은 상단에서 288mm(하단 854), 래치측 가장자리에서 220mm → 마스크 bbox로 결정
    fy = 1 if (cen[1] - by0) < (by1 - cen[1]) else -1     # 1: 도어 상단이 위
    fx = 1 if (bx1 - cen[0]) < (cen[0] - bx0) else -1     # 1: 래치가 오른쪽
    pr = cen + np.array([OFF_R[0] * fx, OFF_R[1] * fy]) * s
    R = local_hole(g2, pr, 32 * s, s)
    if R is None:
        if debug:
            v = ri.copy()
            for b in b2:
                cv2.circle(v, (int(b[0]), int(b[1])), 5, (0, 200, 255), 1)
            cv2.circle(v, (int(cen[0]), int(cen[1])), 8, (255, 0, 0), 2)
            cv2.circle(v, (int(pr[0]), int(pr[1])), 14, (0, 0, 255), 2)
            cv2.rectangle(v, (bx0, by0), (bx1, by1), (0, 255, 0), 2)
            cv2.putText(v, f"no_corner_R s={s:.3f} fx={fx} fy={fy}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imwrite(debug, v, [cv2.IMWRITE_JPEG_QUALITY, 80])
            x, y = int(pr[0]), int(pr[1]); r0 = 120
            z = v[max(0, y - r0):y + r0, max(0, x - r0):x + r0]
            if z.size:
                cv2.imwrite(debug.replace('.jpg', '_zoomR.jpg'), cv2.resize(z, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
            x, y = int(cen[0]), int(cen[1]); r0 = 160
            z = v[max(0, y - r0):y + r0, max(0, x - r0):x + r0]
            cv2.imwrite(debug.replace('.jpg', '_zoomB.jpg'), cv2.resize(z, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        return dict(ok=False, reason='no_corner_R', s=round(s, 3), fx=fx, fy=fy, n_blobs=len(b2))
    # 좌측 모서리 홀: 같은 행, 반대편(힌지측)으로 탐색. 후보 = 행 ±8px 안의 blob들
    edge = bx0 if fx == 1 else bx1
    pl = np.array([edge + CORNER_EDGE * s * fx, R[1]])
    Lh = local_hole(g2, pl, 40 * s, s)
    cand = np.array([Lh]) if Lh is not None else np.zeros((0, 2))
    if len(cand) == 0:
        if debug:
            v = ri.copy()
            for b in b2:
                cv2.circle(v, (int(b[0]), int(b[1])), 5, (0, 200, 255), 1)
            cv2.circle(v, (int(cen[0]), int(cen[1])), 8, (255, 0, 0), 2)
            cv2.circle(v, (int(R[0]), int(R[1])), 12, (0, 0, 255), 2)
            cv2.line(v, (0, int(R[1])), (v.shape[1], int(R[1])), (0, 0, 255), 1)
            cv2.putText(v, f"no_corner_L s={s:.3f} fx={fx} fy={fy}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imwrite(debug, v, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return dict(ok=False, reason='no_corner_L', s=round(s, 3), R=R.tolist(), fx=fx, fy=fy)
    # 가장자리(CORNER_EDGE)에서 가까운 것 우선, 없으면 가장 바깥 것
    dedge = np.abs(np.abs(cand[:, 0] - edge) - CORNER_EDGE * s)
    L = cand[int(np.argmin(dedge))]
    D_px = abs(R[0] - L[0])
    D_mm = D_px / s
    pred = min(CAD_D, key=lambda k: abs(CAD_D[k] - D_mm))
    cropped = bool((edge <= 2) or (edge >= rm.shape[1] - 3))
    out = dict(ok=True, D_mm=round(D_mm, 1), s=round(s, 4), pred=pred, cropped=cropped,
               bolt_pts=quad['n_pts'], bolt_err=round(quad['err'], 1), fx=fx, fy=fy,
               edge_gap_mm=round(abs(L[0] - edge) / s, 1),
               W_mask_mm=round((bx1 - bx0) / s, 1), n_cand=int(len(cand)))
    if debug:
        v = ri.copy()
        for b in b2:
            cv2.circle(v, (int(b[0]), int(b[1])), 5, (0, 200, 255), 1)
        cv2.circle(v, (int(cen[0]), int(cen[1])), 8, (255, 0, 0), 2)
        cv2.circle(v, (int(R[0]), int(R[1])), 12, (0, 0, 255), 2)
        cv2.circle(v, (int(L[0]), int(L[1])), 12, (0, 0, 255), 2)
        cv2.line(v, (int(L[0]), int(L[1])), (int(R[0]), int(R[1])), (0, 0, 255), 2)
        cv2.putText(v, f"D={D_mm:.0f}mm s={s:.3f} -> {pred}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(debug, v, [cv2.IMWRITE_JPEG_QUALITY, 85])
        zs = []
        for pt in (L, R):
            x, y = int(pt[0]), int(pt[1]); r0 = 60
            z = ri[max(0, y - r0):y + r0, max(0, x - r0):x + r0].copy()
            if z.size:
                z = cv2.resize(z, (240, 240), interpolation=cv2.INTER_CUBIC)
                cv2.circle(z, (120, 120), 16, (0, 0, 255), 2); zs.append(z)
        if len(zs) == 2:
            cv2.imwrite(debug.replace('.jpg', '_zoomLR.jpg'), np.hstack(zs))
    return out


def run_datasets(limit):
    base = f'{DOOR}/datasets'
    res, t0 = {}, time.time()
    for cls in sorted(CAD_D):
        res[cls] = []
        files = sorted(glob.glob(f'{base}/{cls}/rgb_*.png'))[:limit]
        for f in files:
            idx = os.path.basename(f)[4:8]
            rgb = cv2.imread(f)
            depth = cv2.imread(f'{base}/{cls}/depth_{idx}.png', cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(f'{base}/{cls}/mask_{idx}.png', 0)
            if rgb is None or depth is None or mask is None:
                continue
            try:
                m = measure(rgb, depth, mask)
            except Exception as e:  # noqa
                m = dict(ok=False, reason=f'exc:{e}')
            m['idx'] = idx
            res[cls].append(m)
        ok = [m for m in res[cls] if m.get('ok')]
        d = np.array([m['D_mm'] for m in ok])
        acc = np.mean([m['pred'] == cls for m in ok]) * 100 if ok else 0
        print(f"{cls:16s} n={len(res[cls]):3d} ok={len(ok):3d} D med={np.median(d) if d.size else 0:7.1f} "
              f"mean={d.mean() if d.size else 0:7.1f} std={d.std() if d.size else 0:5.1f} | CAD={CAD_D[cls]:5d} "
              f"| 정답률(ok중)={acc:5.1f}%  [{time.time() - t0:.0f}s]", flush=True)
        json.dump(res, open(f'{SCR}/hole_pipeline_datasets.json', 'w'), indent=1)
    return res


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'datasets':
        run_datasets(int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9)
    elif mode == 'file':
        rgb = cv2.imread(sys.argv[2]); depth = cv2.imread(sys.argv[3], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(sys.argv[4], 0) if len(sys.argv) > 4 and os.path.exists(sys.argv[4]) else None
        if mask is None:
            seed = np.zeros(depth.shape, np.uint8)
            h, w = depth.shape
            seed[int(h * 0.3):int(h * 0.7), int(w * 0.4):int(w * 0.7)] = 255
            mask = plane_mask(depth, intrinsics(h), seed)
            cv2.imwrite('dbg_planemask.png', mask)
        print(measure(rgb, depth, mask, debug=sys.argv[5] if len(sys.argv) > 5 else None))
