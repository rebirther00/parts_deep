"""CAD STL → 클래스별 홀 랜드마크 3D 좌표 + 도어 로컬 프레임 정의 (cad_holes.json).

도어 로컬 프레임(전 클래스 동일 정의) — 포즈 추정 T_cam→door 의 대상 프레임:
  원점 = 코너 홀(힌지/래치) 중점, X = 힌지→래치, Z = 6홀 평면 법선(카메라 쪽), Y = Z×X
CAD(차체) 좌표계는 부품에서 1~3m 오프셋이라 직접 사용 금지 — T_door_to_cad 로만 연결.

추출 (직교 투영은 다층 판금 가림 때문에 검출에 쓰지 않음, 디버그 외곽선 전용):
  ① 홀 벽면 삼각형(법선⊥두께축) 4점/삼각형 조밀 샘플 → 반경 1.2mm 클러스터
     → 클러스터 PCA 평면에서 원 피팅(기울어진 홀 대응)
  ② 래치 볼트홀 4: 157×96mm 직사각 시그니처 (hole_classifier.bolt_frame 상수)
  ③ 코너 홀 쌍: 거리 ≈ CAD_D + 볼트 직사각 상대 기하(geometry_gate 와 동일 의미:
     래치측 |u| 80~260·|w| 100~300, 힌지 반대편, |Δw|≤90) — 차체 좌표 방향 무관
  ④ 카메라 관측면(±Y): labels/holes 라벨 배치 손지향(chirality) 대조 → 겹층 홀은
     카메라 쪽 층 선택
실행: python 01_extract_cad_holes.py
출력: cad_holes.json + artifacts/cad_holes_debug/<클래스>.png + 검증 표
"""
import collections
import glob
import itertools
import json
import math
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from stl import mesh as stl_mesh

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(BASE, '..', '..'))
DOOR = os.path.join(ROOT, 'class_estimation', 'door_pipeline')
sys.path.insert(0, DOOR)
import hole_classifier as hc
import pose_utils as pu
from hole_classifier import CAD_D  # 코너 홀 거리 정본(mm)

STL_DIR = os.path.join(ROOT, 'cad', 'door_stl')
OUT_JSON = os.path.join(BASE, 'cad_holes.json')
DBG_DIR = os.path.join(BASE, 'artifacts', 'cad_holes_debug')
# 관측면 정본: cad/door_stp/'cad 설명.txt'의 '카메라가 바라봐야 하는 방향'(STP ±Z).
# STP Z = -STL Y 매핑(-Z→+Y, +Z→-Y)은 자유 선택이 확신(마진 1.9~32.6mm)인 6클래스에서
# 전부 일치함을 확인. RH 2종은 코너 행이 볼트 직사각 기준 준대칭이라 이 정본으로 고정.
VIEW_DESC = {'E25_door_LH_FRT': '+Y', 'E25_door_LH_RR': '+Y', 'E25_door_RH': '+Y',
             'E30_E38_door_RH': '+Y', 'E30_door_LH_FRT': '-Y', 'E30_door_LH_RR': '-Y',
             'E38_door_LH_FRT': '-Y', 'E38_door_LH_RR': '-Y'}
RES = 1.0             # 디버그 투영 mm/px
RECT = (157.0, 96.0)  # 볼트홀 직사각(hole_classifier.bolt_frame과 동일 상수)
RECT_DIAG = math.hypot(*RECT)
BOLT_NAMES = ['bolt_tl', 'bolt_tr', 'bolt_bl', 'bolt_br']
LM6 = BOLT_NAMES + ['corner_hinge', 'corner_latch']


def load_mesh(path):
    m = stl_mesh.Mesh.from_file(path)
    tris = m.vectors.astype(np.float64)
    pts = tris.reshape(-1, 3)
    mins, maxs = pts.min(0), pts.max(0)
    thin = int(np.argmin(maxs - mins))
    a0, a1 = [a for a in range(3) if a != thin]
    return tris, mins, maxs, (thin, a0, a1)


def find_holes_3d(tris, axes):
    """홀 벽면 삼각형 조밀 샘플 → 클러스터 → PCA 평면 원 피팅.

    반환: dict(c3=중심 3D, X/Y3/Z, r, rms, cover, n, ax_extent) 리스트."""
    thin, a0, a1 = axes
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    W = tris[np.abs(n[:, thin]) < 0.5]
    if not len(W):
        return []
    P = np.concatenate([W.reshape(-1, 3), W.mean(1)])          # 4점/삼각형
    tree = cKDTree(P)
    pairs = tree.query_pairs(1.2, output_type='ndarray')
    parent = np.arange(len(P))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(len(P))])
    holes = []
    for root, cnt in zip(*np.unique(roots, return_counts=True)):
        if cnt < 24:
            continue
        Q = P[roots == root]
        c = Q.mean(0)
        _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
        e1, e2, ax = Vt[0], Vt[1], Vt[2]                        # ax ≈ 홀 축
        ext = float(np.ptp((Q - c) @ ax))
        if ext > 12:                                            # 판재 두께 수준이어야
            continue
        x, y = (Q - c) @ e1, (Q - c) @ e2
        A = np.stack([2 * x, 2 * y, np.ones(len(Q))], 1)        # Kasa 원 피팅
        try:
            (cx, cy, cc), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        r = math.sqrt(max(cc + cx * cx + cy * cy, 0))
        if not 2.0 <= r <= 25.0:
            continue
        rr = np.hypot(x - cx, y - cy)
        rms = float(np.sqrt(np.mean((rr - r) ** 2)))
        if rms > max(0.5, 0.12 * r):
            continue
        ang = np.arctan2(y - cy, x - cx)
        cover = len(np.unique(((ang + math.pi) / (2 * math.pi) * 12).astype(int) % 12))
        if cover < 8:
            continue
        c3 = c + cx * e1 + cy * e2
        holes.append(dict(c3=c3, X=float(c3[a0]), Y3=float(c3[thin]), Z=float(c3[a1]),
                          r=float(r), rms=rms, cover=int(cover), n=int(cnt), ax_extent=ext))
    return holes


def d3(a, b):
    return float(np.linalg.norm(a['c3'] - b['c3']))


def find_bolt_rect(holes):
    """157×96 직사각 시그니처 최적 4홀 (오차 평균 최소)."""
    shorts = [(a, b) for a, b in itertools.combinations(holes, 2) if abs(d3(a, b) - RECT[1]) < 8]
    best = None
    for (a, b), (c, d) in itertools.combinations(shorts, 2):
        if len({id(a), id(b), id(c), id(d)}) < 4:
            continue
        for p, q in ((c, d), (d, c)):  # a-p, b-q 가 긴 변이 되는 대응
            e = [abs(d3(a, p) - RECT[0]), abs(d3(b, q) - RECT[0]),
                 abs(d3(a, q) - RECT_DIAG), abs(d3(b, p) - RECT_DIAG),
                 abs(d3(a, b) - RECT[1]), abs(d3(p, q) - RECT[1])]
            if max(e) > 10:
                continue
            if best is None or np.mean(e) < best[0]:
                best = (float(np.mean(e)), [a, b, p, q])
    return best


def rect_axes(quad):
    """직사각 중심 + 면내 축(ex=긴 변 157 방향, ey=짧은 변)."""
    cq = np.mean([h['c3'] for h in quad], 0)
    h0 = quad[0]
    lng = min(quad[1:], key=lambda h: abs(d3(h0, h) - RECT[0]))
    sht = min([h for h in quad[1:] if h is not lng], key=lambda h: abs(d3(h0, h) - RECT[1]))
    ex = lng['c3'] - h0['c3']; ex /= np.linalg.norm(ex)
    ey = sht['c3'] - h0['c3']; ey -= ex * np.dot(ey, ex); ey /= np.linalg.norm(ey)
    return cq, ex, ey


def find_corners_by_rect(holes, D, quad):
    """거리 ≈ CAD_D 쌍 중 볼트 직사각 상대 기하(게이트 유사) 통과 후보들 (XZ 중복 제거)."""
    cq, ex, ey = rect_axes(quad)
    qid = {id(h) for h in quad}
    scored = []
    for a, b in itertools.combinations([h for h in holes if id(h) not in qid], 2):
        d = d3(a, b)
        if abs(d - D) > 25:
            continue
        ua, wa = np.dot(a['c3'] - cq, ex), np.dot(a['c3'] - cq, ey)
        ub, wb = np.dot(b['c3'] - cq, ex), np.dot(b['c3'] - cq, ey)
        for lat, hin, (ul, wl), (uh, wh) in ((a, b, (ua, wa), (ub, wb)), (b, a, (ub, wb), (ua, wa))):
            if not (90 <= abs(ul) <= 240 and 110 <= abs(wl) <= 280):   # geometry_gate ±10 여유
                continue
            if np.sign(uh) == np.sign(ul) or abs(wh - wl) > 90:
                continue
            scored.append((abs(d - D), hin, lat))
    scored.sort(key=lambda t: t[0])
    uniq = []                                       # 겹층(같은 XZ) 중복 제거 — 층은 관측면 확정 후 선택
    same = lambda g, h: math.hypot(g['X'] - h['X'], g['Z'] - h['Z']) < 8
    for err, hin, lat in scored:
        if any(same(hin, h0) and same(lat, l0) for _, h0, l0 in uniq):
            continue
        uniq.append((err, hin, lat))
    return uniq[:30]


def load_labels(cls):
    """labels/holes 의 6점 완비 라벨 (원본 픽셀, 이미지 x우/y하)."""
    out = []
    for f in glob.glob(os.path.join(DOOR, 'labels', 'holes', f'datasets__{cls}__*.json')):
        d = json.load(open(f)); p = d.get('points', {}); vis = d.get('visible', {})
        if not all(k in p and vis.get(k, True) for k in LM6):
            continue
        try:
            pts = {k: np.asarray(p[k], float).reshape(2) for k in LM6}
        except (ValueError, TypeError):
            continue
        if all(np.isfinite(v).all() for v in pts.values()):
            out.append(pts)
    return out


def field_features(cls, net, dev, kmax=24):
    """실측 프레임(factory_v2 우선, 없으면 datasets)에서 검출기+depth 평면 역투영으로
    uw_features [ul, wl, uh, wh, D] 중앙값(mm). 쌍둥이 홀 판별의 기준값."""
    files = sorted(glob.glob(os.path.join(DOOR, 'datasets_factory_v2', 'all', cls, 'rgb_*.png'))) or \
            sorted(glob.glob(os.path.join(DOOR, 'datasets', cls, 'rgb_*.png')))
    assert files, f'{cls}: 실측 프레임 없음'
    files = files[::max(1, len(files) // kmax)][:kmax]
    feats = []
    for f in files:
        rgb = cv2.imread(f)
        depth = cv2.imread(f.replace('rgb_', 'depth_'), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        det = hc.detect(net, dev, rgb)
        if len(det['bolt']) < 4 or not det['corner_hinge'] or not det['corner_latch']:
            continue
        pts = {f'b{i}': det['bolt'][i][:2] for i in range(4)}
        pts['h'] = det['corner_hinge'][0][:2]; pts['l'] = det['corner_latch'][0][:2]
        K = pu.intrinsics_for(depth.shape)
        pl = pu.fit_plane(depth, list(pts.values()), K)
        if not pl:
            continue
        c, n, _ = pl
        P3 = pu.landmarks_3d(pts, c, n, K)
        if not P3:
            continue
        feats.append(pu.uw_features(P3['h'], P3['l'], [P3[f'b{i}'] for i in range(4)], n))
    assert len(feats) >= 3, f'{cls}: 실측 특징 부족 ({len(feats)}/{len(files)})'
    return np.median(np.stack(feats), 0), len(feats)


def select_by_field(cands, quad, ff, thin, view):
    """관측면(view=VIEW_DESC 정본) 안에서 실측 uw_features RMS 최소 코너쌍 선택.

    반환: (힌지, 래치, rms_mm, margin_mm, view_free) — view_free는 양면 자유 선택 결과(검증용)."""
    qb = np.stack([q['c3'] for q in quad])
    results = []
    for err, hin, lat in cands:
        for v in ('+Y', '-Y'):
            ncam = np.zeros(3); ncam[thin] = 1.0 if v == '+Y' else -1.0
            fc = pu.uw_features(hin['c3'], lat['c3'], qb, ncam)
            results.append((float(np.sqrt(np.mean((fc - ff) ** 2))), hin, lat, v))
    results.sort(key=lambda t: t[0])
    view_free = results[0][3]
    inview = [r for r in results if r[3] == view]
    rms0, hin0, lat0, _ = inview[0]
    same = lambda g, h: math.hypot(g['X'] - h['X'], g['Z'] - h['Z']) < 8
    margin = next((r - rms0 for r, h, l, v in inview[1:]
                   if not (same(h, hin0) and same(l, lat0))), 1e9)
    return hin0, lat0, rms0, margin, view_free


def chirality_votes(cls, hinge, latch, bc, axes, view):
    """검증용: 라벨 손지향 다수결이 선택된 (쌍, 관측면)과 일치하는지. 반환 (일치?, votes)."""
    _, a0, a1 = axes
    sgn = []
    for pts in load_labels(cls):
        v1 = pts['corner_latch'] - pts['corner_hinge']
        v2 = np.mean([pts[k] for k in BOLT_NAMES], 0) - pts['corner_hinge']
        s = v1[0] * v2[1] - v1[1] * v2[0]
        if np.isfinite(s) and s != 0:
            sgn.append(np.sign(s))
    c = collections.Counter(sgn)
    if not c:
        return None, {}
    maj = c.most_common(1)[0][0]
    sx = -1.0 if view == '+Y' else 1.0
    uv = lambda p: np.array([sx * p[a0], -p[a1]])
    v1, v2 = uv(latch['c3']) - uv(hinge['c3']), uv(bc) - uv(hinge['c3'])
    ok = np.sign(v1[0] * v2[1] - v1[1] * v2[0]) == maj
    return bool(ok), {int(k): v for k, v in c.items()}


def camera_layer(h, holes, view, thin):
    """같은 자리 겹층 홀(XZ 8mm 이내) 중 카메라 쪽 층 선택."""
    ysign = 1.0 if view == '+Y' else -1.0
    same = [g for g in holes if math.hypot(g['X'] - h['X'], g['Z'] - h['Z']) < 8
            and abs(g['Y3'] - h['Y3']) < 25]
    return max(same, key=lambda g: ysign * g['Y3'])


def extract(cls, net, dev):
    tris, mins, maxs, axes = load_mesh(os.path.join(STL_DIR, cls + '.stl'))
    thin, a0, a1 = axes
    holes = find_holes_3d(tris, axes)

    rect = find_bolt_rect(holes)
    assert rect, f'{cls}: 볼트 직사각 실패 (원형 홀 {len(holes)}개)'
    rect_err, quad = rect
    cands = find_corners_by_rect(holes, CAD_D[cls], quad)
    assert cands, f'{cls}: 코너 홀 쌍 후보 없음 (원형 홀 {len(holes)}개)'
    ff, n_field = field_features(cls, net, dev)
    view = VIEW_DESC[cls]
    hinge, latch, fit_mm, margin_mm, view_free = select_by_field(cands, quad, ff, thin, view)
    bc = np.mean([h['c3'] for h in quad], 0)
    chir_ok, votes = chirality_votes(cls, hinge, latch, bc, axes, view)
    hinge, latch = camera_layer(hinge, holes, view, thin), camera_layer(latch, holes, view, thin)
    quad = [camera_layer(q, holes, view, thin) for q in quad]
    D = d3(hinge, latch)

    named = dict(corner_hinge=hinge, corner_latch=latch)

    # 도어 프레임: 원점=코너 중점, X=힌지→래치, Z=6홀 평면 법선(카메라 쪽), Y=Z×X
    P6 = np.stack([h['c3'] for h in [hinge, latch] + quad])
    c6 = P6.mean(0)
    nrm = np.linalg.svd(P6 - c6)[2][2]
    ycam = np.zeros(3); ycam[thin] = 1.0 if view == '+Y' else -1.0
    if np.dot(nrm, ycam) < 0:
        nrm = -nrm
    o = (hinge['c3'] + latch['c3']) / 2
    X = latch['c3'] - hinge['c3']; X /= np.linalg.norm(X)
    Z = nrm - X * np.dot(nrm, X); Z /= np.linalg.norm(Z)
    Y = np.cross(Z, X)
    R = np.stack([X, Y, Z], 1)                     # door→cad 회전 (열 = 도어축)
    to_door = lambda h: R.T @ (h['c3'] - o)

    # 볼트 이름: 도어 프레임 y 내림차순 2개 = t(카메라 시점 위), x 오름차순 = l
    qd = sorted(quad, key=lambda q: -to_door(q)[1])
    top = sorted(qd[:2], key=lambda q: to_door(q)[0])
    bot = sorted(qd[2:], key=lambda q: to_door(q)[0])
    for q, nm in zip(top + bot, BOLT_NAMES):
        named[nm] = q

    plane_off = {k: float(np.dot(named[k]['c3'] - c6, nrm)) for k in LM6}
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = o
    info = dict(
        D_mm=round(D, 2), D_cad=CAD_D[cls],
        field_fit_mm=round(fit_mm, 2), field_margin_mm=round(margin_mm, 2), n_field=n_field,
        field_features={k: round(float(v), 1) for k, v in zip(['ul', 'wl', 'uh', 'wh', 'D'], ff)},
        rect_err=round(rect_err, 2),
        dist_rect_to_latch=round(float(np.linalg.norm(latch['c3'] - bc)), 1),
        view_from=view, view_free=view_free, chirality_ok=chir_ok, chirality_votes=votes,
        n_circular_holes=len(holes),
        holes_cad={k: [round(v, 2) for v in named[k]['c3']] for k in LM6},
        holes_door={k: [round(v, 2) for v in to_door(named[k])] for k in LM6},
        hole_r_mm={k: round(named[k]['r'], 2) for k in LM6},
        plane_offset_mm={k: round(v, 2) for k, v in plane_off.items()},
        coplanarity_mm=round(max(plane_off.values()) - min(plane_off.values()), 2),
        T_door_to_cad=[[round(v, 6) for v in row] for row in T],
    )
    return info, tris, holes, named, mins, axes


def debug_png(cls, tris, holes, named, mins, axes, info):
    thin, a0, a1 = axes
    dims = tris.reshape(-1, 3).max(0) - mins
    W, H = int(dims[a0] / RES) + 3, int(dims[a1] / RES) + 3
    T2 = np.stack([(tris[:, :, a0] - mins[a0]) / RES, (tris[:, :, a1] - mins[a1]) / RES], -1).astype(np.int32)
    grid = np.zeros((H, W), np.uint8)
    for t in T2:
        cv2.fillConvexPoly(grid, t, 255)
    if info['view_from'] == '+Y':                  # 카메라 시점과 일치하도록 좌우 반전
        grid = grid[:, ::-1]
    grid = np.ascontiguousarray(grid[::-1])        # CAD +Z(상)을 이미지 위로
    img = cv2.cvtColor(255 - grid // 3, cv2.COLOR_GRAY2BGR)

    def px(h):
        u = int((h['X'] - mins[a0]) / RES); v = int((h['Z'] - mins[a1]) / RES)
        if info['view_from'] == '+Y':
            u = W - 1 - u
        return u, H - 1 - v
    for h in holes:
        cv2.circle(img, px(h), max(2, int(h['r'] / RES)), (170, 170, 170), 1)
    colors = dict(corner_hinge=(60, 60, 239), corner_latch=(8, 151, 249))
    for k in LM6:
        c = colors.get(k, (246, 130, 59))
        cv2.circle(img, px(named[k]), max(4, int(named[k]['r'] / RES) + 4), c, 2)
        cv2.putText(img, k, (px(named[k])[0] + 10, px(named[k])[1] - 6), 0, 0.6, c, 2)
    cv2.line(img, px(named['corner_hinge']), px(named['corner_latch']), (120, 120, 120), 1)
    cv2.putText(img, f"{cls}  D={info['D_mm']}mm (CAD {info['D_cad']})  view {info['view_from']}",
                (20, 40), 0, 1.0, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(DBG_DIR, cls + '.png'), img)


if __name__ == '__main__':
    os.makedirs(DBG_DIR, exist_ok=True)
    out = dict(meta=dict(
        units='mm', source='cad/door_stl',
        door_frame='origin=corner midpoint, X=hinge->latch, Z=6-hole plane normal(camera side), Y=Z×X',
        note='CAD 좌표계는 차체 기준(부품에서 1~3m 오프셋) — T_door_to_cad로만 사용',
        landmark_names=LM6), classes={})
    net, dev = hc.load_model()
    print(f"{'클래스':17s} {'D_mm':>7s} {'CAD':>5s} {'차이':>5s} {'실측차':>6s} {'마진mm':>6s} "
          f"{'직사각':>5s} {'래치거리':>7s} {'공면성':>6s} 관측면 라벨검증 실측n")
    for cls in sorted(CAD_D):
        info, tris, holes, named, mins, axes = extract(cls, net, dev)
        debug_png(cls, tris, holes, named, mins, axes, info)
        out['classes'][cls] = info
        print(f"{cls:17s} {info['D_mm']:7.1f} {info['D_cad']:5d} {info['D_mm'] - info['D_cad']:+5.1f} "
              f"{info['field_fit_mm']:6.1f} {info['field_margin_mm']:6.1f} {info['rect_err']:5.2f} "
              f"{info['dist_rect_to_latch']:7.1f} {info['coplanarity_mm']:6.2f} "
              f"{info['view_from']:3s} 자유{'=' if info['view_free'] == info['view_from'] else '≠'} "
              f"{'라벨일치' if info['chirality_ok'] else '라벨불일치!'}{info['chirality_votes']} "
              f"{info['n_field']}")
    json.dump(out, open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
    print(f"\n→ {OUT_JSON}\n→ {DBG_DIR}/*.png")
