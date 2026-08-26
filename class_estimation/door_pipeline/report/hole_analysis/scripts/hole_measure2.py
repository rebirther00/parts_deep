"""검증용 측정기 v2: 클래스 라벨(CAD 프라이어)을 써서 사무실 datasets에서 모서리 홀 D를 실측.
warp(호모그래피, 1mm/px) → axis_align → 템플릿 방향/플립(vent_labels meta) → CAD 예측 위치 근방 국소 검출.
스케일: 래치 볼트홀 피치(157×96) 실측 → s_bolt. 보조: bbox 스케일.
"""
import glob, json, os, sys, time
import cv2, numpy as np
sys.path.insert(0, '/home/koceti/parts_deep/class_estimation/door_pipeline')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribute_utils import axis_align, load_templates
from hole_pipeline import plane_homography, warp_plane, intrinsics, local_hole, CAD, CAD_D
LATCH = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holes', 'holes.json')))

DOOR = '/home/koceti/parts_deep/class_estimation/door_pipeline'
BASE = f'{DOOR}/datasets'
META = json.load(open(f'{DOOR}/vent_labels/datasets/meta.json'))
T = load_templates()
OUT = sys.argv[1] if len(sys.argv) > 1 else 'hole_measure2.json'
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
DEBUG_DIR = sys.argv[3] if len(sys.argv) > 3 else None


def orient(img, mask, cls, fl):
    ao, am = axis_align(img, mask)
    th, tw = T[f'{cls}_sil'].shape
    if (ao.shape[1] > ao.shape[0]) != (tw > th):
        ao = cv2.rotate(ao, cv2.ROTATE_90_CLOCKWISE); am = cv2.rotate(am, cv2.ROTATE_90_CLOCKWISE)
    if 'h' in fl:
        ao, am = ao[:, ::-1], am[:, ::-1]
    if 'v' in fl:
        ao, am = ao[::-1, :], am[::-1, :]
    return np.ascontiguousarray(ao), np.ascontiguousarray(am)


def find_bolts(g, am, cls):
    ys, xs = np.where(am > 0); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    W = x1 - x0 + 1; s = W / CAD[cls]['W']
    lat = [h for h in LATCH[cls]['holes'] if h['w'] <= 12]
    bolts, err = [], 0.0
    for h in lat:
        ex = np.array([x1 - (LATCH[cls]['W'] - h['cx']) * s, y0 + h['cy'] * s])
        b = local_hole(g, ex, 18 * s, s)
        bolts.append(b)
        err += (np.linalg.norm(b - ex) if b is not None else 40 * s)
    return bolts, err


def measure(ao, am, cls, dbg=None):
    # 방향 4가설(원본/좌우/상하/둘다): 래치 볼트홀 4개가 가장 잘 맞는 것 채택
    best = None
    for fl in ('none', 'h', 'v', 'hv'):
        a2, m2 = ao, am
        if 'h' in fl: a2, m2 = a2[:, ::-1], m2[:, ::-1]
        if 'v' in fl: a2, m2 = a2[::-1, :], m2[::-1, :]
        a2, m2 = np.ascontiguousarray(a2), np.ascontiguousarray(m2)
        g2 = cv2.cvtColor(a2, cv2.COLOR_BGR2GRAY)
        bolts, err = find_bolts(g2, m2, cls)
        nb = sum(b is not None for b in bolts)
        key = (-nb, err)
        if best is None or key < best[0]:
            best = (key, fl, a2, m2, g2, bolts)
    _, flip, ao, am, g, bolts = best
    ys, xs = np.where(am > 0); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    W = x1 - x0 + 1; cad = CAD[cls]; s = W / cad['W']
    s_bolt = None
    if all(b is not None for b in bolts) and len(bolts) == 4:
        bs = sorted(bolts, key=lambda b: (round(b[1] / (40 * s)), b[0]))
        pw = (abs(bs[1][0] - bs[0][0]) + abs(bs[3][0] - bs[2][0])) / 2
        ph = (abs(bs[2][1] - bs[0][1]) + abs(bs[3][1] - bs[1][1])) / 2
        s_bolt = (pw / 157.0 + ph / 96.0) / 2
        if abs(pw / 157.0 - ph / 96.0) > 0.06:
            s_bolt = None
    exL = np.array([x0 + cad['L'][0] * s, y0 + cad['L'][1] * s])
    exR = np.array([x1 - (cad['W'] - cad['R'][0]) * s, y0 + cad['R'][1] * s])
    L = local_hole(g, exL, 28 * s, s); R = local_hole(g, exR, 28 * s, s)
    out = dict(flip=flip, W_px=int(W), s_bbox=round(s, 3), s_bolt=(round(s_bolt, 3) if s_bolt else None),
               n_bolts=int(sum(b is not None for b in bolts)),
               L=(L.round(1).tolist() if L is not None else None), R=(R.round(1).tolist() if R is not None else None),
               cropped=bool(x0 <= 2 or x1 >= am.shape[1] - 3))
    if L is not None and R is not None:
        D_px = float(R[0] - L[0])
        out['D_px'] = round(D_px, 1)
        out['D_bbox_mm'] = round(D_px / s, 1)
        out['D_mm'] = round(D_px / s_bolt, 1) if s_bolt else None
        out['dy_px'] = round(float(R[1] - L[1]), 1)
    if dbg:
        v = ao.copy()
        for b in bolts:
            if b is not None: cv2.circle(v, (int(b[0]), int(b[1])), 8, (255, 0, 0), 2)
        for pt, ex in ((L, exL), (R, exR)):
            cv2.rectangle(v, (int(ex[0] - 28 * s), int(ex[1] - 28 * s)), (int(ex[0] + 28 * s), int(ex[1] + 28 * s)), (0, 255, 255), 1)
            if pt is not None: cv2.circle(v, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), 2)
        cv2.putText(v, f"{cls} D={out.get('D_mm')} (bbox {out.get('D_bbox_mm')}) CAD={CAD_D[cls]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(dbg, v, [cv2.IMWRITE_JPEG_QUALITY, 80])
        zs = []
        for pt, ex in ((L, exL), (R, exR)):
            c = pt if pt is not None else ex
            x, y = int(c[0]), int(c[1]); r0 = 50
            z = ao[max(0, y - r0):y + r0, max(0, x - r0):x + r0].copy()
            if z.size:
                z = cv2.resize(z, (200, 200), interpolation=cv2.INTER_CUBIC)
                cv2.circle(z, (100, 100), 14, (0, 0, 255) if pt is not None else (0, 255, 255), 2); zs.append(z)
        if len(zs) == 2:
            cv2.imwrite(dbg.replace('.jpg', '_z.jpg'), np.hstack(zs))
    return out


if __name__ == '__main__':
    res = {}; t0 = time.time()
    if DEBUG_DIR: os.makedirs(DEBUG_DIR, exist_ok=True)
    for cls in sorted(CAD_D):
        res[cls] = []
        for row in META[cls][:LIMIT]:
            idx = row['idx']
            rgb = cv2.imread(f'{BASE}/{cls}/rgb_{idx}.png'); depth = cv2.imread(f'{BASE}/{cls}/depth_{idx}.png', cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(f'{BASE}/{cls}/mask_{idx}.png', 0)
            if rgb is None or depth is None or mask is None: continue
            try:
                H = plane_homography(depth, mask, intrinsics(rgb.shape[0]))
                wr, wm = warp_plane(rgb, mask, H)
                ao, am = orient(wr, wm, cls, 'none')
                m = measure(ao, am, cls, dbg=(f'{DEBUG_DIR}/{cls}_{idx}.jpg' if DEBUG_DIR else None))
            except Exception as e:
                m = dict(error=str(e))
            m['idx'] = idx; res[cls].append(m)
        ok = [m for m in res[cls] if m.get('D_mm')]
        d = np.array([m['D_mm'] for m in ok]); db = np.array([m['D_bbox_mm'] for m in res[cls] if m.get('D_bbox_mm')])
        print(f"{cls:16s} n={len(res[cls]):3d} D_bolt n={len(ok):3d} med={np.median(d) if d.size else 0:7.1f} std={d.std() if d.size else 0:5.1f} "
              f"| D_bbox n={db.size:3d} med={np.median(db) if db.size else 0:7.1f} | CAD={CAD_D[cls]}  [{time.time()-t0:.0f}s]", flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)
