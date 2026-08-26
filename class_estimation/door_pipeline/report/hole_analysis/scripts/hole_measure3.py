"""v3: NCC 템플릿 기반 모서리 홀 검출 (라벨 없음).
1) 볼트홀 4개(기하로 안정 검출) 크롭을 모아 평균 템플릿 생성
2) 모서리 홀은 CAD 예측 창 안에서 템플릿 NCC 최대점 선택 (스케일 0.8/1.0)
3) (선택) 고신뢰 모서리 홀 크롭으로 템플릿 재생성 → 재측정 (self-training 1회)
사용: python hole_measure3.py out.json [limit] [debug_dir]
"""
import glob, json, os, sys, time
import cv2, numpy as np
sys.path.insert(0, '/home/koceti/parts_deep/class_estimation/door_pipeline')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribute_utils import axis_align, load_templates
from hole_pipeline import plane_homography, warp_plane, intrinsics, local_hole, CAD, CAD_D
SCR = os.path.dirname(os.path.abspath(__file__))
LATCH = json.load(open(f'{SCR}/holes/holes.json'))
DOOR = '/home/koceti/parts_deep/class_estimation/door_pipeline'
BASE = f'{DOOR}/datasets'
META = json.load(open(f'{DOOR}/vent_labels/datasets/meta.json'))
T = load_templates()
OUT = sys.argv[1] if len(sys.argv) > 1 else 'hole_measure3.json'
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
DEBUG_DIR = sys.argv[3] if len(sys.argv) > 3 else None
TPL = 28   # 템플릿 크기(px, warp 해상도 ≈1.24px/mm → 약 23mm)


def orient0(img, mask):
    ao, am = axis_align(img, mask)
    return np.ascontiguousarray(ao), np.ascontiguousarray(am)


def find_bolts(g, am, cls):
    ys, xs = np.where(am > 0); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    W = x1 - x0 + 1; s = W / CAD[cls]['W']
    lat = [h for h in LATCH[cls]['holes'] if h['w'] <= 12]
    bolts, err = [], 0.0
    for h in lat:
        ex = np.array([x1 - (LATCH[cls]['W'] - h['cx']) * s, y0 + h['cy'] * s])
        b = local_hole(g, ex, 18 * s, s)
        bolts.append(b); err += (np.linalg.norm(b - ex) if b is not None else 40 * s)
    return bolts, err


def orient_by_bolts(ao, am, cls):
    best = None
    for fl in ('none', 'h', 'v', 'hv'):
        a2, m2 = ao, am
        if 'h' in fl: a2, m2 = a2[:, ::-1], m2[:, ::-1]
        if 'v' in fl: a2, m2 = a2[::-1, :], m2[::-1, :]
        a2, m2 = np.ascontiguousarray(a2), np.ascontiguousarray(m2)
        g2 = cv2.cvtColor(a2, cv2.COLOR_BGR2GRAY)
        bolts, err = find_bolts(g2, m2, cls)
        nb = sum(b is not None for b in bolts)
        if best is None or (-nb, err) < best[0]:
            best = ((-nb, err), fl, a2, m2, g2, bolts)
    return best[1:]


def bolt_scale(bolts, s):
    if not all(b is not None for b in bolts) or len(bolts) != 4:
        return None, None
    bs = sorted(bolts, key=lambda b: (round(b[1] / (40 * s)), b[0]))
    pw = (abs(bs[1][0] - bs[0][0]) + abs(bs[3][0] - bs[2][0])) / 2
    ph = (abs(bs[2][1] - bs[0][1]) + abs(bs[3][1] - bs[1][1])) / 2
    if abs(pw / 157.0 - ph / 96.0) > 0.06:
        return None, bs
    return (pw / 157.0 + ph / 96.0) / 2, bs


def crop(g, pt, half):
    x, y = int(round(pt[0])), int(round(pt[1]))
    if x - half < 0 or y - half < 0 or x + half >= g.shape[1] or y + half >= g.shape[0]:
        return None
    return g[y - half:y + half, x - half:x + half].astype(np.float32)


def norm(c):
    c = c - c.mean(); sd = c.std()
    return c / sd if sd > 1e-3 else None


def ncc_pick(g, ex, r, tpl, scales=(1.0, 0.85, 0.7)):
    """예측점 ex 주변 반경 r에서 템플릿 NCC 최대점 → (pt, score, scale)."""
    x, y = int(ex[0]), int(ex[1]); r = int(r)
    x0, y0 = max(0, x - r), max(0, y - r); x1, y1 = min(g.shape[1], x + r), min(g.shape[0], y + r)
    roi = g[y0:y1, x0:x1].astype(np.float32)
    best = None
    for sc in scales:
        t = cv2.resize(tpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if roi.shape[0] <= t.shape[0] or roi.shape[1] <= t.shape[1]:
            continue
        res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        pt = np.array([x0 + ml[0] + t.shape[1] / 2, y0 + ml[1] + t.shape[0] / 2])
        if best is None or mv > best[1]:
            best = (pt, float(mv), sc)
    return best


def prepare(cls, idx):
    rgb = cv2.imread(f'{BASE}/{cls}/rgb_{idx}.png'); depth = cv2.imread(f'{BASE}/{cls}/depth_{idx}.png', cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(f'{BASE}/{cls}/mask_{idx}.png', 0)
    if rgb is None or depth is None or mask is None:
        return None
    cropped = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
    H = plane_homography(depth, mask, intrinsics(rgb.shape[0]))
    wr, wm = warp_plane(rgb, mask, H)
    ao, am = orient0(wr, wm)
    th, tw = T[f'{cls}_sil'].shape
    if (ao.shape[1] > ao.shape[0]) != (tw > th):
        ao = cv2.rotate(ao, cv2.ROTATE_90_CLOCKWISE); am = cv2.rotate(am, cv2.ROTATE_90_CLOCKWISE)
    fl, ao, am, g, bolts = orient_by_bolts(ao, am, cls)
    ys, xs = np.where(am > 0); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    s = (x1 - x0 + 1) / CAD[cls]['W']
    s_bolt, bs = bolt_scale(bolts, s)
    keep_color = idx in ('0000', '0001', '0002', '0003')
    return dict(g=g, ao=(ao if keep_color else None), bolts=bolts, s=s, s_bolt=s_bolt, bbox=(x0, x1, y0, y1), flip=fl, cropped=cropped)


def items():
    for cls in sorted(CAD_D):
        for row in META[cls][:LIMIT]:
            yield cls, row['idx']


if __name__ == '__main__':
    t0 = time.time()
    if DEBUG_DIR: os.makedirs(DEBUG_DIR, exist_ok=True)
    # ── 1) 준비 + 볼트홀 템플릿 수집 ──
    cache = {}
    crops = []
    for cls, idx in items():
        try:
            p = prepare(cls, idx)
        except Exception as e:
            p = None
        if p is None:
            continue
        cache[(cls, idx)] = p
        if p['s_bolt'] is not None:
            for b in p['bolts']:
                c = crop(p['g'], b, TPL // 2)
                if c is not None:
                    n = norm(c)
                    if n is not None:
                        crops.append(n)
    tpl = np.mean(crops, axis=0).astype(np.float32)
    tv = cv2.normalize(tpl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f'{SCR}/tpl_bolt.png', cv2.resize(tv, (TPL * 8, TPL * 8), interpolation=cv2.INTER_NEAREST))
    print(f'[1] 준비 {len(cache)}장, 볼트홀 크롭 {len(crops)}개 → 템플릿 tpl_bolt.png  [{time.time() - t0:.0f}s]', flush=True)

    # ── 2) 모서리 홀 NCC 측정 (라운드 0: 볼트 템플릿, 라운드 1: 모서리 자가 템플릿) ──
    def measure_all(tpl, tag):
        res = {c: [] for c in CAD_D}; corner_crops = []
        for (cls, idx), p in cache.items():
            g, s = p['g'], p['s']; x0, x1, y0, y1 = p['bbox']; cad = CAD[cls]
            exL = np.array([x0 + cad['L'][0] * s, y0 + cad['L'][1] * s])
            exR = np.array([x1 - (cad['W'] - cad['R'][0]) * s, y0 + cad['R'][1] * s])
            L = ncc_pick(g, exL, 28 * s, tpl); R = ncc_pick(g, exR, 28 * s, tpl)
            m = dict(idx=idx, flip=p['flip'], s_bbox=round(s, 4), s_bolt=(round(p['s_bolt'], 4) if p['s_bolt'] else None),
                     cropped=p['cropped'])
            if L and R:
                m.update(L=L[0].round(1).tolist(), R=R[0].round(1).tolist(), scL=round(L[1], 3), scR=round(R[1], 3),
                         D_px=round(float(R[0][0] - L[0][0]), 1), dy_px=round(float(R[0][1] - L[0][1]), 1))
                for pk in (L, R):
                    if pk[1] > 0.55:
                        c = crop(g, pk[0], TPL // 2)
                        if c is not None:
                            n = norm(c)
                            if n is not None: corner_crops.append(n)
            res[cls].append(m)
            if DEBUG_DIR and tag == 'r1' and p['ao'] is not None:
                v = p['ao'].copy()
                for b in p['bolts']:
                    if b is not None: cv2.circle(v, (int(b[0]), int(b[1])), 8, (255, 0, 0), 2)
                zs = []
                for pk, ex in ((L, exL), (R, exR)):
                    if pk: cv2.circle(v, (int(pk[0][0]), int(pk[0][1])), 10, (0, 0, 255), 2)
                    c = pk[0] if pk else ex; x, y = int(c[0]), int(c[1]); r0 = 50
                    z = p['ao'][max(0, y - r0):y + r0, max(0, x - r0):x + r0].copy()
                    if z.size:
                        z = cv2.resize(z, (200, 200), interpolation=cv2.INTER_CUBIC)
                        cv2.circle(z, (100, 100), 14, (0, 0, 255), 2)
                        cv2.putText(z, f"{pk[1]:.2f}" if pk else '-', (5, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); zs.append(z)
                cv2.imwrite(f'{DEBUG_DIR}/{cls}_{idx}.jpg', v, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if len(zs) == 2: cv2.imwrite(f'{DEBUG_DIR}/{cls}_{idx}_z.jpg', np.hstack(zs))
        return res, corner_crops

    res0, cc = measure_all(tpl, 'r0')
    print(f'[2] 라운드0 완료, 고신뢰 모서리 크롭 {len(cc)}개  [{time.time() - t0:.0f}s]', flush=True)
    if len(cc) >= 50:
        tpl2 = np.mean(cc, axis=0).astype(np.float32)
        tv = cv2.normalize(tpl2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(f'{SCR}/tpl_corner.png', cv2.resize(tv, (TPL * 8, TPL * 8), interpolation=cv2.INTER_NEAREST))
        res1, _ = measure_all(tpl2, 'r1')
        print(f'[3] 라운드1(모서리 자가 템플릿) 완료  [{time.time() - t0:.0f}s]', flush=True)
    else:
        res1 = res0
    json.dump(dict(round0=res0, round1=res1), open(OUT, 'w'), indent=1)
    print('DONE', flush=True)
