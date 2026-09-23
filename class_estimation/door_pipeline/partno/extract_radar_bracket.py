"""RADAR 사양 CAD(STL)에서 중앙 보강재 위 브래킷 홀 2개의 도어 프레임 좌표 추출 → partno/radar_bracket.json.

  python partno/extract_radar_bracket.py                       # RR/RH 5클래스 (cad/door_stl/<class>.stl, 없으면 --stl 로 지정)
  python partno/extract_radar_bracket.py --stl E30_door_LH_RR=cad/door_stl/E30_door_LH_RR_02360K.stl
  python partno/extract_radar_bracket.py --only E25_door_RH

방법: 메시를 pos_pipeline/cad_holes.json 의 T_door_to_cad 로 도어 프레임(원점 = 코너 홀 중점, X = 힌지→래치, Z = 카메라 쪽)에
놓고, 중앙 보강재 밴드(|y| 300~700mm)를 1mm/px 로 '카메라 쪽 최대 z' 지도로 렌더한다. 브래킷 판은 보강재보다 카메라 쪽으로
솟은 밝은 사각형이고, 홀 2개는 메시가 없는(투과) 원형 빈 영역이다. 홀 벽면이 원기둥으로 모델링되지 않아 01_extract_cad_holes 의
벽면 클러스터 방식으로는 잡히지 않으므로 빈 영역 검출을 쓴다.
출력(mm, 도어 프레임): holes [[x,y],[x,y]], r_mm, pitch_mm, plate [x0,y0,x1,y1], z_plate(6홀 평면 기준 높이), z_strip
디버그: partno/artifacts/bracket_<class>.png
"""
import argparse, json, math, os, sys
import cv2, numpy as np
from stl import mesh as stl_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.dirname(HERE)
ROOT = os.path.normpath(os.path.join(DOOR, '..', '..'))
CAD_HOLES = os.path.join(ROOT, 'pos_estimation', 'pos_pipeline', 'cad_holes.json')
STL_DIR = os.path.join(ROOT, 'cad', 'door_stl')
OUT_JSON = os.path.join(HERE, 'radar_bracket.json')
RADAR_CLASSES = ['E25_door_LH_RR', 'E30_door_LH_RR', 'E38_door_LH_RR', 'E25_door_RH', 'E30_E38_door_RH']
BAND = (300.0, 700.0)     # 중앙 보강재 밴드 |y| (mm) — 볼트 사각 y 부호 쪽
HOLE_AREA = (500.0, 2500.0)   # 홀 빈 영역 면적 mm² (Ø25~56)
PITCH = (60.0, 140.0)         # 홀 2개 간격 mm


def height_map(tris_door, ylo, yhi):
    """도어 프레임 삼각형 → 밴드 내 카메라 쪽 최대 z 지도 (1mm/px). 반환 zmap(H,W) NaN=빈 영역, xlo."""
    xlo, xhi = tris_door[:, :, 0].min(), tris_door[:, :, 0].max()
    W, H = int(xhi - xlo) + 2, int(yhi - ylo) + 2
    zmap = np.full((H, W), np.nan, np.float32)
    sel = (tris_door[:, :, 1].min(1) < yhi) & (tris_door[:, :, 1].max(1) > ylo)
    for t in tris_door[sel]:
        z = float(t[:, 2].mean())
        pts = np.stack([t[:, 0] - xlo, t[:, 1] - ylo], 1).astype(np.int32)
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            continue
        m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, pts, 255)
        idx = m > 0; cur = zmap[idx]; zmap[idx] = np.where(np.isnan(cur), z, np.maximum(cur, z))
    return zmap, xlo


def find_bracket(zmap, xlo, ylo):
    """빈 원형 영역 후보 → 같은 높이·간격 60~140mm 쌍 중 주변(판)이 가장 높은 쌍. 반환 dict 또는 None."""
    empty = np.isnan(zmap).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(empty, 8)
    H, W = zmap.shape
    cands = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (HOLE_AREA[0] <= a <= HOLE_AREA[1]) or x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        r = math.sqrt(a / math.pi)
        if abs(w - h) > 0.25 * (w + h) or a < 0.72 * w * h:   # 원형성(정사각 외접 대비 π/4=0.785)
            continue
        cx, cy = cent[i]
        ring = np.zeros_like(empty); cv2.circle(ring, (int(cx), int(cy)), int(r + 14), 1, -1); cv2.circle(ring, (int(cx), int(cy)), int(r + 4), 0, -1)
        zr = zmap[ring > 0]; zr = zr[~np.isnan(zr)]
        if len(zr) < 20:
            continue
        cands.append(dict(cx=cx, cy=cy, r=r, area=int(a), z_ring=float(np.median(zr))))
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            d = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])
            if not (PITCH[0] <= d <= PITCH[1]) or abs(a['cy'] - b['cy']) > 12 or abs(a['r'] - b['r']) > 4:
                continue
            score = (a['z_ring'] + b['z_ring']) / 2
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is None:
        return None, cands
    _, a, b = best
    if a['cx'] > b['cx']: a, b = b, a
    z_plate = (a['z_ring'] + b['z_ring']) / 2
    # 판 범위: 홀 주변 z ≥ z_plate−8 연결 영역의 외접 사각형
    plate = ((~np.isnan(zmap)) & (zmap >= z_plate - 8)).astype(np.uint8)
    n2, lab2 = cv2.connectedComponents(plate, connectivity=4)
    seed = lab2[int(a['cy']), int(a['cx'] + a['r'] + 8)]
    ys, xs = np.where(lab2 == seed) if seed else (np.array([]), np.array([]))
    px0, px1, py0, py1 = (xs.min(), xs.max(), ys.min(), ys.max()) if len(xs) else (a['cx'] - 60, b['cx'] + 60, a['cy'] - 40, a['cy'] + 40)
    band_z = zmap[~np.isnan(zmap)]
    return dict(holes=[[round(float(a['cx'] + xlo), 1), round(float(a['cy'] + ylo), 1)], [round(float(b['cx'] + xlo), 1), round(float(b['cy'] + ylo), 1)]],
                r_mm=round(float(a['r'] + b['r']) / 2, 1), pitch_mm=round(math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy']), 1),
                plate=[round(float(px0 + xlo), 1), round(float(py0 + ylo), 1), round(float(px1 + xlo), 1), round(float(py1 + ylo), 1)],
                z_plate=round(z_plate, 1), z_strip=round(float(np.percentile(band_z, 50)), 1)), cands


def debug_png(zmap, xlo, ylo, info, path, title):
    valid = ~np.isnan(zmap); lo, hi = np.nanpercentile(zmap, 1), np.nanpercentile(zmap, 99)
    g = np.zeros(zmap.shape, np.uint8); g[valid] = np.clip((zmap[valid] - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(g, cv2.COLORMAP_VIRIDIS); img[~valid] = 0
    if info:
        for (x, y) in info['holes']:
            cv2.circle(img, (int(x - xlo), int(y - ylo)), int(info['r_mm']), (0, 0, 255), 2)
        x0, y0, x1, y1 = info['plate']; cv2.rectangle(img, (int(x0 - xlo), int(y0 - ylo)), (int(x1 - xlo), int(y1 - ylo)), (255, 255, 255), 1)
    img = np.ascontiguousarray(img[::-1])
    cv2.putText(img, title, (8, 20), 0, 0.55, (255, 255, 255), 1)
    cv2.imwrite(path, img)


def extract(cls, stl_path, cadh):
    T = np.array(cadh[cls]['T_door_to_cad']); R, o = T[:3, :3], T[:3, 3]
    tris = stl_mesh.Mesh.from_file(stl_path).vectors.astype(np.float64)
    td = ((tris.reshape(-1, 3) - o) @ R).reshape(-1, 3, 3)
    sign = 1.0 if cadh[cls]['holes_door']['bolt_tl'][1] > 0 else -1.0     # 도어 본체가 뻗은 y 방향
    ylo, yhi = sorted((sign * BAND[0], sign * BAND[1]))
    zmap, xlo = height_map(td, ylo, yhi)
    info, cands = find_bracket(zmap, xlo, ylo)
    return info, cands, zmap, xlo, ylo


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stl', action='append', default=[], help='클래스=STL경로 (기본 cad/door_stl/<class>.stl 대체)')
    ap.add_argument('--only', action='append', default=[])
    ap.add_argument('--out', default=OUT_JSON)
    a = ap.parse_args()
    stl_map = dict(s.split('=', 1) for s in a.stl)
    cadh = json.load(open(CAD_HOLES))['classes']
    out = json.load(open(a.out)) if os.path.exists(a.out) else dict(meta={}, classes={})
    out['meta'] = dict(units='mm', frame='pos_pipeline/cad_holes.json 도어 프레임(원점 코너 홀 중점, X 힌지→래치, Z 카메라 쪽)',
                       method='카메라 쪽 최대 z 지도(1mm/px)에서 빈 원형 영역 쌍 검출', band_abs_y=list(BAND),
                       note='holes/plate 는 6홀 평면 위 z_plate 만큼 솟아 있음 — 투영 시 시차 보정 필요')
    for cls in (a.only or RADAR_CLASSES):
        stl_path = stl_map.get(cls) or os.path.join(STL_DIR, cls + '.stl')
        if not os.path.isabs(stl_path): stl_path = os.path.join(ROOT, stl_path)
        if not os.path.exists(stl_path):
            print(f"{cls:17s} STL 없음: {stl_path}"); continue
        info, cands, zmap, xlo, ylo = extract(cls, stl_path, cadh)
        png = os.path.join(HERE, 'artifacts', f'bracket_{cls}.png')
        debug_png(zmap, xlo, ylo, info, png, f"{cls} {'bracket ' + str(info['holes']) if info else 'NOT FOUND'}")
        if info:
            info['stl'] = os.path.relpath(stl_path, ROOT)
            out['classes'][cls] = info
            print(f"{cls:17s} holes {info['holes']}  r {info['r_mm']}  pitch {info['pitch_mm']}  plate {info['plate']}  z_plate {info['z_plate']} (strip {info['z_strip']})  ← {info['stl']}")
        else:
            print(f"{cls:17s} 브래킷 미검출 (후보 {len(cands)}: {[(round(c['cx'] + xlo), round(c['cy'] + ylo), round(c['r'], 1)) for c in cands][:6]}) — 비RADAR CAD 이거나 메시 확인 필요 → {png}")
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f"→ {a.out}")
