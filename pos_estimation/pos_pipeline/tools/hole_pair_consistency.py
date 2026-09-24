"""홀 검출·측정 정확도 프록시 — 같은 기종의 홀-홀 거리(6점 15쌍)를 CAD 거리와 비교 (세션 간, 카메라 잡음이 아닌 검출+스케일 오차).

  python tools/hole_pair_consistency.py [--base datasets_factory_v2/all] [--limit N]
프레임마다 정본 파이프라인과 같은 3D 홀 좌표(depth 평면 + 링 국소 depth)를 구해 15쌍 거리를 CAD 와 비교.
편차 = 측정 − CAD (mm). 클래스별·쌍별 평균(편향)과 std/p95(정밀도), 코너 쌍(D)은 픽셀 폭 스케일 값도 병기.
출력: artifacts/hole_pair_consistency.json / .md
"""
import argparse, collections, glob, itertools, json, os, sys, time
import cv2, numpy as np
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BASE)
import pose_solver as ps, pose_utils as pu
DOOR = ps.DOOR; sys.path.insert(0, DOOR)
import hole_classifier as hc
from camera_utils import intrinsics_for_image
ap = argparse.ArgumentParser(); ap.add_argument('--base', default='datasets_factory_v2/all'); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
cad = ps.load_cad(); net, dev = hc.load_model()
files = sorted(glob.glob(os.path.join(DOOR, a.base, '*', 'rgb_*.png')))
if a.limit: files = files[:a.limit]
dev_by = collections.defaultdict(list)     # (cls, pair) -> [dev mm]
dpx_by = collections.defaultdict(list)     # cls -> [pixel-scale D dev]
n_ok = 0; t0 = time.time()
for i, f in enumerate(files):
    cls = os.path.basename(os.path.dirname(f)); e = cad.get(cls)
    if e is None: continue
    rgb = cv2.imread(f); depth = cv2.imread(f.replace('rgb_', 'depth_'), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None: continue
    intr = intrinsics_for_image(f, rgb.shape); K = pu.intrinsics_for(depth.shape, intr)
    det = hc.detect(net, dev, rgb)
    hinge = det['corner_hinge'][0] if det['corner_hinge'] else None; latch = det['corner_latch'][0] if det['corner_latch'] else None
    if hinge is None or latch is None: continue
    named = dict(corner_hinge=hinge[:2], corner_latch=latch[:2]); named.update(ps.correspond_bolts(det['bolt'], hinge, latch, e['pts']))
    if len(named) < 6: continue
    pl = pu.fit_plane(depth, list(named.values()), K)
    if pl is None: continue
    c, n, _ = pl
    P3 = pu.landmarks_3d(named, c, n, K, {k: float(e['pts'][k][2]) for k in named}, depth=depth)
    if P3 is None: continue
    n_ok += 1
    for p, q in itertools.combinations(sorted(named), 2):
        dm = float(np.linalg.norm(P3[p] - P3[q])); dc = float(np.linalg.norm(np.array(e['pts'][p]) - np.array(e['pts'][q])))
        dev_by[(cls, f'{p}-{q}')].append(dm - dc)
    sn = (intr or {}).get('serial')
    if sn in hc.S_PIXEL and intr.get('fx'):
        span = float(np.hypot(hinge[0] - latch[0], hinge[1] - latch[1])); dpx_by[cls].append(span * hc.S_PIXEL[sn] / intr['fx'] - hc.CAD_D[cls])
    if (i + 1) % 500 == 0: print(f'  {i + 1}/{len(files)} ok {n_ok}  {time.time() - t0:.0f}s', flush=True)
def st(v): v = np.array(v); return dict(n=len(v), mean=float(v.mean()), std=float(v.std()), p95=float(np.percentile(np.abs(v), 95)), max=float(np.abs(v).max()))
out = dict(base=a.base, frames=len(files), used=n_ok, per_class_pair={f'{k[0]}|{k[1]}': st(v) for k, v in dev_by.items()},
           corner_pixel={k: st(v) for k, v in dpx_by.items()})
# 클래스별 요약: 15쌍 전체 편차, 코너 쌍(depth·픽셀), 볼트 쌍(157·96)
L = [f"# 홀-홀 거리 CAD 대비 편차 (검출+스케일 정확도 프록시) — {a.base}, {n_ok}/{len(files)}장, {time.strftime('%Y-%m-%d %H:%M')}", "",
     "편차 = 측정 거리 − CAD 거리(mm). 같은 기종은 CAD 거리가 참값이므로 세션 간 편차의 평균이 편향, std/p95 가 정밀도. 카메라 잡음이 아니라 검출 위치·스케일(intrinsics·depth) 오차가 드러난다.", "",
     "| 클래스 | 15쌍 전체: 평균 / std / |p95| | 코너 쌍 D(depth 3D): 평균 / std | 코너 쌍 D(픽셀 폭): 평균 / std | 볼트 장변 157: 평균 / std | 볼트 단변 96: 평균 / std |", "|---|---|---|---|---|---|"]
summary = {}
for cls in sorted({k[0] for k in dev_by}):
    allv = np.concatenate([np.array(v) for k, v in dev_by.items() if k[0] == cls])
    cn = np.array(dev_by[(cls, 'corner_hinge-corner_latch')]); px = np.array(dpx_by.get(cls, [0.0]))
    long_ = np.concatenate([np.array(dev_by[(cls, p)]) for p in ('bolt_bl-bolt_br', 'bolt_tl-bolt_tr') if (cls, p) in dev_by] or [np.zeros(1)])
    short = np.concatenate([np.array(dev_by[(cls, p)]) for p in ('bolt_bl-bolt_tl', 'bolt_br-bolt_tr') if (cls, p) in dev_by] or [np.zeros(1)])
    summary[cls] = dict(all=st(allv), corner_depth=st(cn), corner_pixel=st(px), bolt_long=st(long_), bolt_short=st(short))
    L.append(f"| {cls} | {allv.mean():+.1f} / {allv.std():.1f} / {np.percentile(np.abs(allv), 95):.1f} | {cn.mean():+.1f} / {cn.std():.1f} | {px.mean():+.1f} / {px.std():.1f} | {long_.mean():+.1f} / {long_.std():.1f} | {short.mean():+.1f} / {short.std():.1f} |")
allv = np.concatenate([np.array(v) for v in dev_by.values()])
edges = ('bolt_bl-bolt_br', 'bolt_tl-bolt_tr', 'bolt_bl-bolt_tl', 'bolt_br-bolt_tr'); diags = ('bolt_bl-bolt_tr', 'bolt_br-bolt_tl')
G = {'볼트 변 4쌍(157/96mm, 검출 정확도)': np.concatenate([np.array(v) for k, v in dev_by.items() if k[1] in edges]),
     '볼트 대각 2쌍(184mm)': np.concatenate([np.array(v) for k, v in dev_by.items() if k[1] in diags]),
     '코너 쌍 D depth 3D(700~1,350mm)': np.concatenate([np.array(v) for k, v in dev_by.items() if k[1] == 'corner_hinge-corner_latch']),
     '코너 쌍 D 픽셀 폭': np.concatenate([np.array(v) for v in dpx_by.values()]) if dpx_by else np.zeros(1),
     '전체 15쌍': allv}
L += ["", "## 전체 집계 (정확한 분위수)", "", "| 구간 | n | 편향(평균) | std | |편차| 중앙값 | |편차| p95 | 최대 |", "|---|---:|---:|---:|---:|---:|---:|"]
out['global'] = {}
for k, v in G.items():
    av = np.abs(v); out['global'][k] = dict(n=int(v.size), mean=float(v.mean()), std=float(v.std()), med_abs=float(np.median(av)), p95_abs=float(np.percentile(av, 95)), max_abs=float(av.max()))
    L.append(f"| {k} | {v.size:,} | {v.mean():+.2f} | {v.std():.2f} | {np.median(av):.2f} | {np.percentile(av, 95):.2f} | {av.max():.1f} |")
out['summary'] = summary
os.makedirs(os.path.join(BASE, 'artifacts'), exist_ok=True)
json.dump(out, open(os.path.join(BASE, 'artifacts', 'hole_pair_consistency.json'), 'w'), ensure_ascii=False, indent=1)
open(os.path.join(BASE, 'artifacts', 'hole_pair_consistency.md'), 'w', encoding='utf8').write('\n'.join(L) + '\n'); print('\n'.join(L))
