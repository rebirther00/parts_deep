"""대차 위 도어 위치 변동 통계 — 03_evaluate_field 결과(세션별 집계 T)로 세션 간 x/y/z/θ/tilt 분포를 낸다.

    python tools/analyze_cart_placement.py [--json artifacts/eval_field_datasets_factory_v2_all.json]
    → CART_PLACEMENT_20260916.md (artifacts/ 는 git 미추적이라 루트에 둠)

목적: 로봇 픽업이 흡수해야 할 보정 범위(클래스별 세션 간 편차의 범위·std·p95) 정량화. 세션 내 std(반복 정밀도)와 대비.
좌표: T_cam→door 의 t(mm, 카메라 광학 좌표계 x우/y하/z전방), θ=면내 회전(deg), tilt=평면 기울기(deg).
"""
import argparse, json, os, collections
import numpy as np
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--json', default=os.path.join(BASE, 'artifacts', 'eval_field_datasets_factory_v2_all.json'))
ap.add_argument('--out', default=os.path.join(BASE, 'CART_PLACEMENT_20260916.md')); a = ap.parse_args()
d = json.load(open(a.json))
ses = [r for r in d['sessions'] if r['agg'].get('ok') and r['n_used'] >= 5]
by = collections.defaultdict(list)
for r in ses: by[r['cls']].append(r)
AX = ['x', 'y', 'z', 'theta', 'tilt']
val = lambda r, k: (r['agg']['t'][AX.index(k)] if k in 'xyz' else r['agg'][k + '_deg'])
L = [f"# 대차 위 도어 위치 변동 (2026-09-16) — {os.path.basename(a.json)}", '',
     f"세션 {len(ses)}개(판정 프레임 ≥5), 클래스 {len(by)}. 값 = 세션 집계 자세(프레임 중앙값). 세션 간 = 대차에 놓일 때마다의 편차, 세션 내 std = 정지 프레임 반복 정밀도.", '',
     '## 클래스별 세션 간 편차 (세션 값 − 클래스 중앙값)', '',
     '| 클래스 | 세션 | 축 | 중앙값 | 범위(min~max) | std | p95\\|편차\\| | 세션 내 std med |', '|---|---|---|---|---|---|---|---|']
allrows = collections.defaultdict(list)
for c in sorted(by):
    rr = by[c]
    for k in AX:
        v = np.array([val(r, k) for r in rr]); dev = v - np.median(v); allrows[k].extend(dev.tolist())
        instd = np.median([r['agg']['std'][k] for r in rr])
        L.append(f"| {c} | {len(rr)} | {k} | {np.median(v):.1f} | {v.min():.1f} ~ {v.max():.1f} | {dev.std():.2f} | {np.percentile(np.abs(dev), 95):.2f} | {instd:.2f} |")
L += ['', '## 전 클래스 합산 (세션 값 − 클래스 중앙값)', '', '| 축 | 단위 | std | p95\\|편차\\| | max\\|편차\\| |', '|---|---|---|---|---|']
for k in AX:
    dv = np.array(allrows[k]); L.append(f"| {k} | {'mm' if k in 'xyz' else 'deg'} | {dv.std():.2f} | {np.percentile(np.abs(dv), 95):.2f} | {np.abs(dv).max():.2f} |")
L += ['', '## 세션 목록 (편차 큰 순 상위 15)', '', '| 세션 | 클래스 | x | y | z | θ | tilt | 잔차 med | 판정 |', '|---|---|---|---|---|---|---|---|---|']
score = []
for c, rr in by.items():
    med = {k: np.median([val(r, k) for r in rr]) for k in AX}
    for r in rr:
        dv = {k: val(r, k) - med[k] for k in AX}
        score.append((max(abs(dv['x']), abs(dv['y']), abs(dv['z'])), r, dv))
for s, r, dv in sorted(score, key=lambda t: -t[0])[:15]:
    L.append(f"| {r['session']} | {r['cls']} | {dv['x']:+.1f} | {dv['y']:+.1f} | {dv['z']:+.1f} | {dv['theta']:+.2f} | {dv['tilt']:+.2f} | {r['agg']['rms_med']:.1f} | {r['n_used']}/{r['n']} |")
open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
