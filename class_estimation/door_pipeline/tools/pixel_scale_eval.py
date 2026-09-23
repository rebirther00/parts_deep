"""픽셀 폭 스케일(S_PIXEL) 보정·오프라인 검증 — 17 `--scale depth` 결과 json 하나로 계산 (네트워크 재실행 없음).
rows 에 span_px·fx·z_mm·tilt_deg·D_mm·pred·serial 이 있어야 한다(2026-09-23 이후 17).
  python tools/pixel_scale_eval.py attribute_models/hole_landmarks/eval_classifier_datasets_factory_collect.json
      [--ref-before 2026-09-09] [--S 1478.5] [--unknown_mm 30] [--sessions]
출력: 시리얼별 S(기준 세션 CAD_D·fx/span_px 중앙값)·클래스별 편차 → hole_classifier.S_PIXEL 에 넣을 값,
      depth 판정 vs 픽셀 판정(프레임·세션) 변경 수, 세션 중앙값 |dev| med/p95/max·경보(>15mm)·최소 마진, 가드(z·tilt) 발동 세션.
라벨: 경로에 <날짜>/<클래스>/s_HHMMSS 가 있으면 DB capture_sessions.class_name(정정 라벨), 아니면 rows 의 cls(v2 뷰 = DB 라벨).
"""
import argparse, json, os, re, sqlite3, collections, sys
import numpy as np
DOOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DOOR)
from hole_classifier import CAD_D, judge, UNKNOWN_MM, S_PIXEL, PIXEL_GUARD   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('json')
ap.add_argument('--ref-before', default='2026-09-09', help='S 보정 기준 세션: 이 날짜 이전 (YYYY-MM-DD)')
ap.add_argument('--S', type=float, default=None, help='S 를 고정(미지정: 기준 세션에서 유도)')
ap.add_argument('--unknown_mm', type=float, default=UNKNOWN_MM)
ap.add_argument('--sessions', action='store_true', help='세션별 표 출력')
ap.add_argument('--db', default=os.path.join(DOOR, 'db', 'door_pipeline.db'))
args = ap.parse_args()

res = json.load(open(args.json))
rows = [r for v in res.values() for r in v['rows']]
con = sqlite3.connect(args.db)
# 정정 라벨은 images.class_id(classes.name)에, 무효화는 images.is_valid=0 에 있다 (capture_sessions.class_name 은 폴더명 그대로)
dbcls, invalid = {}, set()
for sd, cls, nv in con.execute("""select s.session_dir, c.name, sum(i.is_valid) from capture_sessions s
        join images i on i.session_id=s.id join classes c on c.id=i.class_id group by s.id, c.name order by 3"""):
    if nv: dbcls[sd] = cls
for (sd,) in con.execute("select s.session_dir from capture_sessions s join images i on i.session_id=s.id group by s.id having sum(i.is_valid)=0"):
    invalid.add(sd)

def sess_of(r):
    """(세션 키, 날짜 YYYY-MM-DD, DB 정정 라벨)"""
    p = r['image']
    m = re.search(r'(\d{8})/([^/]+)/(s_\d{6})/', p)
    if m:
        sd = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"; d = m.group(1)
        return sd, f"{d[:4]}-{d[4:6]}-{d[6:]}", dbcls.get(sd, r['cls'])
    m = re.search(r'rgb_(\d{8})_(s_\d{6})_', os.path.basename(p))
    if m:
        d = m.group(1); return f"{d}/{r['cls']}/{m.group(2)}", f"{d[:4]}-{d[4:6]}-{d[6:]}", r['cls']
    return os.path.dirname(p), '', r['cls']

ok = [r for r in rows if r.get('gate') == 'ok' and r.get('span_px') and r.get('fx') and r.get('serial') is not None]
for r in ok:
    r['sess'], r['date'], r['cls_db'] = sess_of(r)
n_inv = sum(1 for r in ok if r['sess'] in invalid)
ok = [r for r in ok if r['sess'] not in invalid]
print(f"rows {len(rows)}  gate ok+span/fx/serial {len(ok)} (무효 세션 프레임 {n_inv}장 제외, 라벨은 DB 정정 라벨)")
# ① S 보정
S = {}
for sn in sorted(set(r['serial'] for r in ok)):
    ref = [r for r in ok if r['serial'] == sn and r['date'] < args.ref_before and r['cls_db'] in CAD_D]
    if args.S:
        S[sn] = args.S; print(f"serial {sn}: S 고정 {args.S}"); continue
    if not ref:
        S[sn] = S_PIXEL.get(sn); print(f"serial {sn}: 기준 세션 없음 → S_PIXEL {S[sn]}"); continue
    v = np.array([CAD_D[r['cls_db']] * r['fx'] / r['span_px'] for r in ref]); S[sn] = float(np.median(v))
    print(f"serial {sn}: S = {S[sn]:.1f} mm (기준 {len(ref)}장, 프레임 std {100 * v.std() / S[sn]:.2f}%)   현재 S_PIXEL {S_PIXEL.get(sn)}")
    for c in sorted(set(r['cls_db'] for r in ref)):
        vc = np.array([CAD_D[c] * r['fx'] / r['span_px'] for r in ref if r['cls_db'] == c])
        print(f"    {c:18s} S {np.median(vc):7.1f} ({100 * (np.median(vc) / S[sn] - 1):+.2f}%)  n={len(vc)}")
# ② 프레임 판정 비교
chg = collections.Counter(); guard = collections.Counter()
for r in ok:
    if S.get(r['serial']) is None: continue
    r['D_pix'] = r['span_px'] * S[r['serial']] / r['fx']
    r['pred_pix'] = judge(r['D_pix'], None, args.unknown_mm)[0]
    r['warn'] = (r.get('z_mm') is not None and not (PIXEL_GUARD['z'][0] <= r['z_mm'] <= PIXEL_GUARD['z'][1])) or \
                (r.get('tilt_deg') is not None and r['tilt_deg'] > PIXEL_GUARD['tilt'])
    if r['warn']:
        guard[r['sess']] += 1
        if r.get('D_mm'):   # classify 와 동일: 가드 발동 시 depth D 로 폴백
            r['D_pix'], r['pred_pix'] = r['D_mm'], r['pred']
    if r['pred_pix'] != r['pred']: chg[(r['sess'], r['cls_db'], r['pred'], r['pred_pix'])] += 1
ok = [r for r in ok if 'D_pix' in r]
print(f"\n프레임 판정 변경 depth→pixel: {sum(chg.values())}/{len(ok)}  {dict(chg) if chg else ''}")
print(f"depth 판정 정확도 {100 * np.mean([r['pred'] == r['cls_db'] for r in ok]):.2f}%  →  pixel {100 * np.mean([r['pred_pix'] == r['cls_db'] for r in ok]):.2f}%  (DB 라벨 기준, gate ok 프레임)")
# ③ 세션 중앙값 편차
by = collections.defaultdict(list)
for r in ok: by[r['sess']].append(r)
tab = []
for sd, rs in sorted(by.items()):
    c = rs[0]['cls_db']
    if c not in CAD_D: continue
    Dd = np.median([r['D_mm'] for r in rs if r.get('D_mm')]); Dp = np.median([r['D_pix'] for r in rs])
    tab.append(dict(sess=sd, cls=c, n=len(rs), dev_depth=Dd - CAD_D[c], dev_pix=Dp - CAD_D[c],
                    mar_depth=min(r['margin_mm'] for r in rs if r.get('margin_mm') is not None) if any(r.get('margin_mm') is not None for r in rs) else None,
                    mar_pix=min(judge(r['D_pix'], None, args.unknown_mm)[2] for r in rs),
                    z=np.median([r['z_mm'] for r in rs if r.get('z_mm')]) if any(r.get('z_mm') for r in rs) else None,
                    tilt=np.median([r['tilt_deg'] for r in rs if r.get('tilt_deg')]) if any(r.get('tilt_deg') for r in rs) else None,
                    warn=guard.get(sd, 0)))
def st(name, key):
    d = np.abs([t[key] for t in tab]); return f"{name:8s} |dev| med {np.median(d):4.1f} p95 {np.percentile(d, 95):4.1f} max {d.max():4.1f} 경보(>15) {int((d > 15).sum())}/{len(d)}"
print(f"\n세션 {len(tab)}개 (세션 중앙값 D 기준)")
print(" ", st('depth', 'dev_depth')); print(" ", st('pixel', 'dev_pix'))
md = [t['mar_depth'] for t in tab if t['mar_depth'] is not None]; mp = [t['mar_pix'] for t in tab]
print(f"  최소 마진: depth p5 {np.percentile(md, 5):.1f} min {min(md):.1f}  →  pixel p5 {np.percentile(mp, 5):.1f} min {min(mp):.1f}")
print(f"  가드 발동 세션 {sum(1 for t in tab if t['warn'])}개 (프레임 {sum(guard.values())}장): {[(t['sess'], t['warn']) for t in tab if t['warn']][:8]}")
if args.sessions:
    print(f"\n{'session':40s} {'cls':16s} {'n':>3s} {'devD':>6s} {'devP':>6s} {'marD':>5s} {'marP':>5s} {'z':>5s} {'tilt':>5s} warn")
    for t in tab:
        print(f"{t['sess']:40s} {t['cls']:16s} {t['n']:3d} {t['dev_depth']:+6.1f} {t['dev_pix']:+6.1f} {t['mar_depth'] if t['mar_depth'] is not None else float('nan'):5.0f} {t['mar_pix']:5.0f} "
              f"{t['z'] if t['z'] else float('nan'):5.0f} {t['tilt'] if t['tilt'] else float('nan'):5.1f} {t['warn']}")
