"""시간대별 성능 분해 — 홀 판별기 프레임 결과(17 --base datasets_factory_collect json) + 세션 드리프트 표(DB)를 세션 시각으로 묶는다.

    python tools/analyze_time_of_day.py   → report/hole_analysis/time_of_day_20260916/README.md
세션 시각은 세션 폴더명 s_HHMMSS(KST). 판정률·판정 정확도·|D−CAD| p95·세션 K·z·tilt 를 2시간 구간과 날짜별로.
"""
import json, os, sqlite3, collections, sys
import numpy as np
DOOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DOOR)
from hole_classifier import CAD_D
OUT = os.path.join(DOOR, 'report', 'hole_analysis', 'time_of_day_20260916')
BINS = [(6, 9), (9, 11), (11, 13), (13, 15), (15, 17), (17, 21)]

rows = json.load(open(os.path.join(DOOR, 'attribute_models/hole_landmarks/eval_classifier_datasets_factory_collect.json')))['datasets_factory_collect']['rows']
con = sqlite3.connect(os.path.join(DOOR, 'db/door_pipeline.db')); con.row_factory = sqlite3.Row
valid_dirs = {r[0] for r in con.execute("""SELECT s.session_dir FROM images i JOIN capture_sessions s ON s.id=i.session_id
    JOIN classes c ON c.id=i.class_id WHERE i.is_valid AND c.name!='Unknown' GROUP BY s.id""")}
corr = {r[0]: r[1] for r in con.execute("""SELECT s.session_dir, c.name FROM images i JOIN capture_sessions s ON s.id=i.session_id
    JOIN classes c ON c.id=i.class_id GROUP BY s.id""")}


def hour_of(sd):
    t = sd.split('/s_')[1]; return int(t[:2]) + int(t[2:4]) / 60


def bucket(h):
    return next((f'{a:02d}-{b:02d}' for a, b in BINS if a <= h < b), 'other')


fr = []
for r in rows:
    sd = '/'.join(r['image'].split('/')[1:4])
    if sd not in valid_dirs: continue
    cls = corr.get(sd, r['cls'])
    if cls not in CAD_D: continue
    fr.append(dict(sd=sd, cls=cls, h=hour_of(sd), day=sd[:8], pred=r['pred'], D=r['D_mm'], gate=r['gate']))
print('유효 프레임', len(fr))


def stats(sub):
    j = [x for x in sub if x['pred']]
    dev = np.abs([x['D'] - CAD_D[x['cls']] for x in j if x['D']]) if j else np.array([])
    return dict(n=len(sub), sessions=len({x['sd'] for x in sub}), judged=len(j), acc=(100 * np.mean([x['pred'] == x['cls'] for x in j]) if j else float('nan')),
                p95=(np.percentile(dev, 95) if dev.size else float('nan')), unk=sum(x['pred'] == 'unknown' for x in j),
                gates=dict(collections.Counter(x['gate'] for x in sub if x['gate'] != 'ok')))


sm = con.execute("""SELECT m.*, s.session_dir FROM session_hole_metrics m JOIN capture_sessions s ON s.id=m.session_id""").fetchall()
sm = [dict(m, h=hour_of(m['session_dir']), day=m['session_dir'][:8]) for m in sm]


def sm_stats(sub):
    ks = [m['k_session'] for m in sub if m['k_session']]; zs = [m['z_med'] for m in sub if m['z_med']]; ts = [m['tilt_med'] for m in sub if m['tilt_med']]
    return dict(n=len(sub), k=np.median(ks) if ks else float('nan'), kstd=np.std(ks) if len(ks) > 1 else float('nan'),
                z=np.median(zs) if zs else float('nan'), tilt=np.median(ts) if ts else float('nan'),
                warn=sum(1 for m in sub if (m['dev_mm'] is not None and abs(m['dev_mm']) > 15) or (m['k_session'] and m['k_applied'] and abs(m['k_session'] / m['k_applied'] - 1) > 0.01)))


lines = ['# 시간대별 성능 분해 (2026-09-16)', '',
         f'입력: 홀 판별기 프레임 결과 `eval_classifier_datasets_factory_collect.json`(유효 세션·정정 라벨 기준 {len(fr)}프레임, {len({x["sd"] for x in fr})}세션) + 세션 드리프트 표 `session_hole_metrics`({len(sm)}세션). 시각 = 세션 시작 KST.', '',
         '## 시간대(2시간 구간)', '', '| 구간 | 세션 | 프레임 | 판정률 | 판정 정확도 | \\|D−CAD\\| p95 | unknown | 보류 사유 | K_session med (std) | z med | tilt med | 경보 세션 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
for a, b in BINS:
    key = f'{a:02d}-{b:02d}'; s = stats([x for x in fr if bucket(x['h']) == key]); t = sm_stats([m for m in sm if bucket(m['h']) == key])
    if not s['n']: continue
    lines.append(f"| {key}시 | {s['sessions']} | {s['n']} | {100*s['judged']/s['n']:.0f}% | {s['acc']:.1f}% | {s['p95']:.1f} | {s['unk']} | {s['gates'] or '-'} | {t['k']:.4f} ({100*t['kstd']:.2f}%) | {t['z']:.0f} | {t['tilt']:.1f} | {t['warn']}/{t['n']} |")
lines += ['', '## 날짜별', '', '| 날짜 | 세션 | 프레임 | 판정률 | 판정 정확도 | \\|D−CAD\\| p95 | K_session med | z med | tilt med | 경보 |', '|---|---|---|---|---|---|---|---|---|---|']
for day in sorted({x['day'] for x in fr}):
    s = stats([x for x in fr if x['day'] == day]); t = sm_stats([m for m in sm if m['day'] == day])
    lines.append(f"| {day} | {s['sessions']} | {s['n']} | {100*s['judged']/s['n']:.0f}% | {s['acc']:.1f}% | {s['p95']:.1f} | {t['k']:.4f} | {t['z']:.0f} | {t['tilt']:.1f} | {t['warn']}/{t['n']} |")
# 시간대×클래스 판정 정확도 (셀이 작으면 n 병기)
lines += ['', '## 시간대 × 클래스 (판정 정확도 %, n)', '']
keys = [f'{a:02d}-{b:02d}' for a, b in BINS]
lines.append('| 클래스 | ' + ' | '.join(keys) + ' |'); lines.append('|---|' + '---|' * len(keys))
for c in sorted(CAD_D):
    cells = []
    for k in keys:
        sub = [x for x in fr if x['cls'] == c and bucket(x['h']) == k and x['pred']]
        cells.append(f"{100*np.mean([x['pred']==c for x in sub]):.0f}% ({len(sub)})" if sub else '-')
    if any(ch != '-' for ch in cells): lines.append(f'| {c} | ' + ' | '.join(cells) + ' |')
json.dump(dict(frames=fr), open(os.path.join(OUT, 'frames.json'), 'w'))
open(os.path.join(OUT, 'README.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
