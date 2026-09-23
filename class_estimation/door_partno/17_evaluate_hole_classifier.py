"""4채널 홀 판별기 평가 — 고정 test 프레임에서 형상군·레이더·품번 판정을 규칙 검사(door_pipeline/partno/radar_check)와 나란히 비교.

  python 17_evaluate_hole_classifier.py [--set test|all] [--limit N]
산출: report/hole4_<set>.json/.md — 프레임/세션 판정률·정확도, 규칙이 미판정한 어려운 프레임(가림·판 밖)에서의 검출기 판정.
"""
import argparse, collections, json, os, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.normpath(os.path.join(HERE, '..', 'door_pipeline'))
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'partno')); sys.path.insert(0, HERE)
import hole_classifier4 as hc4
from hole_classifier4 import base
import radar_check as rc
import partno_labels as pl
from camera_utils import intrinsics_for_image

ap = argparse.ArgumentParser(); ap.add_argument('--set', choices=['test', 'all'], default='test'); ap.add_argument('--limit', type=int, default=0)
args = ap.parse_args()

if __name__ == '__main__':
    rows = [r for r in pl.load()['rows'] if args.set == 'all' or r['split'] == 'test']
    if args.limit: rows = rows[:args.limit]
    net4, dev = hc4.load_model(); net3, _ = base.load_model(); geom = rc.load_geometry()
    out = []; t0 = time.time()
    for r in rows:
        f = os.path.join(HERE, r['path']); rgb = cv2.imread(f); dp = f.replace('rgb_', 'depth_')
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.exists(dp) else None; intr = intrinsics_for_image(f, rgb.shape)
        d4 = hc4.classify(net4, dev, rgb, depth, intr, geom)
        # 규칙(운영 경로): 정본 3채널 검출 + 브래킷 투영
        rule = rc.check_frame(net3, dev, rgb, depth, r['cls'], intr, geom) if pl.GROUP[r['cls']] != 'FRT' else dict(status='n/a')
        out.append(dict(path=r['path'], cls=r['cls'], radar_gt=r['radar'], part_gt=r['part_idx'], session=r['session_dir'],
                        pred=d4['pred'], gate=d4['gate'], radar4=d4['radar'], radar4_info=d4['radar_info'], rule=rule.get('status'), rule_score=rule.get('score')))
    ms = 1000 * (time.time() - t0) / max(1, len(rows))

    def summarize(items, label):
        c = collections.Counter()
        for x in items:
            c['n'] += 1
            if x['pred']: c['cls_judged'] += 1; c['cls_ok'] += (x['pred'] == x['cls'])
            if x['radar_gt'] >= 0:
                gt = 'radar' if x['radar_gt'] == 1 else 'none'; c['r_n'] += 1
                if x['radar4'] in ('radar', 'none'): c['r4_judged'] += 1; c['r4_ok'] += (x['radar4'] == gt)
                if x['rule'] in ('radar', 'none'): c['rule_judged'] += 1; c['rule_ok'] += (x['rule'] == gt)
            if x['part_gt'] >= 0:
                c['p_n'] += 1
                rad = None if x['radar4'] not in ('radar', 'none') else int(x['radar4'] == 'radar')
                pi = pl.part_index(x['pred'], rad) if x['pred'] in pl.CLASSES else -1
                if pi >= 0: c['p_judged'] += 1; c['p_ok'] += (pi == x['part_gt'])
        pct = lambda a, b: 100 * c[a] / max(1, c[b])
        return dict(label=label, n=c['n'], cls_judged=c['cls_judged'], cls_acc=pct('cls_ok', 'cls_judged'),
                    radar_n=c['r_n'], det_judged=c['r4_judged'], det_acc=pct('r4_ok', 'r4_judged'), rule_judged=c['rule_judged'], rule_acc=pct('rule_ok', 'rule_judged'),
                    part_n=c['p_n'], part_judged=c['p_judged'], part_acc=pct('p_ok', 'p_judged'))
    all_s = summarize(out, '전체'); hard = summarize([x for x in out if x['radar_gt'] >= 0 and x['rule'] not in ('radar', 'none')], '규칙 미판정 프레임')
    # 세션 단위
    sess = collections.defaultdict(list)
    for x in out: sess[x['session']].append(x)
    s = collections.Counter()
    for sd, L in sess.items():
        gt = L[0]['radar_gt']
        if gt < 0: continue
        a4 = hc4.aggregate_radar([dict(radar=x['radar4']) for x in L]); ar = hc4.aggregate_radar([dict(radar=x['rule']) for x in L])
        s['n'] += 1
        if a4['flag'] is not None: s['det_judged'] += 1; s['det_ok'] += (a4['flag'] == gt)
        if ar['flag'] is not None: s['rule_judged'] += 1; s['rule_ok'] += (ar['flag'] == gt)
    res = dict(set=args.set, n=len(out), ms_per_frame=round(ms, 1), frame=all_s, hard=hard,
               session=dict(n=s['n'], det_judged=s['det_judged'], det_acc=100 * s['det_ok'] / max(1, s['det_judged']), rule_judged=s['rule_judged'], rule_acc=100 * s['rule_ok'] / max(1, s['rule_judged'])),
               rows=out)
    os.makedirs(os.path.join(HERE, 'report'), exist_ok=True); stem = os.path.join(HERE, 'report', f'hole4_{args.set}')
    json.dump(res, open(stem + '.json', 'w'), ensure_ascii=False, indent=1, default=float)
    md = [f"# 4채널 홀 판별기 vs 규칙 — {args.set} ({len(out)}장, {ms:.1f}ms/장)", "",
          "| 구간 | n | 형상군 판정/정확도 | 레이더 GT n | 검출기 판정률/정확도 | 규칙 판정률/정확도 | 품번 판정률/정확도 |", "|---|---:|---|---:|---|---|---|"]
    for q in (all_s, hard):
        md.append(f"| {q['label']} | {q['n']} | {q['cls_judged']}/{q['n']} · {q['cls_acc']:.1f}% | {q['radar_n']} | {q['det_judged']}/{q['radar_n']} · {q['det_acc']:.1f}% | {q['rule_judged']}/{q['radar_n']} · {q['rule_acc']:.1f}% | {q['part_judged']}/{q['part_n']} · {q['part_acc']:.1f}% |")
    md += ["", f"세션 단위(레이더 GT {s['n']}세션): 검출기 판정 {s['det_judged']} · 정확도 {res['session']['det_acc']:.1f}% / 규칙 판정 {s['rule_judged']} · 정확도 {res['session']['rule_acc']:.1f}%"]
    open(stem + '.md', 'w', encoding='utf8').write('\n'.join(md) + '\n'); print('\n'.join(md)); print('→', stem + '.md')
