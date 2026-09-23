"""블라인드 평가 — 학습에 쓰지 않은 세션에서 규칙·4채널 검출기·CNN 을 같은 프레임으로 채점한다.

  python 19_blind_eval.py                                  # 기본: DB split 미지정 세션(9/16 오후~9/23 유입 35세션, CNN 학습·val·test 미사용)
  python 19_blind_eval.py --since 20260924 --tag new       # 날짜 이후 유입 세션 (신규 세션 블라인드 루틴: ingest→pull→이 스크립트→/options 확정→재실행 채점)
  python 19_blind_eval.py --detector attribute_models/hole_landmarks_bracket_blind/model.pth --detector-blind
프레임: 로컬 미러 datasets_factory_collect (세션당 ≤20장). 형상군은 모델 자체 판정(DB 라벨 미사용), 규칙의 브래킷 위치는 정본 홀 판별기 판정 클래스로 투영.
GT: DB 클래스 × 사용자 확정 레이더. 확정 전 세션은 '미확정'으로 예측만 기록(확정 뒤 재실행하면 채점).
산출: report/blind_<tag>.md/.json — 모델별 프레임/세션 품번 판정률·정확도.
"""
import argparse, collections, json, os, sqlite3, sys, time
import cv2, numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.normpath(os.path.join(HERE, '..', 'door_pipeline'))
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'db')); sys.path.insert(0, os.path.join(DOOR, 'partno')); sys.path.insert(0, HERE)
import hole_classifier4 as hc4
from hole_classifier4 import base
import radar_check as rc
import partno_labels as pl
import build_dataset as bd
from camera_utils import intrinsics_for_image
from rgbe_utils import RGBETransform

ap = argparse.ArgumentParser()
ap.add_argument('--since', default=None, help='YYYYMMDD 이후 세션 (기본: split 미지정 세션)')
ap.add_argument('--tag', default='unsplit')
ap.add_argument('--runs', nargs='*', default=['partno_multi_448_seed42', 'partno_part14_448_seed42', 'partno_multi_640_seed42'])
ap.add_argument('--detector', default=os.path.join(HERE, 'attribute_models', 'hole_landmarks_bracket', 'model.pth'))
ap.add_argument('--detector-blind', action='store_true', help='검출기가 이 세션들을 학습에 쓰지 않았음(블라인드) — 표에 표시')
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()


def load_cnn(run):
    info = json.load(open(os.path.join(HERE, 'artifacts', run, 'split_info.json')))
    sys.argv = [sys.argv[0], '--mode', info['mode'], '--image_size', str(info['image_size']), '--init', 'seed916']
    import importlib.util
    spec = importlib.util.spec_from_file_location('t02_' + run, os.path.join(HERE, '02_train_multitask.py')); t02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t02)
    net = t02.MTNet(info['mode'], len(pl.CLASSES), len(pl.PARTS)); net.load_state_dict(torch.load(os.path.join(HERE, 'artifacts', run, 'model.pth'), map_location='cpu'))
    return dict(run=run, mode=info['mode'], net=net.eval(), tf=RGBETransform(info['image_size'], is_train=False))


def cnn_predict(m, rgb, dev):
    x = m['tf'](Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits, rl = m['net'].to(dev)(x)
    if m['mode'] == 'part14':
        pi = int(logits.argmax(1)); return dict(part=pi, cls=None, radar=None)
    c = int(logits.argmax(1)); r = int(rl.item() > 0); cls = pl.CLASSES[c]
    return dict(part=pl.part_index(cls, r if pl.GROUP[cls] != 'FRT' else None), cls=cls, radar=(r if pl.GROUP[cls] != 'FRT' else None))


if __name__ == '__main__':
    con = sqlite3.connect(os.path.join(DOOR, 'db', 'door_pipeline.db')); con.row_factory = sqlite3.Row
    q = """SELECT s.id sid, s.session_dir, s.option_radar, s.option_source, c.name cls,
                  (SELECT split FROM images WHERE session_id=s.id LIMIT 1) split,
                  GROUP_CONCAT(i.rgb_path) rgbs, GROUP_CONCAT(i.depth_path) depths
           FROM images i JOIN classes c ON c.id=i.class_id JOIN capture_sessions s ON s.id=i.session_id
           WHERE c.dataset_id=5 AND i.synced_local AND i.is_valid AND c.name!='Unknown' GROUP BY s.id ORDER BY s.session_dir"""
    rows = [r for r in con.execute(q) if (r['session_dir'] >= a.since if a.since else r['split'] is None)]
    if a.limit: rows = rows[:a.limit]
    print(f"블라인드 세션 {len(rows)} ({'split 미지정' if not a.since else a.since + ' 이후'})")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net3, _ = base.load_model(); net4 = hc4.Net().to(dev); net4.load_state_dict(torch.load(a.detector, map_location=dev)); net4.eval()
    geom = rc.load_geometry(); cnns = [load_cnn(r) for r in a.runs if os.path.exists(os.path.join(HERE, 'artifacts', r, 'model.pth'))]
    frames = []; t0 = time.time()
    for si, r in enumerate(rows):
        gt_radar = r['option_radar'] if r['option_source'] == 'user' else None
        gt_part = pl.part_index(r['cls'], None if pl.GROUP[r['cls']] == 'FRT' else gt_radar)
        for rp, dp in zip(r['rgbs'].split(','), r['depths'].split(',')):
            rgb = cv2.imread(str(bd.MIRROR / rp)); depth = cv2.imread(str(bd.MIRROR / dp), cv2.IMREAD_UNCHANGED) if dp else None
            if rgb is None: continue
            intr = intrinsics_for_image(str(bd.MIRROR / rp), rgb.shape)
            # 규칙(운영 경로): 정본 3채널 판정 클래스 + 브래킷 투영
            b = base.classify(net3, dev, rgb, depth, intrinsics=intr); bc = b['pred'] if b['pred'] in pl.CLASSES else None
            rule = rc.check_frame(net3, dev, rgb, depth, bc, intr, geom, det=b['points']) if bc and pl.GROUP[bc] != 'FRT' else dict(status='n/a' if bc else 'no_cls')
            rr = None if rule['status'] not in ('radar', 'none') else int(rule['status'] == 'radar')
            rule_part = pl.part_index(bc, rr if (bc and pl.GROUP[bc] != 'FRT') else None) if bc else -1
            # 4채널 검출기
            d4 = hc4.classify(net4, dev, rgb, depth, intr, geom); c4 = d4['pred'] if d4['pred'] in pl.CLASSES else None
            r4 = None if d4['radar'] not in ('radar', 'none') else int(d4['radar'] == 'radar')
            det_part = pl.part_index(c4, r4 if (c4 and pl.GROUP[c4] != 'FRT') else None) if c4 else -1
            f = dict(session=r['session_dir'], cls=r['cls'], gt_radar=gt_radar, gt_part=gt_part, rule=dict(cls=bc, radar=rr, part=rule_part), det=dict(cls=c4, radar=r4, part=det_part))
            for m in cnns: f[m['run']] = cnn_predict(m, rgb, dev)
            frames.append(f)
        print(f"  [{si + 1}/{len(rows)}] {r['session_dir']:36s} {r['cls']:16s} GT 레이더 {gt_radar}  {time.time() - t0:.0f}s", flush=True)
    models = ['rule', 'det'] + [m['run'] for m in cnns]
    res = {}
    for m in models:
        c = collections.Counter(); sess = collections.defaultdict(list)
        for f in frames:
            p = f[m]['part']; sess[f['session']].append((f['gt_part'], p))
            if f['gt_part'] < 0: c['unconfirmed'] += 1; continue
            c['n'] += 1
            if p >= 0: c['judged'] += 1; c['ok'] += (p == f['gt_part'])
        s_n = s_ok = s_j = 0
        for sd, L in sess.items():
            gt = L[0][0]
            if gt < 0: continue
            votes = [p for _, p in L if p >= 0]; s_n += 1
            if votes: s_j += 1; s_ok += (collections.Counter(votes).most_common(1)[0][0] == gt)
        res[m] = dict(frames=c['n'], judged=c['judged'], correct=c['ok'], unconfirmed=c['unconfirmed'],
                      judged_rate=100 * c['judged'] / max(1, c['n']), acc=100 * c['ok'] / max(1, c['judged']),
                      sessions=s_n, s_judged=s_j, s_correct=s_ok, s_acc=100 * s_ok / max(1, s_j))
    stem = os.path.join(HERE, 'report', f'blind_{a.tag}'); os.makedirs(os.path.dirname(stem), exist_ok=True)
    json.dump(dict(tag=a.tag, since=a.since, sessions=[r['session_dir'] for r in rows], detector=os.path.relpath(a.detector, HERE), detector_blind=a.detector_blind, results=res, frames=frames),
              open(stem + '.json', 'w'), ensure_ascii=False, indent=1, default=str)
    md = [f"# 블라인드 평가 ({a.tag}) — 세션 {len(rows)}, 프레임 {len(frames)}, {time.strftime('%Y-%m-%d %H:%M')}", "",
          f"집합: {'DB split 미지정 세션(CNN 학습·val·test 미사용)' if not a.since else a.since + ' 이후 유입 세션'}. 검출기({os.path.relpath(a.detector, HERE)}): {'블라인드' if a.detector_blind else '이 세션들을 학습에 사용(비블라인드)'}. GT = DB 클래스 × 사용자 확정 레이더.", "",
          "| 모델 | 프레임 n | 판정률 | 품번 정확도 | 세션 n | 세션 판정 | 세션 정확도 | 미확정 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m, r in res.items():
        md.append(f"| {m} | {r['frames']} | {r['judged_rate']:.1f}% | {r['acc']:.1f}% ({r['correct']}/{r['judged']}) | {r['sessions']} | {r['s_judged']} | {r['s_acc']:.1f}% | {r['unconfirmed']} |")
    open(stem + '.md', 'w', encoding='utf8').write('\n'.join(md) + '\n'); print('\n'.join(md)); print('→', stem + '.md')
