"""의사 라벨 생성 — 확정 라벨(DB 클래스 × 사용자 확정 레이더)로 홀 랜드마크 4채널 검출기 학습용 json 을 만든다.

  python 15_make_pseudo_labels.py                 # DB 유효 세션 전체(로컬 20장) + 사람 라벨 134장의 브래킷 보강
  python 15_make_pseudo_labels.py --limit 20      # 디버그

프레임마다:
  ① door_pipeline 검출기(3채널)로 볼트 4·코너 2 검출, geometry_gate 통과 프레임만 채택 → 6점 의사 라벨(검출기 홀드아웃 오차 ~1px)
  ② 세션 레이더 확정값이 1 이면 partno/radar_check.check_frame 로 브래킷 홀 2개 위치(투영 + 국소 탐색) → 'radar' 판정 프레임만 브래킷 점 기록,
     그 외(가림·미정)는 bracket_known=false 로 두어 학습 시 브래킷 채널 손실에서 제외
     확정값 0 또는 FRT 클래스면 브래킷 없음(bracket_known=true, 음성 예)
  ③ split 은 DB images.split(세션 단위: test 는 학습 제외, val 은 모니터링)
사람 라벨 134장(labels/holes, 사무실·현장)은 6점을 그대로 쓰고 브래킷은 규칙 판정으로 보강(세션 GT 없음 → 'radar'/'none' 만 채택).
출력: labels/holes_pseudo/<date>_<sess>_<idx>.json (필드: image, cls, split, radar_gt, points{bolt_1..4, corner_hinge, corner_latch, bracket_1, bracket_2}, bracket_known, src='pseudo')
      labels/holes_pseudo/human_<key>.json (사람 라벨 + 브래킷 보강, src='human+rule')
      labels/pseudo_summary.json
"""
import argparse, collections, glob, json, os, sqlite3, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.normpath(os.path.join(HERE, '..', 'door_pipeline'))
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'db')); sys.path.insert(0, os.path.join(DOOR, 'partno'))
import hole_classifier as hc
from camera_utils import intrinsics_for_image
import build_dataset as bd
import radar_check as rc

OUT = os.path.join(HERE, 'labels', 'holes_pseudo')
ap = argparse.ArgumentParser()
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--min-score', type=float, default=0.4, help='6점 각 피크 최소 점수')
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)


def six_points(det):
    if len(det['bolt']) < 4 or not det['corner_hinge'] or not det['corner_latch']:
        return None
    pts = {f'bolt_{i + 1}': [float(b[0]), float(b[1])] for i, b in enumerate(det['bolt'][:4])}
    pts['corner_hinge'] = [float(det['corner_hinge'][0][0]), float(det['corner_hinge'][0][1])]
    pts['corner_latch'] = [float(det['corner_latch'][0][0]), float(det['corner_latch'][0][1])]
    scores = [b[2] for b in det['bolt'][:4]] + [det['corner_hinge'][0][2], det['corner_latch'][0][2]]
    return pts, min(scores)


def bracket_from_rule(net, dev, rgb, depth, cls, intr, geom, det):
    f = rc.check_frame(net, dev, rgb, depth, cls, intr, geom, det=det)
    if f.get('status') == 'radar':
        dx, dy = f['dxy']
        return {'bracket_1': [float(f['holes_px'][0][0] + dx), float(f['holes_px'][0][1] + dy)],
                'bracket_2': [float(f['holes_px'][1][0] + dx), float(f['holes_px'][1][1] + dy)]}, f
    return None, f


def main():
    con = sqlite3.connect(os.path.join(DOOR, 'db', 'door_pipeline.db')); con.row_factory = sqlite3.Row
    net, dev = hc.load_model(); geom = rc.load_geometry()
    rows = con.execute("""SELECT s.id sid, s.session_dir, s.option_radar, s.option_source, c.name cls,
                                 (SELECT split FROM images WHERE session_id=s.id LIMIT 1) split,
                                 GROUP_CONCAT(i.rgb_path) rgbs, GROUP_CONCAT(i.depth_path) depths
                          FROM images i JOIN classes c ON c.id=i.class_id JOIN capture_sessions s ON s.id=i.session_id
                          WHERE c.dataset_id=5 AND i.synced_local AND i.is_valid AND c.name!='Unknown'
                          GROUP BY s.id ORDER BY s.session_dir""").fetchall()
    if args.limit: rows = rows[:args.limit]
    stat = collections.Counter(); t0 = time.time(); n_files = 0
    for si, r in enumerate(rows):
        radar_gt = r['option_radar'] if r['option_source'] == 'user' else None
        if hc.GROUP[r['cls']] == 'FRT': radar_gt = 0          # FRT: 옵션 없음 → 브래킷 음성
        date, _, sess = r['session_dir'].split('/')
        for rp, dp in zip(r['rgbs'].split(','), r['depths'].split(',')):
            rgb = cv2.imread(str(bd.MIRROR / rp)); depth = cv2.imread(str(bd.MIRROR / dp), cv2.IMREAD_UNCHANGED) if dp else None
            if rgb is None: stat['no_image'] += 1; continue
            intr = intrinsics_for_image(str(bd.MIRROR / rp), rgb.shape)
            det = hc.detect(net, dev, rgb)
            sp = six_points(det)
            hinge = det['corner_hinge'][0] if det['corner_hinge'] else None; latch = det['corner_latch'][0] if det['corner_latch'] else None
            gate = hc.geometry_gate(hc.bolt_frame(det['bolt']), hinge, latch, rgb.shape)
            if sp is None or gate != 'ok' or sp[1] < args.min_score:
                stat['skip_6pt'] += 1; continue
            pts, minsc = sp
            bracket_known, rule = True, None
            if radar_gt == 1:
                br, rule = bracket_from_rule(net, dev, rgb, depth, r['cls'], intr, geom, det)
                if br: pts.update(br); stat['bracket_pos'] += 1
                else: bracket_known = False; stat['bracket_unknown'] += 1
            elif radar_gt == 0:
                stat['bracket_neg'] += 1
            else:
                bracket_known = False; stat['radar_gt_none'] += 1
            idx = os.path.basename(rp).replace('rgb_', '').replace('.png', '')
            rec = dict(image=os.path.relpath(str(bd.MIRROR / rp), HERE), cls=r['cls'], session_dir=r['session_dir'], split=r['split'],
                       radar_gt=radar_gt, points=pts, bracket_known=bracket_known, min_score=round(minsc, 3),
                       rule_status=(rule or {}).get('status'), rule_score=(rule or {}).get('score'), src='pseudo')
            json.dump(rec, open(os.path.join(OUT, f'{date}_{sess}_{idx}.json'), 'w'), ensure_ascii=False)
            n_files += 1
        print(f"  [{si + 1}/{len(rows)}] {r['session_dir']:36s} {r['cls']:16s} radar={radar_gt} split={r['split']}  {dict(stat)}  {time.time() - t0:.0f}s", flush=True)
    # 사람 라벨 134장: 6점 그대로 + 브래킷 규칙 보강
    hstat = collections.Counter()
    for f in sorted(glob.glob(os.path.join(DOOR, 'labels', 'holes', '*.json'))):
        d = json.load(open(f)); P = d['points']; vis = d.get('visible', {})
        need = ['bolt_tl', 'bolt_tr', 'bolt_bl', 'bolt_br', 'corner_hinge', 'corner_latch']
        if not all(P.get(k) and vis.get(k, True) for k in need): hstat['incomplete'] += 1; continue
        img = os.path.join(DOOR, d['image']); rgb = cv2.imread(img)
        if rgb is None: hstat['no_image'] += 1; continue
        dpath = img.replace('rgb_', 'depth_'); depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED) if os.path.exists(dpath) else None
        intr = intrinsics_for_image(img, rgb.shape)
        det = {'bolt': [(*P[k], 1.0) for k in need[:4]], 'corner_hinge': [(*P['corner_hinge'], 1.0)], 'corner_latch': [(*P['corner_latch'], 1.0)]}
        pts = {f'bolt_{i + 1}': [float(P[k][0]), float(P[k][1])] for i, k in enumerate(need[:4])}
        pts['corner_hinge'] = [float(P['corner_hinge'][0]), float(P['corner_hinge'][1])]; pts['corner_latch'] = [float(P['corner_latch'][0]), float(P['corner_latch'][1])]
        bracket_known, radar_gt, rule = True, 0, None
        if d['cls'] in geom:
            br, rule = bracket_from_rule(net, dev, rgb, depth, d['cls'], intr, geom, det)
            if br: pts.update(br); radar_gt = 1; hstat['bracket_pos'] += 1
            elif rule.get('status') == 'none': radar_gt = 0; hstat['bracket_neg'] += 1
            else: bracket_known = False; radar_gt = None; hstat['bracket_unknown'] += 1
        else:
            hstat['frt'] += 1
        rec = dict(image=os.path.relpath(img, HERE), cls=d['cls'], session_dir=None, split='human', key=os.path.basename(f)[:-5],
                   radar_gt=radar_gt, points=pts, bracket_known=bracket_known, rule_status=(rule or {}).get('status'),
                   rule_score=(rule or {}).get('score'), src='human+rule')
        json.dump(rec, open(os.path.join(OUT, 'human_' + os.path.basename(f)), 'w'), ensure_ascii=False)
        n_files += 1
    summary = dict(sessions=len(rows), files=n_files, pseudo=dict(stat), human=dict(hstat), min_score=args.min_score, made_at=time.strftime('%Y-%m-%d %H:%M'))
    json.dump(summary, open(os.path.join(HERE, 'labels', 'pseudo_summary.json'), 'w'), ensure_ascii=False, indent=1)
    print('\n요약:', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
