"""품번(14) 추론 — 임의 폴더의 rgb_*.png(+depth_*.png)에 대해 형상군(홀 판별기 D) × 레이더 옵션(규칙 검사) → 품번.

  python partno/infer_partno.py datasets_factory_collect/20260914/E25_door_RH/s_124221          # 세션 폴더 1개
  python partno/infer_partno.py datasets_factory_v2/test --recursive                             # <class>/rgb_*.png 또는 <date>/<class>/s_*/rgb_*.png
  python partno/infer_partno.py <dir> --cls E25_door_RH        # 형상군을 지정(홀 판정 대신)하고 옵션만 검사
  python partno/infer_partno.py <dir> --frames --json out.json # 프레임별 표 + JSON 저장

프레임: hole_classifier.classify(픽셀 폭 D → 9종 판정) + radar_check.check_frame(브래킷 홀 투영 점수).
세션(폴더): 클래스 = D 중앙값 최근접(hole_classifier.aggregate), 레이더 = 표 다수결(radar_check.aggregate) → part_numbers.json 조회.
FRT 는 옵션 없음(레이더 검사 생략). 학습 모델은 홀 랜드마크 검출기 하나만 쓴다(레이더 판정은 규칙).
"""
import argparse, collections, glob, json, os, sys
import cv2

HERE = os.path.dirname(os.path.abspath(__file__)); DOOR = os.path.dirname(HERE)
sys.path.insert(0, DOOR); sys.path.insert(0, HERE)
import hole_classifier as hc
from camera_utils import intrinsics_for_image
import radar_check as rc

PN = {(p['class_name'], p['radar']): p for p in json.load(open(os.path.join(HERE, 'part_numbers.json')))['parts']}


def part_for(cls, radar):
    if cls is None or cls == hc.UNKNOWN:
        return None
    return PN.get((cls, None)) or (PN.get((cls, radar)) if radar is not None else None)


def group_files(root, recursive):
    pat = os.path.join(root, '**', 'rgb_*.png') if recursive else os.path.join(root, 'rgb_*.png')
    files = sorted(glob.glob(pat, recursive=recursive))
    by = collections.OrderedDict()
    for f in files:
        by.setdefault(os.path.relpath(os.path.dirname(f), root) or '.', []).append(f)
    return by


def infer_dir(net, dev, geom, files, cls_fixed=None):
    rows = []
    for f in files:
        rgb = cv2.imread(f); dp = f.replace('rgb_', 'depth_')
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.exists(dp) else None
        if rgb is None:
            continue
        intr = intrinsics_for_image(f, rgb.shape)
        r = hc.classify(net, dev, rgb, depth, intrinsics=intr)
        cls = cls_fixed or r['pred']
        opt = rc.check_frame(net, dev, rgb, depth, cls, intr, geom, det=r['points']) if cls in geom else dict(status='n/a')
        rows.append(dict(image=f, cls=r['pred'], D_mm=r['D_mm'], gate=r['gate'], margin_mm=r.get('margin_mm'),
                         radar=opt['status'], score=opt.get('score'), used_cls=cls))
    agg = hc.aggregate([dict(gate=x['gate'], D_mm=x['D_mm']) for x in rows])
    cls = cls_fixed or agg['pred']
    radar = rc.aggregate([dict(status=x['radar'], score=x['score']) for x in rows]) if cls in geom else None
    flag = radar['auto_flag'] if radar else None
    part = part_for(cls, flag)
    return dict(n=len(rows), cls=cls, D_mm=agg['D_mm'], n_judged=agg['n_judged'], radar=flag,
                radar_votes=(f"O{radar['n_radar']}/X{radar['n_none']}/판정{radar['n_judged']}" if radar else '해당없음' if cls and 'FRT' in str(cls) else '—'),
                part_no=(part['part_no'] + '-' + part['rev']) if part else None, part_name=part['name'] if part else None, frames=rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--recursive', action='store_true'); ap.add_argument('--cls', default=None)
    ap.add_argument('--frames', action='store_true'); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root = a.root if os.path.isabs(a.root) else os.path.join(DOOR, a.root)
    by = group_files(root, a.recursive)
    if not by:
        raise SystemExit(f'rgb_*.png 없음: {root}')
    net, dev = hc.load_model(); geom = rc.load_geometry()
    out = {}
    print(f"{'폴더':44s} {'n':>3s} {'판정':>3s} {'클래스':16s} {'D':>7s} {'레이더':>6s} {'표':>16s}  품번")
    for d, files in by.items():
        res = infer_dir(net, dev, geom, files, a.cls)
        out[d] = res
        print(f"{d[-44:]:44s} {res['n']:3d} {res['n_judged']:3d} {str(res['cls']):16s} {res['D_mm'] if res['D_mm'] else 0:7.1f} "
              f"{ {1: 'O', 0: 'X', None: '미정'}[res['radar']]:>6s} {res['radar_votes']:>16s}  {res['part_no'] or '—'} {res['part_name'] or ''}")
        if a.frames:
            for x in res['frames']:
                print(f"      {os.path.basename(x['image']):34s} {str(x['cls']):16s} {x['D_mm'] if x['D_mm'] else 0:7.1f} {x['gate']:12s} {x['radar']:10s} {x['score'] if x['score'] is not None else float('nan'):5.2f}")
    if a.json:
        json.dump(out, open(a.json, 'w'), ensure_ascii=False, indent=1, default=str); print(f"→ {a.json}")
