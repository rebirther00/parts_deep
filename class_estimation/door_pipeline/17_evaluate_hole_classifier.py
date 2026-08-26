"""홀 랜드마크 판별기 평가.

  python 17_evaluate_hole_classifier.py                 # ① 공식 test 분할(seed42) ② datasets 전체(라벨 학습분 제외) ③ datasets_field
  python 17_evaluate_hole_classifier.py --base datasets_factory   # 임의 디렉터리(<class>/rgb_*.png)

출력: attribute_models/hole_landmarks/eval_classifier.json + DB evaluation_results(inference_pipeline)
"""
import argparse, glob, json, os, time, collections
import cv2, numpy as np
from hole_classifier import load_model, classify, CAD_D, GROUP

DOOR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DOOR, 'attribute_models', 'hole_landmarks')
ap = argparse.ArgumentParser()
ap.add_argument('--base', default=None, help='평가 디렉터리 (기본: test분할 + datasets 전체 + datasets_field)')
ap.add_argument('--split_info', default=os.path.join(DOOR, 'artifacts', 'rgbe_noaux_448_seed42', 'split_info.json'))
ap.add_argument('--no_db', action='store_true')
ap.add_argument('--oracle_group', action='store_true', help='정답 그룹을 제약으로 사용 (그룹 선판별 상한 평가)')
args = ap.parse_args()


def train_label_keys():
    sp = os.path.join(OUT, 'split.json'); hold = set(json.load(open(sp))['holdout']) if os.path.exists(sp) else set()
    keys = set()
    for f in glob.glob(os.path.join(DOOR, 'labels', 'holes', '*.json')):
        k = os.path.basename(f)[:-5]
        if k not in hold:
            d = json.load(open(f)); keys.add(os.path.normpath(os.path.join(DOOR, d['image'])))
    return keys


def run_set(net, dev, name, files, cls_of):
    rows = []; t0 = time.time()
    for f in files:
        rgb = cv2.imread(f); dp = f.replace('rgb_', 'depth_')
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.exists(dp) else None
        if rgb is None:
            continue
        c0 = cls_of(f)
        r = classify(net, dev, rgb, depth, group=(GROUP.get(c0) if args.oracle_group else None))
        rows.append(dict(image=os.path.relpath(f, DOOR), cls=cls_of(f), pred=r['pred'], D_mm=r['D_mm'], gate=r['gate'],
                         D_src=r['D_src'], margin_mm=r.get('margin_mm')))
    # 집계
    valid = [r for r in rows if r['cls'] in CAD_D]; judged = [r for r in valid if r['pred']]
    acc = np.mean([r['pred'] == r['cls'] for r in judged]) * 100 if judged else 0
    gacc = np.mean([GROUP[r['pred']] == GROUP[r['cls']] for r in judged]) * 100 if judged else 0
    gates = collections.Counter(r['gate'] for r in valid)
    print(f"\n[{name}] {len(rows)}장  ({time.time() - t0:.0f}s)")
    print(f"  판정 {len(judged)}/{len(valid)} ({100 * len(judged) / max(1, len(valid)):.0f}%)  판정 정확도 클래스 {acc:.1f}% / 그룹 {gacc:.1f}%  "
          f"전체 대비 정답 {sum(r['pred'] == r['cls'] for r in judged)}/{len(valid)}   보류 사유 {dict(gates)}")
    print(f"  {'클래스':16s} {'n':>4s} {'판정':>4s} {'정답':>4s} {'정확도':>6s} | {'D med':>7s} {'CAD':>5s} {'차이':>5s} | 오판 내역")
    cm = collections.Counter()
    for c in sorted(CAD_D):
        rr = [r for r in valid if r['cls'] == c]; jj = [r for r in rr if r['pred']]
        if not rr: continue
        ok = sum(r['pred'] == c for r in jj); d = np.array([r['D_mm'] for r in jj])
        wrong = collections.Counter(r['pred'] for r in jj if r['pred'] != c)
        for r in jj: cm[(c, r['pred'])] += 1
        print(f"  {c:16s} {len(rr):4d} {len(jj):4d} {ok:4d} {100 * ok / max(1, len(jj)):5.1f}% | {np.median(d) if d.size else 0:7.1f} {CAD_D[c]:5d} {np.median(d) - CAD_D[c] if d.size else 0:+5.0f} | {dict(wrong) if wrong else ''}")
    return dict(rows=rows, n=len(valid), judged=len(judged), correct=int(sum(r['pred'] == r['cls'] for r in judged)),
                acc_judged=acc, group_acc_judged=gacc, gates=dict(gates))


if __name__ == '__main__':
    net, dev = load_model()
    results = {}
    if args.base:
        base = os.path.join(DOOR, args.base)
        files = sorted(glob.glob(os.path.join(base, '*', 'rgb_*.png')))
        results[args.base] = run_set(net, dev, args.base, files, lambda f: os.path.basename(os.path.dirname(f)))
    else:
        trained = train_label_keys()
        # ① 공식 test 분할
        if os.path.exists(args.split_info):
            sp = json.load(open(args.split_info))
            files = [p if os.path.isabs(p) else os.path.join(DOOR, p) for p in sp['test_paths']]
            files = [f for f in files if os.path.normpath(f) not in trained]
            results['test_split'] = run_set(net, dev, 'test 분할(seed42, 라벨 학습분 제외)', files, lambda f: os.path.basename(os.path.dirname(f)))
        # ② datasets 전체 (라벨 학습분 제외)
        files = [f for f in sorted(glob.glob(os.path.join(DOOR, 'datasets', '*', 'rgb_*.png'))) if os.path.normpath(f) not in trained]
        results['datasets_all'] = run_set(net, dev, 'datasets 전체(라벨 학습분 제외)', files, lambda f: os.path.basename(os.path.dirname(f)))
        # ③ 현장
        files = sorted(glob.glob(os.path.join(DOOR, 'datasets_field', '*', 'rgb_*.png')))
        if files:
            results['datasets_field'] = run_set(net, dev, 'datasets_field(현장)', files, lambda f: os.path.basename(os.path.dirname(f)).split('_s_')[0])
    json.dump(results, open(os.path.join(OUT, 'eval_classifier.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    if not args.no_db:
        from db.db_log import DBLog
        db = DBLog(); mid = db.find_model(weights_path='attribute_models/hole_landmarks/model.pth', name='hole_landmarks_resnet18')
        dsmap = {'test_split': 'door_real', 'datasets_all': 'door_real', 'datasets_field': 'door_field'}
        for k, v in results.items():
            db.log_evaluation(model_id=mid, dataset_name=dsmap.get(k, k), eval_type='inference_pipeline', total_samples=v['n'],
                              correct=v['correct'], accuracy=100.0 * v['correct'] / max(1, v['n']),
                              per_class_results=dict(set=k, judged=v['judged'], acc_judged=v['acc_judged'], group_acc_judged=v['group_acc_judged'], gates=v['gates']),
                              inference_device=str(dev), report_path='attribute_models/hole_landmarks/eval_classifier.json')
        db.close()
