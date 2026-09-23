"""CNN 품번 모델 평가 — 고정 test(또는 전량) 프레임·세션 단위 클래스/레이더/품번 정확도.

  python 03_evaluate_multitask.py --run partno_multi_448_seed42 [--set test|all] [--db-log]
산출: report/cnn_<run>_<set>.json/.md, DB evaluation_results(--db-log 시).
"""
import argparse, collections, json, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.normpath(os.path.join(HERE, '..', 'door_pipeline'))
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'db')); sys.path.insert(0, HERE)
import partno_labels as pl

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True); ap.add_argument('--set', choices=['test', 'all'], default='test')
ap.add_argument('--db-log', action='store_true'); ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()
RUN_DIR = os.path.join(HERE, 'artifacts', args.run)
info = json.load(open(os.path.join(RUN_DIR, 'split_info.json')))
sys.argv = [sys.argv[0], '--mode', info['mode'], '--image_size', str(info['image_size']), '--init', info['init']]   # 02 의 argparse 재사용
import importlib.util
spec = importlib.util.spec_from_file_location('t02', os.path.join(HERE, '02_train_multitask.py')); t02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t02)


def part_pred(mode, logits, rl):
    if mode == 'part14':
        return [int(i) for i in logits.argmax(1)], [None] * len(logits), [None] * len(logits)
    cls = [int(i) for i in logits.argmax(1)]; rad = [int(v > 0) for v in rl]
    parts = [pl.part_index(pl.CLASSES[c], (r if pl.GROUP[pl.CLASSES[c]] != 'FRT' else None)) for c, r in zip(cls, rad)]
    return parts, cls, rad


if __name__ == '__main__':
    man = pl.load(); rows = [r for r in man['rows'] if args.set == 'all' or r['split'] == 'test']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = t02.MTNet(info['mode'], len(pl.CLASSES), len(pl.PARTS)).to(dev)
    net.load_state_dict(torch.load(os.path.join(RUN_DIR, 'model.pth'), map_location=dev)); net.eval()
    dl = DataLoader(t02.DS(rows, False), batch_size=32, shuffle=False, num_workers=args.workers)
    preds = []; t0 = time.time()
    with torch.no_grad():
        for x, c, r, p in dl:
            logits, rl = net(x.to(dev)); pp, cc, rr = part_pred(info['mode'], logits.cpu(), rl.cpu() if rl is not None else None)
            preds += list(zip(pp, cc, rr))
    ms = 1000 * (time.time() - t0) / max(1, len(rows))
    res = collections.Counter(); per = collections.defaultdict(collections.Counter); sess = collections.defaultdict(list)
    for r, (pp, cc, rr) in zip(rows, preds):
        gt_part = r['part_idx']
        if info['mode'] == 'multi':
            res['cls_n'] += 1; res['cls_ok'] += (cc == r['cls_idx'])
            if r['radar'] >= 0: res['radar_n'] += 1; res['radar_ok'] += (rr == r['radar'])
        if gt_part >= 0:
            res['part_n'] += 1; ok = (pp == gt_part); res['part_ok'] += ok
            per[pl.PARTS[gt_part]]['n'] += 1; per[pl.PARTS[gt_part]]['ok'] += ok
        sess[r['session_dir']].append((r, pp, cc, rr))
    # 세션 다수결
    s_ok = s_n = 0; s_wrong = []
    for sd, L in sess.items():
        gts = {x[0]['part_idx'] for x in L}; gt = next(iter(gts)) if len(gts) == 1 else -1
        if gt < 0: continue
        maj = collections.Counter(x[1] for x in L).most_common(1)[0][0]; s_n += 1; s_ok += (maj == gt)
        if maj != gt: s_wrong.append((sd, pl.PARTS[gt], pl.PARTS[maj] if maj >= 0 else None))
    out = dict(run=args.run, set=args.set, n_frames=len(rows), ms_per_frame=round(ms, 1),
               cls_acc=100 * res['cls_ok'] / max(1, res['cls_n']) if info['mode'] == 'multi' else None,
               radar_acc=100 * res['radar_ok'] / max(1, res['radar_n']) if info['mode'] == 'multi' else None, radar_n=res['radar_n'],
               part_acc=100 * res['part_ok'] / max(1, res['part_n']), part_n=res['part_n'], part_ok=res['part_ok'],
               session_acc=100 * s_ok / max(1, s_n), session_n=s_n, session_wrong=s_wrong,
               per_part={k: dict(n=v['n'], ok=v['ok'], acc=100 * v['ok'] / v['n']) for k, v in sorted(per.items())})
    os.makedirs(os.path.join(HERE, 'report'), exist_ok=True); stem = os.path.join(HERE, 'report', f'cnn_{args.run}_{args.set}')
    json.dump(out, open(stem + '.json', 'w'), ensure_ascii=False, indent=1)
    md = [f"# CNN {args.run} — {args.set} ({len(rows)}장, {ms:.1f}ms/장)", "",
          f"- 프레임 품번 정확도 {out['part_ok']}/{out['part_n']} = **{out['part_acc']:.1f}%**" + (f" · 클래스 {out['cls_acc']:.1f}% · 레이더 {out['radar_acc']:.1f}% ({out['radar_n']}장)" if info['mode'] == 'multi' else ''),
          f"- 세션 다수결 품번 정확도 {s_ok}/{s_n} = **{out['session_acc']:.1f}%**" + (f" · 오판 {s_wrong}" if s_wrong else ''), "",
          "| 품번 | n | 정답 | 정확도 |", "|---|---:|---:|---:|"] + [f"| {k} | {v['n']} | {v['ok']} | {v['acc']:.1f}% |" for k, v in out['per_part'].items()]
    open(stem + '.md', 'w', encoding='utf8').write('\n'.join(md) + '\n'); print('\n'.join(md)); print('→', stem + '.md')
    if args.db_log:
        from db_log import DBLog
        db = DBLog(); mid = db.find_model(name=args.run)
        db.log_evaluation(model_id=mid, dataset_name='door_factory_collect', eval_type='cross_domain', total_samples=out['part_n'], correct=out['part_ok'],
                          accuracy=out['part_acc'], per_class_results=dict(set=f'partno_{args.set}', cls_acc=out['cls_acc'], radar_acc=out['radar_acc'], session_acc=out['session_acc'],
                                                                             per_part=out['per_part']), inference_time_ms=ms, inference_device=str(dev), report_path=os.path.relpath(stem + '.json', DOOR))
        db.close(); print('DB 기록')
