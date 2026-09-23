"""세션 5-fold 교차검증 — CNN 품번 모델(02_train_multitask.py) 용. door_pipeline/scripts/session_cv.py 와 같은 fold 규칙(클래스별 시간순 라운드로빈, 세션<3 클래스는 train 고정).

  python scripts/session_cv.py --mode multi --image_size 448 --tag cvA            # fold 마다 manifest 의 split 을 덮어써 학습·평가
fold 결과: report/cv_<tag>_<mode>_<size>.json (fold별 품번 정확도·합산).
"""
import argparse, collections, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import partno_labels as pl
ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='multi'); ap.add_argument('--image_size', type=int, default=448)
ap.add_argument('--k', type=int, default=5); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--tag', required=True)
ap.add_argument('--epochs', type=int, default=30)
a = ap.parse_args()
man = pl.load(); rows = man['rows']
by = collections.defaultdict(list)
for r in rows:
    if r['session_dir']: by[r['cls']].append(r['session_dir'])
fold_of = {}
for cls, ss in by.items():
    uniq = sorted(set(ss))
    for i, sd in enumerate(uniq): fold_of[sd] = (i % a.k) if len(uniq) >= 3 else -1
results = {}; PY = sys.executable; backup = pl.MANIFEST + '.bak'
os.replace(pl.MANIFEST, backup) if not os.path.exists(backup) else None
try:
    for k in range(a.k):
        for r in rows: r['split'] = 'test' if fold_of.get(r['session_dir']) == k else 'train'
        # val: train 세션의 15% 를 세션 단위로 떼어 체크포인트 선택
        tr_sess = sorted({r['session_dir'] for r in rows if r['split'] == 'train'}); vs = set(tr_sess[::7])
        for r in rows:
            if r['split'] == 'train' and r['session_dir'] in vs: r['split'] = 'val'
        json.dump(dict(classes=man['classes'], parts=man['parts'], rows=rows), open(pl.MANIFEST, 'w'), ensure_ascii=False)
        run = f"partno_{a.mode}_{a.image_size}_seed{a.seed}_{a.tag}_fold{k}"; t0 = time.time()
        print(f"=== fold {k}: test 세션 {sum(1 for v in fold_of.values() if v == k)}, test 프레임 {sum(r['split'] == 'test' for r in rows)}", flush=True)
        subprocess.run([PY, '-u', os.path.join(HERE, '02_train_multitask.py'), '--mode', a.mode, '--image_size', str(a.image_size), '--seed', str(a.seed),
                        '--epochs', str(a.epochs), '--tag', f'_{a.tag}_fold{k}'], check=True, stdout=open(os.path.join(HERE, 'logs', f'cv_{a.tag}_fold{k}.log'), 'w'), stderr=subprocess.STDOUT, cwd=HERE)
        subprocess.run([PY, '-u', os.path.join(HERE, '03_evaluate_multitask.py'), '--run', run], check=True, stdout=open(os.path.join(HERE, 'logs', f'cv_{a.tag}_fold{k}_eval.log'), 'w'), stderr=subprocess.STDOUT, cwd=HERE)
        res = json.load(open(os.path.join(HERE, 'report', f'cnn_{run}_test.json')))
        results[k] = dict(run=run, part_acc=res['part_acc'], part_n=res['part_n'], part_ok=res['part_ok'], cls_acc=res['cls_acc'], radar_acc=res['radar_acc'], session_acc=res['session_acc'], minutes=round((time.time() - t0) / 60, 1))
        print(f"fold {k}: 품번 {res['part_acc']:.1f}% ({res['part_ok']}/{res['part_n']}) 세션 {res['session_acc']:.1f}%  {results[k]['minutes']}분", flush=True)
        json.dump(results, open(os.path.join(HERE, 'report', f'cv_{a.tag}_{a.mode}_{a.image_size}.json'), 'w'), indent=1)
finally:
    os.replace(backup, pl.MANIFEST)
tot = sum(r['part_n'] for r in results.values()); ok = sum(r['part_ok'] for r in results.values())
print(f"[{a.k}-fold] 품번 합산 {ok}/{tot} = {100 * ok / max(1, tot):.2f}%  fold별 {[round(r['part_acc'], 1) for r in results.values()]}")
