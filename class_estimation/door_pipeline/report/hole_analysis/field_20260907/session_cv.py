"""세션 그룹 K-fold 교차검증 (DB 미변경, DB 로그 없음).
클래스별 세션을 시간순으로 K개 fold에 라운드로빈 배정 → fold k = test, 나머지 = train(val은 02_train이 프레임 15% 자동).
세션<3 클래스는 train 고정(평가 제외). 각 fold 뷰는 심볼릭 링크로 스크래치에 생성, 학습은 02_train --presplit, 평가는 04.
사용: python session_cv.py <out_root> [K]
"""
import sqlite3, os, sys, json, subprocess, collections, shutil, time
from pathlib import Path
DOOR = Path('/home/koceti/parts_deep/class_estimation/door_pipeline'); MIRROR = DOOR / 'datasets_factory_collect'
PY = os.path.expanduser('~/parts_deep/venv/parts_deep/bin/python')
OUT = Path(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
con = sqlite3.connect(DOOR / 'db' / 'door_pipeline.db'); con.row_factory = sqlite3.Row
rows = con.execute("""SELECT s.id, s.session_dir, c.name cls, s.started_at FROM capture_sessions s
   JOIN datasets d ON d.id=s.dataset_id JOIN images i ON i.session_id=s.id JOIN classes c ON c.id=i.class_id
   WHERE d.name='door_factory_collect' AND c.name!='Unknown' AND i.is_valid AND i.synced_local
   GROUP BY s.id ORDER BY c.name, s.started_at""").fetchall()
by = collections.defaultdict(list)
for r in rows: by[r['cls']].append(dict(r))
fold_of = {}
for cls, ss in by.items():
    for i, r in enumerate(ss):
        fold_of[r['session_dir']] = (i % K) if len(ss) >= 3 else -1     # -1: 항상 train
imgs = {}
for cls, ss in by.items():
    for r in ss:
        imgs[r['session_dir']] = (cls, con.execute("SELECT rgb_path, depth_path, rgb_filename FROM images WHERE session_id=? AND is_valid AND synced_local ORDER BY rgb_filename", (r['id'],)).fetchall())
def build(root, k):
    if root.exists(): shutil.rmtree(root)
    n = collections.Counter()
    for sd, (cls, ims) in imgs.items():
        t = 'test' if fold_of[sd] == k else 'train'
        day, _, sess = sd.split('/')
        d = root / t / cls; d.mkdir(parents=True, exist_ok=True)
        for im in ims:
            idx = im['rgb_filename'].replace('rgb_', '').replace('.png', '')
            (d / f'rgb_{day}_{sess}_{idx}.png').symlink_to(MIRROR / im['rgb_path'])
            if im['depth_path']: (d / f'depth_{day}_{sess}_{idx}.png').symlink_to(MIRROR / im['depth_path'])
            n[(t, cls)] += 1
    return n
results = {}
OUT.mkdir(parents=True, exist_ok=True)
json.dump({'K': K, 'fold_of': fold_of}, open(OUT / 'folds.json', 'w'), ensure_ascii=False, indent=1)
for k in range(K):
    root = OUT / f'fold{k}'; n = build(root, k)
    print(f'\n=== fold {k}: test 세션 {sum(1 for v in fold_of.values() if v == k)}개, test 이미지 {sum(v for (t, c), v in n.items() if t == "test")}장', flush=True)
    run = DOOR / 'artifacts' / f'rgbe_noaux_448_seed42_fold{k}'
    if run.exists(): shutil.rmtree(run)
    env = dict(os.environ, DOOR_DB_LOG='0')
    t0 = time.time()
    subprocess.run([PY, '-u', str(DOOR / '02_train.py'), '--model_type', 'rgbe', '--no_aux', '--image_size', '448', '--seed', '42',
                    '--dataset_dir', str(root), '--presplit'], env=env, check=True, stdout=open(OUT / f'train_fold{k}.log', 'w'), stderr=subprocess.STDOUT, cwd=DOOR)
    subprocess.run([PY, '-u', str(DOOR / '04_evaluate_factory.py'), '--model', str(run / 'model.pth'), '--dataset_dir', str(root / 'test')],
                   env=env, check=True, stdout=open(OUT / f'eval_fold{k}.log', 'w'), stderr=subprocess.STDOUT, cwd=DOOR)
    res = json.load(open(run / 'factory_eval_results.json'))
    results[k] = dict(accuracy=res['accuracy'], total=res['total_samples'], correct=res['correct_samples'], class_acc=res['class_accuracies'],
                      minutes=round((time.time() - t0) / 60, 1))
    print(f'fold {k}: {res["accuracy"]}% ({res["correct_samples"]}/{res["total_samples"]})  {results[k]["minutes"]}분', flush=True)
    json.dump(results, open(OUT / 'cv_results.json', 'w'), ensure_ascii=False, indent=1)
tot = sum(r['total'] for r in results.values()); cor = sum(r['correct'] for r in results.values())
print(f'\n[{K}-fold 세션 CV] 합산 {cor}/{tot} = {100 * cor / tot:.2f}%   fold별', [r['accuracy'] for r in results.values()])
