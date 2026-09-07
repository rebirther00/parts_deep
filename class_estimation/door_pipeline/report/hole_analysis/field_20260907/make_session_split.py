"""DB를 건드리지 않고 세션 단위 70/15/15 분할 뷰를 스크래치에 만든다 (build_dataset.auto-split 규칙과 동일:
클래스별 세션 시간순, test=max(1,round(15%)) 가장 이른 세션, val=다음 max(1,round(15%)), 나머지 train.
DB에 이미 지정된 split은 그대로 존중)."""
import sqlite3, os, json, sys, collections
from pathlib import Path
DOOR = Path('/home/koceti/parts_deep/class_estimation/door_pipeline')
MIRROR = DOOR / 'datasets_factory_collect'
OUT = Path(sys.argv[1])
con = sqlite3.connect(DOOR / 'db' / 'door_pipeline.db'); con.row_factory = sqlite3.Row
rows = con.execute("""SELECT s.id, s.session_dir, c.name cls, s.started_at,
   (SELECT split FROM images WHERE session_id=s.id LIMIT 1) split
   FROM capture_sessions s JOIN datasets d ON d.id=s.dataset_id
   JOIN images i ON i.session_id=s.id JOIN classes c ON c.id=i.class_id
   WHERE d.name='door_factory_collect' AND c.name!='Unknown' AND i.is_valid AND i.synced_local
   GROUP BY s.id ORDER BY c.name, s.started_at""").fetchall()
by = collections.defaultdict(list)
for r in rows: by[r['cls']].append(dict(r))
assign = {}
for cls, ss in by.items():
    n = len(ss)
    if n < 3:  # 세션 3개 미만: 분할 불가 → 전부 train (평가 제외)
        for r in ss: r["split"] = r["split"] or "train"
        for r in ss: assign[r["session_dir"]] = r["split"]
        continue
    target = {'test': max(1, round(.15*n)), 'val': max(1, round(.15*n)) if n >= 3 else 0}
    have = {k: sum(r['split']==k for r in ss) for k in target}
    for r in ss:
        if r['split']: continue
        for k in ('test','val'):
            if have[k] < target[k]: r['split']=k; have[k]+=1; break
        else: r['split']='train'
    for r in ss: assign[r['session_dir']] = r['split']
if OUT.exists():
    import shutil; shutil.rmtree(OUT)
manifest = []; cnt = collections.Counter()
for cls, ss in by.items():
    for r in ss:
        imgs = con.execute("SELECT rgb_path, depth_path, rgb_filename FROM images WHERE session_id=? AND is_valid AND synced_local ORDER BY rgb_filename", (r['id'],)).fetchall()
        for im in imgs:
            day, _, sess = r['session_dir'].split('/')
            idx = im['rgb_filename'].replace('rgb_','').replace('.png','')
            for t in (r['split'], 'all'):
                d = OUT / t / cls; d.mkdir(parents=True, exist_ok=True)
                (d / f"rgb_{day}_{sess}_{idx}.png").symlink_to(MIRROR / im['rgb_path'])
                if im['depth_path']: (d / f"depth_{day}_{sess}_{idx}.png").symlink_to(MIRROR / im['depth_path'])
            manifest.append(dict(session_dir=r['session_dir'], cls=cls, split=r['split'], rgb=im['rgb_path']))
            cnt[(cls, r['split'])] += 1
sess_cnt = collections.Counter((c, s) for c, ss in by.items() for s in [r['split'] for r in ss])
print(f"{'클래스':16s} " + " ".join(f"{k+'(세션/장)':>14s}" for k in ('train','val','test')))
for cls in sorted(by):
    print(f"{cls:16s} " + " ".join(f"{sess_cnt[(cls,k)]:5d}/{cnt[(cls,k)]:<8d}" for k in ('train','val','test')))
print("합계 세션", sum(sess_cnt.values()), "이미지", len(manifest))
(OUT / 'manifest.json').write_text(json.dumps(dict(assign=assign, images=manifest), ensure_ascii=False, indent=1))
