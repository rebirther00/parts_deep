"""공통 평가 뷰: 현재 DB 기준 유효·라벨확정·로컬 세션 전부(Unknown·무효 제외), 세션당 최대 20장 시간축 균등 선별.
세션 3개 미만 클래스(E25_LH_FRT)는 CNN CV에서 시험 불가하므로 공통 집합에서 제외하고 별도 폴더(extra/)에 둔다.
출력: <out>/common/<class>/rgb_<date>_<sess>_<idx>.png(+depth), <out>/extra/..., <out>/manifest.json"""
import sqlite3, sys, json, collections, shutil
from pathlib import Path
DOOR=Path('/home/koceti/parts_deep/class_estimation/door_pipeline'); MIRROR=DOOR/'datasets_factory_collect'; OUT=Path(sys.argv[1]); CAP=20
con=sqlite3.connect(DOOR/'db'/'door_pipeline.db'); con.row_factory=sqlite3.Row
def pick_even(items,n):
    if len(items)<=n: return list(items)
    idx=sorted({round(i*(len(items)-1)/(n-1)) for i in range(n)}); return [items[i] for i in idx]
sess=con.execute("""SELECT s.id, s.session_dir, c.name cls FROM capture_sessions s JOIN datasets d ON d.id=s.dataset_id
 JOIN images i ON i.session_id=s.id JOIN classes c ON c.id=i.class_id
 WHERE d.name='door_factory_collect' AND c.name!='Unknown' AND i.is_valid AND i.synced_local GROUP BY s.id ORDER BY c.name, s.started_at""").fetchall()
n_by=collections.Counter(r['cls'] for r in sess)
if OUT.exists(): shutil.rmtree(OUT)
man=[]; cnt=collections.Counter(); scnt=collections.Counter()
for r in sess:
    ims=con.execute("SELECT rgb_path, depth_path, rgb_filename FROM images WHERE session_id=? AND is_valid AND synced_local ORDER BY rgb_filename",(r['id'],)).fetchall()
    ims=pick_even(ims,CAP); grp='common' if n_by[r['cls']]>=3 else 'extra'
    day,_,ss=r['session_dir'].split('/'); d=OUT/grp/r['cls']; d.mkdir(parents=True,exist_ok=True)
    for im in ims:
        idx=im['rgb_filename'][4:-4]
        (d/f'rgb_{day}_{ss}_{idx}.png').symlink_to(MIRROR/im['rgb_path'])
        if im['depth_path']: (d/f'depth_{day}_{ss}_{idx}.png').symlink_to(MIRROR/im['depth_path'])
        man.append(dict(group=grp, cls=r['cls'], session_dir=r['session_dir'], rgb=im['rgb_path'])); cnt[(grp,r['cls'])]+=1
    scnt[(grp,r['cls'])]+=1
json.dump(dict(cap=CAP, images=man), open(OUT/'manifest.json','w'), ensure_ascii=False, indent=1)
for grp in ('common','extra'):
    print(f'[{grp}] 세션 {sum(v for (g,c),v in scnt.items() if g==grp)}  이미지 {sum(v for (g,c),v in cnt.items() if g==grp)}')
    for c in sorted({c for (g,c) in cnt if g==grp}): print(f'   {c:16s} 세션 {scnt[(grp,c)]:2d}  장 {cnt[(grp,c)]:3d}')
