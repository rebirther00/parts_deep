"""블라인드 평가 뷰 생성 (심볼릭 링크, datasets_blind_unsplit_20261023 와 같은 구조). 2026-11-04.

DB 기준 유효·라벨 확정(Unknown 제외)·로컬 동기화 세션 가운데 조건에 맞는 세션을 세션당 최대 20장(시간축 균등)으로 모아
  <out>/collect/<date>/<DB 클래스>/<sess>/rgb_XXXX.png (+depth)   → 17_evaluate_hole_classifier.py --base <out>/collect  (06 수집 구조)
  <out>/cnn/<DB 클래스>/rgb_<date>_<sess>_<idx>.png (+depth)       → 04_evaluate_factory.py --dataset_dir <out>/cnn
를 만든다. 클래스 폴더명은 DB 정정 라벨(images.class_id)이며, 링크 대상은 미러 datasets_factory_collect(작업자 입력 폴더) 원본이다.

  python tools/build_blind_set.py datasets_blind_new_20261104 --since 20261023 --exclude datasets_blind_unsplit_20261023
  python tools/build_blind_set.py datasets_blind_all_20261104 --unsplit        # split 미지정 세션 전부
  python tools/build_blind_set.py <out> --sessions list.txt                     # 세션 dir 목록 파일
"""
import argparse, collections, json, shutil, sqlite3, sys
from pathlib import Path

DOOR = Path(__file__).resolve().parent.parent
MIRROR = DOOR / 'datasets_factory_collect'
CAP = 20

ap = argparse.ArgumentParser()
ap.add_argument('out', help='출력 디렉터리(door_pipeline 기준 상대 경로)')
ap.add_argument('--since', help='YYYYMMDD 이후(포함) 세션만')
ap.add_argument('--until', help='YYYYMMDD 이전(포함) 세션만')
ap.add_argument('--unsplit', action='store_true', help='images.split 미지정 세션만')
ap.add_argument('--exclude', action='append', default=[], help='이 블라인드 뷰(collect/<date>/*/<sess>)에 이미 있는 세션은 제외')
ap.add_argument('--sessions', help='세션 dir 목록 파일(한 줄 하나) — 그 세션만')
ap.add_argument('--cap', type=int, default=CAP)
a = ap.parse_args()


def pick_even(items, n):
    if n <= 0 or len(items) <= n: return list(items)
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)}); return [items[i] for i in idx]


con = sqlite3.connect(DOOR / 'db' / 'door_pipeline.db'); con.row_factory = sqlite3.Row
sess = con.execute("""SELECT s.id, s.session_dir, c.name cls, s.started_at,
        SUM(i.split IS NOT NULL) n_split
    FROM capture_sessions s JOIN datasets d ON d.id = s.dataset_id
    JOIN images i ON i.session_id = s.id JOIN classes c ON c.id = i.class_id
    WHERE d.name = 'door_factory_collect' AND c.name != 'Unknown' AND i.is_valid AND i.synced_local
    GROUP BY s.id ORDER BY s.session_dir""").fetchall()

only = None
if a.sessions: only = {l.strip() for l in open(a.sessions, encoding='utf-8') if l.strip()}
excluded = set()
for ex in a.exclude:
    for p in (DOOR / ex / 'collect').glob('*/*/s_*'):
        excluded.add(f"{p.parent.parent.name}/{p.name}")      # <date>/<sess>

def keep(r):
    day, _, ss = r['session_dir'].split('/')
    if only is not None and r['session_dir'] not in only: return False
    if a.since and day < a.since: return False
    if a.until and day > a.until: return False
    if a.unsplit and r['n_split']: return False
    if f"{day}/{ss}" in excluded: return False
    return True

sess = [r for r in sess if keep(r)]
out = DOOR / a.out
if out.exists(): shutil.rmtree(out)
man = []; cnt = collections.Counter(); scnt = collections.Counter()
for r in sess:
    ims = con.execute("SELECT rgb_path, depth_path, rgb_filename FROM images WHERE session_id=? AND is_valid AND synced_local ORDER BY rgb_filename", (r['id'],)).fetchall()
    ims = pick_even(ims, a.cap)
    day, _, ss = r['session_dir'].split('/')
    dc = out / 'collect' / day / r['cls'] / ss; dc.mkdir(parents=True, exist_ok=True)
    dn = out / 'cnn' / r['cls']; dn.mkdir(parents=True, exist_ok=True)
    meta = MIRROR / r['session_dir'] / 'meta.json'
    if meta.exists() and not (dc / 'meta.json').exists(): (dc / 'meta.json').symlink_to(meta)
    for im in ims:
        idx = im['rgb_filename'][4:-4]
        (dc / im['rgb_filename']).symlink_to(MIRROR / im['rgb_path'])
        (dn / f'rgb_{day}_{ss}_{idx}.png').symlink_to(MIRROR / im['rgb_path'])
        if im['depth_path']:
            (dc / Path(im['depth_path']).name).symlink_to(MIRROR / im['depth_path'])
            (dn / f'depth_{day}_{ss}_{idx}.png').symlink_to(MIRROR / im['depth_path'])
        man.append(dict(cls=r['cls'], session_dir=r['session_dir'], rgb=im['rgb_path']))
        cnt[r['cls']] += 1
    scnt[r['cls']] += 1
json.dump(dict(cap=a.cap, since=a.since, until=a.until, unsplit=a.unsplit, exclude=a.exclude,
               sessions=[r['session_dir'] for r in sess], images=man), open(out / 'manifest.json', 'w'), ensure_ascii=False, indent=1)
print(f'{a.out}: 세션 {len(sess)}  이미지 {len(man)}  (세션당 ≤{a.cap})')
for c in sorted(cnt): print(f'   {c:16s} 세션 {scnt[c]:3d}  장 {cnt[c]:4d}')
