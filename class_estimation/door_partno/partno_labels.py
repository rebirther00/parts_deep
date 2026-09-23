"""품번 라벨 매니페스트 — datasets_factory_v2 뷰의 프레임마다 (형상군 클래스, 레이더 GT, 품번, split) 을 DB 에서 붙인다.

  python partno_labels.py            # → labels/partno_manifest.json + 요약
GT: 클래스 = DB 정정 라벨(뷰 폴더명), 레이더 = capture_sessions.option_radar (option_source='user' 만; FRT 는 옵션 없음 → -1), 품번 = partno/part_numbers.json.
레이더 미확정 RR/RH 프레임은 radar=-1·part_idx=-1 (레이더/품번 학습·평가에서 제외, 형상군 학습에는 사용).
"""
import glob, json, os, re, sqlite3, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.normpath(os.path.join(HERE, '..', 'door_pipeline'))
sys.path.insert(0, DOOR)
from hole_classifier import CAD_D, GROUP

PN = json.load(open(os.path.join(DOOR, 'partno', 'part_numbers.json')))['parts']
CLASSES = sorted(CAD_D)                                   # 형상군 9 (02_train 의 sorted 폴더 순서와 동일)
PARTS = [p['part_no'] for p in PN]                        # 품번 14 (part_numbers.json 순서)
PART_OF = {(p['class_name'], p['radar']): p['part_no'] for p in PN}
MANIFEST = os.path.join(HERE, 'labels', 'partno_manifest.json')


def part_index(cls, radar):
    p = PART_OF.get((cls, None)) or (PART_OF.get((cls, radar)) if radar in (0, 1) else None)
    return PARTS.index(p) if p else -1


def session_map(db=os.path.join(DOOR, 'db', 'door_pipeline.db')):
    con = sqlite3.connect(db)
    out = {}
    for sd, rad, src in con.execute("SELECT session_dir, option_radar, option_source FROM capture_sessions"):
        date, _, sess = sd.split('/')
        out[(date, sess)] = (sd, rad if src == 'user' else None)
    return out


def build(view=os.path.join(HERE, 'datasets_factory_v2')):
    smap = session_map(); rows = []
    for split in ('train', 'val', 'test'):
        for f in sorted(glob.glob(os.path.join(view, split, '*', 'rgb_*.png'))):
            cls = os.path.basename(os.path.dirname(f)); m = re.search(r'rgb_(\d{8})_(s_\d+)_(\d+)\.png', f)
            sd, rad = smap.get((m.group(1), m.group(2)), (None, None))
            radar = -1 if GROUP[cls] == 'FRT' or rad is None else int(rad)
            rows.append(dict(path=os.path.relpath(f, HERE), cls=cls, cls_idx=CLASSES.index(cls), radar=radar,
                             part_idx=part_index(cls, radar if radar >= 0 else None), split=split, session_dir=sd))
    return rows


def load():
    return json.load(open(MANIFEST))


if __name__ == '__main__':
    rows = build(); json.dump(dict(classes=CLASSES, parts=PARTS, rows=rows), open(MANIFEST, 'w'), ensure_ascii=False)
    c = collections.Counter((r['split'], 'radar' if r['radar'] == 1 else 'none' if r['radar'] == 0 else 'n/a') for r in rows)
    print(f"프레임 {len(rows)}  split×레이더: {dict(sorted(c.items()))}")
    print('품번 미정(part_idx=-1):', sum(r['part_idx'] < 0 for r in rows), '→', MANIFEST)
