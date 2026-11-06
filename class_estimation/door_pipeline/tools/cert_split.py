"""공인 시험용 세션 분할(학습 70 : 테스트 30) 내보내기 — 입회자 시드로 재현 가능.

  python tools/cert_split.py --seed 20261106                 # report/cert_test/split_20261106.json(+csv) 생성
  python tools/cert_split.py --seed 1234 --dry-run           # 표만

규칙(계획서 지표 6 '부품 인식율' 평가환경: 획득 이미지 데이터셋, 학습:테스트 70:30):
  - 모집단 = DB door_factory_collect 유효 세션(is_valid, Unknown 제외, 로컬 표본 ≥1장). 세션당 표본 ≤20장(pull 규약).
  - 학습 측 = CNN 폴백(seed916)의 train/val 세션(DB images.split). 홀 판별기는 실험실 134장 라벨로만 학습해 현장 세션을 쓰지 않았고,
    4채널(브래킷) 검출기는 같은 train/val 세션으로 학습했으므로 두 모델 모두 테스트 풀에 학습 세션이 없다.
  - 테스트 풀 = split 이 train/val 이 아닌 세션(기존 test 23 + 미분할). 전체 세션 수의 30% 를 클래스 비율대로 시드로 추출.
    풀이 목표보다 작은 클래스는 풀 전부.
  - 출력 JSON 에 세션 목록·클래스·장수·레이더 확정값·DB split·GT 출처(작업자 라벨 + 정정 이력 notes)와 파일 자체의 sha256 을 기록.
"""
import argparse, csv, hashlib, json, os, random, sqlite3, sys, time, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DOOR, 'db'))
import build_dataset as bd   # noqa: E402

OUT_DIR = os.path.join(DOOR, 'report', 'cert_test')
TEST_FRAC = 0.30


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sessions(con):
    q = """SELECT s.id sid, s.session_dir, s.class_name, s.option_radar, s.option_source, s.notes,
                  (SELECT split FROM images WHERE session_id=s.id AND is_valid LIMIT 1) split,
                  SUM(i.synced_local) n_local, COUNT(i.id) n_valid
           FROM capture_sessions s JOIN images i ON i.session_id=s.id JOIN classes c ON c.id=i.class_id
           JOIN datasets d ON d.id=c.dataset_id
           WHERE i.is_valid AND c.name!='Unknown' AND d.name=? GROUP BY s.id HAVING n_local>0 ORDER BY s.session_dir"""
    return [dict(r) for r in con.execute(q, (bd.DATASET_NAME,))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join(DOOR, 'db', 'door_pipeline.db'))
    ap.add_argument('--seed', type=int, required=True, help='입회자가 정한 시드')
    ap.add_argument('--frac', type=float, default=TEST_FRAC)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    con = sqlite3.connect(a.db); con.row_factory = sqlite3.Row
    rows = sessions(con)
    train = [r for r in rows if r['split'] in ('train', 'val')]
    pool = [r for r in rows if r['split'] not in ('train', 'val')]
    by_cls = collections.defaultdict(list)
    for r in pool:
        by_cls[r['class_name']].append(r)
    n_total = len(rows)
    rng = random.Random(a.seed)
    test, short = [], {}
    for cls in sorted({r['class_name'] for r in rows}):
        n_cls = sum(1 for r in rows if r['class_name'] == cls)
        want = int(round(n_cls * a.frac))
        cand = sorted(by_cls.get(cls, []), key=lambda r: r['session_dir'])
        rng.shuffle(cand)
        if want > len(cand):
            short[cls] = (want, len(cand)); want = len(cand)
        test += cand[:want]
    test.sort(key=lambda r: r['session_dir'])
    test_ids = {r['sid'] for r in test}
    rest = [r for r in pool if r['sid'] not in test_ids]

    def pack(r):
        return dict(session_dir=r['session_dir'], class_name=r['class_name'], n_frames=int(r['n_local']), n_valid=int(r['n_valid']),
                    option_radar=r['option_radar'], option_source=r['option_source'], db_split=r['split'],
                    label_corrected=bool(r['notes'] and ('정정' in r['notes'] or 'relabel' in r['notes'].lower())))
    out = dict(
        kind='cert_split', created_at=time.strftime('%Y-%m-%d %H:%M:%S'), seed=a.seed, test_frac=a.frac, dataset=bd.DATASET_NAME,
        rule='테스트 풀 = CNN train/val 제외 세션(홀 판별기·4채널 검출기도 현장 train/val 외 세션은 미학습), 클래스 비율대로 시드 추출, 세션당 표본 ≤20장',
        totals=dict(sessions=n_total, train_val_sessions=len(train), pool_sessions=len(pool), test_sessions=len(test),
                    test_frames=sum(int(r['n_local']) for r in test), train_val_frames=sum(int(r['n_local']) for r in train)),
        per_class={cls: dict(total=sum(1 for r in rows if r['class_name'] == cls), train_val=sum(1 for r in train if r['class_name'] == cls),
                             pool=len(by_cls.get(cls, [])), test=sum(1 for r in test if r['class_name'] == cls),
                             test_frames=sum(int(r['n_local']) for r in test if r['class_name'] == cls))
                   for cls in sorted({r['class_name'] for r in rows})},
        shortfall=short, test=[pack(r) for r in test], train_val=[pack(r) for r in train], pool_unused=[pack(r) for r in rest])
    print(f"세션 {n_total} (학습측 train/val {len(train)}, 풀 {len(pool)}) → 테스트 {len(test)}세션 {out['totals']['test_frames']}장  시드 {a.seed}")
    for cls, v in out['per_class'].items():
        print(f"  {cls:18s} 전체 {v['total']:3d}  학습측 {v['train_val']:3d}  풀 {v['pool']:3d}  테스트 {v['test']:3d} ({v['test_frames']}장)")
    if short:
        print("  풀 부족 클래스:", short)
    if a.dry_run:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    jp = os.path.join(OUT_DIR, f'split_{a.seed}.json'); cp = os.path.join(OUT_DIR, f'split_{a.seed}.csv')
    json.dump(out, open(jp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    with open(cp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['role', 'session_dir', 'class_name', 'n_frames', 'option_radar', 'db_split', 'label_corrected'])
        for role, lst in (('test', out['test']), ('train_val', out['train_val']), ('pool_unused', out['pool_unused'])):
            for r in lst:
                w.writerow([role, r['session_dir'], r['class_name'], r['n_frames'], r['option_radar'], r['db_split'], int(r['label_corrected'])])
    h = sha256(jp)
    open(jp + '.sha256', 'w').write(f"{h}  {os.path.basename(jp)}\n")
    print(f"저장 {jp}\n     {cp}\n     sha256 {h}")


if __name__ == '__main__':
    main()
