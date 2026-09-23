"""품번(14) 정확도 평가 — 형상군(홀 판별기) × 레이더(규칙 검사) 조합을 품번으로 묶어 채점.

  python partno/evaluate_partno.py                # 세션 단위(전량) + 프레임 단위(고정 test 분할) → partno/eval_partno.json/.md + DB 기록
  python partno/evaluate_partno.py --gt auto      # 사용자 확정이 없는 세션은 자동 판정을 GT 로 간주(일치도 확인용, 공식 수치 아님)
  python partno/evaluate_partno.py --db-log       # 공식 수치(GT=사용자 확정)를 DB evaluation_results 에 기록

GT 품번 = DB 정정 클래스 × 사용자 확정 레이더(capture_sessions.option_source='user'). FRT 는 클래스만으로 품번 확정.
예측 품번 = 홀 판별기 판정 클래스 × 자동 레이더(session_radar_check.auto_flag).
  세션 단위: 클래스 = session_hole_metrics.pred_major(20_session_drift.py, 픽셀 폭 D 다수결)
  프레임 단위: 클래스 = 17_evaluate_hole_classifier.py 최신 test json 의 프레임 판정, 레이더 = 그 프레임 세션의 자동 판정
판정률 = 품번이 나온 비율(클래스 판정 + (RR/RH 면) 레이더 판정 모두 있음), 판정 정확도 = 판정된 것 중 정답 비율 — 홀 판별기 관례와 동일.
GT 가 없는 세션(RR/RH 미확정)은 '미확정'으로 분모에서 빼고 수를 따로 적는다.
"""
import argparse, collections, glob, json, os, re, sqlite3, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); DOOR = os.path.dirname(HERE)
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'db'))
PN = {(p['class_name'], p['radar']): p for p in json.load(open(os.path.join(HERE, 'part_numbers.json')))['parts']}
EVAL_JSON = os.path.join(DOOR, 'attribute_models', 'hole_landmarks', 'eval_classifier_datasets_factory_v2_test_pixel.json')


def part_key(cls, radar):
    """품번 키 또는 None(미판정). FRT 는 radar 무관."""
    if not cls or cls == 'unknown' or cls == 'Unknown':
        return None
    p = PN.get((cls, None))
    if p: return p['part_no']
    if radar is None: return None
    p = PN.get((cls, radar)); return p['part_no'] if p else None


def needs_radar(cls):
    return bool(cls) and (cls, None) not in PN and cls in {k[0] for k in PN}


def session_table(con, gt_mode):
    rows = con.execute("""
        SELECT s.id, s.session_dir, s.option_radar, s.option_source,
               (SELECT c.name FROM images i JOIN classes c ON c.id=i.class_id WHERE i.session_id=s.id LIMIT 1) cls,
               (SELECT split FROM images WHERE session_id=s.id LIMIT 1) split,
               COALESCE((SELECT MIN(is_valid) FROM images WHERE session_id=s.id), 1) valid,
               m.pred_major, m.n_pred_major, m.n_judged AS n_hole, r.auto_flag, r.n_judged AS n_radar_judged
        FROM capture_sessions s LEFT JOIN session_hole_metrics m ON m.session_id=s.id LEFT JOIN session_radar_check r ON r.session_id=s.id
        ORDER BY s.session_dir""").fetchall()
    out = []
    for r in rows:
        if not r['valid'] or not r['cls'] or r['cls'] == 'Unknown' or r['pred_major'] is None and r['n_hole'] is None:
            continue
        gt_radar = r['option_radar'] if (r['option_source'] == 'user' or (gt_mode == 'auto' and r['option_radar'] is not None)) else None
        gt = part_key(r['cls'], gt_radar); gt_missing = needs_radar(r['cls']) and gt_radar is None
        pred = part_key(r['pred_major'], r['auto_flag'])
        out.append(dict(session_dir=r['session_dir'], cls=r['cls'], split=r['split'], gt=gt, gt_missing=gt_missing, pred=pred,
                        pred_cls=r['pred_major'], gt_radar=gt_radar, auto_flag=r['auto_flag'], radar_src=r['option_source']))
    return out


def score(items, label):
    scored = [x for x in items if not x['gt_missing']]
    judged = [x for x in scored if x['pred']]
    correct = sum(x['pred'] == x['gt'] for x in judged)
    res = dict(label=label, n=len(items), n_gt=len(scored), n_gt_missing=sum(x['gt_missing'] for x in items),
               judged=len(judged), correct=correct,
               judged_rate=100.0 * len(judged) / max(1, len(scored)), acc_judged=100.0 * correct / max(1, len(judged)),
               acc_overall=100.0 * correct / max(1, len(scored)))
    per = collections.defaultdict(lambda: dict(n=0, judged=0, correct=0, missing=0))
    for x in items:
        d = per[x['gt'] or ('?' + x['cls'])]
        if x['gt_missing']: d['missing'] += 1; continue
        d['n'] += 1
        if x['pred']: d['judged'] += 1; d['correct'] += x['pred'] == x['gt']
    res['per_part'] = {k: v for k, v in sorted(per.items())}
    res['wrong'] = [dict(session=x.get('session_dir') or x.get('image'), gt=x['gt'], pred=x['pred']) for x in judged if x['pred'] != x['gt']]
    return res


def frame_table(con, sess_items, eval_json):
    """17 test json 프레임 판정 × 세션 자동 레이더 → 프레임 품번."""
    if not os.path.exists(eval_json):
        return None, None
    d = json.load(open(eval_json)); key = next(iter(d)); rows = d[key]['rows']
    by_sess = {}
    for x in sess_items:
        date, _, sess = x['session_dir'].split('/'); by_sess[(date, sess)] = x
    out = []
    for r in rows:
        m = re.search(r'rgb_(\d{8})_(s_\d+)_\d+\.png', r['image'])
        if not m: continue
        x = by_sess.get((m.group(1), m.group(2)))
        if x is None: continue
        gt_radar = x['gt_radar']; gt = part_key(r['cls'], gt_radar); gt_missing = needs_radar(r['cls']) and gt_radar is None
        out.append(dict(image=r['image'], cls=r['cls'], gt=gt, gt_missing=gt_missing, pred=part_key(r['pred'], x['auto_flag'])))
    return out, key


def md_report(sess_all, sess_test, frames, gt_mode, key):
    L = [f"# 품번(14) 정확도 — {time.strftime('%Y-%m-%d %H:%M')}  (GT 레이더: {'사용자 확정만' if gt_mode == 'user' else '사용자 확정 + 자동(미확정분)'})", ""]
    for r in (sess_all, sess_test, frames):
        if not r: continue
        L += [f"## {r['label']}", "",
              f"- 대상 {r['n']} (GT 확정 {r['n_gt']}, 레이더 미확정 {r['n_gt_missing']}) · 판정 {r['judged']}/{r['n_gt']} = **{r['judged_rate']:.1f}%** · 판정 정확도 {r['correct']}/{r['judged']} = **{r['acc_judged']:.1f}%** · 전체 대비 {r['acc_overall']:.1f}%", "",
              "| 품번(GT) | n | 판정 | 정답 | 정확도 | 레이더 미확정 |", "|---|---:|---:|---:|---:|---:|"]
        for k, v in r['per_part'].items():
            L.append(f"| {k} | {v['n']} | {v['judged']} | {v['correct']} | {100 * v['correct'] / max(1, v['judged']):.1f}% | {v['missing']} |")
        if r['wrong']:
            L += ["", "오판: " + "; ".join(f"{w['session']} {w['gt']}→{w['pred']}" for w in r['wrong'][:20])]
        L.append("")
    return "\n".join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join(DOOR, 'db', 'door_pipeline.db'))
    ap.add_argument('--gt', choices=['user', 'auto'], default='user')
    ap.add_argument('--eval-json', default=EVAL_JSON); ap.add_argument('--db-log', action='store_true')
    a = ap.parse_args()
    con = sqlite3.connect(a.db); con.row_factory = sqlite3.Row
    items = session_table(con, a.gt)
    sess_all = score(items, f"세션 단위 · 전량 ({len(items)}세션)")
    test_items = [x for x in items if x['split'] == 'test']
    sess_test = score(test_items, f"세션 단위 · 고정 test 분할 ({len(test_items)}세션)")
    fr, key = frame_table(con, items, a.eval_json)
    frames = score(fr, f"프레임 단위 · 고정 test 분할 ({len(fr)}장, 17 {os.path.basename(a.eval_json)})") if fr else None
    md = md_report(sess_all, sess_test, frames, a.gt, key)
    print(md)
    stem = os.path.join(HERE, 'eval_partno' + ('' if a.gt == 'user' else '_gtauto'))
    open(stem + '.md', 'w', encoding='utf8').write(md)
    json.dump(dict(gt_mode=a.gt, session_all=sess_all, session_test=sess_test, frame_test=frames, sessions=items),
              open(stem + '.json', 'w'), ensure_ascii=False, indent=1, default=str)
    print(f"→ {stem}.md / .json")
    if a.db_log and a.gt == 'user' and frames and frames['n_gt']:
        from db_log import DBLog
        db = DBLog(a.db); mid = db.find_model(weights_path='attribute_models/hole_landmarks/model.pth', name='hole_landmarks_resnet18')
        db.log_evaluation(model_id=mid, dataset_name='door_factory_collect', eval_type='inference_pipeline',
                          total_samples=frames['n_gt'], correct=frames['correct'], accuracy=frames['acc_overall'],
                          per_class_results=dict(set='partno_test_frames', judged=frames['judged'], acc_judged=frames['acc_judged'],
                                                 session_all=dict(n_gt=sess_all['n_gt'], judged=sess_all['judged'], correct=sess_all['correct']),
                                                 session_test=dict(n_gt=sess_test['n_gt'], judged=sess_test['judged'], correct=sess_test['correct'])),
                          report_path=os.path.relpath(stem + '.json', DOOR))
        db.close(); print("DB evaluation_results 기록 (partno_test_frames)")
