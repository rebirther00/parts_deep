"""Read-only re-tabulations for review round 3 (manuscript v14); writes reports/review3_checks_v14.json.

No model is run and no stored prediction is changed. Sources: the frozen snapshot (operational records in
data/snapshots/v1/source.sqlite) and the stored predictions of extension_v3 / fold0_test_v3 / extension_v12 / extension_v12s.

  camera      door-plane tilt and per-session pixel scale in the operational records before / after the camera re-adjustment
  distance    pixel-width and depth-plane door decisions and relative residuals before / after the re-adjustment
  operational depth-path image decisions in the stored operational evaluation of 2026-09-23
  cnn_rule    image-level rejection of registered images and the share of fully rejected / fully accepted doors
  e23         measured D of the E23 doors and its distance to the nearest registered CAD value
"""
import collections
import csv
import glob
import json
from pathlib import Path
import sqlite3
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT, now, read_json, snapshot

OUT = ROOT / 'reports/review3_checks_v14.json'
SPLIT_DATE = '20260921'            # first session with a changed view (2026-09-21 15:57); the last normal session is 2026-09-18 13:26
BUMPED = ('20260921/E38_door_LH_RR/s_155719', '20260922/E38_door_LH_RR/s_080559')   # after the bump, before the re-adjustment (08:06-08:10 on 09-22)


def period(session_dir):
    return 'before' if session_dir[:8] < SPLIT_DATE else 'after'


def state(session_dir):
    return 'normal' if session_dir[:8] < SPLIT_DATE else 'bumped' if session_dir in BUMPED else 'readjusted'


def vote(rows):
    c = collections.Counter(r['pred'] for r in rows if r['pred'] is not None).most_common(2)
    return None if not c or (len(c) > 1 and c[0][1] == c[1][1]) else c[0][0]


def doors(mode, cad):
    files = sorted(glob.glob(str(ROOT / f'reports/extension_v3/holes_field8_f[1-4]_s42_v3_{mode}_predictions.jsonl'))) \
        + [str(ROOT / f'reports/fold0_test_v3/holes_field8_f0_s42_v2_{mode}_test_predictions.jsonl')]
    g = collections.defaultdict(list)
    for f in files:
        for line in open(f):
            r = json.loads(line); g[r['group_id']].append(r)
    out = {}
    for k, rr in g.items():
        pred = vote(rr); c = rr[0]['class_name']
        out[k] = dict(cls=c, ok=pred == c, rejected=pred is None,
                      rel=[100 * (r['measurement'] / cad[c] - 1) for r in rr if r.get('measurement') is not None])
    return out


def main():
    sp, _, _, _ = snapshot(); cfg = read_json(sp / 'protocol.json'); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    elig = {int(r['session_id']): r for r in csv.DictReader(open(sp / 'sessions.csv'))}
    con = sqlite3.connect(f'file:{sp / "source.sqlite"}?mode=ro', uri=True); con.row_factory = sqlite3.Row
    res = dict(created_at=now(), split_date=SPLIT_DATE,
               author_statement='2026-11-11: an operator bumped the camera by mistake and it was re-adjusted; date and time are not recorded. The author agrees with the reading of the records: bumped by 2026-09-21, re-adjusted on 2026-09-22.',
               timeline='last normal session 2026-09-18 13:26; changed view in the sessions of 2026-09-21 15:57 and 2026-09-22 08:05 (5 and 1 images); re-adjusted view from the session of 2026-09-22 08:10',
               images='reports/camera_readjustment_202609/ (scripts/make_camera_change_sheet_v14.py)')

    rows = [dict(r) for r in con.execute('select * from session_hole_metrics where model_id=12') if r['session_id'] in elig]
    cam = {}
    for key in ('tilt_med', 's_session', 'z_med'):
        cam[key] = {}
        for p in ('normal', 'bumped', 'readjusted'):
            x = sorted(r[key] for r in rows if r[key] is not None and state(elig[r['session_id']]['session_dir']) == p)
            cam[key][p] = dict(sessions=len(x), median=statistics.median(x), p05=x[int(.05 * len(x))], p95=x[min(len(x) - 1, int(.95 * len(x)))])
    res['camera'] = dict(source='session_hole_metrics, operational evaluation of 2026-11-04 (model 12), 361 eligible sessions', **cam)

    dist = {}
    for mode in ('pixel', 'depth'):
        d = doors(mode, cad); dist[mode] = {}
        for p in ('before', 'after'):
            sel = {k: v for k, v in d.items() if period(k) == p}; rel = [x for v in sel.values() for x in v['rel']]
            dist[mode][p] = dict(doors=len(sel), wrong=sum(not v['ok'] and not v['rejected'] for v in sel.values()),
                                 rejected=sum(v['rejected'] for v in sel.values()), image_residual_median_pct=statistics.median(rel),
                                 image_residual_min_pct=min(rel), image_residual_max_pct=max(rel),
                                 failed_by_class=dict(collections.Counter(v['cls'] for v in sel.values() if not v['ok'])),
                                 doors_by_class=dict(collections.Counter(v['cls'] for v in sel.values())))
        dist[mode]['failed_dates'] = dict(sorted(collections.Counter(k[:8] for k, v in d.items() if not v['ok']).items()))
    res['distance'] = dict(source='five-fold out-of-fold TEST predictions of the field-adapted hole detectors (extension_v3, fold0_test_v3)', **dist)

    r7 = [dict(r) for r in con.execute('select * from session_hole_metrics where model_id=7') if r['session_id'] in elig]
    res['operational'] = dict(
        source='session_hole_metrics, operational evaluation of 2026-09-23 (model 7, depth path); pred_major is the pixel-width decision recomputed for the whole period',
        sessions=len(r7), depth_path_majority_wrong=sum(1 for r in r7 if r['n_judged'] and 2 * (r['n_wrong'] or 0) > r['n_judged']),
        depth_path_some_image_wrong=sum(1 for r in r7 if (r['n_wrong'] or 0) > 0),
        switch_record='git commit d9ee5ee (2026-09-23), hole_classifier.SCALE_DEFAULT comment: 185 sessions, 3,762 images, 0 decisions changed by the switch from the depth path to pixel width')

    rule = dict(door_rule='run_extension_v5.vote: mode of the accepted images; a door is rejected only when all its images are rejected or the vote is tied',
                mahalanobis_registered={}, output_scores_accepted_unregistered=collections.Counter())
    for seed, f in ((42, 'reports/extension_v12/cnn_unseen_predictions.jsonl'), (123, 'reports/extension_v12s/s123/cnn_unseen_predictions.jsonl'),
                    (456, 'reports/extension_v12s/s456/cnn_unseen_predictions.jsonl')):
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for line in open(ROOT / f):
            r = json.loads(line)
            if r.get('condition', 'unseen') == 'unseen':
                agg[(r['kind'], r['score_name'])][(r['excluded'], r['group_id'], r['class_name'] == r['excluded'])].append(r['accepted'])
        for (kind, score), dd in agg.items():
            known = [v for k, v in dd.items() if not k[2]]; unknown = [v for k, v in dd.items() if k[2]]
            if score == 'maha':
                n = sum(len(v) for v in known)
                rule['mahalanobis_registered'][f'{kind}:s{seed}'] = dict(images=n, images_rejected_pct=100 * sum(v.count(False) for v in known) / n,
                                                                         doors=len(known), doors_all_images_rejected=sum(1 for v in known if not any(v)))
            if score in ('msp', 'maxlogit', 'energy'):
                for v in unknown:
                    if any(v):
                        rule['output_scores_accepted_unregistered']['doors'] += 1
                        rule['output_scores_accepted_unregistered']['all_images_accepted' if all(v) else 'some_images_accepted'] += 1
    rule['output_scores_accepted_unregistered'] = dict(rule['output_scores_accepted_unregistered'])
    res['cnn_rule'] = rule

    e = [json.loads(line) for line in open(ROOT / 'reports/e23_openset_v3/holes_field8_f0_unregistered_predictions.jsonl')]
    m = [r['measurement'] for r in e if r.get('measurement') is not None]; near = [min(abs(x - c) for c in cad.values()) for x in m]
    res['e23'] = dict(images=len(e), measured_D_median_mm=statistics.median(m), nearest_registered_distance_min_mm=min(near),
                      nearest_registered_distance_median_mm=statistics.median(near))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(dict(camera={k: {p: round(v[p]['median'], 2) for p in v} for k, v in cam.items()},
                          distance={m_: {p: (dist[m_][p]['doors'], dist[m_][p]['wrong'], dist[m_][p]['rejected'], round(dist[m_][p]['image_residual_median_pct'], 2))
                                         for p in ('before', 'after')} for m_ in dist},
                          operational={k: res['operational'][k] for k in ('sessions', 'depth_path_majority_wrong')},
                          maha={k: (round(v['images_rejected_pct'], 1), v['doors_all_images_rejected']) for k, v in rule['mahalanobis_registered'].items()},
                          accepted=rule['output_scores_accepted_unregistered'], e23=res['e23']), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
