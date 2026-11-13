"""Descriptive analysis of the finished extension_v3 runs (no model is run, nothing is re-thresholded).

Writes reports/extension_v3/analysis.json:
  - relative distance error D/D_CAD - 1 for pixel and depth (5-fold OOF of holes_field8, temporal hole model)
  - depth failure breakdown (wrong class vs abstain, direction of the wrong class)
  - CNN with the val-95%-coverage softmax threshold fixed in v1 (metrics_val95coverage.json, already computed by eval-cnn)
  - the door both CNNs miss, as seen by the hole method
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT, dump, now, read_json, read_jsonl, snapshot  # noqa: E402

EXT = ROOT / 'reports/extension_v3'
F0 = ROOT / 'reports/fold0_test_v3'


def hole_files(mode):
    out = [F0 / f'holes_field8_f0_s42_v2_{mode}_test_predictions.jsonl']
    out += [EXT / f'holes_field8_f{k}_s42_v3_{mode}_predictions.jsonl' for k in (1, 2, 3, 4)]
    return out


def error_stats(rows, cad):
    e = [(r['group_id'], 100 * (r['measurement'] / cad[r['class_name']] - 1), r['measurement'] - cad[r['class_name']])
         for r in rows if r['measurement'] is not None]
    rel = np.array([x[1] for x in e]); mm = np.array([x[2] for x in e])
    per = defaultdict(list)
    for g, v, _ in e:
        per[g].append(v)
    med = np.array([np.median(v) for v in per.values()])
    within = np.array([np.std(v) for v in per.values() if len(v) > 1])
    return dict(frames_measured=len(e), frames_total=len(rows), p5=float(np.percentile(rel, 5)), median=float(np.median(rel)),
                p95=float(np.percentile(rel, 95)), min=float(rel.min()), max=float(rel.max()), abs_max_mm=float(np.abs(mm).max()),
                door_median_min=float(med.min()), door_median_max=float(med.max()), between_door_std=float(np.std(med)),
                within_door_std_median=float(np.median(within)), doors=len(per))


def session_votes(rows):
    per = defaultdict(list)
    for r in rows:
        per[r['group_id']].append(r)
    out = {}
    for g, rs in per.items():
        c = Counter(r['pred'] for r in rs if r['pred'] is not None)
        top = c.most_common(2)
        pred = None if not top or (len(top) > 1 and top[0][1] == top[1][1]) else top[0][0]
        out[g] = dict(true=rs[0]['class_name'], pred=pred)
    return out


def depth_breakdown(rows, cad):
    s = session_votes(rows)
    wrong = [v for v in s.values() if v['pred'] is not None and v['pred'] != v['true']]
    abst = [v for v in s.values() if v['pred'] is None]
    larger = sum(cad[v['pred']] > cad[v['true']] for v in wrong)
    fr = Counter((r['class_name'], r['pred']) for r in rows if r['pred'] is not None and r['pred'] != r['class_name'])
    fr_abst = Counter(r['class_name'] for r in rows if r['pred'] is None)
    return dict(sessions=len(s), correct=sum(v['pred'] == v['true'] for v in s.values()), wrong=len(wrong), abstained=len(abst),
                wrong_to_larger_class=int(larger), abstained_classes=dict(Counter(v['true'] for v in abst)),
                wrong_pairs=dict(Counter(f"{v['true']}->{v['pred']}" for v in wrong)),
                wrong_frames=sum(fr.values()), wrong_frame_pairs={f'{a}->{b}': n for (a, b), n in fr.items()},
                abstained_frames=sum(fr_abst.values()), abstained_frame_classes=dict(fr_abst))


def cnn_threshold(kind):
    folds, tot = {}, Counter()
    for k in range(5):
        m = read_json(ROOT / f'reports/{kind}_f{k}_s42_test/metrics_val95coverage.json')
        rep = read_json(ROOT / f'reports/{kind}_f{k}_s42_test/report.json')
        f, s = m['frame'], m['session']
        folds[str(k)] = dict(threshold=rep['val95coverage_threshold'], session=[s['correct'], s['n']], session_abstained=s['abstained'],
                             session_wrong=s['judged'] - s['correct'], frame_coverage=f['coverage'], frame=[f['correct'], f['n']])
        tot.update(dict(sc=s['correct'], sn=s['n'], sa=s['abstained'], sw=s['judged'] - s['correct'], fc=f['correct'], fn=f['n'], fj=f['judged']))
    m = read_json(ROOT / f'reports/{kind}_temporal_s42_test/metrics_val95coverage.json')
    rep = read_json(ROOT / f'reports/{kind}_temporal_s42_test/report.json')
    f, s = m['frame'], m['session']
    abst = [p['session_dir'] for p in m['session_predictions'] if p['pred'] is None]
    wrong = [(p['session_dir'], p['pred']) for p in m['session_predictions'] if p['pred'] is not None and p['pred'] != p['class_name']]
    temporal = dict(threshold=rep['val95coverage_threshold'], session=[s['correct'], s['n']], session_abstained=s['abstained'],
                    session_wrong=s['judged'] - s['correct'], frame_coverage=f['coverage'], frame=[f['correct'], f['n']],
                    abstained_sessions=abst, wrong_sessions=wrong)
    return dict(per_fold=folds, oof=dict(session=[tot['sc'], tot['sn']], session_abstained=tot['sa'], session_wrong=tot['sw'],
                                          frame=[tot['fc'], tot['fn']], frame_coverage=tot['fj'] / tot['fn']), temporal=temporal)


def missed_door(group):
    out = {}
    for name, files in (('hole8_pixel_oof', hole_files('pixel')), ('hole_temporal_pixel', [EXT / 'holes_field_temporal_s42_pixel_predictions.jsonl'])):
        rows = [r for f in files for r in read_jsonl(f) if r['group_id'] == group]
        ms = [r['measurement'] for r in rows if r['measurement'] is not None]
        out[name] = dict(frames=len(rows), correct=sum(r['pred'] == r['class_name'] for r in rows),
                         D_median=float(np.median(ms)) if ms else None, D_min=float(min(ms)) if ms else None, D_max=float(max(ms)) if ms else None)
    for kind in ('rgb', 'rgbe'):
        for split in [f'f{k}' for k in range(5)] + ['temporal']:
            p = ROOT / f'reports/{kind}_{split}_s42_test/predictions.jsonl'
            rows = [r for r in read_jsonl(p) if r['group_id'] == group]
            if rows:
                out[f'{kind}_{split}'] = dict(frames=len(rows), preds=dict(Counter(r['pred'] for r in rows)),
                                             confidence_median=float(np.median([r['confidence'] for r in rows])))
    return out


def main():
    sp, _, frames, splits = snapshot()
    cfg = read_json(sp / 'protocol.json'); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    res = dict(created_at=now(), source='reports/extension_v3 + reports/fold0_test_v3 + reports/{rgb,rgbe}_*_s42_test')
    for mode in ('pixel', 'depth'):
        oof = [r for f in hole_files(mode) for r in read_jsonl(f)]
        tmp = read_jsonl(EXT / f'holes_field_temporal_s42_{mode}_predictions.jsonl')
        res[f'error_{mode}'] = dict(oof=error_stats(oof, cad), temporal=error_stats(tmp, cad))
        if mode == 'depth':
            res['depth_breakdown'] = dict(oof=depth_breakdown(oof, cad), temporal=depth_breakdown(tmp, cad))
    res['cnn_threshold'] = {k: cnn_threshold(k) for k in ('rgb', 'rgbe')}
    res['cnn_missed_door'] = {g: missed_door(g) for g in ('20261022/E25_door_LH_FRT/s_084658', '20261022/E25_door_LH_FRT/s_123253')}
    tparts = {k: {r['group_id'] for r in v} for k, v in __import__('door8.core', fromlist=['partition']).partition(frames, splits, 'temporal').items()}
    cls = {r['group_id']: r['class_name'] for r in frames}
    res['temporal_split_classes'] = {k: dict(Counter(cls[g] for g in v)) for k, v in tparts.items()}
    dump(EXT / 'analysis.json', res)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:9000])


if __name__ == '__main__':
    main()
