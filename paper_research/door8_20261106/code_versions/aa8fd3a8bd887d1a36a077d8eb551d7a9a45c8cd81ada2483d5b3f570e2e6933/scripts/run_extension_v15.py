"""Extension v15 (review round 4 of manuscript v14), registered in config/extension_v15_protocol.json. No training.

  scale    A  scale S fit on one camera state and applied to the other (stored features of the five field-adapted detectors)
  errors   B  the five in-list input-error sessions: fold-0 hole detector and fold-0 CNNs (inference on the excluded images)
  retab    C  descriptive re-tabulations of stored predictions and operational records
  collect     reports/extension_v15/collected.json
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import sqlite3
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from door8.core import setup_runtime
setup_runtime()

import numpy as np
import torch

from door8.core import ROOT, SOURCE, dump, jsonl, now, partition, read_json, read_jsonl, sha256, snapshot
from door8.geometry import classify_measurement, fit_scale
from door8.learning import CNNData, Classifier, HoleNet, cnn_predict, load_run, make_loader
import run_extension_v3 as V3
import run_extension_v5 as V5
import run_extension_v12 as V12

PROTOCOL = ROOT / 'config/extension_v15_protocol.json'
OUT = ROOT / 'reports/extension_v15'
FOLDS = ('0', '1', '2', '3', '4')
CHANGE_DATE = '20260921'
BUMPED = ('20260921/E38_door_LH_RR/s_155719', '20260922/E38_door_LH_RR/s_080559')
ERROR_SESSIONS = [('20260903/E30_door_LH_RR/s_151736', 'E30_door_LH_RR', 'E25_door_LH_RR'),
                  ('20260903/E30_door_LH_RR/s_183159', 'E30_door_LH_RR', 'E30_E38_door_RH'),
                  ('20260905/E30_door_LH_RR/s_123451', 'E30_door_LH_RR', 'E30_E38_door_RH'),
                  ('20260914/E30_door_LH_RR/s_170035', 'E30_door_LH_RR', 'E30_E38_door_RH'),
                  ('20261029/E30_door_LH_FRT/s_075858', 'E30_door_LH_FRT', 'E38_door_LH_FRT')]


def state(session_dir):
    return 'normal' if session_dir[:8] < CHANGE_DATE else 'changed'


def judge(rows, pool, cal, cfg, extra):
    out = []
    for row in rows:
        p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg)
        out.append({k: row[k] for k in ('frame_id', 'group_id', 'class_name')} | dict(pred=p, measurement=v, reason=why) | extra)
    return out


def summary(preds, cad, band):
    s = dict(doors=V12.door_counts(preds), frames=V12.frame_counts(preds))
    if any(r['measurement'] is not None for r in preds):
        st = V12.residual_stats(preds, cad, band)
        rel = [r['measurement'] / cad[r['class_name']] - 1 for r in preds if r['measurement'] is not None]
        s.update(residual_median=float(np.median(rel)), residual_range=st['range'], residual_rms=st['rms'], max_abs_mm=st['max_abs_mm'],
                 frames_out_of_band=st['frames_out_of_band'], max_utilisation=st['max_utilisation'])
    return s


# ---------------------------------------------------------------- A
def scale():
    sp, frames, splits, cfg = V12.ctx(); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); band = V5.bands(cad)
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), modes={}); dumped = []
    for mode in ('pixel', 'depth'):
        pooled = collections.defaultdict(list); scales = {}
        for f in FOLDS:
            parts = partition(frames, splits, f); pool = V5.hole_pool(f)
            stored = {r['frame_id']: r for r in read_jsonl(V5.test_prediction_file(f, mode))}
            cal_all = fit_scale(parts['train'], pool, cad, mode); scales[f] = dict(all=cal_all['camera_scales'])
            ref = judge(parts['test'], pool, cal_all, cfg, dict(fold=f, mode=mode, scale_from='all'))
            if any(stored[r['frame_id']]['pred'] != r['pred'] for r in ref):
                raise ValueError('Reference decisions differ from the stored predictions')
            for direction, src, dst in (('forward', 'normal', 'changed'), ('reverse', 'changed', 'normal')):
                tr = [r for r in parts['train'] if state(r['session_dir']) == src]; te = [r for r in parts['test'] if state(r['session_dir']) == dst]
                cal = fit_scale(tr, pool, cad, mode)
                scales[f][direction] = dict(scale=cal['camera_scales'], train_sessions=len({r['group_id'] for r in tr}), valid_train_sessions=cal['valid_train_sessions'])
                p = judge(te, pool, cal, cfg, dict(fold=f, mode=mode, scale_from=src, direction=direction))
                pooled[direction] += p; dumped += p
                keep = {r['frame_id'] for r in te}; pooled[direction + '_reference'] += [r for r in ref if r['frame_id'] in keep]
        m = dict(scale_by_fold=scales)
        for key, preds in pooled.items():
            m[key] = summary(preds, cad, band)
        fb = [r for r in pooled['forward'] if r['group_id'] in BUMPED]; votes = V5.vote(fb)
        m['forward_bumped_sessions'] = {g: dict(true=v['true'], pred=v['pred'], frames=v['frames'],
                                                measured=[r['measurement'] for r in fb if r['group_id'] == g]) for g, v in votes.items()}
        m['forward_readjusted_only'] = summary([r for r in pooled['forward'] if r['group_id'] not in BUMPED], cad, band)
        res['modes'][mode] = m
    dump(OUT / 'scale_transfer.json', res); jsonl(OUT / 'scale_transfer_predictions.jsonl', dumped)
    print(json.dumps({mode: {k: (v['doors'], round(100 * v.get('residual_median', float('nan')), 3), [round(100 * x, 2) for x in v.get('residual_range', [])],
                                 round(v.get('max_utilisation', float('nan')), 3))
                             for k, v in m.items() if isinstance(v, dict) and 'doors' in v} for mode, m in res['modes'].items()}, indent=1))


# ---------------------------------------------------------------- B
def error_rows(cfg):
    base = (SOURCE / 'datasets_factory_collect').resolve(); rows = []
    for s, operator, corrected in ERROR_SESSIONS:
        d = base / s; meta = read_json(d / 'meta.json') if (d / 'meta.json').exists() else {}
        intr = meta.get('intrinsics') or dict(serial=54910212, width=1920, height=1200, fx=1269.746, fy=1269.746, cx=961.669, cy=598.594)
        found = sorted(d.glob('rgb_*.png'))
        if not found:
            raise ValueError(f'No images for {s}')
        for rgb in found:
            dep = d / rgb.name.replace('rgb_', 'depth_')
            rows.append(dict(frame_id=f'{s}:{rgb.name}', session_dir=s, group_id=s, class_name=corrected, class_index=cfg['classes'].index(corrected),
                             operator_input=operator, rgb=str(rgb), depth=str(dep), intrinsics=intr, rgb_sha256=sha256(rgb), depth_sha256=sha256(dep),
                             rgb_bytes=rgb.stat().st_size, rgb_mtime_ns=rgb.stat().st_mtime_ns,
                             depth_bytes=dep.stat().st_size, depth_mtime_ns=dep.stat().st_mtime_ns))
    return rows


def errors():
    sp, frames, splits, cfg = V12.ctx(); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); parts = partition(frames, splits, '0')
    known = {r['group_id'] for r in frames}
    if any(s in known for s, _, _ in ERROR_SESSIONS):
        raise ValueError('An input-error session is inside the evaluation snapshot')
    rows = error_rows(cfg); jsonl(OUT / 'input_errors_manifest.jsonl', rows)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per = {s: dict(operator_input=o, corrected=c, images=sum(r['session_dir'] == s for r in rows)) for s, o, c in ERROR_SESSIONS}
    rp, _ = load_run('holes_field8_f0_s42_v2', sp)
    net = HoleNet(pretrained=False).to(dev); net.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True)); net.eval()
    feats = V3.extract(net, rows, dev); jsonl(OUT / 'input_errors_hole_features.jsonl', list(feats.values()))
    pool = {r['frame_id']: r for r in read_jsonl(ROOT / 'features/holes_field8_f0_s42_v2_development/rows.jsonl')}
    cal = fit_scale(parts['train'], pool, cad, 'pixel'); preds = []
    for row in rows:
        p, v, why = classify_measurement(row, feats[row['frame_id']], cal, cfg)
        preds.append(dict(frame_id=row['frame_id'], session_dir=row['session_dir'], pred=p, measurement=v, reason=why))
    jsonl(OUT / 'input_errors_hole_predictions.jsonl', preds); so = V3.session_outcome(preds)
    for s in per:
        m = [r['measurement'] for r in preds if r['session_dir'] == s and r['measurement'] is not None]
        per[s]['hole'] = dict(decision=so.get(s), measured_D_median=float(np.median(m)) if m else None, images_measured=len(m),
                              image_decisions=dict(collections.Counter(str(r['pred']) for r in preds if r['session_dir'] == s)))
    for run in ('rgb_f0_s42', 'rgbe_f0_s42'):
        rp, rm = load_run(run, sp)
        cnn = Classifier(rm['model_kind'], len(cfg['classes']), pretrained=False).to(dev)
        cnn.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True))
        loader = make_loader(CNNData(rows, rm['model_kind'], rm['image_size'], False), 16, False, 2, 42)
        cp, _ = cnn_predict(cnn, loader, rows, cfg['classes'], dev); jsonl(OUT / f'input_errors_{run}_predictions.jsonl', cp)
        so = V3.session_outcome(cp)
        for s in per:
            per[s][run] = dict(decision=so.get(s), image_decisions=dict(collections.Counter(r['pred'] for r in cp if r['session_dir'] == s)),
                               confidence_median=float(np.median([r['confidence'] for r in cp if r['session_dir'] == s])))
    for s, d in per.items():
        d['agrees'] = {k: ('operator_input' if d[k]['decision'] == d['operator_input'] else 'corrected' if d[k]['decision'] == d['corrected'] else 'neither')
                       for k in ('hole', 'rgb_f0_s42', 'rgbe_f0_s42')}
    m = sum(d['hole']['decision'] != d['operator_input'] for d in per.values())
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), scale=cal['camera_scales'], sessions=per,
               hole_disagrees_with_operator_input=m, door_accuracy_if_operator_inputs_were_right=[361, 361 + m, 361 / (361 + m)])
    dump(OUT / 'input_errors.json', res)
    print(json.dumps({s: dict(images=d['images'], operator=d['operator_input'], corrected=d['corrected'], hole=(d['hole']['decision'], d['hole']['measured_D_median']),
                              rgb=d['rgb_f0_s42']['image_decisions'], rgbe=d['rgbe_f0_s42']['image_decisions'], agrees=d['agrees']) for s, d in per.items()},
                     ensure_ascii=False, indent=1))
    print('bound', res['door_accuracy_if_operator_inputs_were_right'])


# ---------------------------------------------------------------- C
def retab():
    sp, frames, splits, cfg = V12.ctx()
    elig = {int(r['session_id']): r for r in csv.DictReader(open(sp / 'sessions.csv'))}
    excluded = {int(r['session_id']): r for r in csv.DictReader(open(sp / 'excluded_sessions.csv'))}
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL))
    # C1 depth-plane failures by window
    per = collections.defaultdict(list)
    for f in FOLDS:
        for r in read_jsonl(V5.test_prediction_file(f, 'depth')):
            per[r['group_id']].append(r)
    votes = V5.vote([r for rr in per.values() for r in rr]); failed = {g for g, v in votes.items() if v['pred'] != v['true']}
    window = lambda g: 'before_0921' if g[:8] < '20260921' else '0921_to_0923' if g[:8] <= '20260923' else 'after_0923'
    res['C1'] = dict(doors=dict(collections.Counter(window(g) for g in votes)), failed=dict(collections.Counter(window(g) for g in failed)),
                     failed_0921_to_0923=sorted(g for g in failed if window(g) == '0921_to_0923'),
                     bumped={g: dict(true=votes[g]['true'], pred=votes[g]['pred']) for g in BUMPED})
    # C2, C3 operational depth path
    con = sqlite3.connect(f'file:{sp / "source.sqlite"}?mode=ro', uri=True); con.row_factory = sqlite3.Row
    c2 = {}
    for date, mid in (('2026-09-23', 7), ('2026-11-04', 12)):
        rows = [dict(r) for r in con.execute('select * from session_hole_metrics where model_id=?', (mid,)) if r['session_id'] in elig]
        wrong = {elig[r['session_id']]['session_dir'] for r in rows if r['n_judged'] and 2 * (r['n_wrong'] or 0) > r['n_judged']}
        none = {elig[r['session_id']]['session_dir'] for r in rows if not r['n_judged']}
        c2[date] = dict(model_id=mid, sessions=len(rows), sessions_by_state=dict(collections.Counter(state(elig[r['session_id']]['session_dir']) for r in rows)),
                        majority_wrong=len(wrong), majority_wrong_by_window=dict(collections.Counter(window(g) for g in wrong)),
                        none_judged=len(none), majority_wrong_also_failed_in_manuscript=len(wrong & failed),
                        manuscript_failed_not_majority_wrong=len(failed - wrong - none))
    res['C2'] = c2
    r7 = [dict(r) for r in con.execute('select session_id from session_hole_metrics where model_id=7')]
    ids = {r['session_id'] for r in r7}; inlist = ids & set(elig); err = ids & {s for s, r in excluded.items() if r['reason'].startswith(('E23', 'human'))}
    res['C3'] = dict(rows_model7=len(ids), among_361=len(inlist), input_error_sessions=len(err), other=len(ids - inlist - err),
                     among_361_by_state=dict(collections.Counter(state(elig[s]['session_dir']) for s in inlist)),
                     last_session=max(elig[s]['session_dir'] for s in inlist))
    # C4 Mahalanobis: rejected registered images inside fully rejected doors
    c4 = {}
    for seed, f in ((42, 'reports/extension_v12/cnn_unseen_predictions.jsonl'), (123, 'reports/extension_v12s/s123/cnn_unseen_predictions.jsonl'),
                    (456, 'reports/extension_v12s/s456/cnn_unseen_predictions.jsonl')):
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for line in open(ROOT / f):
            r = json.loads(line)
            if r.get('condition', 'unseen') == 'unseen' and r['score_name'] == 'maha' and r['class_name'] != r['excluded']:
                agg[r['kind']][(r['excluded'], r['group_id'])].append(r['accepted'])
        for kind, doors in agg.items():
            rej = sum(v.count(False) for v in doors.values()); full = sum(len(v) for v in doors.values() if not any(v))
            c4[f'{kind}:s{seed}'] = dict(images_rejected=rej, in_fully_rejected_doors=full, share=full / rej,
                                         doors_fully_rejected=sum(1 for v in doors.values() if not any(v)),
                                         doors_partly_rejected=sum(1 for v in doors.values() if any(v) and not all(v)), doors=len(doors))
    res['C4'] = c4
    # C5 annotated field images by camera state
    _, _, labels = V12.load_field_labels(V3.LABELSET, sp); c5 = dict(annotated_40=dict(collections.Counter(state(r['group_id']) for r in labels)), folds={})
    for f in FOLDS:
        meta = read_json(ROOT / 'runs' / V12.hole_run(f) / 'run.json')
        c5['folds'][f] = dict(train_field_sessions=dict(collections.Counter(state(g) for g in meta['train_group_ids'])),
                              selection_field_sessions=dict(collections.Counter(state(g) for g in meta['val_group_ids'])))
    res['C5'] = c5
    dump(OUT / 'retabulations.json', res); print(json.dumps(res, ensure_ascii=False, indent=1))


def collect():
    a = read_json(OUT / 'scale_transfer.json'); b = read_json(OUT / 'input_errors.json'); c = read_json(OUT / 'retabulations.json')
    for x in (a, b, c):
        if x['protocol_sha256'] != sha256(PROTOCOL):
            raise ValueError('Protocol changed')
    keep = ('forward', 'forward_reference', 'forward_readjusted_only', 'reverse', 'reverse_reference')
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL),
               scale_transfer={mode: {k: m[k] for k in keep} | dict(bumped=m['forward_bumped_sessions'],
                                                                    scales={f: {d: v[d]['scale'] if d != 'all' else v[d] for d in ('all', 'forward', 'reverse')}
                                                                            for f, v in m['scale_by_fold'].items()}) for mode, m in a['modes'].items()},
               input_errors=dict(bound=b['door_accuracy_if_operator_inputs_were_right'],
                                 sessions={s: dict(images=d['images'], operator_input=d['operator_input'], corrected=d['corrected'],
                                                   hole=d['hole']['decision'], hole_D=d['hole']['measured_D_median'],
                                                   rgb=d['rgb_f0_s42']['decision'], rgbe=d['rgbe_f0_s42']['decision'], agrees=d['agrees']) for s, d in b['sessions'].items()}),
               retabulations={k: c[k] for k in ('C1', 'C2', 'C3', 'C4', 'C5')})
    dump(OUT / 'collected.json', out); print('collected')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('cmd', choices=('scale', 'errors', 'retab', 'collect'))
    x = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    dict(scale=scale, errors=errors, retab=retab, collect=collect)[x.cmd]()
