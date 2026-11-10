"""Fold-0 TEST evaluation registered in config/fold0_test_v3_protocol.json.

Main: holes_field8_f0_s42_v2. Sensitivity: holes_field24_f0_s42_v2.
Reference: operational hole-landmark weights frozen 2026-08-26 (before the first snapshot session),
run through the SAME peak extraction, geometry gate, TRAIN-only calibration and CAD decision.
Nothing outside this research folder is modified; the operational module is only imported.
"""
from collections import Counter
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import setup_runtime
setup_runtime()

import cv2
import numpy as np
import torch

from door8.core import (ROOT, SOURCE, check_source, dump, jsonl, now, partition, read_json,
                        read_jsonl, require_fresh_dir, sha256, snapshot)
from door8.features import peaks
from door8.field_data import field_partitions, load_field_labels
from door8.field_experiments import check_adaptation_provenance, localization
from door8.geometry import bolt_frame, classify_measurement, depth_measure, fit_scale, geometry_gate
from door8.learning import HoleNet, letterbox, load_run, tensor_image
from door8.metrics import paired_session_comparison, summarize

PROTOCOL = ROOT / 'config/fold0_test_v3_protocol.json'
FOLD = '0'
LABELSET = 'field40_20261109_v1'
OPS_WEIGHTS = SOURCE / 'attribute_models/hole_landmarks/model.pth'
OPS_LABEL_DIR = SOURCE / 'labels/holes'
OPS_SPLIT = SOURCE / 'attribute_models/hole_landmarks/split.json'


def load_models(plan, sp, frames, splits, dev):
    models = {}
    for key in ('holes_field8_f0_s42_v2', 'holes_field24_f0_s42_v2'):
        rp, rm = load_run(key, sp)
        check_adaptation_provenance(rm, sp, frames, splits, FOLD)
        if rm['weights_sha256'] != plan['weights_sha256'][key]:
            raise ValueError(f'{key}: weights differ from the registered protocol')
        net = HoleNet(pretrained=False).to(dev)
        net.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True)); net.eval()
        models[key] = net
    if sha256(OPS_WEIGHTS) != plan['weights_sha256']['operational_20260826']:
        raise ValueError('Operational reference weights differ from the registered protocol')
    sys.path.insert(0, str(SOURCE))
    import hole_classifier as hc  # operational module, import only
    sd = torch.load(OPS_WEIGHTS, map_location=dev, weights_only=True)
    if sd['head.6.weight'].shape[0] != 3:
        raise ValueError('Expected the 3-channel 2026-08-26 model')
    ops = hc.Net(3).to(dev); ops.load_state_dict(sd); ops.eval()
    models['operational_20260826'] = ops
    return models


def ops_provenance(frames):
    """The reference detector's own training labels must not come from any snapshot session."""
    snap_sessions = {r['session_dir'] for r in frames}
    holdout = set(read_json(OPS_SPLIT)['holdout'])
    keys = sorted(p.stem for p in OPS_LABEL_DIR.glob('*.json'))
    train = [k for k in keys if k not in holdout]
    field = [k for k in keys if k.startswith('field__')]
    # field__<class>_s_<hhmmss>__<idx>; the only field session is the excluded 2026-08-10 pre-protocol one.
    field_sessions = sorted({k.split('__')[1] for k in field})
    if field_sessions != ['E25_door_RH_s_091317']:
        raise ValueError(f'Unexpected operational field label sessions: {field_sessions}')
    overlap = [s for s in snap_sessions if s.endswith('/E25_door_RH/s_091317')]
    labeled_at = max(read_json(OPS_LABEL_DIR / (k + '.json')).get('labeled_at', '') for k in keys)
    if overlap or labeled_at >= '2026-08-27':
        raise ValueError(f'Operational labels overlap the snapshot period: {overlap} {labeled_at}')
    return dict(label_files=len(keys), train=len(train), holdout=len(holdout),
                field_label_files=len(field), field_in_train=sum(k in train for k in field),
                field_label_sessions=['20260810/E25_door_RH/s_091317 (excluded pre-protocol session)'],
                latest_label_time=labeled_at, overlap_with_snapshot_sessions=overlap,
                weights_file_mtime=str(np.datetime64(int(OPS_WEIGHTS.stat().st_mtime), 's')),
                db_training_session='door_pipeline training_sessions id 6: 2026-08-26 15:40:49-15:49:19 KST, 107/27, 80 epochs')


@torch.no_grad()
def feature(net, small, M, im, depth, row, hp, dev):
    with torch.autocast(device_type=dev.type, enabled=dev.type == 'cuda'):
        hm = net(tensor_image(small)[None].to(dev))[0].float().cpu().numpy()
    det = {k: peaks(hm[c], n, hp['peak_threshold'], M, hp['stride'])
           for c, (k, n) in enumerate([('bolt', 4), ('hinge', 1), ('latch', 1)])}
    hinge = det['hinge'][0] if det['hinge'] else None
    latch = det['latch'][0] if det['latch'] else None
    f = dict(frame_id=row['frame_id'], rgb_sha256=row['rgb_sha256'], points=det,
             gate=geometry_gate(bolt_frame(det['bolt']), hinge, latch, im.shape), span_normalized=None,
             max_heat=[float(v) for v in hm.reshape(3, -1).max(1)])
    if hinge is not None and latch is not None:
        f['span_normalized'] = float(np.linalg.norm(np.asarray(hinge[:2]) - latch[:2]) / row['intrinsics']['fx'])
        m = depth_measure(depth, det['bolt'] + [hinge, latch], hinge, latch, row['intrinsics'])
        if m:
            f.update(m)
    return f


def development_train_features(key, sp, parts):
    fp = ROOT / 'features' / f'{key}_development'
    fm = read_json(fp / 'features.json')
    if fm['status'] != 'complete' or fm['snapshot_sha256'] != sha256(sp / 'COMPLETE.json') or sha256(fp / 'rows.jsonl') != fm['rows_sha256']:
        raise ValueError(f'{key}: development features invalid')
    idx = {r['frame_id']: r for r in read_jsonl(fp / 'rows.jsonl')}
    need = {r['frame_id'] for r in parts['train']}
    if not need <= set(idx):
        raise ValueError(f'{key}: TRAIN features incomplete')
    return {k: idx[k] for k in need}, fm['rows_sha256']


def relative_errors(preds, cad):
    rel = [r['measurement'] / cad[r['class_name']] - 1 for r in preds if r.get('measurement') is not None and r['reason'] == 'ok']
    by = {}
    for r in preds:
        if r.get('measurement') is not None and r['reason'] == 'ok':
            by.setdefault(r['session_dir'], []).append(r['measurement'] / cad[r['class_name']] - 1)
    sm = [float(np.median(v)) for v in by.values()]
    if not rel:
        return None
    q = lambda a, p: float(np.quantile(a, p))
    return dict(n=len(rel), frame_p05=q(rel, .05), frame_median=q(rel, .5), frame_p95=q(rel, .95),
                frame_abs_max=float(np.max(np.abs(rel))), session_median_min=min(sm), session_median_max=max(sm),
                session_median_sd=float(np.std(sm)), sessions=len(sm))


def error_pairs(preds):
    return [dict(true=t, pred=p, frames=n) for (t, p), n in
            Counter((r['class_name'], r['pred']) for r in preds if r['pred'] and r['pred'] != r['class_name']).most_common()]


def main():
    plan = read_json(PROTOCOL)
    sp, _, frames, splits = snapshot()
    cfg = read_json(sp / 'protocol.json'); hp = cfg['holes']
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    parts = partition(frames, splits, FOLD)
    lp, _, labels = load_field_labels(LABELSET, sp)
    roles = field_partitions(labels, frames, splits, FOLD)
    if len(parts['test']) != 1478 or len({r['group_id'] for r in parts['test']}) != 74:
        raise ValueError('Unexpected fold-0 TEST size')
    prov = ops_provenance(frames)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_models(plan, sp, frames, splits, dev)
    out = require_fresh_dir('reports/fold0_test_v3')
    feat_dirs = {k: require_fresh_dir(f'features/{k}_f0_test_v3') for k in models}
    started = now()

    feats = {k: {} for k in models}
    test_ids = {r['frame_id'] for r in parts['test']}
    rows = parts['test'] + parts['train'] + parts['val']
    for i, row in enumerate(rows):
        check_source(row)
        im = cv2.imread(row['rgb']); depth = cv2.imread(row['depth'], cv2.IMREAD_UNCHANGED)
        if im is None or depth is None or depth.shape != im.shape[:2]:
            raise ValueError('Unreadable/unaligned source pair')
        small, M = letterbox(im, hp['width'], hp['height'])
        for k, net in models.items():
            if row['frame_id'] in test_ids or k == 'operational_20260826':
                feats[k][row['frame_id']] = feature(net, small, M, im, depth, row, hp, dev)
        if (i + 1) % 500 == 0:
            print(f'frames {i + 1}/{len(rows)}', flush=True)

    for k, d in feat_dirs.items():
        jsonl(d / 'rows.jsonl', list(feats[k].values()))
        dump(d / 'features.json', dict(status='complete', model=k, fold=FOLD, n=len(feats[k]),
                                        subset='test' if k != 'operational_20260826' else 'train_val_test',
                                        protocol_sha256=sha256(PROTOCOL), rows_sha256=sha256(d / 'rows.jsonl'),
                                        snapshot_sha256=sha256(sp / 'COMPLETE.json'), created_at=now()))

    rgb_pred = read_jsonl(ROOT / 'reports/rgb_f0_s42_test/predictions.jsonl')
    rgbe_pred = read_jsonl(ROOT / 'reports/rgbe_f0_s42_test/predictions.jsonl')
    summary = dict(protocol=str(PROTOCOL.relative_to(ROOT)), protocol_sha256=sha256(PROTOCOL), started_at=started,
                   test_frames=len(parts['test']), test_sessions=len({r['group_id'] for r in parts['test']}),
                   operational_reference_provenance=prov, results={})
    train_dev_hash = {}
    for k in models:
        if k == 'operational_20260826':
            pool = feats[k]
        else:
            tr, train_dev_hash[k] = development_train_features(k, sp, parts)
            pool = {**tr, **feats[k]}
        res = {}
        for mode in ('pixel', 'depth'):
            cal = fit_scale(parts['train'], pool, cad, mode)
            if set(cal['train_frame_ids']) != {r['frame_id'] for r in parts['train']}:
                raise ValueError('Calibration leakage')
            subsets = [('test', parts['test'])] + ([('val', parts['val'])] if k == 'operational_20260826' else [])
            for name, subset in subsets:
                preds = []
                for row in subset:
                    p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg)
                    preds.append({kk: row[kk] for kk in ['frame_id', 'session_dir', 'group_id', 'class_name']} |
                                 dict(pred=p, measurement=v, reason=why))
                m = summarize(preds, cfg['classes'])
                tag = f'{k}_{mode}_{name}'
                jsonl(out / f'{tag}_predictions.jsonl', preds); dump(out / f'{tag}_metrics.json', m)
                entry = dict(calibration_scales=cal['camera_scales'], valid_train_sessions=cal['valid_train_sessions'],
                             frame={x: m['frame'][x] for x in ['n', 'correct', 'abstained', 'coverage', 'accuracy_all', 'macro_f1']},
                             session={x: m['session'][x] for x in ['n', 'correct', 'abstained', 'coverage', 'accuracy_all', 'macro_f1', 'wilson95_accuracy_all']},
                             per_class_session_recall={c: [v['correct'], v['n']] for c, v in m['session']['per_class'].items()},
                             reasons=dict(Counter(r['reason'] for r in preds)), relative_error=relative_errors(preds, cad),
                             misclassified_pairs=error_pairs(preds))
                if name == 'test':
                    entry['paired_vs_rgb'] = paired_session_comparison(preds, rgb_pred, cfg['classes'])
                res[f'{mode}_{name}'] = entry
        loc, pts = localization(roles['test'], feats[k] if k != 'operational_20260826' else pool)
        jsonl(out / f'{k}_test_localization_points.jsonl', pts)
        res['localization_test_role_field8'] = loc
        mh = np.array([feats[k][fid]['max_heat'] for fid in test_ids])
        res['test_max_heat_median_bolt_hinge_latch'] = np.median(mh, 0).round(3).tolist()
        summary['results'][k] = res
    for name, pred in (('rgb_f0_s42', rgb_pred), ('rgbe_f0_s42', rgbe_pred)):
        m = summarize(pred, cfg['classes'])
        summary['results'][name] = dict(
            frame={x: m['frame'][x] for x in ['n', 'correct', 'coverage', 'accuracy_all', 'macro_f1']},
            session={x: m['session'][x] for x in ['n', 'correct', 'coverage', 'accuracy_all', 'macro_f1', 'wilson95_accuracy_all']},
            misclassified_sessions=[dict(session=s['session_dir'], true=s['class_name'], pred=s['pred'])
                                    for s in m['session_predictions'] if s['pred'] != s['class_name']])
    summary['development_train_features_rows_sha256'] = train_dev_hash
    summary['finished_at'] = now()
    dump(out / 'summary.json', summary)
    brief = {k: {kk: (v['session']['correct'], v['session']['n'], v['frame']['correct'], v['frame']['n'])
                 for kk, v in r.items() if isinstance(v, dict) and 'session' in v} for k, r in summary['results'].items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
