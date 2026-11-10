"""Extension experiments registered in config/extension_v3_protocol.json.

  train-holes --fold {1..4|temporal} --run NAME --select {nested8|all_train}
  eval-holes  --fold K --run NAME            (TRAIN+TEST features, train-only scale, TEST metrics)
  eval-ops    --fold K                       (operational 2026-08-26 detector, features already frozen)
  e23                                        (unregistered-class experiment, fold-0 models)
  collect                                    (5-fold OOF + temporal + E23 summary)
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import setup_runtime
setup_runtime()

import cv2
import numpy as np
import torch
from torch import nn

from door8.core import (ROOT, SOURCE, capture_code, check_source, dump, jsonl, now, partition, read_json,
                        read_jsonl, require_fresh_dir, sha256, snapshot)
from door8.features import peaks
from door8.field_data import field_partitions, load_field_labels, nested_budget
from door8.field_experiments import check_adaptation_provenance, localization, validation_loss
from door8.geometry import bolt_frame, classify_measurement, depth_measure, fit_scale, geometry_gate
from door8.learning import (CNNData, Classifier, HoleData, HoleNet, cnn_predict, heatmap_loss, letterbox,
                            load_run, make_loader, pretrained_resnet, seed_all, tensor_image)
from door8.metrics import paired_session_comparison, summarize, wilson

PROTOCOL = ROOT / 'config/extension_v3_protocol.json'
LABELSET = 'field40_20261109_v1'
OPS_FEATURES = ROOT / 'features/operational_20260826_f0_test_v3'
OPS_WEIGHTS = SOURCE / 'attribute_models/hole_landmarks/model.pth'
HP = dict(epochs=80, batch_size=4, learning_rate=1e-4, width=1280, height=768, stride=4, sigma=2.0, peak_threshold=0.15)
E23_SESSIONS = ['20260903/E25_door_LH_FRT/s_124446', '20260903/E25_door_LH_FRT/s_132914',
                '20260903/E25_door_LH_FRT/s_141947', '20260905/E25_door_LH_FRT/s_150506',
                '20260912/E25_door_LH_FRT/s_094850', '20260912/E25_door_LH_FRT/s_103329']
E23 = ('E23_door_LH_FRT', 458)


# ---------------------------------------------------------------- hole training
def hole_inputs(fold, select):
    sp, _, frames, splits = snapshot()
    lp, _, labels = load_field_labels(LABELSET, sp)
    roles = field_partitions(labels, frames, splits, fold)
    classes = read_json(sp / 'protocol.json')['classes']
    if select == 'nested8':
        selected = nested_budget(roles['train'], classes, 8, str(fold), 20261109)
    elif select == 'all_train':
        selected = list(roles['train'])
    else:
        raise ValueError(select)
    lab = read_jsonl(sp / 'annotations_lab.jsonl'); ls = read_json(sp / 'annotations_lab_split.json')
    tr = [r for r in lab if r['annotation_id'] in ls['train']] + selected
    lv = [r for r in lab if r['annotation_id'] in ls['val']]
    va = roles['val']
    parts = partition(frames, splits, fold)
    held = {r['group_id'] for r in parts['val'] + parts['test']}
    if {r['group_id'] for r in selected} & held or not va:
        raise ValueError('Field supervision leakage or empty field VAL')
    for r in tr + va + lv:
        if sha256(r['rgb']) != r['rgb_sha256']:
            raise ValueError('Annotation image changed')
    return sp, lp, tr, va, lv, selected


def train_holes(run, fold, select, seed=42, workers=2, memory_fraction=.12):
    sp, lp, tr, va, lv, selected = hole_inputs(fold, select)
    dst = require_fresh_dir('runs/' + run); seed_all(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    meta = dict(model_kind='holes', training_regime='human_manual_lab_plus_field', snapshot='v1',
                snapshot_sha256=sha256(sp / 'COMPLETE.json'), labelset=LABELSET, labelset_sha256=sha256(lp / 'COMPLETE.json'),
                extension_protocol_sha256=sha256(PROTOCOL), fold=str(fold), seed=seed, selection=select,
                field_train_budget=len(selected), field_val_images=len(va), lab_train_images=94, lab_val_images=24,
                train_annotation_ids=[r['annotation_id'] for r in tr], val_annotation_ids=[r['annotation_id'] for r in va],
                train_group_ids=sorted({r['group_id'] for r in selected}), val_group_ids=sorted({r['group_id'] for r in va}),
                initialization='ImageNet1K_V1', initialization_sha256=sha256(pretrained_resnet()), pilot=False,
                requested_epochs=HP['epochs'], hyperparameters=HP, status='running', started_at=now())
    meta.update(capture_code()); dump(dst / 'run.json', meta)
    t0 = time.perf_counter()
    net = HoleNet().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=HP['learning_rate'])
    dl = make_loader(HoleData(tr, HP, True), HP['batch_size'], True, workers, seed)
    vl = make_loader(HoleData(va, HP, False), HP['batch_size'], False, workers, seed)
    ll = make_loader(HoleData(lv, HP, False), HP['batch_size'], False, workers, seed)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, HP['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda'); best = float('inf'); hist = []
    for ep in range(HP['epochs']):
        net.train(); losses = []
        for x, y, m in dl:
            x, y, m = x.to(dev), y.to(dev), m.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == 'cuda'):
                loss = heatmap_loss(net(x), y, m)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        fl = validation_loss(net, vl, dev); lab_l = validation_loss(net, ll, dev)
        hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(losses)), field_val_loss=fl, lab_val_loss_monitor_only=lab_l))
        if fl < best:
            best = fl; meta.update(best_epoch=ep + 1, best_field_val_loss=fl); torch.save(net.state_dict(), dst / 'model.pth')
        sched.step(); dump(dst / 'history.json', hist); dump(dst / 'run.json', meta)
        print(json.dumps(dict(run=run, **hist[-1])), flush=True)
    meta.update(status='complete', finished_at=now(), seconds=time.perf_counter() - t0, weights_sha256=sha256(dst / 'model.pth'))
    dump(dst / 'run.json', meta)


# ---------------------------------------------------------------- features / evaluation
@torch.no_grad()
def feature(net, im, depth, row, dev):
    small, M = letterbox(im, HP['width'], HP['height'])
    with torch.autocast(device_type=dev.type, enabled=dev.type == 'cuda'):
        hm = net(tensor_image(small)[None].to(dev))[0].float().cpu().numpy()
    det = {k: peaks(hm[c], n, HP['peak_threshold'], M, HP['stride'])
           for c, (k, n) in enumerate([('bolt', 4), ('hinge', 1), ('latch', 1)])}
    hinge = det['hinge'][0] if det['hinge'] else None
    latch = det['latch'][0] if det['latch'] else None
    f = dict(frame_id=row['frame_id'], rgb_sha256=row['rgb_sha256'], points=det,
             gate=geometry_gate(bolt_frame(det['bolt']), hinge, latch, im.shape), span_normalized=None)
    if hinge is not None and latch is not None:
        f['span_normalized'] = float(np.linalg.norm(np.asarray(hinge[:2]) - latch[:2]) / row['intrinsics']['fx'])
        m = depth_measure(depth, det['bolt'] + [hinge, latch], hinge, latch, row['intrinsics'])
        if m:
            f.update(m)
    return f


def extract(net, rows, dev, verify=True):
    out = {}
    for i, row in enumerate(rows):
        if verify:
            check_source(row)
        im = cv2.imread(row['rgb']); depth = cv2.imread(row['depth'], cv2.IMREAD_UNCHANGED)
        if im is None or depth is None or depth.shape != im.shape[:2]:
            raise ValueError('Unreadable/unaligned source pair')
        out[row['frame_id']] = feature(net, im, depth, row, dev)
        if (i + 1) % 1000 == 0:
            print(f'features {i + 1}/{len(rows)}', flush=True)
    return out


def evaluate(dst, tag, parts, pool, cfg, extra=None):
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); res = {}
    for mode in ('pixel', 'depth'):
        cal = fit_scale(parts['train'], pool, cad, mode)
        if set(cal['train_frame_ids']) != {r['frame_id'] for r in parts['train']}:
            raise ValueError('Calibration leakage')
        preds = []
        for row in parts['test']:
            p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg)
            preds.append({k: row[k] for k in ['frame_id', 'session_dir', 'group_id', 'class_name']} | dict(pred=p, measurement=v, reason=why))
        m = summarize(preds, cfg['classes'])
        jsonl(dst / f'{tag}_{mode}_predictions.jsonl', preds); dump(dst / f'{tag}_{mode}_metrics.json', m)
        dump(dst / f'{tag}_{mode}_calibration.json', {k: cal[k] for k in ('mode', 'camera_scales', 'valid_train_sessions')})
        res[mode] = dict(scale=cal['camera_scales'], frame=[m['frame']['correct'], m['frame']['n']],
                         session=[m['session']['correct'], m['session']['n']], session_abstained=m['session']['abstained'],
                         wilson=m['session']['wilson95_accuracy_all'], reasons=dict(Counter(r['reason'] for r in preds)))
    res.update(extra or {})
    dump(dst / f'{tag}_summary.json', res)
    print(json.dumps({tag: res}, ensure_ascii=False), flush=True)
    return res


def eval_holes(fold, run):
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    parts = partition(frames, splits, fold)
    rp, rm = load_run(run, sp); check_adaptation_provenance(rm, sp, frames, splits, fold)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_per_process_memory_fraction(.12) if dev.type == 'cuda' else None
    net = HoleNet(pretrained=False).to(dev)
    net.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True)); net.eval()
    fdir = require_fresh_dir(f'features/{run}_trainTest')
    pool = extract(net, parts['train'] + parts['test'], dev)
    jsonl(fdir / 'rows.jsonl', list(pool.values()))
    dump(fdir / 'features.json', dict(status='complete', run=run, fold=str(fold), subset='train_test', n=len(pool),
                                      rows_sha256=sha256(fdir / 'rows.jsonl'), created_at=now()))
    lp, _, labels = load_field_labels(LABELSET, sp); roles = field_partitions(labels, frames, splits, fold)
    loc, _ = localization(roles['test'], pool) if roles['test'] else (None, None)
    dst = ROOT / 'reports/extension_v3'; dst.mkdir(exist_ok=True)
    return evaluate(dst, run, parts, pool, cfg, dict(localization_test_role=loc, fold=str(fold),
                                                     weights_sha256=rm['weights_sha256']))


def ops_pool():
    fm = read_json(OPS_FEATURES / 'features.json')
    if sha256(OPS_FEATURES / 'rows.jsonl') != fm['rows_sha256']:
        raise ValueError('Operational features changed')
    return {r['frame_id']: r for r in read_jsonl(OPS_FEATURES / 'rows.jsonl')}


def eval_ops(fold):
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    parts = partition(frames, splits, fold)
    dst = ROOT / 'reports/extension_v3'; dst.mkdir(exist_ok=True)
    return evaluate(dst, f'operational_20260826_{fold}', parts, ops_pool(), cfg, dict(fold=str(fold)))


# ---------------------------------------------------------------- E23 open set
def e23_rows():
    base = (SOURCE / 'datasets_factory_collect').resolve(); rows = []
    for s in E23_SESSIONS:
        d = base / s; meta = read_json(d / 'meta.json') if (d / 'meta.json').exists() else {}
        intr = meta.get('intrinsics') or dict(serial=54910212, width=1920, height=1200, fx=1269.746, fy=1269.746, cx=961.669, cy=598.594)
        for rgb in sorted(d.glob('rgb_*.png')):
            dep = d / rgb.name.replace('rgb_', 'depth_')
            rows.append(dict(frame_id=f'{s}:{rgb.name}', session_dir=s, group_id=s, class_name=E23[0], class_index=0,
                             rgb=str(rgb), depth=str(dep), intrinsics=intr, rgb_sha256=sha256(rgb), depth_sha256=sha256(dep),
                             rgb_bytes=rgb.stat().st_size, rgb_mtime_ns=rgb.stat().st_mtime_ns,
                             depth_bytes=dep.stat().st_size, depth_mtime_ns=dep.stat().st_mtime_ns))
    if len(rows) != 120:
        raise ValueError(f'Expected 120 E23 frames, found {len(rows)}')
    return rows


def session_outcome(preds):
    by = {}
    for r in preds:
        by.setdefault(r['session_dir'], []).append(r['pred'])
    out = {}
    for s, pp in by.items():
        votes = Counter(p for p in pp if p).most_common()
        out[s] = votes[0][0] if votes and (len(votes) == 1 or votes[0][1] > votes[1][1]) else None
    return out


def e23():
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    parts = partition(frames, splits, '0')
    dst = require_fresh_dir('reports/e23_openset_v3')
    rows = e23_rows(); jsonl(dst / 'manifest.jsonl', rows)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_reg = copy.deepcopy(cfg); cfg_reg['classes'] = cfg['classes'] + [E23[0]]
    cfg_reg['cad_hole_distance_mm'] = cfg['cad_hole_distance_mm'] + [E23[1]]
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    summary = dict(protocol_sha256=sha256(PROTOCOL), frames=len(rows), sessions=len(E23_SESSIONS), results={})
    # hole models
    f8_dev = {r['frame_id']: r for r in read_jsonl(ROOT / 'features/holes_field8_f0_s42_v2_development/rows.jsonl')}
    ops = ops_pool()
    rp, _ = load_run('holes_field8_f0_s42_v2', sp)
    net8 = HoleNet(pretrained=False).to(dev); net8.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True)); net8.eval()
    sys.path.insert(0, str(SOURCE)); import hole_classifier as hc
    netops = hc.Net(3).to(dev); netops.load_state_dict(torch.load(OPS_WEIGHTS, map_location=dev, weights_only=True)); netops.eval()
    for name, net, train_pool in (('holes_field8_f0', net8, f8_dev), ('operational_20260826', netops, ops)):
        feats = extract(net, rows, dev)
        jsonl(dst / f'{name}_features.jsonl', list(feats.values()))
        cal = fit_scale(parts['train'], train_pool, cad, 'pixel')
        res = {}
        for cond, c in (('unregistered', cfg), ('registered', cfg_reg)):
            preds = []
            for row in rows:
                p, v, why = classify_measurement(row, feats[row['frame_id']], cal, c)
                preds.append(dict(frame_id=row['frame_id'], session_dir=row['session_dir'], pred=p, measurement=v, reason=why))
            jsonl(dst / f'{name}_{cond}_predictions.jsonl', preds)
            so = session_outcome(preds)
            res[cond] = dict(frame_pred=dict(Counter(str(p['pred']) for p in preds)), frame_reason=dict(Counter(p['reason'] for p in preds)),
                             session_pred=dict(Counter(str(v) for v in so.values())),
                             measured_D_median=float(np.median([p['measurement'] for p in preds if p['measurement'] is not None])) if any(p['measurement'] for p in preds) else None)
        res['scale'] = cal['camera_scales']
        summary['results'][name] = res
    # CNN classifiers (closed set) with the v1 VAL-95%-coverage threshold
    for run in ('rgb_f0_s42', 'rgbe_f0_s42'):
        rp, rm = load_run(run, sp)
        net = Classifier(rm['model_kind'], len(cfg['classes']), pretrained=False).to(dev)
        net.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True))
        loader = make_loader(CNNData(rows, rm['model_kind'], rm['image_size'], False), 16, False, 2, 42)
        preds, _ = cnn_predict(net, loader, rows, cfg['classes'], dev)
        thr = read_json(ROOT / f'reports/{run}_test/report.json')['val95coverage_threshold']
        rej = [dict(r, pred=r['pred'] if r['confidence'] >= thr else None) for r in preds]
        jsonl(dst / f'{run}_predictions.jsonl', preds)
        summary['results'][run] = dict(
            threshold=thr,
            closed_set=dict(frame_pred=dict(Counter(r['pred'] for r in preds)), session_pred=dict(Counter(str(v) for v in session_outcome(preds).values()))),
            thresholded=dict(frame_pred=dict(Counter(str(r['pred']) for r in rej)), session_pred=dict(Counter(str(v) for v in session_outcome(rej).values()))),
            confidence_median=float(np.median([r['confidence'] for r in preds])),
            confidence_min=float(np.min([r['confidence'] for r in preds])))
    summary['finished_at'] = now(); dump(dst / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- OOF collection
def collect():
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    ext = ROOT / 'reports/extension_v3'; f0 = ROOT / 'reports/fold0_test_v3'
    def preds_for(method, fold):
        if method == 'hole8':
            p = f0 / 'holes_field8_f0_s42_v2_pixel_test_predictions.jsonl' if fold == '0' else ext / f'holes_field8_f{fold}_s42_v3_pixel_predictions.jsonl'
        elif method == 'hole8_depth':
            p = f0 / 'holes_field8_f0_s42_v2_depth_test_predictions.jsonl' if fold == '0' else ext / f'holes_field8_f{fold}_s42_v3_depth_predictions.jsonl'
        elif method == 'ops':
            p = f0 / 'operational_20260826_pixel_test_predictions.jsonl' if fold == '0' else ext / f'operational_20260826_{fold}_pixel_predictions.jsonl'
        else:
            p = ROOT / f'reports/{method}_f{fold}_s42_test/predictions.jsonl'
        return read_jsonl(p) if p.exists() else None
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), cv5={}, per_fold={}, temporal={})
    allp = {}
    for method in ('hole8', 'hole8_depth', 'ops', 'rgb', 'rgbe'):
        rows = []; ok = True
        for k in '01234':
            p = preds_for(method, k)
            if p is None:
                ok = False; continue
            m = summarize(p, cfg['classes'])
            out['per_fold'].setdefault(method, {})[k] = dict(session=[m['session']['correct'], m['session']['n']], frame=[m['frame']['correct'], m['frame']['n']])
            rows += p
        if ok:
            m = summarize(rows, cfg['classes']); allp[method] = rows
            out['cv5'][method] = dict(session=[m['session']['correct'], m['session']['n']], session_abstained=m['session']['abstained'],
                                      wilson=m['session']['wilson95_accuracy_all'], frame=[m['frame']['correct'], m['frame']['n']],
                                      frame_abstained=m['frame']['abstained'], macro_f1=m['session']['macro_f1'],
                                      wrong_sessions=[dict(s=s['session_dir'], true=s['class_name'], pred=s['pred']) for s in m['session_predictions'] if s['pred'] != s['class_name']])
    for a in ('hole8', 'ops', 'rgbe'):
        if a in allp and 'rgb' in allp:
            out['cv5'].setdefault('paired_vs_rgb', {})[a] = paired_session_comparison(allp[a], allp['rgb'], cfg['classes'])
    tparts = partition(frames, splits, 'temporal')
    for name, p in (('hole', ext / 'holes_field_temporal_s42_pixel_predictions.jsonl'), ('hole_depth', ext / 'holes_field_temporal_s42_depth_predictions.jsonl'),
                    ('ops', ext / 'operational_20260826_temporal_pixel_predictions.jsonl'),
                    ('rgb', ROOT / 'reports/rgb_temporal_s42_test/predictions.jsonl'), ('rgbe', ROOT / 'reports/rgbe_temporal_s42_test/predictions.jsonl')):
        if p.exists():
            pr = read_jsonl(p); m = summarize(pr, cfg['classes'])
            out['temporal'][name] = dict(session=[m['session']['correct'], m['session']['n']], session_abstained=m['session']['abstained'],
                                         wilson=m['session']['wilson95_accuracy_all'], frame=[m['frame']['correct'], m['frame']['n']],
                                         wrong_sessions=[dict(s=s['session_dir'], true=s['class_name'], pred=s['pred']) for s in m['session_predictions'] if s['pred'] != s['class_name']])
    out['temporal']['test_sessions'] = len({r['group_id'] for r in tparts['test']})
    dump(ext / 'collected.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_fold'}, ensure_ascii=False, indent=1)[:6000])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('train-holes'); a.add_argument('--fold', required=True); a.add_argument('--run', required=True)
    a.add_argument('--select', required=True, choices=['nested8', 'all_train']); a.add_argument('--memory-fraction', type=float, default=.12)
    a = sub.add_parser('eval-holes'); a.add_argument('--fold', required=True); a.add_argument('--run', required=True)
    a = sub.add_parser('eval-ops'); a.add_argument('--fold', required=True)
    sub.add_parser('e23'); sub.add_parser('collect')
    x = p.parse_args()
    if x.cmd == 'train-holes':
        train_holes(x.run, x.fold, x.select, memory_fraction=x.memory_fraction)
    elif x.cmd == 'eval-holes':
        eval_holes(x.fold, x.run)
    elif x.cmd == 'eval-ops':
        eval_ops(x.fold)
    elif x.cmd == 'e23':
        e23()
    else:
        collect()
