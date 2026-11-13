"""Analyses registered in config/extension_v12_protocol.json. No model is trained.

  cnn-infer --run NAME        A: float32 forward of an existing CNN; logits + Mahalanobis distances for VAL/TEST(/E23)
  cnn-analyze                 A: msp / maxlogit / energy / maha in float64: unseen, seen, registered operating point, E23
  hole-register               B: unseen class identified after registering only its CAD value (strict detectors)
  hole-gate                   C: geometry gate on versus off
  subpixel-extract --fold K   D: features with quadratic sub-pixel peaks for the fold's TRAIN+TEST frames
  subpixel-eval               D: integer versus refined peaks
  frames                      E: door decision from the first n frames
  timing                      F: per-image processing time on this PC
  collect                     reports/extension_v12/collected.json
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from door8.core import setup_runtime
setup_runtime()

import cv2
import numpy as np
import torch

from door8.core import ROOT, check_source, dump, jsonl, now, partition, read_json, read_jsonl, sha256, snapshot
from door8.field_data import field_partitions, load_field_labels
from door8.field_experiments import localization
from door8.geometry import bolt_frame, classify_measurement, fit_scale, geometry_gate
from door8.learning import CNNData, Classifier, HoleNet, letterbox, load_run, make_loader, tensor_image
from door8.metrics import wilson
import run_extension_v3 as V3
import run_extension_v5 as V5
import run_extension_v6 as V6

PROTOCOL = ROOT / 'config/extension_v12_protocol.json'
OUT = ROOT / 'reports/extension_v12'
CACHE = ROOT / 'cache/extension_v12'
FOLDS = ('0', '1', '2', '3', '4')
KINDS = ('rgb', 'rgbe')
SCORES = ('msp', 'maxlogit', 'energy', 'maha')


def ctx():
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    return sp, frames, splits, cfg


def hole_run(fold):
    return 'holes_field8_f0_s42_v2' if str(fold) == '0' else f'holes_field8_f{fold}_s42_v3'


def lse(a):
    m = a.max(1, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.log(np.exp(a - m).sum(1)) + m[:, 0]


def door_counts(rows):
    v = V5.vote(rows)
    return dict(n=len(v), correct=sum(x['pred'] == x['true'] for x in v.values()), abstained=sum(x['pred'] is None for x in v.values()),
                wrong=sum(x['pred'] is not None and x['pred'] != x['true'] for x in v.values()))


def frame_counts(rows):
    return dict(n=len(rows), correct=sum(r['pred'] == r['class_name'] for r in rows), abstained=sum(r['pred'] is None for r in rows),
                wrong=sum(r['pred'] is not None and r['pred'] != r['class_name'] for r in rows))


# ---------------------------------------------------------------- A: CNN forward
def cnn_runs():
    sp, frames, splits, cfg = ctx(); runs = []
    for kind in KINDS:
        runs += [dict(run=f'{kind}_loco_{c}_f0_s42', group='unseen', kind=kind, fold='0', exclude=c) for c in cfg['classes']]
        runs += [dict(run=f'{kind}_f{k}_s42', group='seen', kind=kind, fold=k, exclude=None) for k in FOLDS]
        runs += [dict(run=f'{kind}_temporal_s42', group='temporal', kind=kind, fold='temporal', exclude=None)]
    return runs


def stored_test_file(run, group):
    return ROOT / (f'reports/extension_v5/{run}_predictions.jsonl' if group == 'unseen' else f'reports/{run}_test/predictions.jsonl')


@torch.no_grad()
def forward(net, rows, kind, size, dev, workers):
    loader = make_loader(CNNData(rows, kind, size, False), 32, False, workers, 42)
    L = np.zeros((len(rows), net.head[-1].out_features), np.float32); Fe = np.zeros((len(rows), 512), np.float32)
    for x, _, idx in loader:
        f = net.backbone(x.to(dev)); z = net.head(f)                      # float32 forward, no autocast
        L[idx.numpy()] = z.float().cpu().numpy(); Fe[idx.numpy()] = f.float().cpu().numpy()
    return L, Fe


def maha_fit(F, y, n):
    F = F.astype(np.float64); mu = np.stack([F[y == c].mean(0) for c in range(n)])
    R = F - mu[y]; S = R.T @ R / len(F)
    return mu, np.linalg.pinv(S, hermitian=True), int(np.linalg.matrix_rank(S, hermitian=True))


def maha_dist(F, mu, P):
    F = F.astype(np.float64); out = np.empty((len(F), len(mu)))
    for c in range(len(mu)):
        d = F - mu[c]; out[:, c] = ((d @ P) * d).sum(1)
    return out


def cnn_infer(run, workers=4, memory_fraction=.2):
    spec = next(r for r in cnn_runs() if r['run'] == run)
    sp, frames, splits, cfg = ctx(); rp, rm = load_run(run, sp)
    classes = rm['classes']; idx = {c: i for i, c in enumerate(classes)}; byid = {r['frame_id']: r for r in frames}
    parts = partition(frames, splits, rm['fold'])
    tr = [byid[i] for i in rm['train_frame_ids']]; va = [byid[i] for i in rm['val_frame_ids']]; te = parts['test']
    if {r['group_id'] for r in tr + va} & {r['group_id'] for r in te}:
        raise ValueError('Leakage')
    if spec['exclude']:
        if rm['excluded_class'] != spec['exclude'] or any(r['class_name'] == spec['exclude'] for r in tr + va):
            raise ValueError('Excluded class in training data')
    elif [r['frame_id'] for r in parts['train']] != rm['train_frame_ids'] or [r['frame_id'] for r in parts['val']] != rm['val_frame_ids']:
        if {r['frame_id'] for r in parts['train']} != set(rm['train_frame_ids']) or {r['frame_id'] for r in parts['val']} != set(rm['val_frame_ids']):
            raise ValueError('Run does not match the partition')
    sets = dict(train=tr, val=va, test=te)
    if spec['group'] == 'seen' and spec['fold'] == '0':
        sets['e23'] = V3.e23_rows()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    net = Classifier(rm['model_kind'], len(classes), pretrained=False).to(dev)
    net.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True)); net.eval()
    res = {}
    for name, rows in sets.items():
        rows = [dict(r, class_index=idx.get(r['class_name'], 0)) for r in rows]
        res[name] = (rows,) + forward(net, rows, rm['model_kind'], rm['image_size'], dev, workers)
        print(f'{run} {name} {len(rows)}', flush=True)
    rows, _, Ftr = res['train']; y = np.array([r['class_index'] for r in rows])
    mu, P, rank = maha_fit(Ftr, y, len(classes))
    arrays = dict(classes=np.array(classes))
    for name in sets:
        if name == 'train':
            continue
        rows, L, Fe = res[name]
        arrays[f'{name}_ids'] = np.array([r['frame_id'] for r in rows]); arrays[f'{name}_logits'] = L
        arrays[f'{name}_maha'] = maha_dist(Fe, mu, P)
    (CACHE / 'cnn').mkdir(parents=True, exist_ok=True); np.savez_compressed(CACHE / 'cnn' / f'{run}.npz', **arrays)
    check = {}
    for name, path in (('test', stored_test_file(run, spec['group'])), ('val', rp / 'val_predictions.jsonl')):
        stored = {r['frame_id']: r for r in read_jsonl(path)}; rows, L, _ = res[name]
        if set(stored) != {r['frame_id'] for r in rows}:
            raise ValueError(f'Stored {name} predictions do not match')
        new = [classes[int(j)] for j in L.argmax(1)]
        check[name] = dict(n=len(rows), argmax_mismatch=sum(stored[r['frame_id']]['pred'] != p for r, p in zip(rows, new)),
                           stored_confidence_exactly_one=sum(stored[r['frame_id']]['confidence'] == 1.0 for r in rows))
    meta = dict(**spec, classes=classes, protocol_sha256=sha256(PROTOCOL), weights_sha256=rm['weights_sha256'],
                train_frames=len(res['train'][0]), covariance_rank=rank, consistency=check, torch=torch.__version__, device=str(dev), created_at=now())
    (OUT / 'cnn_infer').mkdir(parents=True, exist_ok=True); dump(OUT / 'cnn_infer' / f'{run}.json', meta)
    print(json.dumps(dict(run=run, consistency=check, rank=rank)), flush=True)


# ---------------------------------------------------------------- A: CNN analysis
def load_cnn(run):
    meta = read_json(OUT / 'cnn_infer' / f'{run}.json')
    if meta['protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('Protocol changed')
    z = np.load(CACHE / 'cnn' / f'{run}.npz', allow_pickle=False)
    return meta, {k: z[k] for k in z.files}


def score_table(logits, maha, allowed):
    z = logits[:, allowed].astype(np.float64); n = len(z); j = z.argmax(1); zmax = z[np.arange(n), j]
    zo = z.copy(); zo[np.arange(n), j] = -np.inf
    return j, dict(msp=zmax - lse(zo), maxlogit=zmax, energy=lse(z), maha=-maha[:, allowed].min(1))


def scored_rows(a, name, classes, allowed, info):
    j, s = score_table(a[f'{name}_logits'], a[f'{name}_maha'], allowed); out = {k: [] for k in SCORES}
    for n, fid in enumerate(a[f'{name}_ids'].tolist()):
        base = dict(frame_id=fid, group_id=info[fid]['group_id'], class_name=info[fid]['class_name'], pred=classes[allowed[int(j[n])]])
        for k in SCORES:
            out[k].append(dict(base, score=float(s[k][n])))
    return out


def threshold(rows):
    v = sorted(r['score'] for r in rows); t = v[int(.05 * len(v))]
    return t, float(np.mean([x >= t for x in v]))


def apply(rows, t):
    return [dict(r, pred=r['pred'] if r['score'] >= t else None) for r in rows]


def cnn_analyze():
    sp, frames, splits, cfg = ctx(); classes8 = cfg['classes']
    info = {r['frame_id']: r for r in frames}; e23 = {r['frame_id']: r for r in V3.e23_rows()}
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), unseen={}, seen={}, registered={}, e23={}, consistency={})
    dumped = []
    for spec in cnn_runs():
        res['consistency'][spec['run']] = read_json(OUT / 'cnn_infer' / f"{spec['run']}.json")['consistency']
    # unseen: 7-output models, fold-0 TEST
    for kind in KINDS:
        runs = {k: {} for k in SCORES}
        for c in classes8:
            meta, a = load_cnn(f'{kind}_loco_{c}_f0_s42'); cl = meta['classes']; allowed = list(range(len(cl)))
            va = scored_rows(a, 'val', cl, allowed, info); te = scored_rows(a, 'test', cl, allowed, info)
            for k in SCORES:
                t, acc = threshold(va[k]); m = V5.loco_metrics(te[k], apply(te[k], t), c, score=lambda r: r['score'])
                m.update(threshold=t, val_acceptance=acc); runs[k][c] = m
                dumped += [dict(r, kind=kind, excluded=c, score_name=k, accepted=r['score'] >= t, condition='unseen') for r in te[k]]
        res['unseen'][kind] = {k: dict(pooled=V5.pool_runs(runs[k]),
                                       per_class={c: dict(unknown_doors=[m['unknown_doors']['rejected'], m['unknown_doors']['n']],
                                                          accepted_as=m['unknown_doors']['accepted_as'], known=m['known_doors_threshold'],
                                                          auroc=m['auroc_frame'], val_acceptance=m['val_acceptance']) for c, m in runs[k].items()})
                               for k in SCORES}
    # seen: 8-output models with output c masked, fold 0 and five folds
    for kind in KINDS:
        runs = {k: {} for k in SCORES}
        for f in FOLDS:
            meta, a = load_cnn(f'{kind}_f{f}_s42'); cl = meta['classes']
            for c in classes8:
                ci = cl.index(c); allowed = [i for i in range(len(cl)) if i != ci]
                va = scored_rows(a, 'val', cl, allowed, info); te = scored_rows(a, 'test', cl, allowed, info)
                for k in SCORES:
                    t, acc = threshold([r for r in va[k] if r['class_name'] != c])
                    m = V5.loco_metrics(te[k], apply(te[k], t), c, score=lambda r: r['score'])
                    m.update(threshold=t, val_acceptance=acc); runs[k][f'{f}:{c}'] = m
        res['seen'][kind] = {k: dict(fold0=V5.pool_runs({n: m for n, m in runs[k].items() if n.startswith('0:')}), oof=V5.pool_runs(runs[k]))
                             for k in SCORES}
    # registered operating point: 8 outputs, five-fold OOF and temporal
    for kind in KINDS:
        res['registered'][kind] = {}
        for name, folds in (('oof', FOLDS), ('temporal', ('temporal',))):
            closed = []; opened = {k: [] for k in SCORES}; acc = {k: [] for k in SCORES}
            for f in folds:
                meta, a = load_cnn(f'{kind}_f{f}_s42' if f != 'temporal' else f'{kind}_temporal_s42'); cl = meta['classes']; allowed = list(range(len(cl)))
                va = scored_rows(a, 'val', cl, allowed, info); te = scored_rows(a, 'test', cl, allowed, info); closed += te['msp']
                for k in SCORES:
                    t, ac = threshold(va[k]); opened[k] += apply(te[k], t); acc[k].append(ac)
            res['registered'][kind][name] = dict(closed=dict(frames=frame_counts(closed), doors=door_counts(closed)),
                                                 thresholded={k: dict(frames=frame_counts(opened[k]), doors=door_counts(opened[k]),
                                                                      val_acceptance=[min(acc[k]), max(acc[k])]) for k in SCORES})
    # E23 on the 8-class fold-0 models
    for kind in KINDS:
        meta, a = load_cnn(f'{kind}_f0_s42'); cl = meta['classes']; allowed = list(range(len(cl)))
        va = scored_rows(a, 'val', cl, allowed, info); te = scored_rows(a, 'test', cl, allowed, info); ev = scored_rows(a, 'e23', cl, allowed, e23)
        res['e23'][kind] = {}
        for k in SCORES:
            t, _ = threshold(va[k]); op = apply(ev[k], t); v = V5.vote(op)
            res['e23'][kind][k] = dict(doors=len(v), doors_rejected=sum(x['pred'] is None for x in v.values()),
                                       accepted_as=dict(Counter(x['pred'] for x in v.values() if x['pred'] is not None)),
                                       frames_rejected=sum(r['pred'] is None for r in op), frames=len(op),
                                       auroc=V5.auroc([r['score'] for r in te[k]], [r['score'] for r in ev[k]]),
                                       known_test_frames_judged=float(np.mean([r['score'] >= t for r in te[k]])))
    OUT.mkdir(parents=True, exist_ok=True); dump(OUT / 'cnn_scores.json', res); jsonl(OUT / 'cnn_unseen_predictions.jsonl', dumped)
    for kind in KINDS:
        for k in SCORES:
            p = res['unseen'][kind][k]['pooled']; s = res['seen'][kind][k]
            print(kind, k, 'unseen', p['unknown_doors'], p['known_door_decisions'], round(p['auroc_mean'], 3), round(p['auroc_min'], 3),
                  '| seen f0', s['fold0']['unknown_doors'], s['fold0']['known_door_decisions'], '| oof', s['oof']['unknown_doors'], s['oof']['known_door_decisions'])


# ---------------------------------------------------------------- B: register an unseen class by its CAD value
def hole_register():
    sp, frames, splits, cfg = ctx(); parts = partition(frames, splits, '0')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); band = V5.bands(cad); res = {}; rows_out = []
    for c in cfg['classes']:
        run = V6.strict_run(c); fdir = ROOT / f'features/{run}_trainTest'; fm = read_json(fdir / 'features.json')
        if sha256(fdir / 'rows.jsonl') != fm['rows_sha256']:
            raise ValueError('Strict features changed')
        pool = {r['frame_id']: r for r in read_jsonl(fdir / 'rows.jsonl')}
        g = V5.cfg_without(cfg, c); cad7 = dict(zip(g['classes'], g['cad_hole_distance_mm']))
        cal = fit_scale([r for r in parts['train'] if r['class_name'] != c], pool, cad7, 'pixel')       # no calibration data of class c
        preds = []
        for row in parts['test']:
            p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg)                      # all 8 CAD values registered
            preds.append({k: row[k] for k in ('frame_id', 'group_id', 'class_name')} | dict(pred=p, measurement=v, reason=why, new_class=c))
        new = [r for r in preds if r['class_name'] == c]; old = [r for r in preds if r['class_name'] != c]
        rel = [r['measurement'] / cad[c] - 1 for r in new if r['measurement'] is not None]
        per = defaultdict(list)
        for r in new:
            if r['measurement'] is not None:
                per[r['group_id']].append(r['measurement'] / cad[c] - 1)
        util = [(np.median(v) / band[c][1] if np.median(v) > 0 else np.median(v) / band[c][0]) for v in per.values()]
        vn = V5.vote(new)
        res[c] = dict(scale=cal['camera_scales'], new_doors=door_counts(new), new_frames=frame_counts(new),
                      new_wrong_as=dict(Counter(x['pred'] for x in vn.values() if x['pred'] not in (None, c))),
                      other_doors=door_counts(old), relative_residual=[float(min(rel)), float(max(rel))] if rel else None,
                      max_utilisation=float(max(util)) if util else None)
        rows_out += preds
    tot = lambda key: {k: sum(res[c][key][k] for c in res) for k in ('n', 'correct', 'abstained', 'wrong')}
    pooled = dict(new_doors=tot('new_doors'), new_frames=tot('new_frames'), other_doors=tot('other_doors'))
    pooled['new_doors_wilson'] = wilson(pooled['new_doors']['correct'], pooled['new_doors']['n'])
    pooled['max_utilisation'] = max(res[c]['max_utilisation'] for c in res)
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), per_class=res, pooled=pooled)
    OUT.mkdir(parents=True, exist_ok=True); dump(OUT / 'hole_register.json', out); jsonl(OUT / 'hole_register_predictions.jsonl', rows_out)
    print(json.dumps(pooled))


# ---------------------------------------------------------------- C: geometry gate on / off
def hole_gate():
    sp, frames, splits, cfg = ctx(); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); res = {}; store = {}
    for split in FOLDS + ('temporal',):
        parts = partition(frames, splits, split); pool = V5.hole_pool(split); cal = fit_scale(parts['train'], pool, cad, 'pixel')
        stored = {r['frame_id']: r for r in read_jsonl(V5.test_prediction_file(split, 'pixel'))}
        for gates in (True, False):
            preds = []
            for row in parts['test']:
                p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg, gates=gates)
                preds.append({k: row[k] for k in ('frame_id', 'group_id', 'class_name')} | dict(pred=p, measurement=v, reason=why,
                             gate=pool[row['frame_id']].get('gate')))
            if gates and any(stored[r['frame_id']]['pred'] != r['pred'] for r in preds):
                raise ValueError('Gate-on decisions differ from the stored predictions')
            store[(split, gates)] = preds
    for name, folds in (('oof', FOLDS), ('temporal', ('temporal',))):
        res[name] = {}
        for gates in (True, False):
            preds = [r for f in folds for r in store[(f, gates)]]
            gated = [r for r in preds if r['gate'] != 'ok']
            res[name]['on' if gates else 'off'] = dict(frames=frame_counts(preds), doors=door_counts(preds),
                                                         reasons=dict(Counter(r['reason'] for r in preds)),
                                                         frames_failing_gate=dict(frame_counts(gated), by_gate=dict(Counter(r['gate'] for r in gated))))
    # E23 with the 8-class catalog (fold-0 detector)
    parts = partition(frames, splits, '0'); cal = fit_scale(parts['train'], V5.hole_pool('0'), cad, 'pixel')
    rows = read_jsonl(ROOT / 'reports/e23_openset_v3/manifest.jsonl')
    feats = {r['frame_id']: r for r in read_jsonl(ROOT / 'reports/e23_openset_v3/holes_field8_f0_features.jsonl')}
    res['e23'] = {}
    for gates in (True, False):
        preds = []
        for row in rows:
            p, v, why = classify_measurement(row, feats[row['frame_id']], cal, cfg, gates=gates)
            preds.append(dict(frame_id=row['frame_id'], session_dir=row['session_dir'], pred=p, measurement=v, reason=why))
        so = V3.session_outcome(preds)
        res['e23']['on' if gates else 'off'] = dict(doors=len(so), doors_rejected=sum(v is None for v in so.values()),
                                                      frames=len(preds), frames_rejected=sum(r['pred'] is None for r in preds),
                                                      reasons=dict(Counter(r['reason'] for r in preds)))
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), **res)
    OUT.mkdir(parents=True, exist_ok=True); dump(OUT / 'hole_gate.json', out); print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- D: sub-pixel peaks
def peaks_refined(hm, count, thr, M, stride):
    """Same selection and suppression as door8.features.peaks; returns integer and quadratically refined points."""
    o = np.asarray(hm, np.float32); a = o.copy(); inv = cv2.invertAffineTransform(M); H, W = o.shape; integer = []; refined = []
    for _ in range(count):
        i = int(a.argmax()); y, x = np.unravel_index(i, a.shape); score = float(a[y, x])
        if score < thr:
            break
        dx = dy = 0.
        if 0 < x < W - 1:
            l, r = float(o[y, x - 1]), float(o[y, x + 1]); den = l - 2 * score + r
            if den < 0:
                dx = float(np.clip(.5 * (l - r) / den, -.5, .5))
        if 0 < y < H - 1:
            l, r = float(o[y - 1, x]), float(o[y + 1, x]); den = l - 2 * score + r
            if den < 0:
                dy = float(np.clip(.5 * (l - r) / den, -.5, .5))
        p = inv @ np.array([x * stride, y * stride, 1.]); q = inv @ np.array([(x + dx) * stride, (y + dy) * stride, 1.])
        integer.append([float(p[0]), float(p[1]), score]); refined.append([float(q[0]), float(q[1]), score])
        a[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = -np.inf
    return integer, refined


@torch.no_grad()
def subpixel_extract(fold, memory_fraction=.12):
    sp, frames, splits, cfg = ctx(); parts = partition(frames, splits, fold); rp, rm = load_run(hole_run(fold), sp); HP = V3.HP
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    net = HoleNet(pretrained=False).to(dev); net.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True)); net.eval()
    rows = parts['train'] + parts['test']; out = []
    for i, row in enumerate(rows):
        check_source(row); im = cv2.imread(row['rgb'])
        if im is None:
            raise ValueError('Unreadable RGB')
        small, M = letterbox(im, HP['width'], HP['height'])
        with torch.autocast(device_type=dev.type, enabled=dev.type == 'cuda'):                         # identical to the stored extraction
            hm = net(tensor_image(small)[None].to(dev))[0].float().cpu().numpy()
        det_i, det_r = {}, {}
        for c, (k, n) in enumerate([('bolt', 4), ('hinge', 1), ('latch', 1)]):
            det_i[k], det_r[k] = peaks_refined(hm[c], n, HP['peak_threshold'], M, HP['stride'])
        hinge = det_r['hinge'][0] if det_r['hinge'] else None; latch = det_r['latch'][0] if det_r['latch'] else None
        f = dict(frame_id=row['frame_id'], points=det_r, points_integer=det_i,
                 gate=geometry_gate(bolt_frame(det_r['bolt']), hinge, latch, im.shape), span_normalized=None)
        if hinge is not None and latch is not None:
            f['span_normalized'] = float(np.linalg.norm(np.asarray(hinge[:2]) - latch[:2]) / row['intrinsics']['fx'])
        out.append(f)
        if (i + 1) % 1000 == 0:
            print(f'subpixel f{fold} {i + 1}/{len(rows)}', flush=True)
    (CACHE / 'subpixel').mkdir(parents=True, exist_ok=True); jsonl(CACHE / 'subpixel' / f'f{fold}.jsonl', out)
    dump(CACHE / 'subpixel' / f'f{fold}.json', dict(fold=fold, run=hole_run(fold), weights_sha256=rm['weights_sha256'], n=len(out),
                                                    protocol_sha256=sha256(PROTOCOL), created_at=now()))
    print(f'subpixel f{fold} done {len(out)}', flush=True)


def residual_stats(preds, cad, band):
    meas = [r for r in preds if r['measurement'] is not None]
    rel = np.array([r['measurement'] / cad[r['class_name']] - 1 for r in meas])
    per = defaultdict(list)
    for r, v in zip(meas, rel):
        per[(r['class_name'], r['group_id'])].append(v)
    util = [(np.median(v) / band[c][1] if np.median(v) > 0 else np.median(v) / band[c][0]) for (c, _), v in per.items()]
    out_band = sum(not (band[r['class_name']][0] <= v <= band[r['class_name']][1]) for r, v in zip(meas, rel))
    return dict(frames_measured=len(meas), percentiles_5_50_95=[float(x) for x in np.percentile(rel, [5, 50, 95])],
                range=[float(rel.min()), float(rel.max())], rms=float(np.sqrt(np.mean(rel ** 2))), mean_abs=float(np.mean(np.abs(rel))),
                max_abs_mm=float(max(abs(r['measurement'] - cad[r['class_name']]) for r in meas)), frames_out_of_band=int(out_band),
                max_utilisation=float(max(util)), within_door_sd_median=float(np.median([np.std(v) for v in per.values()])),
                between_door_sd=float(np.std([np.median(v) for v in per.values()])))


def subpixel_eval():
    sp, frames, splits, cfg = ctx(); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); band = V5.bands(cad)
    lp, _, labels = load_field_labels(V3.LABELSET, sp); res = dict(folds={})
    allp = dict(integer=[], refined=[]); pts = dict(integer=[], refined=[]); mism = 0; total = 0
    for f in FOLDS:
        if read_json(CACHE / 'subpixel' / f'f{f}.json')['protocol_sha256'] != sha256(PROTOCOL):
            raise ValueError('Protocol changed')
        parts = partition(frames, splits, f); sub = {r['frame_id']: r for r in read_jsonl(CACHE / 'subpixel' / f'f{f}.jsonl')}
        base = V5.hole_pool(f)
        for fid, r in sub.items():
            total += 1
            same = all(sorted(p[:2] for p in r['points_integer'][k]) == sorted(p[:2] for p in base[fid]['points'][k]) for k in ('bolt', 'hinge', 'latch'))
            mism += not same
        scales = {}
        for name, pool in (('integer', base), ('refined', sub)):
            cal = fit_scale(parts['train'], pool, cad, 'pixel'); scales[name] = cal['camera_scales']; preds = []
            for row in parts['test']:
                p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg)
                preds.append({k: row[k] for k in ('frame_id', 'group_id', 'class_name')} | dict(pred=p, measurement=v, reason=why))
            if name == 'integer':
                stored = {r['frame_id']: r for r in read_jsonl(V5.test_prediction_file(f, 'pixel'))}
                if any(stored[r['frame_id']]['pred'] != r['pred'] for r in preds):
                    raise ValueError('Integer decisions differ from the stored predictions')
            allp[name] += preds
            roles = field_partitions(labels, frames, splits, f)
            pts[name] += [p for p in localization(roles['test'], pool, role='test')[1]]
        res['folds'][f] = dict(scale=scales)
    for name in ('integer', 'refined'):
        e = [p['error_px'] for p in pts[name]]; m = [x for x in e if x is not None]
        loc = dict(points=len(e), matched=len(m), pck8=sum(x <= 8 for x in m) / len(e), median_px=float(np.median(m)), p95_px=float(np.quantile(m, .95)),
                   mean_px=float(np.mean(m)))
        res[name] = dict(frames=frame_counts(allp[name]), doors=door_counts(allp[name]), residual=residual_stats(allp[name], cad, band),
                         localization_test_role=loc)
    res.update(created_at=now(), protocol_sha256=sha256(PROTOCOL), integer_peak_mismatch_frames=int(mism), frames_compared=int(total))
    OUT.mkdir(parents=True, exist_ok=True); dump(OUT / 'subpixel.json', res); print(json.dumps({k: res[k] for k in ('integer', 'refined', 'integer_peak_mismatch_frames')}, indent=1))


# ---------------------------------------------------------------- E: frames per decision
def frames_per_decision():
    order = lambda r: int(re.search(r'_(\d+)\.png$', r['frame_id']).group(1))
    src = dict(hole=[V5.test_prediction_file(f, 'pixel') for f in FOLDS],
               rgb=[ROOT / f'reports/rgb_f{f}_s42_test/predictions.jsonl' for f in FOLDS],
               rgbe=[ROOT / f'reports/rgbe_f{f}_s42_test/predictions.jsonl' for f in FOLDS])
    res = {}
    for name, files in src.items():
        per = defaultdict(list)
        for p in files:
            for r in read_jsonl(p):
                per[r['group_id']].append(r)
        res[name] = {}
        for n in (1, 3, 5, 10, 20):
            rows = [r for rr in per.values() for r in sorted(rr, key=order)[:n]]
            res[name][str(n)] = dict(doors=door_counts(rows), frames=frame_counts(rows))
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), **res)
    OUT.mkdir(parents=True, exist_ok=True); dump(OUT / 'frames_per_decision.json', out)
    print(json.dumps({k: {n: v['doors'] for n, v in d.items()} for k, d in res.items()}, indent=1))


# ---------------------------------------------------------------- F: timing
@torch.no_grad()
def timing():
    sp, frames, splits, cfg = ctx(); parts = partition(frames, splits, '0'); HP = V3.HP
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); cal = fit_scale(parts['train'], V5.hole_pool('0'), cad, 'pixel')
    dev = torch.device('cuda'); rows = parts['test'][::max(1, len(parts['test']) // 220)][:220]
    rp, _ = load_run(hole_run('0'), sp); hole = HoleNet(pretrained=False).to(dev)
    hole.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True)); hole.eval()
    cp, cm = load_run('rgb_f0_s42', sp); cnn = Classifier('rgb', len(cfg['classes']), pretrained=False).to(dev)
    cnn.load_state_dict(torch.load(cp / 'model.pth', map_location=dev, weights_only=True)); cnn.eval()
    t = dict(decode=[], hole=[], cnn=[])
    for i, row in enumerate(rows):
        t0 = time.perf_counter(); im = cv2.imread(row['rgb']); t1 = time.perf_counter()
        small, M = letterbox(im, HP['width'], HP['height'])
        with torch.autocast(device_type='cuda'):
            hm = hole(tensor_image(small)[None].to(dev))[0].float().cpu().numpy()
        det = {k: V3.peaks(hm[c], n, HP['peak_threshold'], M, HP['stride']) for c, (k, n) in enumerate([('bolt', 4), ('hinge', 1), ('latch', 1)])}
        hinge = det['hinge'][0] if det['hinge'] else None; latch = det['latch'][0] if det['latch'] else None
        f = dict(gate=geometry_gate(bolt_frame(det['bolt']), hinge, latch, im.shape), span_normalized=None)
        if hinge is not None and latch is not None:
            f['span_normalized'] = float(np.linalg.norm(np.asarray(hinge[:2]) - latch[:2]) / row['intrinsics']['fx'])
        classify_measurement(row, f, cal, cfg); torch.cuda.synchronize(); t2 = time.perf_counter()
        sq, _ = letterbox(im, cm['image_size'], cm['image_size'])
        with torch.autocast(device_type='cuda'):
            cnn(tensor_image(sq)[None].to(dev)).float().softmax(1).cpu()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        if i >= 20:
            t['decode'].append(t1 - t0); t['hole'].append(t2 - t1); t['cnn'].append(t3 - t2)
    st = lambda v: dict(n=len(v), mean_ms=float(np.mean(v) * 1e3), median_ms=float(np.median(v) * 1e3), p95_ms=float(np.quantile(v, .95) * 1e3))
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               image='1920x1200 PNG', decode=st(t['decode']), hole_pipeline=st(t['hole']), cnn_forward=st(t['cnn']),
               note='batch 1, mixed precision as in the stored evaluations; decode = cv2.imread of the RGB PNG; edge device not measured')
    OUT.mkdir(parents=True, exist_ok=True); dump(OUT / 'timing.json', out); print(json.dumps(out, indent=1))


def collect():
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL))
    for k in ('cnn_scores', 'hole_register', 'hole_gate', 'subpixel', 'frames_per_decision', 'timing'):
        p = OUT / f'{k}.json'; out[k] = read_json(p) if p.exists() else None
    out['missing'] = [k for k, v in out.items() if v is None]
    dump(OUT / 'collected.json', out); print('missing', out['missing'])


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('cnn-infer'); a.add_argument('--run', required=True)
    a = sub.add_parser('subpixel-extract'); a.add_argument('--fold', required=True)
    for name in ('cnn-analyze', 'hole-register', 'hole-gate', 'subpixel-eval', 'frames', 'timing', 'collect', 'list-jobs'):
        sub.add_parser(name)
    args = ap.parse_args()
    if args.cmd == 'cnn-infer':
        cnn_infer(args.run)
    elif args.cmd == 'subpixel-extract':
        subpixel_extract(args.fold)
    elif args.cmd == 'list-jobs':
        for r in cnn_runs():
            print('cnn-infer --run ' + r['run'])
        for f in FOLDS:
            print('subpixel-extract --fold ' + f)
    else:
        {'cnn-analyze': cnn_analyze, 'hole-register': hole_register, 'hole-gate': hole_gate, 'subpixel-eval': subpixel_eval,
         'frames': frames_per_decision, 'timing': timing, 'collect': collect}[args.cmd]()


if __name__ == '__main__':
    main()
