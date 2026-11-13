"""Extension v16 (review round 5): unseen-class condition and CAD registration on folds 1-4, registered in
config/extension_v16_protocol.json.

The fold-0 functions of extension_v5 / v6 / v12 are reused. Their source is patched in memory: the literal fold '0'
becomes the module variable _FOLD, and the line that reads the LAB monitor ids falls back to the LAB split (folds 1-4
base runs do not store them). The differences are written to reports/extension_v16/code_patches.diff.

  check                                    patched code reproduces the stored fold-0 inputs and TEST predictions
  train-hole --exclude C --fold K          strict hole detector
  eval-hole  --exclude C --fold K          rejection with CAD minus c (TRAIN + TEST features)
  train-cnn  --kind K --exclude C --fold K 7-output CNN
  eval-cnn   --kind K --exclude C --fold K legacy evaluation + float32 forward for the four scores
  val-hole   --exclude C --fold K [--seed S]  validation features and predictions of a strict detector (C2)
  analyze    --fold K                      per-fold summary (CNN scores, hole rejection, CAD registration)
  list-jobs                                job chains for the queue
"""
import argparse
import difflib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from door8.core import setup_runtime
setup_runtime()

import torch

from door8.core import ROOT, dump, jsonl, now, partition, read_json, read_jsonl, require_fresh_dir, sha256, snapshot
from door8.geometry import classify_measurement, fit_scale
from door8.learning import HoleNet, load_run
import run_extension_v3 as V3
import run_extension_v5 as V5
import run_extension_v6 as V6
import run_extension_v12 as V12

PROTOCOL = ROOT / 'config/extension_v16_protocol.json'
OUT = ROOT / 'reports/extension_v16'
CACHE = ROOT / 'cache/extension_v16'
NEW_FOLDS = ('1', '2', '3', '4')
KINDS = V12.KINDS
SCORES = V12.SCORES
PATCHES = [(V6, 'strict_inputs', [("'0'", '_FOLD', 1),
                                  ("base['lab_monitor_annotation_ids']", "(base.get('lab_monitor_annotation_ids') or _LAB_VAL)", 1)]),
           (V6, 'train_strict', [("'0'", '_FOLD', 1)]),
           (V6, 'eval_strict', [("'0'", '_FOLD', 4)]),
           (V5, 'train_cnn_loco', [("'0'", '_FOLD', 2)]),
           (V5, 'eval_cnn_loco', [("'0'", '_FOLD', 1)]),
           (V12, 'hole_register', [("'0'", '_FOLD', 1)])]


def patch_sources():
    """Rebind the six functions to fold-parametrised copies of their own source; return the unified diff."""
    diff = []
    for module, name, reps in PATCHES:
        src = inspect.getsource(getattr(module, name)); new = src
        for old, rep, n in reps:
            if new.count(old) != n:
                raise ValueError(f'{module.__name__}.{name}: expected {n} x {old}')
            new = new.replace(old, rep)
        exec(compile(new, f'<{module.__name__}.{name} fold-parametrised>', 'exec'), module.__dict__)
        diff += list(difflib.unified_diff(src.splitlines(), new.splitlines(), f'{module.__name__}.{name}', f'{module.__name__}.{name} (v16)', lineterm=''))
    sp, _, _, _ = snapshot(); V6._LAB_VAL = read_json(sp / 'annotations_lab_split.json')['val']
    return '\n'.join(diff) + '\n'


DIFF = patch_sources()


def classes8():
    sp, _, _, _ = snapshot()
    return read_json(sp / 'protocol.json')['classes']


def hole_name(c, fold, seed=42):
    return f'holes_strict_loco_{c}_f{fold}_s{seed}'


def cnn_name(kind, c, fold):
    return f'{kind}_loco_{c}_f{fold}_s42'


def bind(fold, out=None):
    """Point the reused modules at this protocol, this fold's run names and this fold's output directories."""
    out = out or OUT / f'f{fold}'; out.mkdir(parents=True, exist_ok=True)
    V5._FOLD = V6._FOLD = V12._FOLD = str(fold)
    V5.PROTOCOL = V6.PROTOCOL = V12.PROTOCOL = PROTOCOL
    V5.OUT = V6.OUT = V12.OUT = out; V12.CACHE = CACHE / f'f{fold}'
    V6.BASE_RUN = V12.hole_run(fold); V6.strict_run = lambda c: hole_name(c, fold)
    V12.cnn_runs = lambda: [dict(run=cnn_name(kind, c, fold), group='unseen', kind=kind, fold=str(fold), exclude=c) for kind in KINDS for c in classes8()]
    V12.stored_test_file = lambda run, group: out / f'{run}_predictions.jsonl'
    return out


def check():
    """Fold 0: the patched input builder and decision path must reproduce what is stored for the eight strict runs."""
    bind('0', out=ROOT / 'cache/_v16_check'); sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    parts = partition(frames, splits, '0'); res = {}
    for c in cfg['classes']:
        run = hole_name(c, '0'); rm = read_json(ROOT / 'runs' / run / 'run.json')
        _, _, base, tr, va, lv = V6.strict_inputs(c)
        same_inputs = ([r['annotation_id'] for r in tr] == rm['train_annotation_ids'] and [r['annotation_id'] for r in va] == rm['val_annotation_ids']
                       and [r['annotation_id'] for r in lv] == rm['lab_monitor_annotation_ids'])
        pool = {r['frame_id']: r for r in read_jsonl(ROOT / f'features/{run}_trainTest/rows.jsonl')}
        g = V5.cfg_without(cfg, c); cad7 = dict(zip(g['classes'], g['cad_hole_distance_mm']))
        cal = fit_scale([r for r in parts['train'] if r['class_name'] != c], pool, cad7, 'pixel')
        preds = V6.clean(V6.hole_preds(parts, pool, cal, g, c, V6._FOLD))
        stored = read_jsonl(ROOT / f'reports/extension_v6/{run}_predictions.jsonl')
        same_preds = [(r['frame_id'], r['pred'], r['measurement']) for r in preds] == [(r['frame_id'], r['pred'], r['measurement']) for r in stored]
        res[c] = dict(inputs=same_inputs, predictions=same_preds, n=len(preds))
    ok = all(v['inputs'] and v['predictions'] for v in res.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'code_patches.diff').write_text(DIFF, encoding='utf-8')
    dump(OUT / 'check_fold0.json', dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), ok=ok, per_class=res))
    print(json.dumps(dict(ok=ok, per_class=res), indent=1))
    if not ok:
        raise SystemExit('fold-0 reproduction failed')


def val_hole(c, fold, seed=42, memory_fraction=.12):
    """C2: validation features and predictions of a strict detector (inference only)."""
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); parts = partition(frames, splits, str(fold))
    run = hole_name(c, fold, seed); rp, rm = load_run(run, sp)
    if rm['excluded_class'] != c or str(rm['fold']) != str(fold):
        raise ValueError('Run mismatch')
    out = OUT / (f'f{fold}' if seed == 42 and str(fold) != '0' else f'f{fold}_s{seed}'); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    net = HoleNet(pretrained=False).to(dev); net.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True)); net.eval()
    val = [r for r in parts['val'] if r['class_name'] != c]
    fdir = require_fresh_dir(f'features/{run}_val'); vpool = V3.extract(net, val, dev)
    jsonl(fdir / 'rows.jsonl', list(vpool.values()))
    dump(fdir / 'features.json', dict(status='complete', run=run, fold=str(fold), subset='val_without_excluded_class', n=len(vpool),
                                      rows_sha256=sha256(fdir / 'rows.jsonl'), weights_sha256=rm['weights_sha256'], created_at=now()))
    pool = {r['frame_id']: r for r in read_jsonl(ROOT / f'features/{run}_trainTest/rows.jsonl')}
    g = V5.cfg_without(cfg, c); cad7 = dict(zip(g['classes'], g['cad_hole_distance_mm']))
    cal = fit_scale([r for r in parts['train'] if r['class_name'] != c], pool, cad7, 'pixel'); preds = []
    for row in val:
        p, v, why = classify_measurement(row, vpool[row['frame_id']], cal, g)
        preds.append({k: row[k] for k in ('frame_id', 'group_id', 'class_name')}
                     | dict(pred=p, measurement=v, reason=why, nearest_mm=None if v is None else min(abs(v - x) for x in cad7.values()), fold=str(fold), excluded=c, seed=seed))
    jsonl(out / f'{run}_val_predictions.jsonl', preds)
    print(json.dumps(dict(run=run, val_images=len(preds), measured=sum(r['measurement'] is not None for r in preds))), flush=True)


def analyze(fold):
    out = bind(fold); sp, frames, splits, cfg = V12.ctx(); cl8 = cfg['classes']; info = {r['frame_id']: r for r in frames}
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), fold=str(fold), seed=42, cnn={}, cnn_legacy={}, consistency={}); dumped = []
    for kind in KINDS:                                              # same computation as the unseen block of V12.cnn_analyze
        runs = {k: {} for k in SCORES}; legacy = {}
        for c in cl8:
            run = cnn_name(kind, c, fold); meta, a = V12.load_cnn(run); cl = meta['classes']; allowed = list(range(len(cl)))
            res['consistency'][run] = meta['consistency']
            va = V12.scored_rows(a, 'val', cl, allowed, info); te = V12.scored_rows(a, 'test', cl, allowed, info)
            for k in SCORES:
                t, acc = V12.threshold(va[k]); m = V5.loco_metrics(te[k], V12.apply(te[k], t), c, score=lambda r: r['score'])
                m.update(threshold=t, val_acceptance=acc); runs[k][c] = m
                dumped += [dict(r, kind=kind, excluded=c, score_name=k, accepted=r['score'] >= t, condition='unseen', fold=str(fold), seed=42, split='test') for r in te[k]]
                dumped += [dict(r, kind=kind, excluded=c, score_name=k, condition='unseen', fold=str(fold), seed=42, split='val') for r in va[k]]
            legacy[c] = read_json(out / f'{run}_metrics.json')
        res['cnn'][kind] = {k: dict(pooled=V5.pool_runs(runs[k]),
                                    per_class={c: dict(unknown_doors=[m['unknown_doors']['rejected'], m['unknown_doors']['n']],
                                                       accepted_as=m['unknown_doors']['accepted_as'], known=m['known_doors_threshold'],
                                                       auroc=m['auroc_frame'], threshold=m['threshold'], val_acceptance=m['val_acceptance']) for c, m in runs[k].items()})
                            for k in SCORES}
        res['cnn_legacy'][kind] = dict(pooled=V5.pool_runs(legacy))
    strict = {c: read_json(out / f'{hole_name(c, fold)}_metrics.json') for c in cl8}
    res['hole_reject'] = dict(pooled=V5.pool_runs(strict),
                              per_class={c: dict(unknown_doors=[m['unknown_doors']['rejected'], m['unknown_doors']['n']],
                                                 accepted_as=m['unknown_doors']['accepted_as'], known=m['known_doors_threshold'],
                                                 auroc=m['auroc_frame'], min_distance_mm=m['unknown_min_distance_mm'], best_epoch=m['best_epoch'],
                                                 scale=m['scale'], localization=m['test_role_localization']['all'],
                                                 excluded_class_points=m['excluded_class_test_role_points']) for c, m in strict.items()})
    V12.hole_register()                                             # writes out/hole_register.json with this fold's strict detectors
    res['hole_register'] = read_json(out / 'hole_register.json')
    dump(out / 'analysis.json', res); jsonl(out / 'cnn_unseen_scores.jsonl', dumped)
    print(json.dumps(dict(fold=fold, hole_reject=res['hole_reject']['pooled'], hole_register=res['hole_register']['pooled'],
                          cnn={kind: {k: res['cnn'][kind][k]['pooled'] for k in SCORES} for kind in KINDS}), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('train-hole', 'eval-hole', 'val-hole'):
        a = sub.add_parser(name); a.add_argument('--exclude', required=True); a.add_argument('--fold', required=True)
        if name == 'val-hole':
            a.add_argument('--seed', type=int, default=42)
    for name in ('train-cnn', 'eval-cnn'):
        a = sub.add_parser(name); a.add_argument('--kind', choices=KINDS, required=True); a.add_argument('--exclude', required=True); a.add_argument('--fold', required=True)
    a = sub.add_parser('analyze'); a.add_argument('--fold', required=True)
    sub.add_parser('check'); sub.add_parser('list-jobs')
    x = ap.parse_args()
    if x.cmd == 'check':
        return check()
    if x.cmd == 'list-jobs':
        for kind in ('hole',) + KINDS:                              # hole jobs first: they carry the main result
            for k in NEW_FOLDS:
                for c in classes8():
                    print(f'{kind} {c} {k}')
        return
    if x.cmd == 'val-hole':
        return val_hole(x.exclude, x.fold, x.seed)
    if x.fold not in NEW_FOLDS:
        raise ValueError('Only folds 1-4 are trained in this extension')
    bind(x.fold)
    if x.cmd == 'train-hole':
        V6.train_strict(x.exclude, seed=42)
    elif x.cmd == 'eval-hole':
        V6.eval_strict(x.exclude)
    elif x.cmd == 'train-cnn':
        V5.train_cnn_loco(x.kind, x.exclude, cnn_name(x.kind, x.exclude, x.fold), seed=42, workers=4)
    elif x.cmd == 'eval-cnn':
        run = cnn_name(x.kind, x.exclude, x.fold); V5.eval_cnn_loco(run); V12.cnn_infer(run)
    else:
        analyze(x.fold)


if __name__ == '__main__':
    main()
