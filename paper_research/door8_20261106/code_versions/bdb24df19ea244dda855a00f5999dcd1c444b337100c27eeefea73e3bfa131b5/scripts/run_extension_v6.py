"""Experiments registered in config/extension_v6_protocol.json (equal-exposure unregistered-class comparison).

  seen                          both methods saw class c; no training: hole S fixed + CAD minus c, CNN 8-class output masked
  train-strict --exclude CLASS  fold-0 hole detector trained without any class-c image (LAB and field)
  eval-strict  --exclude CLASS  fold-0 TRAIN+TEST features, S refit without c, CAD minus c, TEST-role localization
  collect                       reports/extension_v6/collected.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from door8.core import setup_runtime
setup_runtime()

import numpy as np
import torch
from torch import nn

from door8.core import ROOT, capture_code, dump, jsonl, now, partition, read_json, read_jsonl, require_fresh_dir, sha256, snapshot
from door8.field_data import field_partitions, load_field_labels
from door8.field_experiments import localization, validation_loss
from door8.geometry import classify_measurement, fit_scale
from door8.learning import HoleData, HoleNet, heatmap_loss, load_run, make_loader, pretrained_resnet, seed_all
import run_extension_v3 as V3
import run_extension_v5 as V5

PROTOCOL = ROOT / 'config/extension_v6_protocol.json'
OUT = ROOT / 'reports/extension_v6'
BASE_RUN = 'holes_field8_f0_s42_v2'
FOLDS = ('0', '1', '2', '3', '4')


def strict_run(c):
    return f'holes_strict_loco_{c}_f0_s42'


def hole_preds(parts, pool, cal, g, c, fold):
    cad7 = dict(zip(g['classes'], g['cad_hole_distance_mm'])); preds = []
    for row in parts['test']:
        p, v, why = classify_measurement(row, pool[row['frame_id']], cal, g)
        s = -min(abs(v - x) for x in cad7.values()) if v is not None else float('-inf')
        preds.append({k: row[k] for k in ('frame_id', 'group_id', 'class_name')} | dict(pred=p, measurement=v, reason=why, score=s,
                     fold=fold, excluded=c))
    return preds


def hole_extra(preds, c):
    unk = [r for r in preds if r['class_name'] == c]
    meas = [r['measurement'] for r in unk if r['measurement'] is not None]
    return dict(unknown_measured_D_median=float(np.median(meas)) if meas else None,
                unknown_min_distance_mm=float(-max(r['score'] for r in unk)) if unk and max(r['score'] for r in unk) > float('-inf') else None)


def clean(rows):
    return [dict(r, score=None if r['score'] == float('-inf') else r['score']) for r in rows]   # None = gate failed


# ---------------------------------------------------------------- seen condition (no training)
def masked(rows, classes, c):
    ci = classes.index(c); out = []
    for r in rows:
        p = np.asarray(r['probabilities'], float); rest = p.sum() - p[ci]
        if rest <= 0:
            raise ValueError('Masked probability mass underflowed')
        q = p / rest; q[ci] = -1.0; j = int(q.argmax())
        out.append(dict(frame_id=r['frame_id'], group_id=r['group_id'], class_name=r['class_name'], pred=classes[j], confidence=float(q[j])))
    return out


def seen():
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), hole={}, cnn={'rgb': {}, 'rgbe': {}})
    OUT.mkdir(parents=True, exist_ok=True); hp_all = []; cnn_all = []
    for k in FOLDS:
        parts = partition(frames, splits, k); pool = V5.hole_pool(k); test_ids = {r['frame_id'] for r in parts['test']}
        cal = fit_scale(parts['train'], pool, cad, 'pixel')            # S fixed: 8 classes, as in the main evaluation
        for c in cfg['classes']:
            g = V5.cfg_without(cfg, c); preds = hole_preds(parts, pool, cal, g, c, k)
            m = V5.loco_metrics(preds, preds, c, score=lambda r: r['score'])
            m.update(scale=cal['camera_scales'], **hole_extra(preds, c)); res['hole'][f'{k}:{c}'] = m; hp_all += preds
        for kind in ('rgb', 'rgbe'):
            rm = read_json(ROOT / f'runs/{kind}_f{k}_s42/run.json'); classes = rm['classes']
            test = read_jsonl(ROOT / f'reports/{kind}_f{k}_s42_test/predictions.jsonl')
            val = read_jsonl(ROOT / f'runs/{kind}_f{k}_s42/val_predictions.jsonl')
            if {r['frame_id'] for r in test} != test_ids or {r['frame_id'] for r in val} != {r['frame_id'] for r in parts['val']}:
                raise ValueError(f'stored predictions do not match fold {k}')
            for c in cfg['classes']:
                mv = sorted(r['confidence'] for r in masked([r for r in val if r['class_name'] != c], classes, c))
                thr = mv[int(.05 * len(mv))]
                closed = masked(test, classes, c)
                opened = [dict(r, pred=r['pred'] if r['confidence'] >= thr else None) for r in closed]
                m = V5.loco_metrics(closed, opened, c, score=lambda r: r['confidence'])
                m.update(threshold=thr, val_acceptance=float(np.mean([x >= thr for x in mv])))
                res['cnn'][kind][f'{k}:{c}'] = m
                cnn_all += [dict(r, kind=kind, fold=k, excluded=c, accepted=r['confidence'] >= thr) for r in closed]
    jsonl(OUT / 'seen_hole_predictions.jsonl', clean(hp_all)); jsonl(OUT / 'seen_cnn_predictions.jsonl', cnn_all)
    res['pooled'] = {}
    for name, runs in (('hole', res['hole']), ('rgb', res['cnn']['rgb']), ('rgbe', res['cnn']['rgbe'])):
        res['pooled'][name] = dict(fold0=V5.pool_runs({kk: v for kk, v in runs.items() if kk.startswith('0:')}), oof=V5.pool_runs(runs))
    dump(OUT / 'seen.json', res)
    print(json.dumps(res['pooled'], indent=1))


# ---------------------------------------------------------------- unseen condition: strict hole detector
def strict_inputs(c):
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    if c not in cfg['classes']:
        raise ValueError(c)
    base = read_json(ROOT / f'runs/{BASE_RUN}/run.json')
    lp, _, labels = load_field_labels(base['labelset'], sp)
    if sha256(lp / 'COMPLETE.json') != base['labelset_sha256']:
        raise ValueError('Field labels changed')
    field = {r['annotation_id']: r for r in labels}; lab = {r['annotation_id']: r for r in read_jsonl(sp / 'annotations_lab.jsonl')}
    if set(field) & set(lab):
        raise ValueError('Annotation id collision')
    get = lambda i: field[i] if i in field else lab[i]
    tr = [get(i) for i in base['train_annotation_ids']]; va = [field[i] for i in base['val_annotation_ids']]
    lv = [lab[i] for i in base['lab_monitor_annotation_ids']]
    if len(tr) != 102 or len(va) != 8 or len(lv) != 24:
        raise ValueError('Base run inputs not as expected')
    roles = field_partitions(labels, frames, splits, '0')
    if {r['annotation_id'] for r in tr if r['annotation_id'] in field} - {r['annotation_id'] for r in roles['train']} \
            or {r['annotation_id'] for r in va} != {r['annotation_id'] for r in roles['val']}:
        raise ValueError('Field supervision crossed fold roles')
    keep = lambda rr: [r for r in rr if r['class_name'] != c]
    tr, va, lv = keep(tr), keep(va), keep(lv)
    for r in tr + va + lv:
        if sha256(r['rgb']) != r['rgb_sha256']:
            raise ValueError('Annotation image changed')
    return sp, lp, base, tr, va, lv


def train_strict(c, seed=42, workers=2, memory_fraction=.12):
    sp, lp, base, tr, va, lv = strict_inputs(c); run = strict_run(c); HP = base['hyperparameters']
    dst = require_fresh_dir('runs/' + run); seed_all(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    nf = sum(r['annotation_id'].startswith('field_') for r in tr)
    meta = dict(model_kind='holes', training_regime='human_manual_lab_plus_field_strict_loco', snapshot='v1',
                snapshot_sha256=sha256(sp / 'COMPLETE.json'), labelset=base['labelset'], labelset_sha256=sha256(lp / 'COMPLETE.json'),
                extension_protocol_sha256=sha256(PROTOCOL), base_run=BASE_RUN, excluded_class=c, fold='0', seed=seed,
                field_train_images=nf, lab_train_images=len(tr) - nf, field_val_images=len(va), lab_monitor_images=len(lv),
                train_annotation_ids=[r['annotation_id'] for r in tr], val_annotation_ids=[r['annotation_id'] for r in va],
                lab_monitor_annotation_ids=[r['annotation_id'] for r in lv],
                train_classes=dict(Counter(r['class_name'] for r in tr)), val_classes=dict(Counter(r['class_name'] for r in va)),
                initialization='ImageNet1K_V1', initialization_sha256=sha256(pretrained_resnet()), pilot=False,
                requested_epochs=HP['epochs'], hyperparameters=HP, checkpoint_selection='minimum_field_validation_heatmap_loss',
                status='running', started_at=now())
    if c in meta['train_classes'] or c in meta['val_classes'] or any(r['class_name'] == c for r in lv):
        raise ValueError('Excluded class leaked into training inputs')
    meta.update(capture_code()); dump(dst / 'run.json', meta)
    t0 = time.perf_counter()
    net = HoleNet().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=HP['learning_rate'])
    dl = make_loader(HoleData(tr, HP, True), HP['batch_size'], True, workers, seed)
    vl = make_loader(HoleData(va, HP, False), HP['batch_size'], False, workers, seed)
    ll = make_loader(HoleData(lv, HP, False), HP['batch_size'], False, workers, seed)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, HP['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda'); best = float('inf'); hist = []
    for ep in range(HP['epochs']):
        net.train(); losses = []; te = time.perf_counter()
        for x, y, m in dl:
            x, y, m = x.to(dev), y.to(dev), m.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == 'cuda'):
                loss = heatmap_loss(net(x), y, m)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        fl = validation_loss(net, vl, dev); lab_l = validation_loss(net, ll, dev)
        hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(losses)), field_val_loss=fl, lab_val_loss_monitor_only=lab_l,
                         seconds=time.perf_counter() - te))
        if fl < best:
            best = fl; meta.update(best_epoch=ep + 1, best_field_val_loss=fl); torch.save(net.state_dict(), dst / 'model.pth')
        sched.step(); dump(dst / 'history.json', hist); dump(dst / 'run.json', meta)
        print(json.dumps(dict(run=run, **hist[-1])), flush=True)
    meta.update(status='complete', finished_at=now(), seconds=time.perf_counter() - t0, weights_sha256=sha256(dst / 'model.pth'))
    dump(dst / 'run.json', meta)


def eval_strict(c, memory_fraction=.12):
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); parts = partition(frames, splits, '0')
    run = strict_run(c); rp, rm = load_run(run, sp)
    if rm['extension_protocol_sha256'] != sha256(PROTOCOL) or rm['excluded_class'] != c:
        raise ValueError('Run/protocol mismatch')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    net = HoleNet(pretrained=False).to(dev)
    net.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True)); net.eval()
    fdir = require_fresh_dir(f'features/{run}_trainTest')
    pool = V3.extract(net, parts['train'] + parts['test'], dev)
    jsonl(fdir / 'rows.jsonl', list(pool.values()))
    dump(fdir / 'features.json', dict(status='complete', run=run, fold='0', subset='train_test', n=len(pool),
                                      rows_sha256=sha256(fdir / 'rows.jsonl'), weights_sha256=rm['weights_sha256'], created_at=now()))
    g = V5.cfg_without(cfg, c); cad7 = dict(zip(g['classes'], g['cad_hole_distance_mm']))
    cal = fit_scale([r for r in parts['train'] if r['class_name'] != c], pool, cad7, 'pixel')
    preds = hole_preds(parts, pool, cal, g, c, '0')
    m = V5.loco_metrics(preds, preds, c, score=lambda r: r['score'])
    _, _, labels = load_field_labels(rm['labelset'], sp); roles = field_partitions(labels, frames, splits, '0')
    loc, points = localization(roles['test'], pool, role='test')
    cid = {r['frame_id'] for r in roles['test'] if r['class_name'] == c}
    m.update(run=run, excluded_class=c, scale=cal['camera_scales'], best_epoch=rm.get('best_epoch'), weights_sha256=rm['weights_sha256'],
             test_role_localization=loc, excluded_class_test_role_points=[p for p in points if p['frame_id'] in cid], **hole_extra(preds, c))
    OUT.mkdir(parents=True, exist_ok=True)
    jsonl(OUT / f'{run}_predictions.jsonl', clean(preds)); dump(OUT / f'{run}_metrics.json', m)
    print(json.dumps({k: m[k] for k in ('unknown_doors', 'known_doors_threshold', 'auroc_frame', 'unknown_min_distance_mm')}), flush=True)


# ---------------------------------------------------------------- collect
def collect():
    sp, _, _, _ = snapshot(); classes = read_json(sp / 'protocol.json')['classes']
    s = read_json(OUT / 'seen.json'); v5 = read_json(ROOT / 'reports/extension_v5/collected.json')
    strict = {c: read_json(OUT / f'{strict_run(c)}_metrics.json') for c in classes if (OUT / f'{strict_run(c)}_metrics.json').exists()}
    v5cnn = {kind: {c: read_json(ROOT / f'reports/extension_v5/{kind}_loco_{c}_f0_s42_metrics.json') for c in classes} for kind in ('rgb', 'rgbe')}

    def row(m, extra=()):
        return dict(unknown_doors=[m['unknown_doors']['rejected'], m['unknown_doors']['n']], accepted_as=m['unknown_doors']['accepted_as'],
                    known=m['known_doors_threshold'], auroc=m['auroc_frame'], **{k: m.get(k) for k in extra})
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), missing_strict=[c for c in classes if c not in strict],
               seen=dict(per_class={c: dict(hole=row(s['hole'][f'0:{c}'], ('unknown_min_distance_mm',)),
                                            rgb=row(s['cnn']['rgb'][f'0:{c}'], ('threshold',)), rgbe=row(s['cnn']['rgbe'][f'0:{c}'], ('threshold',)))
                                    for c in classes}, pooled=s['pooled']),
               unseen=dict(per_class={c: dict(hole=row(strict[c], ('unknown_min_distance_mm', 'best_epoch')) if c in strict else None,
                                              rgb=row(v5cnn['rgb'][c]), rgbe=row(v5cnn['rgbe'][c])) for c in classes},
                           pooled=dict(hole=V5.pool_runs(strict) if strict else None, rgb=v5['cnn']['rgb']['pooled'], rgbe=v5['cnn']['rgbe']['pooled'])),
               unequal_v5=dict(hole=v5['hole'], rgb=v5['cnn']['rgb']['pooled'], rgbe=v5['cnn']['rgbe']['pooled']),
               strict_localization={c: dict(all=strict[c]['test_role_localization']['all'], per_channel=strict[c]['test_role_localization']['per_channel'],
                                            excluded_class_points=strict[c]['excluded_class_test_role_points']) for c in strict})
    dump(OUT / 'collected.json', out)
    print(json.dumps(dict(seen=out['seen']['pooled'], unseen=out['unseen']['pooled'], missing=out['missing_strict']), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('seen'); sub.add_parser('collect')
    a = sub.add_parser('train-strict'); a.add_argument('--exclude', required=True)
    a = sub.add_parser('eval-strict'); a.add_argument('--exclude', required=True)
    x = p.parse_args()
    {'seen': lambda: seen(), 'collect': lambda: collect(),
     'train-strict': lambda: train_strict(x.exclude), 'eval-strict': lambda: eval_strict(x.exclude)}[x.cmd]()


if __name__ == '__main__':
    main()
