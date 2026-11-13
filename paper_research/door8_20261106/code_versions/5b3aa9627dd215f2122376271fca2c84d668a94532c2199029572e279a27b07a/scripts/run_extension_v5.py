"""Experiments registered in config/extension_v5_protocol.json.

  train-cnn-loco --kind {rgb|rgbe} --exclude CLASS --run NAME   (fold 0, 7 classes, v1 CNN recipe)
  eval-cnn-loco  --run NAME                                     (fold-0 TEST incl. the excluded class)
  hole-loco                                                     (no retraining; CAD list minus c, S refit without c)
  margin                                                        (TRAIN-session decisions predict TEST reliability?)
  collect                                                       (reports/extension_v5/collected.json + 결과.md)
"""
import argparse
import collections
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import setup_runtime
setup_runtime()

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from door8.core import (ROOT, capture_code, dump, jsonl, now, outpath, partition, read_json, read_jsonl,
                        require_fresh_dir, sha256, snapshot)
from door8.geometry import classify_measurement, fit_scale
from door8.learning import CNNData, Classifier, cnn_predict, make_loader, pretrained_resnet, seed_all
from door8.metrics import summarize, wilson

PROTOCOL = ROOT / 'config/extension_v5_protocol.json'
OUT = ROOT / 'reports/extension_v5'
F0 = ROOT / 'reports/fold0_test_v3'
EXT3 = ROOT / 'reports/extension_v3'


# ---------------------------------------------------------------- helpers
def vote(rows):
    """Session decision: majority of judged frames; tie or nothing judged = None."""
    per = defaultdict(list)
    for r in rows:
        per[r['group_id']].append(r)
    out = {}
    for g, rr in per.items():
        c = Counter(r['pred'] for r in rr if r['pred'] is not None).most_common(2)
        pred = None if not c or (len(c) > 1 and c[0][1] == c[1][1]) else c[0][0]
        out[g] = dict(true=rr[0]['class_name'], pred=pred, frames=len(rr))
    return out


def auroc(known, unknown):
    """P(score_known > score_unknown) + 0.5 P(tie); -inf allowed."""
    k = np.asarray(known, float); u = np.asarray(unknown, float)
    if not len(k) or not len(u):
        return None
    allv = np.concatenate([k, u]); order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv)); sv = allv[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    rk = ranks[:len(k)].sum()
    return float((rk - len(k) * (len(k) + 1) / 2) / (len(k) * len(u)))


def bands(cad):
    """Relative correct-decision band (lo<0<hi) per class: neighbour midpoint and 30 mm reject, inner one."""
    names = sorted(cad, key=cad.get); out = {}
    for i, c in enumerate(names):
        d = cad[c]
        lo = d - 30 if i == 0 else max(d - 30, (d + cad[names[i - 1]]) / 2)
        hi = d + 30 if i == len(names) - 1 else min(d + 30, (d + cad[names[i + 1]]) / 2)
        out[c] = ((lo - d) / d, (hi - d) / d)
    return out


def cfg_without(cfg, c):
    g = copy.deepcopy(cfg); keep = [i for i, x in enumerate(cfg['classes']) if x != c]
    g['classes'] = [cfg['classes'][i] for i in keep]; g['cad_hole_distance_mm'] = [cfg['cad_hole_distance_mm'][i] for i in keep]
    return g


def hole_pool(split):
    if split == '0':
        files = [ROOT / 'features/holes_field8_f0_s42_v2_development/rows.jsonl', ROOT / 'features/holes_field8_f0_s42_v2_f0_test_v3/rows.jsonl']
    elif split == 'temporal':
        files = [ROOT / 'features/holes_field_temporal_s42_trainTest/rows.jsonl']
    else:
        files = [ROOT / f'features/holes_field8_f{split}_s42_v3_trainTest/rows.jsonl']
    return {r['frame_id']: r for f in files for r in read_jsonl(f)}


def test_prediction_file(split, mode):
    if split == '0':
        return F0 / f'holes_field8_f0_s42_v2_{mode}_test_predictions.jsonl'
    if split == 'temporal':
        return EXT3 / f'holes_field_temporal_s42_{mode}_predictions.jsonl'
    return EXT3 / f'holes_field8_f{split}_s42_v3_{mode}_predictions.jsonl'


# ---------------------------------------------------------------- CNN leave-one-class-out
def train_cnn_loco(kind, exclude, run, seed=42, workers=4):
    sp, summary, rows, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); parts = partition(rows, splits, '0')
    if exclude not in cfg['classes']:
        raise ValueError(exclude)
    classes = [c for c in cfg['classes'] if c != exclude]; idx = {c: i for i, c in enumerate(classes)}
    tr = [dict(r, class_index=idx[r['class_name']]) for r in parts['train'] if r['class_name'] != exclude]
    va = [dict(r, class_index=idx[r['class_name']]) for r in parts['val'] if r['class_name'] != exclude]
    runpath = require_fresh_dir('runs/' + run); seed_all(seed)
    hp = cfg['cnn']; dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Classifier(kind, len(classes)).to(dev)
    dl = make_loader(CNNData(tr, kind, hp['image_size'], True), hp['batch_size'], True, workers, seed)
    vl = make_loader(CNNData(va, kind, hp['image_size'], False), hp['batch_size'], False, workers, seed)
    count = collections.Counter(r['class_index'] for r in tr)
    weights = torch.tensor([(len(tr) / max(count[i], 1)) ** .5 for i in range(len(classes))], device=dev)
    weights /= weights.mean(); opt = torch.optim.Adam(net.parameters(), lr=hp['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=.5, patience=3)
    meta = dict(model_kind=kind, snapshot='v1', snapshot_sha256=sha256(sp / 'COMPLETE.json'), protocol_sha256=summary['protocol_sha256'],
                extension_protocol_sha256=sha256(PROTOCOL), fold='0', seed=seed, classes=classes, excluded_class=exclude,
                image_size=hp['image_size'], training_regime='field_class_labels_leave_one_class_out',
                initialization='ImageNet1K_V1', initialization_sha256=sha256(pretrained_resnet()),
                train_frame_ids=[r['frame_id'] for r in tr], val_frame_ids=[r['frame_id'] for r in va],
                train_group_ids=sorted({r['group_id'] for r in tr}), val_group_ids=sorted({r['group_id'] for r in va}),
                pilot=False, requested_epochs=hp['epochs'], started_at=now(), status='running', hyperparameters=hp, device=str(dev))
    meta.update(capture_code()); dump(runpath / 'run.json', meta)
    hist = []; best = (-1., float('-inf')); wait = 0; t0 = time.perf_counter()
    for ep in range(hp['epochs']):
        net.train(); total = 0.; seen = 0; et = time.perf_counter()
        for x, y, _ in dl:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == 'cuda'):
                loss = F.cross_entropy(net(x), y, weight=weights)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.)
            scaler.step(opt); scaler.update(); total += float(loss.detach()) * len(y); seen += len(y)
        pred, vloss = cnn_predict(net, vl, va, classes, dev)
        met = summarize(pred, classes); score = (met['session']['balanced_accuracy'], -vloss)
        scheduler.step(vloss)
        rec = dict(epoch=ep + 1, train_loss=total / seen, val_loss=vloss, val_session_balanced_accuracy=score[0], seconds=time.perf_counter() - et)
        hist.append(rec); dump(runpath / 'history.json', hist)
        if score > best:
            best = score; wait = 0; torch.save(net.state_dict(), outpath(runpath / 'model.pth')); jsonl(runpath / 'val_predictions.jsonl', pred)
            meta.update(best_epoch=ep + 1, best_val_session_balanced_accuracy=score[0]); dump(runpath / 'run.json', meta)
        else:
            wait += 1
        print(json.dumps(dict(run=run, **rec)), flush=True)
        if wait >= hp['patience']:
            break
    meta.update(status='complete', finished_at=now(), seconds=time.perf_counter() - t0, weights_sha256=sha256(runpath / 'model.pth'))
    dump(runpath / 'run.json', meta)


def eval_cnn_loco(run, workers=4):
    sp, _, rows, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); parts = partition(rows, splits, '0')
    rp = ROOT / 'runs' / run; rm = read_json(rp / 'run.json')
    if rm['status'] != 'complete' or rm['extension_protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('Run incomplete or protocol changed')
    classes, exclude = rm['classes'], rm['excluded_class']; idx = {c: i for i, c in enumerate(classes)}
    if set(rm['train_group_ids']) & {r['group_id'] for r in parts['test']}:
        raise ValueError('Leakage')
    test = [dict(r, class_index=idx.get(r['class_name'], 0)) for r in parts['test']]       # dummy index for the unknown class
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Classifier(rm['model_kind'], len(classes), pretrained=False).to(dev)
    net.load_state_dict(torch.load(rp / 'model.pth', map_location=dev, weights_only=True))
    loader = make_loader(CNNData(test, rm['model_kind'], rm['image_size'], False), cfg['cnn']['batch_size'], False, workers, rm['seed'])
    pred, _ = cnn_predict(net, loader, test, classes, dev)
    conf = sorted(r['confidence'] for r in read_jsonl(rp / 'val_predictions.jsonl')); thr = conf[int(.05 * len(conf))]
    closed = [dict(r) for r in pred]
    opened = [dict(r, pred=r['pred'] if r['confidence'] >= thr else None) for r in pred]
    OUT.mkdir(parents=True, exist_ok=True)
    jsonl(OUT / f'{run}_predictions.jsonl', [dict(r, unknown=r['class_name'] == exclude, accepted=r['confidence'] >= thr) for r in pred])
    m = loco_metrics(closed, opened, exclude, score=lambda r: r['confidence'])
    m.update(run=run, kind=rm['model_kind'], excluded_class=exclude, threshold=thr,
             val_acceptance=float(np.mean([c >= thr for c in conf])), best_epoch=rm.get('best_epoch'))
    dump(OUT / f'{run}_metrics.json', m); print(json.dumps({k: m[k] for k in ('unknown_doors', 'known_doors_threshold', 'auroc_frame')}))


def loco_metrics(closed, opened, exclude, score):
    unk_c = [r for r in closed if r['class_name'] == exclude]; kn_c = [r for r in closed if r['class_name'] != exclude]
    unk_o = [r for r in opened if r['class_name'] == exclude]; kn_o = [r for r in opened if r['class_name'] != exclude]
    uv, kv, kvc = vote(unk_o), vote(kn_o), vote(kn_c)
    return dict(
        unknown_doors=dict(n=len(uv), rejected=sum(v['pred'] is None for v in uv.values()),
                           accepted_as=dict(Counter(v['pred'] for v in uv.values() if v['pred'] is not None))),
        unknown_frames=dict(n=len(unk_o), rejected=sum(r['pred'] is None for r in unk_o)),
        unknown_closed_set_destination=dict(Counter(v['pred'] for v in vote(unk_c).values())),
        known_doors_threshold=dict(n=len(kv), correct=sum(v['pred'] == v['true'] for v in kv.values()),
                                   abstained=sum(v['pred'] is None for v in kv.values()),
                                   wrong=sum(v['pred'] is not None and v['pred'] != v['true'] for v in kv.values())),
        known_doors_closed=dict(n=len(kvc), correct=sum(v['pred'] == v['true'] for v in kvc.values())),
        known_frames_judged=sum(r['pred'] is not None for r in kn_o) / max(len(kn_o), 1),
        auroc_frame=auroc([score(r) for r in kn_o], [score(r) for r in unk_o]))


# ---------------------------------------------------------------- hole leave-one-class-out (no retraining)
def hole_loco():
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), runs={})
    OUT.mkdir(parents=True, exist_ok=True); allpred = []
    for k in ('0', '1', '2', '3', '4'):
        parts = partition(frames, splits, k); pool = hole_pool(k)
        if not {r['frame_id'] for r in parts['train'] + parts['test']} <= set(pool):
            raise ValueError(f'missing features fold {k}')
        for c in cfg['classes']:
            g = cfg_without(cfg, c); cad7 = dict(zip(g['classes'], g['cad_hole_distance_mm']))
            cal = fit_scale([r for r in parts['train'] if r['class_name'] != c], pool, cad7, 'pixel')
            preds = []
            for row in parts['test']:
                p, v, why = classify_measurement(row, pool[row['frame_id']], cal, g)
                s = -min(abs(v - x) for x in cad7.values()) if v is not None else float('-inf')
                preds.append({kk: row[kk] for kk in ('frame_id', 'group_id', 'class_name')} | dict(pred=p, measurement=v, reason=why, score=s,
                             fold=k, excluded=c))
            m = loco_metrics(preds, preds, c, score=lambda r: r['score'])
            m.update(scale=cal['camera_scales'], unknown_measured_D_median=float(np.median([r['measurement'] for r in preds
                     if r['class_name'] == c and r['measurement'] is not None])) if any(r['class_name'] == c and r['measurement'] is not None for r in preds) else None,
                     unknown_min_distance_mm=float(-max(r['score'] for r in preds if r['class_name'] == c)) if any(r['class_name'] == c for r in preds) else None)
            res['runs'][f'{k}:{c}'] = m; allpred += preds
    jsonl(OUT / 'hole_loco_predictions.jsonl', [dict(r, score=None if r['score'] == float('-inf') else r['score']) for r in allpred])  # None = gate failed
    res['fold0'] = pool_runs({kk: v for kk, v in res['runs'].items() if kk.startswith('0:')})
    res['oof'] = pool_runs(res['runs'])
    dump(OUT / 'hole_loco.json', res); print(json.dumps({'fold0': res['fold0'], 'oof': res['oof']}, indent=1))


def pool_runs(runs):
    tot = Counter()
    for m in runs.values():
        tot.update(dict(ud=m['unknown_doors']['n'], ur=m['unknown_doors']['rejected'], uf=m['unknown_frames']['n'], ufr=m['unknown_frames']['rejected'],
                        kd=m['known_doors_threshold']['n'], kc=m['known_doors_threshold']['correct'], ka=m['known_doors_threshold']['abstained'],
                        kw=m['known_doors_threshold']['wrong']))
    return dict(unknown_doors=[tot['ur'], tot['ud']], unknown_doors_wilson=wilson(tot['ur'], tot['ud']), unknown_frames=[tot['ufr'], tot['uf']],
                known_door_decisions=dict(n=tot['kd'], correct=tot['kc'], abstained=tot['ka'], wrong=tot['kw']),
                auroc_mean=float(np.mean([m['auroc_frame'] for m in runs.values() if m['auroc_frame'] is not None])),
                auroc_min=float(np.min([m['auroc_frame'] for m in runs.values() if m['auroc_frame'] is not None])))


# ---------------------------------------------------------------- margin: TRAIN decisions predict TEST reliability?
def door_stats(preds, cad, band):
    sess = vote(preds); n = len(sess); fail = sum(v['pred'] != v['true'] for v in sess.values())
    meas = [r for r in preds if r['measurement'] is not None]
    out_band = sum(not (band[r['class_name']][0] <= r['measurement'] / cad[r['class_name']] - 1 <= band[r['class_name']][1]) for r in meas)
    per = defaultdict(list)
    for r in meas:
        per[(r['class_name'], r['group_id'])].append(r['measurement'] / cad[r['class_name']] - 1)
    util = defaultdict(float)
    for (c, _), v in per.items():
        m = float(np.median(v)); u = m / band[c][1] if m > 0 else m / band[c][0]
        util[c] = max(util[c], u)
    return dict(doors=n, failed=fail, frames_measured=len(meas), frames_out_of_band=out_band, utilisation=dict(util))


def margin():
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); band = bands(cad)
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), bands=band, splits={})
    for k in ('0', '1', '2', '3', '4', 'temporal'):
        parts = partition(frames, splits, k); pool = hole_pool(k)
        for mode in ('pixel', 'depth'):
            cal = fit_scale(parts['train'], pool, cad, mode)
            tp = read_jsonl(test_prediction_file(k, mode))
            if {r['frame_id'] for r in tp} != {r['frame_id'] for r in parts['test']}:
                raise ValueError('test predictions do not match split')
            train_preds = []
            for row in parts['train']:
                p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg)
                train_preds.append({kk: row[kk] for kk in ('frame_id', 'group_id', 'class_name')} | dict(pred=p, measurement=v, reason=why))
            res['splits'][f'{k}:{mode}'] = dict(scale=cal['camera_scales'], train=door_stats(train_preds, cad, band), test=door_stats(tp, cad, band))
    for mode in ('pixel', 'depth'):
        for name, keys in (('five_fold', ['0', '1', '2', '3', '4']), ('temporal', ['temporal'])):
            agg = {}
            for side in ('train', 'test'):
                d = [res['splits'][f'{k}:{mode}'][side] for k in keys]
                n, f = sum(x['doors'] for x in d), sum(x['failed'] for x in d)
                fm, fo = sum(x['frames_measured'] for x in d), sum(x['frames_out_of_band'] for x in d)
                util = {c: max(x['utilisation'].get(c, 0.) for x in d) for c in cfg['classes']}
                agg[side] = dict(doors=n, failed=f, rate=f / n, wilson=wilson(f, n), frames_measured=fm, frames_out_of_band=fo,
                                 frame_rate=fo / fm, max_utilisation=max(util.values()), utilisation=util)
            res[f'{name}:{mode}'] = agg
    OUT.mkdir(parents=True, exist_ok=True); dump(OUT / 'margin.json', res)
    print(json.dumps({k: {s: {kk: v[s][kk] for kk in ('doors', 'failed', 'rate', 'frame_rate', 'max_utilisation')} for s in ('train', 'test')}
                      for k, v in res.items() if ':' in k and k.split(':')[0] in ('five_fold', 'temporal')}, indent=1))


# ---------------------------------------------------------------- collect
def collect():
    sp, _, _, _ = snapshot(); cfg = read_json(sp / 'protocol.json')
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), cnn={}, hole=read_json(OUT / 'hole_loco.json')['fold0'],
               hole_oof=read_json(OUT / 'hole_loco.json')['oof'], margin={k: v for k, v in read_json(OUT / 'margin.json').items() if ':' in k and not k[0].isdigit()})
    for kind in ('rgb', 'rgbe'):
        runs = {}
        for c in cfg['classes']:
            p = OUT / f'{kind}_loco_{c}_f0_s42_metrics.json'
            runs[c] = read_json(p) if p.exists() else None
        done = {c: m for c, m in runs.items() if m}
        out['cnn'][kind] = dict(per_class={c: dict(unknown_doors=[m['unknown_doors']['rejected'], m['unknown_doors']['n']],
                                                   accepted_as=m['unknown_doors']['accepted_as'], closed_set=m['unknown_closed_set_destination'],
                                                   known=m['known_doors_threshold'], auroc=m['auroc_frame'], threshold=m['threshold'])
                                           for c, m in done.items()},
                                pooled=pool_runs(done) if done else None, missing=[c for c, m in runs.items() if not m])
    hole_runs = read_json(OUT / 'hole_loco.json')['runs']
    out['hole_per_class'] = {c: dict(unknown_doors=[hole_runs[f'0:{c}']['unknown_doors']['rejected'], hole_runs[f'0:{c}']['unknown_doors']['n']],
                                     min_distance_mm=hole_runs[f'0:{c}']['unknown_min_distance_mm'], auroc=hole_runs[f'0:{c}']['auroc_frame'],
                                     known=hole_runs[f'0:{c}']['known_doors_threshold']) for c in cfg['classes']}
    dump(OUT / 'collected.json', out); print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('train-cnn-loco'); a.add_argument('--kind', choices=['rgb', 'rgbe'], required=True)
    a.add_argument('--exclude', required=True); a.add_argument('--run', required=True); a.add_argument('--workers', type=int, default=4)
    a = sub.add_parser('eval-cnn-loco'); a.add_argument('--run', required=True)
    sub.add_parser('hole-loco'); sub.add_parser('margin'); sub.add_parser('collect')
    args = ap.parse_args()
    if args.cmd == 'train-cnn-loco':
        train_cnn_loco(args.kind, args.exclude, args.run, workers=args.workers)
    elif args.cmd == 'eval-cnn-loco':
        eval_cnn_loco(args.run)
    elif args.cmd == 'hole-loco':
        hole_loco()
    elif args.cmd == 'margin':
        margin()
    else:
        collect()


if __name__ == '__main__':
    main()
