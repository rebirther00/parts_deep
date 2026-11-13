"""Extension v19 (review round 8 of manuscript v18): the post-hoc template-matching diagnostic T1b of extension_v18b,
unchanged, in three further evaluations registered in config/extension_v19_protocol.json. CPU only.

  e23                              E1: fold-0 T1b on the six E23 LH FRT doors (eight-value list; 458 mm registered)
  loco    --split S --cls C        E2: one unseen-class run (templates, search box, selection and S without class C)
  loco-collect                     reports/extension_v19/loco.json
  fixed   --split S --cand I       E3: NCC candidate I (0-5) used without the selection step
  fixed-collect                    reports/extension_v19/fixed_settings.json
"""
import argparse
import collections
import copy
from multiprocessing import Pool
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from door8.core import setup_runtime
setup_runtime()

import cv2
import numpy as np

from door8.core import ROOT, dump, jsonl, now, partition, read_json, read_jsonl, sha256, snapshot
from door8.field_data import load_field_labels
from door8.geometry import fit_scale
from door8.metrics import wilson
import run_extension_v3 as V3
import run_extension_v5 as V5
import run_extension_v12 as V12
import run_extension_v16e as E
import run_extension_v18 as V18

PROTOCOL = ROOT / 'config/extension_v19_protocol.json'
OUT = ROOT / 'reports/extension_v19'
CACHE = ROOT / 'cache/extension_v19'
SPLITS = E.SPLITS; KEYS = E.KEYS; CANDIDATES = E.CANDIDATES
VARIANT = 'gapfree'                                   # T1b: detection of 'gap', order-free errors
E23 = ('E23_door_LH_FRT', 458)


# ---------------------------------------------------------------- shared
_W = {}


def _init(cand, s):
    cv2.setNumThreads(1); _W.update(cand=cand, s=s)


def _one(row):
    g = E.gray(row['rgb']); t = time.perf_counter(); det = V18.detect(g, VARIANT, _W['cand'], _W['s']); dt = time.perf_counter() - t
    f = dict(frame_id=row['frame_id'], gate='missing_corner', span_normalized=None, detection=det, seconds=dt)
    if det:
        f.update(gate='ok', span_normalized=V18.gap_px(det) / row['intrinsics']['fx'])
    return f


def detect_rows(rows, cand, s, procs):
    with Pool(procs, initializer=_init, initargs=(cand, s)) as pool:
        return pool.map(_one, rows, chunksize=16)


def context():
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    return sp, frames, splits, cfg, dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))


def store(dst, feats, extra):
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / 'features.jsonl').exists():
        raise FileExistsError(dst / 'features.jsonl')
    jsonl(dst / 'features.jsonl', feats)
    dump(dst / 'features.json', dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), n=len(feats), rows_sha256=sha256(dst / 'features.jsonl'), **extra))


def count(xs):
    return dict(n=len(xs), correct=sum(x['pred'] == x['true'] for x in xs), rejected=sum(x['pred'] is None for x in xs),
                wrong=sum(x['pred'] not in (None, x['true']) for x in xs))


def choose(rows):
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]['pck8'], rows[i]['median_px'] if rows[i]['median_px'] is not None else 1e9, i))
    return order[0]


# ---------------------------------------------------------------- E1
def e23(procs):
    sp, frames, splits, cfg, cad = context(); parts = partition(frames, splits, '0')
    sel = read_json(V18.OUT / VARIANT / 'select_0.json'); cand = CANDIDATES[sel['selected_index']]; tr, _, _ = E.supervision('0')
    s = V18.setup(VARIANT, cand, tr, E.gray(tr[0]['rgb']).shape)
    pool0 = {r['frame_id']: r for r in read_jsonl(V18.CACHE / VARIANT / '0' / 'features.jsonl')}; cal = fit_scale(parts['train'], pool0, cad, 'pixel')
    ref = read_json(V18.OUT / VARIANT / 'eval_0.json')['scale']
    if any(abs(cal['camera_scales'][k] - v) > 1e-9 for k, v in ref.items()):
        raise ValueError('S differs from the fold-0 T1b evaluation')
    rows = list(read_jsonl(ROOT / 'reports/e23_openset_v3/manifest.jsonl'))
    if len(rows) != 120 or {r['class_name'] for r in rows} != {E23[0]}:
        raise ValueError('Unexpected E23 manifest')
    feats = detect_rows(rows, cand, s, procs); store(CACHE / 'e23', feats, dict(candidate=sel['selected'], region=s['reg']))
    pool = {f['frame_id']: f for f in feats}
    cfg9 = copy.deepcopy(cfg); cfg9['classes'] = cfg['classes'] + [E23[0]]; cfg9['cad_hole_distance_mm'] = cfg['cad_hole_distance_mm'] + [E23[1]]
    p8 = V18.judge(rows, pool, cal, cfg, 'e23'); p9 = V18.judge(rows, pool, cal, cfg9, 'e23')
    OUT.mkdir(parents=True, exist_ok=True); jsonl(OUT / 'e23_predictions_8.jsonl', p8); jsonl(OUT / 'e23_predictions_9.jsonl', p9)
    v8 = V5.vote(p8); v9 = V5.vote(p9); meas = [r['measurement'] for r in p8 if r['measurement'] is not None]; gaps = [V18.gap_px(f['detection']) for f in feats if f['detection']]
    per = collections.defaultdict(list)
    for r in p8:
        if r['measurement'] is not None:
            per[r['group_id']].append(r['measurement'])
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), candidate=sel['selected'], region=s['reg'], scale=cal['camera_scales'], doors=len(v8), images=len(rows),
               unregistered=dict(doors_rejected=sum(x['pred'] is None for x in v8.values()), doors_accepted_as=dict(collections.Counter(x['pred'] for x in v8.values() if x['pred'] is not None)),
                                 images_rejected=sum(r['pred'] is None for r in p8), images_accepted_as=dict(collections.Counter(r['pred'] for r in p8 if r['pred'] is not None)),
                                 reasons=dict(collections.Counter(r['reason'] for r in p8))),
               registered_458=dict(doors_identified=sum(x['pred'] == E23[0] for x in v9.values()), doors_rejected=sum(x['pred'] is None for x in v9.values()),
                                   doors_other=dict(collections.Counter(x['pred'] for x in v9.values() if x['pred'] not in (None, E23[0]))),
                                   images_identified=sum(r['pred'] == E23[0] for r in p9), reasons=dict(collections.Counter(r['reason'] for r in p9))),
               measured_mm=dict(n=len(meas), median=float(np.median(meas)), min=float(min(meas)), max=float(max(meas))) if meas else None,
               measured_mm_door_medians={g: float(np.median(v)) for g, v in sorted(per.items())},
               detection_gap_px=dict(n=len(gaps), median=float(np.median(gaps)), min=float(min(gaps)), max=float(max(gaps))) if gaps else None,
               no_detection=sum(not f['detection'] for f in feats))
    dump(OUT / 'e23.json', out)
    print('E23', out['candidate'], 'unregistered', out['unregistered'], '| registered 458', out['registered_458'], '| D', out['measured_mm'], '| gap px', out['detection_gap_px'])


# ---------------------------------------------------------------- E2
def loco(split, c, procs):
    sp, frames, splits, cfg, cad = context(); dst = OUT / 'loco'; name = f'{split}_{c}'
    if c not in cfg['classes'] or split == 'temporal':
        raise ValueError('loco needs a fold 0-4 and a registered class')
    if (dst / f'run_{name}.json').exists():
        raise FileExistsError(dst / f'run_{name}.json')
    tr, va, _ = E.supervision(split); tr = [r for r in tr if r['class_name'] != c]; va = [r for r in va if r['class_name'] != c]
    shape = E.gray(tr[0]['rgb']).shape; table = []
    for cand in CANDIDATES:
        s = V18.setup(VARIANT, cand, tr, shape); errs = []
        for row in va:
            errs += V18.errors_free(V18.detect(E.gray(row['rgb']), VARIANT, cand, s), row['points'])
        table.append(dict(candidate=E.tag(cand), region=s['reg'], **E.pck(errs)))
    i = choose(table); cand = CANDIDATES[i]; s = V18.setup(VARIANT, cand, tr, shape)
    parts = partition(frames, splits, split); train = [r for r in parts['train'] if r['class_name'] != c]
    feats = detect_rows(train + parts['test'], cand, s, procs); store(CACHE / 'loco' / name, feats, dict(split=split, removed=c, candidate=E.tag(cand), region=s['reg']))
    pool = {f['frame_id']: f for f in feats}; cal = fit_scale(train, pool, cad, 'pixel'); cfg7 = V5.cfg_without(cfg, c)
    p7 = V18.judge(parts['test'], pool, cal, cfg7, split); p8 = V18.judge(parts['test'], pool, cal, cfg, split)
    dst.mkdir(parents=True, exist_ok=True); jsonl(dst / f'predictions7_{name}.jsonl', p7); jsonl(dst / f'predictions8_{name}.jsonl', p8)
    v7 = V5.vote(p7); v8 = V5.vote(p8); cad7 = np.array(cfg7['cad_hole_distance_mm'], float)
    unk = [x for x in v7.values() if x['true'] == c]; dist = [float(np.abs(cad7 - r['measurement']).min()) for r in p7 if r['class_name'] == c and r['measurement'] is not None]
    _, _, labels = load_field_labels(V3.LABELSET, sp); x0, y0, x1, y1 = s['reg']
    pts = [r['points'][k] for r in labels if r['class_name'] == c for k in KEYS]; inside = sum(x0 <= p[0] < x1 and y0 <= p[1] < y1 for p in pts)
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), split=split, removed=c, train_images=len(tr), val_images=len(va), validation=table, selected=E.tag(cand), region=s['reg'],
               scale=cal['camera_scales'], train_sessions=len({r['group_id'] for r in train}),
               unknown=dict(n=len(unk), rejected=sum(x['pred'] is None for x in unk), accepted_as=dict(collections.Counter(x['pred'] for x in unk if x['pred'] is not None))),
               known=count([x for x in v7.values() if x['true'] != c]), new=count([x for x in v8.values() if x['true'] == c]), other=count([x for x in v8.values() if x['true'] != c]),
               new_wrong_as=dict(collections.Counter(x['pred'] for x in v8.values() if x['true'] == c and x['pred'] not in (None, c))),
               unknown_images=dict(n=sum(r['class_name'] == c for r in p7), accepted=sum(r['class_name'] == c and r['pred'] is not None for r in p7),
                                   unmeasured=sum(r['class_name'] == c and r['measurement'] is None for r in p7), min_distance_to_registered_mm=min(dist) if dist else None),
               removed_class_holes=dict(points=len(pts), inside_search_box=inside),
               seconds_per_image_single_thread=float(np.mean([f['seconds'] for f in feats])))
    dump(dst / f'run_{name}.json', out)
    print(name, out['selected'], 'S', {k: round(v, 1) for k, v in out['scale'].items()}, 'unknown', out['unknown'], 'new', out['new'], 'known', out['known'], 'other', out['other'], 'holes inside', inside, '/', len(pts))


def loco_collect():
    sp, frames, splits, cfg, cad = context(); runs = {f'{s}_{c}': read_json(OUT / 'loco' / f'run_{s}_{c}.json') for s in SPLITS[:5] for c in cfg['classes']}
    tot = {k: collections.Counter() for k in ('known', 'new', 'other')}; unk = collections.Counter(); acc = collections.Counter(); nw = collections.Counter(); img = collections.Counter(); dist = []
    for r in runs.values():
        unk.update(n=r['unknown']['n'], rejected=r['unknown']['rejected']); acc.update({f"{r['removed']}->{k}": v for k, v in r['unknown']['accepted_as'].items()})
        nw.update({f"{r['removed']}->{k}": v for k, v in r['new_wrong_as'].items()}); img.update(n=r['unknown_images']['n'], accepted=r['unknown_images']['accepted'], unmeasured=r['unknown_images']['unmeasured'])
        if r['unknown_images']['min_distance_to_registered_mm'] is not None:
            dist.append(r['unknown_images']['min_distance_to_registered_mm'])
        for k in tot:
            tot[k].update(r[k])
    per = {}
    for c in cfg['classes']:
        rr = [runs[f'{s}_{c}'] for s in SPLITS[:5]]
        per[c] = dict(doors=sum(r['unknown']['n'] for r in rr), unknown_rejected=sum(r['unknown']['rejected'] for r in rr), new_correct=sum(r['new']['correct'] for r in rr),
                      new_rejected=sum(r['new']['rejected'] for r in rr), new_wrong=sum(r['new']['wrong'] for r in rr),
                      holes_inside_search_box=[f"{r['removed_class_holes']['inside_search_box']}/{r['removed_class_holes']['points']}" for r in rr],
                      min_distance_to_registered_mm=min([r['unknown_images']['min_distance_to_registered_mm'] for r in rr if r['unknown_images']['min_distance_to_registered_mm'] is not None], default=None))
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), runs=len(runs),
               unknown=dict(unk, wilson=wilson(unk['rejected'], unk['n'])), new=dict(tot['new'], wilson=wilson(tot['new']['correct'], tot['new']['n'])),
               known=dict(tot['known']), other=dict(tot['other']), accepted_as=dict(acc), new_wrong_as=dict(nw), unknown_images=dict(img),
               min_distance_to_registered_mm=min(dist) if dist else None, per_class=per,
               selected={k: r['selected'] for k, r in runs.items()}, scale={k: round(list(r['scale'].values())[0], 1) for k, r in runs.items()})
    dump(OUT / 'loco.json', out)
    print('unknown', out['unknown'], '| new', out['new'], '| known', out['known'], '| other', out['other']); print('accepted_as', out['accepted_as']); print('new_wrong_as', out['new_wrong_as'])
    for c, v in per.items():
        print(' ', c, v)


# ---------------------------------------------------------------- E3
def fixed(split, idx, procs):
    cand = CANDIDATES[idx]
    if cand['kind'] != 'ncc':
        raise ValueError('fixed settings are the six NCC candidates')
    sp, frames, splits, cfg, cad = context(); dst = OUT / 'fixed'; name = f'{split}_c{idx}'
    if (dst / f'run_{name}.json').exists():
        raise FileExistsError(dst / f'run_{name}.json')
    tr, _, _ = E.supervision(split); s = V18.setup(VARIANT, cand, tr, E.gray(tr[0]['rgb']).shape); parts = partition(frames, splits, split)
    feats = detect_rows(parts['train'] + parts['test'], cand, s, procs); store(CACHE / 'fixed' / name, feats, dict(split=split, candidate=E.tag(cand), region=s['reg']))
    pool = {f['frame_id']: f for f in feats}; cal = fit_scale(parts['train'], pool, cad, 'pixel'); preds = V18.judge(parts['test'], pool, cal, cfg, split)
    dst.mkdir(parents=True, exist_ok=True); jsonl(dst / f'predictions_{name}.jsonl', preds)
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), split=split, candidate=E.tag(cand), index=idx, scale=cal['camera_scales'], doors=V12.door_counts(preds), frames=V12.frame_counts(preds))
    dump(dst / f'run_{name}.json', out); print(name, out['candidate'], 'S', {k: round(v, 1) for k, v in out['scale'].items()}, 'doors', out['doors'])


def fixed_collect():
    idx = [i for i, c in enumerate(CANDIDATES) if c['kind'] == 'ncc']; res = {}
    for i in idx:
        oof = [r for s in SPLITS[:5] for r in read_jsonl(OUT / 'fixed' / f'predictions_{s}_c{i}.jsonl')]; d = V12.door_counts(oof); t = read_json(OUT / 'fixed' / f'run_temporal_c{i}.json')
        res[E.tag(CANDIDATES[i])] = dict(five_fold=d, five_fold_wilson=wilson(d['correct'], d['n']), frames=V12.frame_counts(oof), temporal=t['doors'],
                                         per_fold={s: read_json(OUT / 'fixed' / f'run_{s}_c{i}.json')['doors']['correct'] for s in SPLITS[:5]},
                                         scale={s: round(list(read_json(OUT / 'fixed' / f'run_{s}_c{i}.json')['scale'].values())[0], 1) for s in SPLITS})
    five = [v['five_fold']['correct'] for v in res.values()]; temp = [v['temporal']['correct'] for v in res.values()]
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), settings=res, five_fold_range=[min(five), max(five)], temporal_range=[min(temp), max(temp)])
    dump(OUT / 'fixed_settings.json', out)
    for k, v in res.items():
        print(k, 'five-fold', v['five_fold'], v['per_fold'], '| temporal', v['temporal'])
    print('range five-fold', out['five_fold_range'], 'temporal', out['temporal_range'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('e23', 'loco', 'loco-collect', 'fixed', 'fixed-collect')); ap.add_argument('--split', choices=SPLITS); ap.add_argument('--cls')
    ap.add_argument('--cand', type=int); ap.add_argument('--procs', type=int, default=6)
    x = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if x.cmd == 'e23':
        e23(x.procs)
    elif x.cmd == 'loco':
        loco(x.split, x.cls, x.procs)
    elif x.cmd == 'loco-collect':
        loco_collect()
    elif x.cmd == 'fixed':
        fixed(x.split, x.cand, x.procs)
    else:
        fixed_collect()
