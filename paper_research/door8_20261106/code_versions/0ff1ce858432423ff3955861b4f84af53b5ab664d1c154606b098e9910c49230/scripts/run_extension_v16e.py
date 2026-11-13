"""Extension v16e (review round 5, bundle E): classical image-processing baseline for the corner-hole detection,
registered in config/extension_v16e_protocol.json. CPU only; the decision stage is the one of the learned detector.

  select  --split S     validation PCK8 of every candidate, choice of one (no TEST data)
  detect  --split S     corner detections of the chosen candidate on the split's TRAIN and TEST frames
  eval    --split S     S fit on TRAIN, decisions on TEST, localisation, comparison with the learned detector
  collect               reports/extension_v16e/collected.json
"""
import argparse
import collections
import json
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
from door8.field_data import field_partitions, load_field_labels
from door8.geometry import classify_measurement, fit_scale
from door8.metrics import wilson
import run_extension_v3 as V3
import run_extension_v5 as V5
import run_extension_v12 as V12

PROTOCOL = ROOT / 'config/extension_v16e_protocol.json'
OUT = ROOT / 'reports/extension_v16e'
CACHE = ROOT / 'cache/extension_v16e'
SPLITS = ('0', '1', '2', '3', '4', 'temporal')
CANDIDATES = [dict(kind='ncc', r=r, m=m) for r in (16, 24, 32) for m in (40, 80)] + [dict(kind='log', sigma=s, m=m) for s in (1.5, 2.5) for m in (40, 80)]
KEYS = ('corner_latch', 'corner_hinge')
CHANGE_DATE = '20260921'


def tag(c):
    return f"ncc r={c['r']} m={c['m']}" if c['kind'] == 'ncc' else f"log sigma={c['sigma']} m={c['m']}"


def base_run(split):
    return 'holes_field_temporal_s42' if split == 'temporal' else V12.hole_run(split)


def supervision(split):
    """Field training and validation annotations of the learned detector of this split."""
    sp, _, frames, splits = snapshot(); _, _, labels = load_field_labels(V3.LABELSET, sp); by = {r['annotation_id']: r for r in labels}
    rm = read_json(ROOT / 'runs' / base_run(split) / 'run.json')
    tr = [by[i] for i in rm['train_annotation_ids'] if i in by]; va = [by[i] for i in rm['val_annotation_ids'] if i in by]
    roles = field_partitions(labels, frames, splits, split)
    if {r['annotation_id'] for r in tr} - {r['annotation_id'] for r in roles['train']} or {r['annotation_id'] for r in va} - {r['annotation_id'] for r in roles['val']}:
        raise ValueError('Field supervision crossed split roles')
    return tr, va, roles['test']


def gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f'Unreadable image {path}')
    return im


def region(train, m, shape):
    pts = np.array([r['points'][k] for r in train for k in KEYS], float); h, w = shape
    x0, y0 = np.floor(pts.min(0) - m).astype(int); x1, y1 = np.ceil(pts.max(0) + m).astype(int)
    return int(max(x0, 0)), int(max(y0, 0)), int(min(x1, w)), int(min(y1, h))


def templates(train, r):
    sets = {k: [] for k in KEYS}
    for row in train:
        g = gray(row['rgb'])
        for k in KEYS:
            x, y = (int(round(v)) for v in row['points'][k])
            if y - r < 0 or x - r < 0 or y + r + 1 > g.shape[0] or x + r + 1 > g.shape[1]:
                continue
            sets[k].append(g[y - r:y + r + 1, x - r:x + r + 1].copy())
    return sets


def detect_ncc(g, reg, sets, r):
    x0, y0, x1, y1 = reg; out = {}
    xa, ya, xb, yb = max(x0 - r, 0), max(y0 - r, 0), min(x1 + r, g.shape[1]), min(y1 + r, g.shape[0])   # so that a centre anywhere in the region is reachable
    roi = g[ya:yb, xa:xb]
    for k in KEYS:
        best = None
        for t in sets[k]:
            res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED); best = res if best is None else np.maximum(best, res)
        _, score, _, loc = cv2.minMaxLoc(best)
        out[k] = (float(loc[0] + xa + r), float(loc[1] + ya + r), float(score))
    return out


def detect_log(g, reg, sigma):
    x0, y0, x1, y1 = reg; roi = g[y0:y1, x0:x1].astype(np.float32)
    resp = cv2.Laplacian(cv2.GaussianBlur(roi, (0, 0), sigma), cv2.CV_32F, ksize=3) * sigma * sigma      # dark blob on a bright door: positive
    peak = (resp == cv2.dilate(resp, np.ones((5, 5), np.uint8))) & (resp > 0)
    ys, xs = np.nonzero(peak)
    if len(xs) == 0:
        return {}
    v = resp[ys, xs]; i = int(v.argmax()); far = np.abs(xs - xs[i]) >= 500
    if not far.any():
        return {}
    j = int(np.flatnonzero(far)[v[far].argmax()])
    return dict(a=(float(xs[i] + x0), float(ys[i] + y0), float(v[i])), b=(float(xs[j] + x0), float(ys[j] + y0), float(v[j])))


def detect(g, cand, reg, sets):
    return detect_ncc(g, reg, sets, cand['r']) if cand['kind'] == 'ncc' else detect_log(g, reg, cand['sigma'])


def errors(det, points):
    """Errors (px) of the two corner points; for the log candidate the better assignment."""
    gt = [np.array(points[k], float) for k in KEYS]
    if not det:
        return [None, None]
    if 'a' in det:
        a, b = np.array(det['a'][:2]), np.array(det['b'][:2])
        e1 = [np.linalg.norm(a - gt[0]), np.linalg.norm(b - gt[1])]; e2 = [np.linalg.norm(b - gt[0]), np.linalg.norm(a - gt[1])]
        return [float(x) for x in (e1 if sum(e1) <= sum(e2) else e2)]
    return [float(np.linalg.norm(np.array(det[k][:2]) - g)) for k, g in zip(KEYS, gt)]


def pck(errs):
    e = [x for x in errs if x is not None]
    return dict(points=len(errs), pck8=sum(x <= 8 for x in e) / max(len(errs), 1), median_px=float(np.median(e)) if e else None,
                max_px=float(max(e)) if e else None)


def select(split):
    tr, va, _ = supervision(split); shape = gray(tr[0]['rgb']).shape; rows = []
    for cand in CANDIDATES:
        reg = region(tr, cand['m'], shape); sets = templates(tr, cand['r']) if cand['kind'] == 'ncc' else None; errs = []
        for row in va:
            errs += errors(detect(gray(row['rgb']), cand, reg, sets), row['points'])
        rows.append(dict(candidate=tag(cand), region=reg, **pck(errs)))
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]['pck8'], rows[i]['median_px'] if rows[i]['median_px'] is not None else 1e9, i))
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), split=split, base_run=base_run(split), train_images=len(tr), val_images=len(va),
               candidates=rows, selected=rows[order[0]]['candidate'], selected_index=order[0])
    dump(OUT / f'select_{split}.json', out)
    print(split, 'selected', out['selected'], [(r['candidate'], round(r['pck8'], 3), None if r['median_px'] is None else round(r['median_px'], 1)) for r in rows])


_W = {}


def _init(split):
    cv2.setNumThreads(1)
    sel = read_json(OUT / f'select_{split}.json'); cand = CANDIDATES[sel['selected_index']]; tr, _, _ = supervision(split)
    shape = gray(tr[0]['rgb']).shape
    _W.update(cand=cand, reg=region(tr, cand['m'], shape), sets=templates(tr, cand['r']) if cand['kind'] == 'ncc' else None)


def _one(row):
    g = gray(row['rgb']); t = time.perf_counter(); det = detect(g, _W['cand'], _W['reg'], _W['sets']); dt = time.perf_counter() - t
    f = dict(frame_id=row['frame_id'], gate='missing_corner', span_normalized=None, detection=det, seconds=dt)
    if det:
        p = list(det.values()) if 'a' in det else [det[k] for k in KEYS]
        f.update(gate='ok', span_normalized=float(np.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]) / row['intrinsics']['fx']))
    return f


def run_detect(split, procs=6):
    sp, _, frames, splits = snapshot(); parts = partition(frames, splits, split); rows = parts['train'] + parts['test']
    dst = CACHE / split; dst.mkdir(parents=True, exist_ok=True)
    if (dst / 'features.jsonl').exists():
        raise FileExistsError(dst / 'features.jsonl')
    t0 = time.perf_counter()
    with Pool(procs, initializer=_init, initargs=(split,)) as pool:
        feats = pool.map(_one, rows, chunksize=16)
    jsonl(dst / 'features.jsonl', feats)
    dump(dst / 'features.json', dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), split=split, n=len(feats), rows_sha256=sha256(dst / 'features.jsonl'),
                                     selected=read_json(OUT / f'select_{split}.json')['selected'], wall_seconds=time.perf_counter() - t0,
                                     seconds_per_image_single_thread=float(np.mean([f['seconds'] for f in feats]))))
    print(split, len(feats), 'frames', round(time.perf_counter() - t0), 's wall; detection', round(1000 * float(np.mean([f['seconds'] for f in feats])), 1), 'ms/image')


def state(session_dir):
    return 'before' if session_dir[:8] < CHANGE_DATE else 'after'


def evaluate(split):
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); parts = partition(frames, splits, split)
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); band = V5.bands(cad)
    fm = read_json(CACHE / split / 'features.json')
    if fm['protocol_sha256'] != sha256(PROTOCOL) or sha256(CACHE / split / 'features.jsonl') != fm['rows_sha256']:
        raise ValueError('Features or protocol changed')
    pool = {r['frame_id']: r for r in read_jsonl(CACHE / split / 'features.jsonl')}
    cal = fit_scale(parts['train'], pool, cad, 'pixel'); preds = []
    for row in parts['test']:
        p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg)
        preds.append({k: row[k] for k in ('frame_id', 'group_id', 'class_name')} | dict(pred=p, measurement=v, reason=why, split=split))
    jsonl(OUT / f'predictions_{split}.jsonl', preds)
    _, _, test_labels = supervision(split); errs = []
    for lab in test_labels:
        errs += errors(pool[lab['frame_id']]['detection'], lab['points'])
    learned = {r['frame_id']: r for r in read_jsonl(V5.test_prediction_file(split, 'pixel'))}
    votes = V5.vote(preds); lv = V5.vote(list(learned.values()))
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), split=split, selected=fm['selected'], scale=cal['camera_scales'],
               seconds_per_image_single_thread=fm['seconds_per_image_single_thread'], doors=V12.door_counts(preds), frames=V12.frame_counts(preds),
               frame_reasons=dict(collections.Counter(r['reason'] for r in preds)), localisation_test_role=pck(errs),
               residual=V12.residual_stats(preds, cad, band) if any(r['measurement'] is not None for r in preds) else None,
               failed_doors=[dict(door=g, true=v['true'], pred=v['pred'], state=state(g)) for g, v in sorted(votes.items()) if v['pred'] != v['true']],
               by_state={s: V12.door_counts([r for r in preds if state(r['group_id']) == s]) for s in ('before', 'after')},
               differs_from_learned=sum(votes[g]['pred'] != lv[g]['pred'] for g in votes), learned_doors=V12.door_counts(list(learned.values())))
    dump(OUT / f'eval_{split}.json', res)
    print(split, res['selected'], 'doors', res['doors'], 'frames', res['frames'], 'loc', res['localisation_test_role'], 'ms', round(1000 * res['seconds_per_image_single_thread'], 1))


def collect():
    ev = {s: read_json(OUT / f'eval_{s}.json') for s in SPLITS}; sel = {s: read_json(OUT / f'select_{s}.json') for s in SPLITS}
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); band = V5.bands(cad)
    oof = [r for s in SPLITS[:5] for r in read_jsonl(OUT / f'predictions_{s}.jsonl')]
    d = V12.door_counts(oof); f = V12.frame_counts(oof); t = ev['temporal']
    cls = collections.Counter((v['true'], 'correct' if v['pred'] == v['true'] else 'rejected' if v['pred'] is None else 'wrong') for v in V5.vote(oof).values())
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL),
               five_fold=dict(doors=d, doors_wilson=wilson(d['correct'], d['n']), frames=f, image_accuracy=f['correct'] / f['n'], image_coverage=1 - f['abstained'] / f['n'],
                              residual=V12.residual_stats(oof, cad, band), by_class={c: {k: cls[(c, k)] for k in ('correct', 'rejected', 'wrong')} for c in cfg['classes']},
                              by_state={s: V12.door_counts([r for r in oof if state(r['group_id']) == s]) for s in ('before', 'after')},
                              failed_doors=[x for s in SPLITS[:5] for x in ev[s]['failed_doors']],
                              localisation_test_role=pck([None] * 0) if False else dict(points=sum(ev[s]['localisation_test_role']['points'] for s in SPLITS[:5]),
                                                                                         pck8_mean=float(np.mean([ev[s]['localisation_test_role']['pck8'] for s in SPLITS[:5]]))),
                              differs_from_learned=sum(ev[s]['differs_from_learned'] for s in SPLITS[:5])),
               temporal=dict(doors=t['doors'], doors_wilson=wilson(t['doors']['correct'], t['doors']['n']), frames=t['frames'], failed_doors=t['failed_doors'],
                             localisation_test_role=t['localisation_test_role'], selected=t['selected']),
               selected={s: sel[s]['selected'] for s in SPLITS}, validation={s: sel[s]['candidates'] for s in SPLITS},
               seconds_per_image_single_thread=float(np.mean([ev[s]['seconds_per_image_single_thread'] for s in SPLITS])))
    dump(OUT / 'collected.json', out)
    print(json.dumps({k: out[k] for k in ('selected', 'seconds_per_image_single_thread')}, ensure_ascii=False))
    print('five-fold doors', d, 'frames', f, '| temporal doors', t['doors'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('cmd', choices=('select', 'detect', 'eval', 'collect')); ap.add_argument('--split', choices=SPLITS)
    ap.add_argument('--procs', type=int, default=6)
    x = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if x.cmd == 'select':
        select(x.split)
    elif x.cmd == 'detect':
        run_detect(x.split, x.procs)
    elif x.cmd == 'eval':
        evaluate(x.split)
    else:
        collect()
