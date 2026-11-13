"""Extension v16w (review round 5, bundle E, alternative): outer-width baseline registered in
config/extension_v16w_protocol.json. CPU only; no hole detection and no point annotations.

  measure    W_px of every candidate on all frames of the snapshot (cache/extension_v16w/features.jsonl)
  evaluate   selection on VAL, decisions on TEST for variants B and A, the unseen-class test of variant A,
             reports/extension_v16w/{predictions_*.jsonl, collected.json}
"""
import argparse
import collections
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
from door8.metrics import wilson
import run_extension_v5 as V5
import run_extension_v12 as V12

PROTOCOL = ROOT / 'config/extension_v16w_protocol.json'
OUT = ROOT / 'reports/extension_v16w'
CACHE = ROOT / 'cache/extension_v16w'
SPLITS = ('0', '1', '2', '3', '4', 'temporal')
CANDIDATES = [dict(k=k, c=c) for k in (1.0, .8) for c in (15, 41)]          # listed order of the protocol
NOMINAL_MM = 1450.0; REJECT_MM = 30.0; CHANGE_DATE = '20260921'


def tag(c):
    return f"k={c['k']} c={c['c']}"


def width(g, k, c):
    """Median row extent (px) of the largest bright component in the middle half of its height; None when there is no usable component."""
    b = cv2.GaussianBlur(g, (0, 0), 2); t, _ = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (b > k * t).astype(np.uint8); mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((c, c), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    if stats[i, cv2.CC_STAT_AREA] < .05 * g.size:
        return None
    x0, y0, w, h = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    ya, yb = y0 + int(round(.25 * h)), y0 + int(round(.75 * h))
    if yb - ya < 20:
        return None
    sub = lab[ya:yb, x0:x0 + w] == i; rows = sub.any(1)
    if rows.sum() < 20:
        return None
    sub = sub[rows]; first = sub.argmax(1); last = sub.shape[1] - 1 - sub[:, ::-1].argmax(1)
    return float(np.median(last - first + 1))


def _one(row):
    cv2.setNumThreads(1)
    g = cv2.imread(str(row['rgb']), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Unreadable image {row['rgb']}")
    t = time.perf_counter(); w = {tag(c): width(g, c['k'], c['c']) for c in CANDIDATES}; dt = (time.perf_counter() - t) / len(CANDIDATES)
    return dict(frame_id=row['frame_id'], width_px=w, seconds_per_candidate=dt)


def measure(procs):
    sp, _, frames, splits = snapshot(); parts = partition(frames, splits, '0')
    rows = {r['frame_id']: r for part in ('train', 'val', 'test') for r in parts[part]}
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / 'features.jsonl').exists():
        raise FileExistsError(CACHE / 'features.jsonl')
    t0 = time.perf_counter()
    with Pool(procs) as pool:
        feats = pool.map(_one, list(rows.values()), chunksize=16)
    jsonl(CACHE / 'features.jsonl', feats)
    dump(CACHE / 'features.json', dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), n=len(feats), rows_sha256=sha256(CACHE / 'features.jsonl'),
                                       wall_seconds=time.perf_counter() - t0, seconds_per_image_single_thread=float(np.mean([f['seconds_per_candidate'] for f in feats])),
                                       measured={tag(c): sum(f['width_px'][tag(c)] is not None for f in feats) for c in CANDIDATES}))
    print(len(feats), 'frames', round(time.perf_counter() - t0), 's wall', read_json(CACHE / 'features.json')['measured'])


# ---------------------------------------------------------------- decisions
def session_medians(rows, x):
    """{session: (class, median of x over its measured frames)}"""
    per = collections.defaultdict(list); cls = {}
    for r in rows:
        v = x.get(r['frame_id'])
        if v is not None:
            per[r['group_id']].append(v); cls[r['group_id']] = r['class_name']
    return {g: (cls[g], float(np.median(v))) for g, v in per.items()}


def fit_refs(train, x):
    by = collections.defaultdict(list)
    for c, m in session_medians(train, x).values():
        by[c].append(m)
    return {c: float(np.median(v)) for c, v in by.items()}


def fit_line(train, x, cad):
    sm = list(session_medians(train, x).values()); xs = np.array([m for _, m in sm]); ys = np.array([cad[c] for c, _ in sm], float)
    a, b = np.polyfit(xs, ys, 1); res = ys - (a * xs + b)
    return float(a), float(b), float(np.sqrt(np.mean(res ** 2))), float(np.abs(res).max())


def decide(rows, value, refs):
    """value: {frame_id: number in mm or None}; refs: {class: mm}. Nearest reference, rejected beyond 30 mm."""
    names = list(refs); vals = np.array([refs[n] for n in names]); out = []
    for r in rows:
        v = value.get(r['frame_id']); pred = None; why = 'no_measurement'
        if v is not None:
            d = np.abs(vals - v); i = int(d.argmin()); pred, why = (names[i], 'ok') if d[i] <= REJECT_MM else (None, 'reject_30mm')
        out.append({k: r[k] for k in ('frame_id', 'group_id', 'class_name')} | dict(pred=pred, measurement=v, reason=why))
    return out


def val_key(preds):
    d = V12.door_counts(preds); f = V12.frame_counts(preds)
    return d['correct'] / max(d['n'], 1), f['correct'] / max(f['n'], 1)


def state(g):
    return 'before' if g[:8] < CHANGE_DATE else 'after'


def summarise(preds):
    d = V12.door_counts(preds); f = V12.frame_counts(preds); v = V5.vote(preds)
    cls = collections.Counter((x['true'], 'correct' if x['pred'] == x['true'] else 'rejected' if x['pred'] is None else 'wrong') for x in v.values())
    return dict(doors=d, doors_wilson=wilson(d['correct'], d['n']), frames=f, image_accuracy=f['correct'] / f['n'], image_coverage=1 - f['abstained'] / f['n'],
                by_class={c: {k: cls[(c, k)] for k in ('correct', 'rejected', 'wrong')} for c in sorted({x['true'] for x in v.values()})},
                by_state={s: V12.door_counts([r for r in preds if state(r['group_id']) == s]) for s in ('before', 'after')},
                wrong_as=dict(collections.Counter(f"{x['true']}->{x['pred']}" for x in v.values() if x['pred'] not in (None, x['true']))))


def evaluate():
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    fm = read_json(CACHE / 'features.json')
    if fm['protocol_sha256'] != sha256(PROTOCOL) or sha256(CACHE / 'features.jsonl') != fm['rows_sha256']:
        raise ValueError('Features or protocol changed')
    feats = {r['frame_id']: r for r in read_jsonl(CACHE / 'features.jsonl')}; fx = {r['frame_id']: r['intrinsics']['fx'] for r in frames}
    norm = {tag(c): {f: (None if r['width_px'][tag(c)] is None else r['width_px'][tag(c)] / fx[f]) for f, r in feats.items()} for c in CANDIDATES}   # W_px / fx
    mm = {t: {f: (None if v is None else NOMINAL_MM * v) for f, v in x.items()} for t, x in norm.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), seconds_per_image_single_thread=fm['seconds_per_image_single_thread'], measured=fm['measured'],
               variant_B={}, variant_A={}, selection={})
    oof = dict(B=[], A=[]); learned_oof = []
    for split in SPLITS:
        parts = partition(frames, splits, split); sel = dict(B=[], A=[])
        for c in CANDIDATES:
            t = tag(c)
            sel['B'].append(val_key(decide(parts['val'], mm[t], fit_refs(parts['train'], mm[t]))))
            a, b, _, _ = fit_line(parts['train'], norm[t], cad)
            sel['A'].append(val_key(decide(parts['val'], {f: (None if v is None else a * v + b) for f, v in norm[t].items()}, cad)))
        learned = list(read_jsonl(V5.test_prediction_file(split, 'pixel'))); lv = V5.vote(learned)
        res['selection'][split] = {}
        for var in ('B', 'A'):
            i = max(range(len(CANDIDATES)), key=lambda j: (sel[var][j][0], sel[var][j][1], -j)); t = tag(CANDIDATES[i])
            res['selection'][split][var] = dict(selected=t, validation={tag(c): dict(door_accuracy=s[0], image_accuracy=s[1]) for c, s in zip(CANDIDATES, sel[var])})
            if var == 'B':
                refs = fit_refs(parts['train'], mm[t]); preds = decide(parts['test'], mm[t], refs); sm = session_medians(parts['train'], mm[t])
                sd = {c: float(np.std([m for cc, m in sm.values() if cc == c])) for c in refs}; order = sorted(refs.values())
                extra = dict(reference_mm=refs, smallest_reference_gap_mm=float(np.min(np.diff(order))), within_class_sd_mm=sd, classes_without_reference=sorted(set(cad) - set(refs)))
            else:
                a, b, rms, mx = fit_line(parts['train'], norm[t], cad)
                preds = decide(parts['test'], {f: (None if v is None else a * v + b) for f, v in norm[t].items()}, cad)
                extra = dict(a=a, b=b, train_session_residual_rms_mm=rms, train_session_residual_max_mm=mx)
            for r in preds:
                r.update(split=split, variant=var, candidate=t)
            jsonl(OUT / f'predictions_{var}_{split}.jsonl', preds); v = V5.vote(preds)
            s = summarise(preds); s.update(selected=t, differs_from_learned=sum(v[g]['pred'] != lv[g]['pred'] for g in v), learned_doors=V12.door_counts(learned), **extra)
            res['variant_' + var][split] = s
            if split != 'temporal':
                oof[var] += preds
            print(split, var, t, 'doors', s['doors'], 'frames', s['frames'])
        if split != 'temporal':
            learned_oof += learned
    for var in ('B', 'A'):
        s = summarise(oof[var]); v = V5.vote(oof[var]); lv = V5.vote(learned_oof)
        s['differs_from_learned'] = sum(v[g]['pred'] != lv[g]['pred'] for g in v); res['variant_' + var]['five_fold'] = s
        print('five-fold', var, 'doors', s['doors'], 'frames', s['frames'], 'before/after', s['by_state'])

    # ---- unseen-class test of variant A (folds 0-4)
    un = dict(per_run={}, unknown=collections.Counter(), known=collections.Counter(), new=collections.Counter(), other=collections.Counter(), accepted_as=collections.Counter(),
              new_wrong_as=collections.Counter())
    for split in SPLITS[:5]:
        parts = partition(frames, splits, split)
        for c in cfg['classes']:
            tr = [r for r in parts['train'] if r['class_name'] != c]; va = [r for r in parts['val'] if r['class_name'] != c]; cad7 = {k: v for k, v in cad.items() if k != c}
            keys = []
            for cand in CANDIDATES:
                t = tag(cand); a, b, _, _ = fit_line(tr, norm[t], cad7)
                keys.append(val_key(decide(va, {f: (None if v is None else a * v + b) for f, v in norm[t].items()}, cad7)))
            i = max(range(len(CANDIDATES)), key=lambda j: (keys[j][0], keys[j][1], -j)); t = tag(CANDIDATES[i]); a, b, rms, _ = fit_line(tr, norm[t], cad7)
            dist = {f: (None if v is None else a * v + b) for f, v in norm[t].items()}
            v7 = V5.vote(decide(parts['test'], dist, cad7)); v8 = V5.vote(decide(parts['test'], dist, cad))
            unk = [x for x in v7.values() if x['true'] == c]; kn = [x for x in v7.values() if x['true'] != c]
            new = [x for x in v8.values() if x['true'] == c]; oth = [x for x in v8.values() if x['true'] != c]
            cnt = lambda xs: dict(n=len(xs), correct=sum(x['pred'] == x['true'] for x in xs), rejected=sum(x['pred'] is None for x in xs),
                                  wrong=sum(x['pred'] not in (None, x['true']) for x in xs))
            run = dict(selected=t, a=a, b=b, train_session_residual_rms_mm=rms, unknown_rejected=sum(x['pred'] is None for x in unk), unknown_n=len(unk), known=cnt(kn), new=cnt(new), other=cnt(oth))
            un['per_run'][f'{split}:{c}'] = run
            un['unknown'].update(rejected=run['unknown_rejected'], n=len(unk))
            for key, d in (('known', run['known']), ('new', run['new']), ('other', run['other'])):
                un[key].update(d)
            un['accepted_as'].update(f"{c}->{x['pred']}" for x in unk if x['pred'] is not None)
            un['new_wrong_as'].update(f"{c}->{x['pred']}" for x in new if x['pred'] not in (None, c))
    res['variant_A_unseen'] = dict(unknown=dict(un['unknown']), known=dict(un['known']), new=dict(un['new']), other=dict(un['other']), accepted_as=dict(un['accepted_as']),
                                   new_wrong_as=dict(un['new_wrong_as']),
                                   per_class={c: dict(unknown_rejected=sum(un['per_run'][f'{s}:{c}']['unknown_rejected'] for s in SPLITS[:5]),
                                                      unknown_n=sum(un['per_run'][f'{s}:{c}']['unknown_n'] for s in SPLITS[:5]),
                                                      new_correct=sum(un['per_run'][f'{s}:{c}']['new']['correct'] for s in SPLITS[:5])) for c in cfg['classes']},
                                   per_run=un['per_run'])
    print('A unseen: unknown', dict(un['unknown']), 'known', dict(un['known']), 'new', dict(un['new']), 'other', dict(un['other']))
    dump(OUT / 'collected.json', res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('cmd', choices=('measure', 'evaluate')); ap.add_argument('--procs', type=int, default=6)
    x = ap.parse_args()
    measure(x.procs) if x.cmd == 'measure' else evaluate()
