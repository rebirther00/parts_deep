"""Extension v18 (review round 7 and the external review of manuscript v17): post-hoc diagnostics registered in
config/extension_v18_protocol.json. CPU only. Nothing of extension_v16e / v16w is overwritten.

  same-hole                      T0: same-hole detections in the stored detections of extension_v16e
  select  --variant V --split S  validation PCK8 of every candidate, choice of one (no TEST data)
  detect  --variant V --split S  corner detections of the chosen candidate on the split's TRAIN and TEST frames
  eval    --variant V --split S  S fitted on TRAIN, decisions on TEST, localisation, comparison with the learned detector
  collect --variant V            reports/extension_v18/<V>/collected.json
  plane                          W0: plane distance in the stored features of the learned detectors
  width                          W1: outer-width baseline (variant B) with the TRAIN-only constant

V = gap     T1, primary: the pair of positions with the largest response sum at >= 500 px horizontal distance
    perkey  T2, secondary: one search box per corner-hole key, maxima taken independently
    gapfree T1b, addendum config/extension_v18b_protocol.json: the detections of 'gap'; the validation score that selects the
            candidate uses the better of the two assignments of the detected points to the annotated holes

  loc-free --variant V           test-role localisation of a variant, key-wise and for the better assignment (addendum)
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
from door8.geometry import classify_measurement, fit_scale
from door8.metrics import wilson
import run_extension_v5 as V5
import run_extension_v12 as V12
import run_extension_v16e as E
import run_extension_v16w as W

PROTOCOL = ROOT / 'config/extension_v18_protocol.json'
PROTOCOL_B = ROOT / 'config/extension_v18b_protocol.json'      # addendum: variant gapfree, loc-free
OUT = ROOT / 'reports/extension_v18'
CACHE = ROOT / 'cache/extension_v18'
SPLITS = E.SPLITS; KEYS = E.KEYS; CANDIDATES = E.CANDIDATES
VARIANTS = ('gap', 'perkey', 'gapfree')
GAP = 500            # px, horizontal; the separation of the registered LoG candidates
SAME_HOLE = 100      # px; two detections closer than this are one hole (review round 7)


# ---------------------------------------------------------------- detection
def best_pair(a, b, gap):
    """(i, j) maximising a[i] + b[j] subject to |i - j| >= gap; None when the arrays are not longer than gap."""
    n = len(a)
    if n <= gap:
        return None
    idx = np.arange(n)

    def prefix(v):
        m = np.maximum.accumulate(v); return m, np.maximum.accumulate(np.where(v >= m, idx, 0))
    ma, ia = prefix(a); mb, ib = prefix(b)
    s1 = ma[:n - gap] + b[gap:]          # i <= j - gap
    s2 = mb[:n - gap] + a[gap:]          # j <= i - gap
    k1, k2 = int(s1.argmax()), int(s2.argmax())
    return (int(ia[k1]), k1 + gap) if s1[k1] >= s2[k2] else (k2 + gap, int(ib[k2]))


def response(g, box, ts, r):
    """Maximum over the templates of the normalised cross-correlation; offsets that turn a map index into an image coordinate."""
    x0, y0, x1, y1 = box
    xa, ya, xb, yb = max(x0 - r, 0), max(y0 - r, 0), min(x1 + r, g.shape[1]), min(y1 + r, g.shape[0])     # as run_extension_v16e.detect_ncc
    roi = g[ya:yb, xa:xb]; best = None
    for t in ts:
        res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED); best = res if best is None else np.maximum(best, res)
    return best, xa + r, ya + r


def detect_gap(g, reg, sets, r):
    maps = {}
    for k in KEYS:
        m, ox, oy = response(g, reg, sets[k], r); maps[k] = np.nan_to_num(m, nan=-1., posinf=-1., neginf=-1.)
    pair = best_pair(maps[KEYS[0]].max(0), maps[KEYS[1]].max(0), GAP)
    if pair is None:
        return {}
    out = {}
    for k, x in zip(KEYS, pair):
        y = int(maps[k][:, x].argmax()); out[k] = (float(x + ox), float(y + oy), float(maps[k][y, x]))
    return out


def detect_perkey(g, regs, sets, r):
    out = {}
    for k in KEYS:
        m, ox, oy = response(g, regs[k], sets[k], r); _, score, _, loc = cv2.minMaxLoc(m)
        out[k] = (float(loc[0] + ox), float(loc[1] + oy), float(score))
    return out


def region_key(train, m, shape, k):
    pts = np.array([r['points'][k] for r in train], float); h, w = shape
    x0, y0 = np.floor(pts.min(0) - m).astype(int); x1, y1 = np.ceil(pts.max(0) + m).astype(int)
    return int(max(x0, 0)), int(max(y0, 0)), int(min(x1, w)), int(min(y1, h))


def setup(variant, cand, tr, shape):
    reg = E.region(tr, cand['m'], shape)
    if cand['kind'] == 'log':
        return dict(reg=reg, sets=None, regs=None)
    return dict(reg=reg, sets=E.templates(tr, cand['r']), regs={k: region_key(tr, cand['m'], shape, k) for k in KEYS} if variant == 'perkey' else None)


def detect(g, variant, cand, s):
    if cand['kind'] == 'log':
        return E.detect_log(g, s['reg'], cand['sigma'])
    return detect_perkey(g, s['regs'], s['sets'], cand['r']) if variant == 'perkey' else detect_gap(g, s['reg'], s['sets'], cand['r'])


def proto(variant):
    return PROTOCOL_B if variant == 'gapfree' else PROTOCOL


def errors_free(det, pts):
    """Errors (px) of the two corner points for the better of the two assignments (as run_extension_v16e.errors does for LoG)."""
    gt = [np.array(pts[k], float) for k in KEYS]
    if not det:
        return [None, None]
    a, b = (np.array(p[:2]) for p in points(det))
    e1 = [np.linalg.norm(a - gt[0]), np.linalg.norm(b - gt[1])]; e2 = [np.linalg.norm(b - gt[0]), np.linalg.norm(a - gt[1])]
    return [float(x) for x in (e1 if sum(e1) <= sum(e2) else e2)]


def points(det):
    return list(det.values()) if 'a' in det else [det[k] for k in KEYS]


def gap_px(det):
    p = points(det); return float(np.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]))


# ---------------------------------------------------------------- T0
def same_hole():
    sp, _, frames, splits = snapshot(); out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), threshold_px=SAME_HOLE, splits={}); tot = collections.Counter()
    for split in SPLITS:
        parts = partition(frames, splits, split); pool = {r['frame_id']: r for r in read_jsonl(E.CACHE / split / 'features.jsonl')}; res = {}
        for role in ('train', 'test'):
            g = {r['frame_id']: (gap_px(pool[r['frame_id']]['detection']) if pool[r['frame_id']]['detection'] else None) for r in parts[role]}
            same = {f: v is not None and v < SAME_HOLE for f, v in g.items()}; per = collections.defaultdict(list)
            for r in parts[role]:
                per[r['group_id']].append(same[r['frame_id']])
            res[role] = dict(images=len(g), no_detection=sum(v is None for v in g.values()), same_hole=sum(same.values()), share=sum(same.values()) / len(g),
                             doors=len(per), doors_majority_same_hole=sum(sum(v) > len(v) / 2 for v in per.values()))
            if role == 'test':
                key = 'temporal' if split == 'temporal' else 'five_fold'
                tot.update({f'{key}_images': len(g), f'{key}_same_hole': sum(same.values()), f'{key}_doors': len(per),
                            f'{key}_doors_majority': res[role]['doors_majority_same_hole']})
        out['splits'][split] = res; print(split, {k: (v['same_hole'], v['images'], round(v['share'], 3), v['doors_majority_same_hole'], v['doors']) for k, v in res.items()})
    out['test_totals'] = {k: dict(images=tot[f'{k}_images'], same_hole=tot[f'{k}_same_hole'], share=tot[f'{k}_same_hole'] / tot[f'{k}_images'],
                                  doors=tot[f'{k}_doors'], doors_majority_same_hole=tot[f'{k}_doors_majority']) for k in ('five_fold', 'temporal')}
    print(out['test_totals']); OUT.mkdir(parents=True, exist_ok=True); dump(OUT / 'same_hole.json', out)


# ---------------------------------------------------------------- T1 / T2
def select(variant, split):
    tr, va, _ = E.supervision(split); shape = E.gray(tr[0]['rgb']).shape; rows = []; err = errors_free if variant == 'gapfree' else E.errors
    for cand in CANDIDATES:
        s = setup(variant, cand, tr, shape); errs = []
        for row in va:
            errs += err(detect(E.gray(row['rgb']), variant, cand, s), row['points'])
        rows.append(dict(candidate=E.tag(cand), region=s['reg'], regions_per_key=s['regs'], **E.pck(errs)))
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]['pck8'], rows[i]['median_px'] if rows[i]['median_px'] is not None else 1e9, i))
    out = dict(created_at=now(), protocol_sha256=sha256(proto(variant)), variant=variant, split=split, base_run=E.base_run(split), train_images=len(tr), val_images=len(va),
               candidates=rows, selected=rows[order[0]]['candidate'], selected_index=order[0])
    (OUT / variant).mkdir(parents=True, exist_ok=True); dump(OUT / variant / f'select_{split}.json', out)
    print(variant, split, 'selected', out['selected'], [(r['candidate'], round(r['pck8'], 3), None if r['median_px'] is None else round(r['median_px'], 1)) for r in rows])


_W = {}


def _init(variant, split):
    cv2.setNumThreads(1)
    sel = read_json(OUT / variant / f'select_{split}.json'); cand = CANDIDATES[sel['selected_index']]; tr, _, _ = E.supervision(split)
    _W.update(variant=variant, cand=cand, s=setup(variant, cand, tr, E.gray(tr[0]['rgb']).shape))


def _one(row):
    g = E.gray(row['rgb']); t = time.perf_counter(); det = detect(g, _W['variant'], _W['cand'], _W['s']); dt = time.perf_counter() - t
    f = dict(frame_id=row['frame_id'], gate='missing_corner', span_normalized=None, detection=det, seconds=dt)
    if det:
        f.update(gate='ok', span_normalized=gap_px(det) / row['intrinsics']['fx'])
    return f


def run_detect(variant, split, procs):
    sp, _, frames, splits = snapshot(); parts = partition(frames, splits, split); rows = parts['train'] + parts['test']
    dst = CACHE / variant / split; dst.mkdir(parents=True, exist_ok=True)
    if (dst / 'features.jsonl').exists():
        raise FileExistsError(dst / 'features.jsonl')
    t0 = time.perf_counter()
    with Pool(procs, initializer=_init, initargs=(variant, split)) as pool:
        feats = pool.map(_one, rows, chunksize=16)
    jsonl(dst / 'features.jsonl', feats)
    dump(dst / 'features.json', dict(created_at=now(), protocol_sha256=sha256(proto(variant)), variant=variant, split=split, n=len(feats), rows_sha256=sha256(dst / 'features.jsonl'),
                                     selected=read_json(OUT / variant / f'select_{split}.json')['selected'], wall_seconds=time.perf_counter() - t0,
                                     seconds_per_image_single_thread=float(np.mean([f['seconds'] for f in feats]))))
    print(variant, split, len(feats), 'frames', round(time.perf_counter() - t0), 's wall; detection', round(1000 * float(np.mean([f['seconds'] for f in feats])), 1), 'ms/image')


def judge(rows, pool, cal, cfg, split):
    preds = []
    for row in rows:
        p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg)
        preds.append({k: row[k] for k in ('frame_id', 'group_id', 'class_name')} | dict(pred=p, measurement=v, reason=why, split=split))
    return preds


def evaluate(variant, split):
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); parts = partition(frames, splits, split)
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); band = V5.bands(cad); src = CACHE / variant / split; dst = OUT / variant
    fm = read_json(src / 'features.json')
    if fm['protocol_sha256'] != sha256(proto(variant)) or sha256(src / 'features.jsonl') != fm['rows_sha256']:
        raise ValueError('Features or protocol changed')
    pool = {r['frame_id']: r for r in read_jsonl(src / 'features.jsonl')}
    cal = fit_scale(parts['train'], pool, cad, 'pixel'); preds = judge(parts['test'], pool, cal, cfg, split)
    jsonl(dst / f'predictions_{split}.jsonl', preds)
    cal_l = fit_scale(parts['train'], V5.hole_pool(split), cad, 'pixel'); preds_l = judge(parts['test'], pool, cal_l, cfg, split)          # supplement: S of the learned detector
    jsonl(dst / f'predictions_learnedS_{split}.jsonl', preds_l)
    _, _, test_labels = E.supervision(split); errs = []
    for lab in test_labels:
        errs += E.errors(pool[lab['frame_id']]['detection'], lab['points'])
    learned = {r['frame_id']: r for r in read_jsonl(V5.test_prediction_file(split, 'pixel'))}
    votes = V5.vote(preds); lv = V5.vote(list(learned.values()))
    same = lambda rows: sum(bool(pool[r['frame_id']]['detection']) and gap_px(pool[r['frame_id']]['detection']) < SAME_HOLE for r in rows)
    res = dict(created_at=now(), protocol_sha256=sha256(proto(variant)), variant=variant, split=split, selected=fm['selected'], scale=cal['camera_scales'], scale_learned=cal_l['camera_scales'],
               seconds_per_image_single_thread=fm['seconds_per_image_single_thread'], doors=V12.door_counts(preds), frames=V12.frame_counts(preds),
               frame_reasons=dict(collections.Counter(r['reason'] for r in preds)), localisation_test_role=E.pck(errs),
               residual=V12.residual_stats(preds, cad, band) if any(r['measurement'] is not None for r in preds) else None,
               failed_doors=[dict(door=g, true=v['true'], pred=v['pred'], state=E.state(g)) for g, v in sorted(votes.items()) if v['pred'] != v['true']],
               by_state={s: V12.door_counts([r for r in preds if E.state(r['group_id']) == s]) for s in ('before', 'after')},
               differs_from_learned=sum(votes[g]['pred'] != lv[g]['pred'] for g in votes), learned_doors=V12.door_counts(list(learned.values())),
               same_hole=dict(train=same(parts['train']), train_images=len(parts['train']), test=same(parts['test']), test_images=len(parts['test'])),
               learned_scale=dict(doors=V12.door_counts(preds_l), frames=V12.frame_counts(preds_l)))
    dump(dst / f'eval_{split}.json', res)
    print(variant, split, res['selected'], 'S', {k: round(v, 1) for k, v in res['scale'].items()}, 'doors', res['doors'], 'frames', res['frames'], 'loc', res['localisation_test_role'],
          '| learned S doors', res['learned_scale']['doors'])


def collect(variant):
    dst = OUT / variant; ev = {s: read_json(dst / f'eval_{s}.json') for s in SPLITS}; sel = {s: read_json(dst / f'select_{s}.json') for s in SPLITS}
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); band = V5.bands(cad)
    oof = [r for s in SPLITS[:5] for r in read_jsonl(dst / f'predictions_{s}.jsonl')]; oof_l = [r for s in SPLITS[:5] for r in read_jsonl(dst / f'predictions_learnedS_{s}.jsonl')]
    d = V12.door_counts(oof); f = V12.frame_counts(oof); t = ev['temporal']; dl = V12.door_counts(oof_l); fl = V12.frame_counts(oof_l)
    cls = collections.Counter((v['true'], 'correct' if v['pred'] == v['true'] else 'rejected' if v['pred'] is None else 'wrong') for v in V5.vote(oof).values())
    out = dict(created_at=now(), protocol_sha256=sha256(proto(variant)), variant=variant,
               five_fold=dict(doors=d, doors_wilson=wilson(d['correct'], d['n']), frames=f, image_accuracy=f['correct'] / f['n'], image_coverage=1 - f['abstained'] / f['n'],
                              residual=V12.residual_stats(oof, cad, band) if any(r['measurement'] is not None for r in oof) else None,
                              by_class={c: {k: cls[(c, k)] for k in ('correct', 'rejected', 'wrong')} for c in cfg['classes']},
                              by_state={s: V12.door_counts([r for r in oof if E.state(r['group_id']) == s]) for s in ('before', 'after')},
                              failed_doors=[x for s in SPLITS[:5] for x in ev[s]['failed_doors']],
                              localisation_test_role=dict(points=sum(ev[s]['localisation_test_role']['points'] for s in SPLITS[:5]),
                                                          within_8px=sum(round(ev[s]['localisation_test_role']['pck8'] * ev[s]['localisation_test_role']['points']) for s in SPLITS[:5]),
                                                          max_px=max(ev[s]['localisation_test_role']['max_px'] for s in SPLITS[:5])),
                              differs_from_learned=sum(ev[s]['differs_from_learned'] for s in SPLITS[:5]),
                              same_hole_test=sum(ev[s]['same_hole']['test'] for s in SPLITS[:5])),
               temporal=dict(doors=t['doors'], doors_wilson=wilson(t['doors']['correct'], t['doors']['n']), frames=t['frames'], image_accuracy=t['frames']['correct'] / t['frames']['n'],
                             image_coverage=1 - t['frames']['abstained'] / t['frames']['n'], failed_doors=t['failed_doors'], localisation_test_role=t['localisation_test_role'],
                             selected=t['selected'], same_hole_test=t['same_hole']['test']),
               learned_scale=dict(five_fold=dict(doors=dl, doors_wilson=wilson(dl['correct'], dl['n']), frames=fl), temporal=t['learned_scale']),
               scale={s: ev[s]['scale'] for s in SPLITS}, scale_learned={s: ev[s]['scale_learned'] for s in SPLITS},
               selected={s: sel[s]['selected'] for s in SPLITS}, validation={s: sel[s]['candidates'] for s in SPLITS},
               seconds_per_image_single_thread=float(np.mean([ev[s]['seconds_per_image_single_thread'] for s in SPLITS])))
    dump(dst / 'collected.json', out)
    print(variant, json.dumps(out['selected'], ensure_ascii=False))
    print(variant, 'five-fold doors', d, 'frames', f, '| temporal doors', t['doors'], '| learned S: five-fold', dl, 'temporal', t['learned_scale']['doors'])


def loc_free(variant):
    """Test-role localisation from the stored detections: key-wise (as registered) and for the better assignment."""
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL_B), variant=variant, splits={}); tot = collections.Counter(); mx = 0.
    for split in SPLITS:
        pool = {r['frame_id']: r for r in read_jsonl(CACHE / variant / split / 'features.jsonl')}; _, _, labs = E.supervision(split); kw = []; fr = []
        for lab in labs:
            det = pool[lab['frame_id']]['detection']; kw += E.errors(det, lab['points']); fr += errors_free(det, lab['points'])
        out['splits'][split] = dict(key_wise=E.pck(kw), order_free=E.pck(fr))
        if split != 'temporal':
            tot.update(points=len(fr), key_wise_within_8px=sum(x is not None and x <= 8 for x in kw), order_free_within_8px=sum(x is not None and x <= 8 for x in fr))
            mx = max([mx] + [x for x in fr if x is not None])
        print(variant, split, 'key-wise', out['splits'][split]['key_wise'], 'order-free', out['splits'][split]['order_free'])
    out['five_fold'] = dict(tot, order_free_max_px=mx); print(variant, 'five-fold', out['five_fold'])
    dump(OUT / variant / 'localisation_order_free.json', out)


# ---------------------------------------------------------------- W0 / W1
def plane_values(rows, pool):
    """Frame values and session medians of plane_z_mm of the images with a fitted plane."""
    per = collections.defaultdict(list)
    for r in rows:
        z = pool[r['frame_id']].get('plane_z_mm')
        if z is not None and np.isfinite(z):
            per[r['group_id']].append(z)
    return [z for v in per.values() for z in v], {g: float(np.median(v)) for g, v in per.items()}


def summary(rows, pool):
    fr, se = plane_values(rows, pool)
    return dict(images=len(rows), images_with_plane=len(fr), sessions=len(se), median_images_mm=float(np.median(fr)), median_of_session_medians_mm=float(np.median(list(se.values()))),
                session_median_range_mm=[float(min(se.values())), float(max(se.values()))])


def plane():
    sp, _, frames, splits = snapshot(); out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), splits={}); oof_rows = []; oof_pool = {}
    for split in SPLITS:
        parts = partition(frames, splits, split); pool = V5.hole_pool(split)
        out['splits'][split] = dict(train=summary(parts['train'], pool), test=summary(parts['test'], pool))
        if split != 'temporal':
            oof_rows += parts['test']; oof_pool.update({r['frame_id']: pool[r['frame_id']] for r in parts['test']})
        print(split, 'train', out['splits'][split]['train']['median_of_session_medians_mm'], 'test', out['splits'][split]['test']['median_of_session_medians_mm'])
    out['five_fold_test'] = summary(oof_rows, oof_pool)
    out['five_fold_test_by_state'] = {s: summary([r for r in oof_rows if E.state(r['group_id']) == s], oof_pool) for s in ('before', 'after')}
    print('five-fold test', out['five_fold_test']); print({s: (v['median_images_mm'], v['median_of_session_medians_mm'], v['sessions']) for s, v in out['five_fold_test_by_state'].items()})
    OUT.mkdir(parents=True, exist_ok=True); dump(OUT / 'plane_distance.json', out)


def width():
    sp, _, frames, splits = snapshot()
    fm = read_json(W.CACHE / 'features.json')
    if sha256(W.CACHE / 'features.jsonl') != fm['rows_sha256']:
        raise ValueError('Width features changed')
    feats = {r['frame_id']: r for r in read_jsonl(W.CACHE / 'features.jsonl')}; fx = {r['frame_id']: r['intrinsics']['fx'] for r in frames}
    norm = {W.tag(c): {f: (None if r['width_px'][W.tag(c)] is None else r['width_px'][W.tag(c)] / fx[f]) for f, r in feats.items()} for c in W.CANDIDATES}
    reg = read_json(W.OUT / 'collected.json'); out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), registered_constant_mm=W.NOMINAL_MM, splits={}); oof = []; diff5 = 0
    for split in SPLITS:
        parts = partition(frames, splits, split); _, se = plane_values(parts['train'], V5.hole_pool(split)); const = float(np.median(list(se.values())))
        mm = {t: {f: (None if v is None else const * v) for f, v in x.items()} for t, x in norm.items()}
        keys = [W.val_key(W.decide(parts['val'], mm[W.tag(c)], W.fit_refs(parts['train'], mm[W.tag(c)]))) for c in W.CANDIDATES]
        i = max(range(len(W.CANDIDATES)), key=lambda j: (keys[j][0], keys[j][1], -j)); t = W.tag(W.CANDIDATES[i])
        preds = W.decide(parts['test'], mm[t], W.fit_refs(parts['train'], mm[t]))
        for r in preds:
            r.update(split=split, candidate=t)
        jsonl(OUT / f'predictions_width_train_constant_{split}.jsonl', preds)
        v = V5.vote(preds); rv = V5.vote(list(read_jsonl(W.OUT / f'predictions_B_{split}.jsonl'))); differ = sorted(g for g in v if v[g]['pred'] != rv[g]['pred'])
        s = W.summarise(preds)
        out['splits'][split] = dict(constant_mm=const, selected=t, registered_selected=reg['variant_B'][split]['selected'], doors=s['doors'], frames=s['frames'],
                                    registered_doors=reg['variant_B'][split]['doors'], doors_decision_differs=len(differ), differing_doors=differ)
        if split != 'temporal':
            oof += preds; diff5 += len(differ)
        print(split, 'const', round(const, 1), t, '(registered', reg['variant_B'][split]['selected'] + ')', 'doors', s['doors'], 'registered', reg['variant_B'][split]['doors'], 'differ', len(differ))
    s = W.summarise(oof)
    out['five_fold'] = dict(doors=s['doors'], doors_wilson=s['doors_wilson'], frames=s['frames'], image_accuracy=s['image_accuracy'], image_coverage=s['image_coverage'],
                            registered_doors=reg['variant_B']['five_fold']['doors'], doors_decision_differs=diff5)
    print('five-fold doors', s['doors'], 'registered', reg['variant_B']['five_fold']['doors'], 'differ', diff5)
    dump(OUT / 'outer_width_train_constant.json', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('same-hole', 'select', 'detect', 'eval', 'collect', 'loc-free', 'plane', 'width')); ap.add_argument('--variant', choices=VARIANTS)
    ap.add_argument('--split', choices=SPLITS); ap.add_argument('--procs', type=int, default=6)
    x = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if x.cmd == 'same-hole':
        same_hole()
    elif x.cmd == 'select':
        select(x.variant, x.split)
    elif x.cmd == 'detect':
        run_detect(x.variant, x.split, x.procs)
    elif x.cmd == 'eval':
        evaluate(x.variant, x.split)
    elif x.cmd == 'collect':
        collect(x.variant)
    elif x.cmd == 'loc-free':
        loc_free(x.variant)
    elif x.cmd == 'plane':
        plane()
    else:
        width()
