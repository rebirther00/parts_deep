"""Extension v16, bundles C and D summary (config/extension_v16_protocol.json). Re-tabulation of stored outputs only.

  margins   C1  reject-distance interval and per-class rejection margins (unseen condition)
  points    C2  door-level operating points: manuscript point, validation-door threshold, post-hoc point; door AUROC
  pooled    D   unseen condition and CAD registration pooled over the five folds (seed 42)
  collect       reports/extension_v16/collected.json

Scopes: fold0_s42, fold0_s123, fold0_s456 (eight runs each) and allfolds_s42 (forty runs; needs the v16 queue).
"""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from door8.core import setup_runtime
setup_runtime()

import numpy as np

from door8.core import ROOT, dump, now, read_json, read_jsonl, sha256, snapshot
from door8.metrics import wilson
import run_extension_v5 as V5
import run_extension_v12 as V12
import run_extension_v12s as V12S
import run_extension_v16 as X

PROTOCOL = X.PROTOCOL; OUT = X.OUT
ORIG = dict(PROTOCOL=V12.PROTOCOL, OUT=V12.OUT, CACHE=V12.CACHE)          # extension_v12 binding (fold 0, seed 42)
KINDS = V12.KINDS; SCORES = V12.SCORES
NEG = float('-inf')


def ctx():
    sp, _, frames, _ = snapshot(); cfg = read_json(sp / 'protocol.json')
    return cfg, {r['frame_id']: r for r in frames}


def scopes(want_all=True):
    s = {f'fold0_s{seed}': [('0', seed)] for seed in (42, 123, 456)}
    if want_all:
        s['allfolds_s42'] = [(k, 42) for k in ('0', '1', '2', '3', '4')]
    return s


def hole_files(c, fold, seed):
    run = X.hole_name(c, fold, seed)
    if fold == '0':
        test = ROOT / (f'reports/extension_v6/{run}_predictions.jsonl' if seed == 42 else f'reports/extension_v12s/s{seed}/{run}_predictions.jsonl')
        metrics = test.with_name(f'{run}_metrics.json'); val = OUT / f'f0_s{seed}' / f'{run}_val_predictions.jsonl'
    else:
        test = OUT / f'f{fold}' / f'{run}_predictions.jsonl'; metrics = OUT / f'f{fold}' / f'{run}_metrics.json'; val = OUT / f'f{fold}' / f'{run}_val_predictions.jsonl'
    return test, metrics, val


def hole_rows(c, fold, seed, cfg):
    """TEST and VAL rows of a strict run with a threshold-free prediction (nearest registered class) and score = -nearest distance."""
    g = V5.cfg_without(cfg, c); cad7 = dict(zip(g['classes'], g['cad_hole_distance_mm'])); test, _, val = hole_files(c, fold, seed)

    def conv(r):
        v = r['measurement']
        if v is None:
            return dict(frame_id=r['frame_id'], group_id=r['group_id'], class_name=r['class_name'], pred=None, score=NEG, measurement=None)
        k = min(cad7, key=lambda n: abs(cad7[n] - v))
        return dict(frame_id=r['frame_id'], group_id=r['group_id'], class_name=r['class_name'], pred=k, score=-abs(cad7[k] - v), measurement=v)
    return [conv(r) for r in read_jsonl(test)], ([conv(r) for r in read_jsonl(val)] if val.exists() else None)


def bind_cnn(fold, seed):
    if fold == '0' and seed == 42:
        V12.PROTOCOL, V12.OUT, V12.CACHE = ORIG['PROTOCOL'], ORIG['OUT'], ORIG['CACHE']; return lambda kind, c: f'{kind}_loco_{c}_f0_s42'
    if fold == '0':
        V12S.bind(seed); return lambda kind, c: V12S.cnn_name(kind, c, seed)
    X.bind(fold); return lambda kind, c: X.cnn_name(kind, c, fold)


def cnn_rows(kind, c, fold, seed, info):
    name = bind_cnn(fold, seed); meta, a = V12.load_cnn(name(kind, c)); cl = meta['classes']; allowed = list(range(len(cl)))
    return V12.scored_rows(a, 'test', cl, allowed, info), V12.scored_rows(a, 'val', cl, allowed, info)


def door_scores(rows):
    d = collections.defaultdict(lambda: NEG)
    for r in rows:
        d[r['group_id']] = max(d[r['group_id']], r['score'])
    return dict(d)


def at(rows, t, c):
    """Counts at image threshold t with the session vote of the manuscript."""
    opened = [dict(r, pred=r['pred'] if r['score'] >= t and r['score'] > NEG else None) for r in rows]
    v = V5.vote(opened); unk = {g: x for g, x in v.items() if x['true'] == c}; kn = {g: x for g, x in v.items() if x['true'] != c}
    return dict(unknown_rejected=sum(x['pred'] is None for x in unk.values()), unknown_n=len(unk), known_correct=sum(x['pred'] == x['true'] for x in kn.values()),
                known_rejected=sum(x['pred'] is None for x in kn.values()), known_wrong=sum(x['pred'] not in (None, x['true']) for x in kn.values()), known_n=len(kn),
                rejected_known_doors=sorted(g for g, x in kn.items() if x['pred'] is None))


def run_points(test, val, c, t_manuscript):
    ds = door_scores(test); cls = {r['group_id']: r['class_name'] for r in test}
    kn = [s for g, s in ds.items() if cls[g] != c]; un = [s for g, s in ds.items() if cls[g] == c]
    res = dict(manuscript=dict(threshold=t_manuscript, **at(test, t_manuscript, c)), door_auroc=V5.auroc(kn, un),
               known_doors_without_score=sum(s == NEG for s in kn))
    finite = [s for s in kn if s > NEG]; t_test = min(finite)
    res['posthoc'] = dict(threshold=t_test, **at(test, t_test, c))
    if val is not None:
        vs = [s for s in door_scores(val).values()]; fin = [s for s in vs if s > NEG]; t_val = min(fin)
        res['validation_door'] = dict(threshold=t_val, validation_doors=len(vs), validation_doors_without_score=len(vs) - len(fin), **at(test, t_val, c))
    return res


def pool_points(runs, n_boot=2000):
    """runs: {(fold, class): run_points(...)}; pooled counts, distinct rejected doors and a door-level bootstrap interval."""
    out = dict(door_auroc_mean=float(np.mean([r['door_auroc'] for r in runs.values()])), door_auroc_min=float(np.min([r['door_auroc'] for r in runs.values()])))
    for op in ('manuscript', 'validation_door', 'posthoc'):
        if not all(op in r for r in runs.values()):
            continue
        tot = collections.Counter(); rej = collections.Counter(); reg = collections.Counter()
        for r in runs.values():
            p = r[op]
            for k in ('unknown_rejected', 'unknown_n', 'known_correct', 'known_rejected', 'known_wrong', 'known_n'):
                tot[k] += p[k]
            for g in p['rejected_known_doors']:
                rej[g] += 1
        doors = sorted({g for r in runs.values() for g in r['all_known_doors']})
        for r in runs.values():
            for g in r['all_known_doors']:
                reg[g] += 1
        a = np.array([rej[g] for g in doors], float); b = np.array([reg[g] for g in doors], float)
        rng = np.random.default_rng(20261112); idx = rng.integers(0, len(doors), size=(n_boot, len(doors)))
        boot = a[idx].sum(1) / b[idx].sum(1)
        out[op] = dict(unknown_rejected=[tot['unknown_rejected'], tot['unknown_n']], known=[tot['known_correct'], tot['known_rejected'], tot['known_wrong']],
                       known_n=tot['known_n'], known_rejection_rate=tot['known_rejected'] / tot['known_n'],
                       known_rejection_rate_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                       distinct_known_doors=len(doors), distinct_known_doors_rejected_at_least_once=len(rej),
                       thresholds=[float(r[op]['threshold']) for r in runs.values()])
    return out


def points(want_all):
    cfg, info = ctx(); res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), scopes={})
    for scope, runs in scopes(want_all).items():
        hole = {}; cnn = {(kind, k): {} for kind in KINDS for k in SCORES}; missing_val = 0
        for fold, seed in runs:
            for c in cfg['classes']:
                te, va = hole_rows(c, fold, seed, cfg); missing_val += va is None
                r = run_points(te, va, c, -30.0); r['all_known_doors'] = sorted({x['group_id'] for x in te if x['class_name'] != c}); hole[(fold, c)] = r
                for kind in KINDS:
                    te_k, va_k = cnn_rows(kind, c, fold, seed, info)
                    for k in SCORES:
                        t_img, _ = V12.threshold(va_k[k]); r = run_points(te_k[k], va_k[k], c, t_img)
                        r['all_known_doors'] = sorted({x['group_id'] for x in te_k[k] if x['class_name'] != c}); cnn[(kind, k)][(fold, c)] = r
        s = dict(runs=len(hole), hole=pool_points(hole), hole_validation_predictions_missing=missing_val)
        if 'validation_door' in s['hole']:
            s['hole']['validation_tau_mm'] = [-t for t in s['hole']['validation_door']['thresholds']]
        s['hole']['posthoc_tau_mm'] = [-t for t in s['hole']['posthoc']['thresholds']]
        for (kind, k), rr in cnn.items():
            s[f'{kind}:{k}'] = pool_points(rr)
        res['scopes'][scope] = s
        print(scope, 'hole', {op: (s['hole'][op]['unknown_rejected'], s['hole'][op]['known']) for op in ('manuscript', 'validation_door', 'posthoc') if op in s['hole']},
              'door AUROC', round(s['hole']['door_auroc_mean'], 4))
        for kind in KINDS:
            for k in SCORES:
                p = s[f'{kind}:{k}']
                print('   ', kind, k, {op: (p[op]['unknown_rejected'], p[op]['known']) for op in ('manuscript', 'validation_door', 'posthoc')}, 'door AUROC', round(p['door_auroc_mean'], 3), round(p['door_auroc_min'], 3))
    dump(OUT / ('c2_operating_points.json' if want_all else 'c2_operating_points_fold0.json'), res)


def margins(want_all):
    cfg, _ = ctx(); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), scopes={})
    nominal = {c: min(abs(cad[c] - cad[o]) for o in cad if o != c) for c in cad}
    for scope, runs in scopes(want_all).items():
        per_run = {}; per_class = collections.defaultdict(list)
        for fold, seed in runs:
            for c in cfg['classes']:
                te, _ = hole_rows(c, fold, seed, cfg)
                reg = [r for r in te if r['class_name'] != c]; unr = [r for r in te if r['class_name'] == c]
                a = max(abs(r['measurement'] - cad[r['class_name']]) for r in reg if r['measurement'] is not None)
                a_near = max(-r['score'] for r in reg if r['measurement'] is not None)
                b = min(-r['score'] for r in unr if r['measurement'] is not None)
                per_run[f'{fold}:{c}'] = dict(registered_max_abs_residual_mm=a, registered_max_nearest_mm=a_near, registered_unmeasured=sum(r['measurement'] is None for r in reg),
                                              registered_images=len(reg), unregistered_min_distance_mm=b, unregistered_unmeasured=sum(r['measurement'] is None for r in unr),
                                              unregistered_images=len(unr))
                per_class[c].append(b)
        amax = max(v['registered_max_abs_residual_mm'] for v in per_run.values()); bmin = min(v['unregistered_min_distance_mm'] for v in per_run.values())
        res['scopes'][scope] = dict(runs=len(per_run), registered_max_abs_residual_mm=amax, unregistered_min_distance_mm=bmin, tau_interval_mm=[amax, bmin],
                                    tau_30_inside=amax <= 30 < bmin, registered_unmeasured=sum(v['registered_unmeasured'] for v in per_run.values()),
                                    unregistered_unmeasured=sum(v['unregistered_unmeasured'] for v in per_run.values()),
                                    per_class={c: dict(cad_mm=cad[c], gap_to_nearest_cad_mm=nominal[c], nominal_rejection_margin_mm=nominal[c] - 30,
                                                       observed_min_distance_mm=min(v), observed_rejection_margin_mm=min(v) - 30) for c, v in per_class.items()},
                                    per_run=per_run)
        print(scope, 'tau interval', round(amax, 1), '-', round(bmin, 1), {c.replace('_door', ''): round(min(v) - 30, 1) for c, v in per_class.items()})
    dump(OUT / ('c1_rejection_margin.json' if want_all else 'c1_rejection_margin_fold0.json'), res)


def pooled():
    cfg, _ = ctx(); cl8 = cfg['classes']; v12 = read_json(ROOT / 'reports/extension_v12/collected.json'); res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL))
    strict = {}; reg = {}; ana = {}
    for fold in ('0', '1', '2', '3', '4'):
        for c in cl8:
            strict[(fold, c)] = read_json(hole_files(c, fold, 42)[1])
        reg[fold] = read_json(ROOT / 'reports/extension_v12/hole_register.json' if fold == '0' else OUT / f'f{fold}' / 'hole_register.json')
        if fold != '0':
            ana[fold] = read_json(OUT / f'f{fold}' / 'analysis.json')
    res['hole_reject'] = dict(pooled=V5.pool_runs({f'{f}:{c}': m for (f, c), m in strict.items()}),
                              per_fold={f: V5.pool_runs({c: strict[(f, c)] for c in cl8}) for f in ('0', '1', '2', '3', '4')},
                              per_class={c: dict(unknown_doors=[sum(strict[(f, c)]['unknown_doors']['rejected'] for f in '01234'), sum(strict[(f, c)]['unknown_doors']['n'] for f in '01234')],
                                                 accepted_as=dict(sum((collections.Counter(strict[(f, c)]['unknown_doors']['accepted_as']) for f in '01234'), collections.Counter())),
                                                 min_distance_mm=min(strict[(f, c)]['unknown_min_distance_mm'] for f in '01234'),
                                                 known=[sum(strict[(f, c)]['known_doors_threshold'][k] for f in '01234') for k in ('correct', 'abstained', 'wrong')]) for c in cl8},
                              failures=[dict(fold=f, removed=c, accepted_as=m['unknown_doors']['accepted_as'], min_distance_mm=m['unknown_min_distance_mm'])
                                        for (f, c), m in strict.items() if m['unknown_doors']['rejected'] != m['unknown_doors']['n']],
                              localisation=dict(test_role_points=sum(m['test_role_localization']['all']['visible_points'] for m in strict.values()),
                                                within_8px=sum(round(m['test_role_localization']['all']['pck8'] * m['test_role_localization']['all']['visible_points']) for m in strict.values()),
                                                excluded_class_points=sum(len(m['excluded_class_test_role_points']) for m in strict.values()),
                                                excluded_class_median_px=float(np.median([p['error_px'] for m in strict.values() for p in m['excluded_class_test_role_points'] if p['error_px'] is not None])),
                                                excluded_class_max_px=float(max(p['error_px'] for m in strict.values() for p in m['excluded_class_test_role_points'] if p['error_px'] is not None))))
    tot = lambda key: {k: sum(reg[f]['pooled'][key][k] for f in reg) for k in ('n', 'correct', 'abstained', 'wrong')}
    nd = tot('new_doors')
    res['hole_register'] = dict(new_doors=nd, new_doors_wilson=wilson(nd['correct'], nd['n']), new_frames=tot('new_frames'), other_doors=tot('other_doors'),
                                max_utilisation=max(reg[f]['pooled']['max_utilisation'] for f in reg), max_utilisation_per_fold={f: reg[f]['pooled']['max_utilisation'] for f in reg},
                                per_class={c: dict(new_doors=[sum(reg[f]['per_class'][c]['new_doors'][k] for f in reg) for k in ('correct', 'abstained', 'wrong')],
                                                   max_utilisation=max(reg[f]['per_class'][c]['max_utilisation'] for f in reg),
                                                   wrong_as=dict(sum((collections.Counter(reg[f]['per_class'][c]['new_wrong_as']) for f in reg), collections.Counter()))) for c in cl8},
                                scale_range=[min(list(reg[f]['per_class'][c]['scale'].values())[0] for f in reg for c in cl8), max(list(reg[f]['per_class'][c]['scale'].values())[0] for f in reg for c in cl8)])
    res['cnn'] = {}
    for kind in KINDS:
        for k in SCORES:
            pc = {('0', c): v for c, v in v12['cnn_scores']['unseen'][kind][k]['per_class'].items()}
            for f in ana:
                pc.update({(f, c): v for c, v in ana[f]['cnn'][kind][k]['per_class'].items()})
            known = lambda v, key: v['known'][key]
            res['cnn'][f'{kind}:{k}'] = dict(unknown_doors=[sum(v['unknown_doors'][0] for v in pc.values()), sum(v['unknown_doors'][1] for v in pc.values())],
                                             known=[sum(known(v, key) for v in pc.values()) for key in ('correct', 'abstained', 'wrong')],
                                             auroc_mean=float(np.mean([v['auroc'] for v in pc.values()])), auroc_min=float(np.min([v['auroc'] for v in pc.values()])),
                                             per_fold_unknown={f: [sum(v['unknown_doors'][0] for (ff, _), v in pc.items() if ff == f), sum(v['unknown_doors'][1] for (ff, _), v in pc.items() if ff == f)] for f in '01234'},
                                             per_class_unknown={c: [sum(v['unknown_doors'][0] for (_, cc), v in pc.items() if cc == c), sum(v['unknown_doors'][1] for (_, cc), v in pc.items() if cc == c)] for c in cl8},
                                             accepted_as=dict(sum((collections.Counter({f'{c}->{d}': n for d, n in v['accepted_as'].items()}) for (_, c), v in pc.items()), collections.Counter())))
    res['consistency'] = {f: dict(test_frames=sum(v['test']['n'] for v in ana[f]['consistency'].values()), test_argmax_mismatch=sum(v['test']['argmax_mismatch'] for v in ana[f]['consistency'].values()),
                                  val_frames=sum(v['val']['n'] for v in ana[f]['consistency'].values()), val_argmax_mismatch=sum(v['val']['argmax_mismatch'] for v in ana[f]['consistency'].values())) for f in ana}
    dump(OUT / 'd_pooled.json', res)
    print('hole reject', res['hole_reject']['pooled']); print('failures', res['hole_reject']['failures'])
    print('register', res['hole_register']['new_doors'], res['hole_register']['other_doors'], 'max U', round(res['hole_register']['max_utilisation'], 3))
    for name, v in res['cnn'].items():
        print(f"{name:14s} unknown {v['unknown_doors']} known {v['known']} auroc {v['auroc_mean']:.3f} ({v['auroc_min']:.3f}) per fold {v['per_fold_unknown']}")


def collect():
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL))
    for key, f in (('d_pooled', 'd_pooled.json'), ('c1', 'c1_rejection_margin.json'), ('c2', 'c2_operating_points.json'), ('production_plan', 'production_plan_check.json')):
        d = read_json(OUT / f)
        if d['protocol_sha256'] != sha256(PROTOCOL):
            raise ValueError(f'Protocol changed: {f}')
        if key == 'c1':
            d = dict(scopes={s: {k: v for k, v in x.items() if k != 'per_run'} for s, x in d['scopes'].items()})
        out[key] = d
    dump(OUT / 'collected.json', out); print('collected')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('cmd', choices=('margins', 'points', 'pooled', 'collect')); ap.add_argument('--fold0-only', action='store_true')
    x = ap.parse_args()
    dict(margins=lambda: margins(not x.fold0_only), points=lambda: points(not x.fold0_only), pooled=pooled, collect=collect)[x.cmd]()
