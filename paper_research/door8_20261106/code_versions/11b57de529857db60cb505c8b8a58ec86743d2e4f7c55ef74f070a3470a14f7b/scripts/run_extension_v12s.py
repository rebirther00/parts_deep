"""Seed repetition of the unseen-class condition, registered in config/extension_v12s_protocol.json.

The training and evaluation code of extension_v5 / v6 / v12 is reused unchanged; only the run names, the registered
protocol file and the output directories are rebound for the new seed.

  train-hole --exclude CLASS --seed S     strict hole detector (run_extension_v6.train_strict)
  eval-hole  --exclude CLASS --seed S     rejection with CAD minus c (run_extension_v6.eval_strict)
  train-cnn  --kind K --exclude CLASS --seed S   7-output CNN (run_extension_v5.train_cnn_loco)
  eval-cnn   --kind K --exclude CLASS --seed S   legacy evaluation + float32 forward for the four scores
  analyze    --seed S                     CNN scores (unseen), hole rejection pooled, hole registration by CAD
  collect                                 reports/extension_v12s/collected.json (seeds 42, 123, 456)
  list-jobs                               job chains for the queue
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from door8.core import setup_runtime
setup_runtime()

from door8.core import ROOT, dump, jsonl, now, read_json, sha256, snapshot
import run_extension_v5 as V5
import run_extension_v6 as V6
import run_extension_v12 as V12

PROTOCOL = ROOT / 'config/extension_v12s_protocol.json'
OUT = ROOT / 'reports/extension_v12s'
CACHE = ROOT / 'cache/extension_v12s'
NEW_SEEDS = (123, 456)
KINDS = V12.KINDS
SCORES = V12.SCORES


def classes8():
    sp, _, _, _ = snapshot()
    return read_json(sp / 'protocol.json')['classes']


def hole_name(c, seed):
    return f'holes_strict_loco_{c}_f0_s{seed}'


def cnn_name(kind, c, seed):
    return f'{kind}_loco_{c}_f0_s{seed}'


def bind(seed):
    """Point the reused modules at this protocol, this seed's run names and this seed's output directories."""
    if seed not in NEW_SEEDS:
        raise ValueError('Seed not registered')
    out = OUT / f's{seed}'; out.mkdir(parents=True, exist_ok=True)
    V5.PROTOCOL = V6.PROTOCOL = V12.PROTOCOL = PROTOCOL
    V5.OUT = V6.OUT = V12.OUT = out; V12.CACHE = CACHE / f's{seed}'
    V6.strict_run = lambda c: hole_name(c, seed)
    V12.cnn_runs = lambda: [dict(run=cnn_name(kind, c, seed), group='unseen', kind=kind, fold='0', exclude=c)
                            for kind in KINDS for c in classes8()]
    V12.stored_test_file = lambda run, group: out / f'{run}_predictions.jsonl'
    return out


def analyze(seed):
    out = bind(seed); sp, frames, splits, cfg = V12.ctx(); cl8 = cfg['classes']; info = {r['frame_id']: r for r in frames}
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), seed=seed, cnn={}, cnn_legacy={}, consistency={}); dumped = []
    for kind in KINDS:                                              # same computation as the unseen block of V12.cnn_analyze
        runs = {k: {} for k in SCORES}; legacy = {}
        for c in cl8:
            run = cnn_name(kind, c, seed); meta, a = V12.load_cnn(run); cl = meta['classes']; allowed = list(range(len(cl)))
            res['consistency'][run] = meta['consistency']
            va = V12.scored_rows(a, 'val', cl, allowed, info); te = V12.scored_rows(a, 'test', cl, allowed, info)
            for k in SCORES:
                t, acc = V12.threshold(va[k]); m = V5.loco_metrics(te[k], V12.apply(te[k], t), c, score=lambda r: r['score'])
                m.update(threshold=t, val_acceptance=acc); runs[k][c] = m
                dumped += [dict(r, kind=kind, excluded=c, score_name=k, accepted=r['score'] >= t, condition='unseen', seed=seed) for r in te[k]]
            legacy[c] = read_json(out / f'{run}_metrics.json')
        res['cnn'][kind] = {k: dict(pooled=V5.pool_runs(runs[k]),
                                    per_class={c: dict(unknown_doors=[m['unknown_doors']['rejected'], m['unknown_doors']['n']],
                                                       accepted_as=m['unknown_doors']['accepted_as'], known=m['known_doors_threshold'],
                                                       auroc=m['auroc_frame'], val_acceptance=m['val_acceptance']) for c, m in runs[k].items()})
                            for k in SCORES}
        res['cnn_legacy'][kind] = dict(pooled=V5.pool_runs(legacy),
                                       per_class={c: dict(unknown_doors=[m['unknown_doors']['rejected'], m['unknown_doors']['n']],
                                                          accepted_as=m['unknown_doors']['accepted_as'], known=m['known_doors_threshold'],
                                                          auroc=m['auroc_frame'], best_epoch=m.get('best_epoch')) for c, m in legacy.items()})
    strict = {c: read_json(out / f'{hole_name(c, seed)}_metrics.json') for c in cl8}
    res['hole_reject'] = dict(pooled=V5.pool_runs(strict),
                              per_class={c: dict(unknown_doors=[m['unknown_doors']['rejected'], m['unknown_doors']['n']],
                                                 accepted_as=m['unknown_doors']['accepted_as'], known=m['known_doors_threshold'],
                                                 auroc=m['auroc_frame'], min_distance_mm=m['unknown_min_distance_mm'], best_epoch=m['best_epoch'],
                                                 localization=m['test_role_localization']['all']) for c, m in strict.items()})
    V12.hole_register()                                             # writes out/hole_register.json with this seed's strict detectors
    res['hole_register'] = read_json(out / 'hole_register.json')
    dump(out / 'analysis.json', res); jsonl(out / 'cnn_unseen_predictions.jsonl', dumped)
    print(json.dumps(dict(seed=seed, hole_reject=res['hole_reject']['pooled'], hole_register=res['hole_register']['pooled'],
                          cnn={kind: {k: res['cnn'][kind][k]['pooled'] for k in SCORES} for kind in KINDS}), indent=1))


def summary_row(p):
    k = p['known_door_decisions']
    return dict(unknown_doors=p['unknown_doors'], wilson=p['unknown_doors_wilson'], known=[k['correct'], k['abstained'], k['wrong']],
                auroc_mean=p['auroc_mean'], auroc_min=p['auroc_min'])


def collect():
    v6 = read_json(ROOT / 'reports/extension_v6/collected.json'); v12 = read_json(ROOT / 'reports/extension_v12/collected.json')
    per = {42: dict(hole_reject=summary_row(v6['unseen']['pooled']['hole']), hole_register=v12['hole_register']['pooled'],
                    cnn={kind: {k: summary_row(v12['cnn_scores']['unseen'][kind][k]['pooled']) for k in SCORES} for kind in KINDS},
                    cnn_legacy={kind: summary_row(v6['unseen']['pooled'][kind]) for kind in KINDS})}
    missing = []
    for s in NEW_SEEDS:
        p = OUT / f's{s}' / 'analysis.json'
        if not p.exists():
            missing.append(s); continue
        a = read_json(p)
        if a['protocol_sha256'] != sha256(PROTOCOL):
            raise ValueError('Protocol changed')
        per[s] = dict(hole_reject=summary_row(a['hole_reject']['pooled']), hole_register=a['hole_register']['pooled'],
                      cnn={kind: {k: summary_row(a['cnn'][kind][k]['pooled']) for k in SCORES} for kind in KINDS},
                      cnn_legacy={kind: summary_row(a['cnn_legacy'][kind]['pooled']) for kind in KINDS},
                      consistency=dict(test_frames=sum(v['test']['n'] for v in a['consistency'].values()),
                                       test_argmax_mismatch=sum(v['test']['argmax_mismatch'] for v in a['consistency'].values()),
                                       val_frames=sum(v['val']['n'] for v in a['consistency'].values()),
                                       val_argmax_mismatch=sum(v['val']['argmax_mismatch'] for v in a['consistency'].values())))
    seeds = sorted(per)

    def across(get):
        rows = [get(per[s]) for s in seeds]
        return dict(seeds=seeds, rejected=[r['unknown_doors'][0] for r in rows], n=rows[0]['unknown_doors'][1],
                    known=[r['known'] for r in rows], auroc_mean=[r['auroc_mean'] for r in rows], auroc_min=[r['auroc_min'] for r in rows],
                    rejected_range=[min(r['unknown_doors'][0] for r in rows), max(r['unknown_doors'][0] for r in rows)],
                    known_abstained_range=[min(r['known'][1] for r in rows), max(r['known'][1] for r in rows)],
                    known_wrong_range=[min(r['known'][2] for r in rows), max(r['known'][2] for r in rows)],
                    auroc_mean_range=[min(r['auroc_mean'] for r in rows), max(r['auroc_mean'] for r in rows)],
                    auroc_min_overall=min(r['auroc_min'] for r in rows))
    table = dict(hole=across(lambda d: d['hole_reject']))
    for kind in KINDS:
        for k in SCORES:
            table[f'{kind}:{k}'] = across(lambda d, kind=kind, k=k: d['cnn'][kind][k])
        table[f'{kind}:legacy_float32_softmax'] = across(lambda d, kind=kind: d['cnn_legacy'][kind])
    reg = dict(seeds=seeds, new_doors=[per[s]['hole_register']['new_doors'] for s in seeds],
               new_frames=[per[s]['hole_register']['new_frames'] for s in seeds],
               other_doors=[per[s]['hole_register']['other_doors'] for s in seeds],
               max_utilisation=[per[s]['hole_register']['max_utilisation'] for s in seeds])
    out = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), missing_seeds=missing, per_seed={str(s): per[s] for s in seeds},
               unseen_rejection=table, registration_by_cad=reg)
    dump(OUT / 'collected.json', out)
    print('missing', missing)
    for name, t in table.items():
        print(f"{name:28s} rejected {t['rejected']}/{t['n']}  known c/a/w {t['known']}  auroc mean {[round(x, 3) for x in t['auroc_mean']]} min {round(t['auroc_min_overall'], 3)}")
    print('register', [(d['correct'], d['abstained'], d['wrong']) for d in reg['new_doors']], 'other', [(d['correct'], d['abstained'], d['wrong']) for d in reg['other_doors']],
          'maxU', [round(x, 3) for x in reg['max_utilisation']])


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('train-hole', 'eval-hole'):
        a = sub.add_parser(name); a.add_argument('--exclude', required=True); a.add_argument('--seed', type=int, required=True)
    for name in ('train-cnn', 'eval-cnn'):
        a = sub.add_parser(name); a.add_argument('--kind', choices=KINDS, required=True); a.add_argument('--exclude', required=True)
        a.add_argument('--seed', type=int, required=True)
    a = sub.add_parser('analyze'); a.add_argument('--seed', type=int, required=True)
    sub.add_parser('collect'); sub.add_parser('list-jobs')
    x = ap.parse_args()
    if x.cmd == 'list-jobs':
        for s in NEW_SEEDS:
            for c in classes8():
                print(f'hole {c} {s}'); print(f'rgb {c} {s}'); print(f'rgbe {c} {s}')
        return
    if x.cmd == 'collect':
        return collect()
    bind(x.seed)
    if x.cmd == 'train-hole':
        V6.train_strict(x.exclude, seed=x.seed)
    elif x.cmd == 'eval-hole':
        V6.eval_strict(x.exclude)
    elif x.cmd == 'train-cnn':
        V5.train_cnn_loco(x.kind, x.exclude, cnn_name(x.kind, x.exclude, x.seed), seed=x.seed, workers=4)
    elif x.cmd == 'eval-cnn':
        run = cnn_name(x.kind, x.exclude, x.seed); V5.eval_cnn_loco(run); V12.cnn_infer(run)
    else:
        analyze(x.seed)


if __name__ == '__main__':
    main()
