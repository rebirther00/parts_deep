"""Extension v17 (review round 6, check 1): checkpoint sensitivity of the fold-1 unseen-condition CNNs, registered in
config/extension_v17_protocol.json.

The 16 fold-1 CNNs of extension_v16 are trained again with the same recipe, inputs and seed, but the patience rule does
not stop the training (40 epochs). Two checkpoints of each training run are kept: RULE (what the registered rule selects
in this run) and LAST (epoch 40). Both are evaluated exactly as the stored runs.

  prepare                                   write the source difference (reports/extension_v17/code_patches.diff)
  train   --kind K --exclude C              one training run -> runs/*_v17last and runs/*_v17rule
  eval    --kind K --exclude C --ckpt X     legacy evaluation + re-extracted logits / features (X = rule | last)
  analyze                                   STORED / RULE / LAST tabulation -> reports/extension_v17/analysis.json
  list-jobs                                 job list for the queue
"""
import argparse
import difflib
import inspect
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from door8.core import setup_runtime
setup_runtime()

from door8.core import ROOT, dump, now, read_json, require_fresh_dir, sha256, snapshot
import run_extension_v5 as V5
TRAIN_SRC = inspect.getsource(V5.train_cnn_loco)                      # the file's source, before run_extension_v16 rebinds it
import run_extension_v12 as V12
import run_extension_v16 as X
import run_extension_v16c as C

PROTOCOL = ROOT / 'config/extension_v17_protocol.json'
OUT = ROOT / 'reports/extension_v17'
CACHE = ROOT / 'cache/extension_v17'
FOLD = '1'
KINDS = V12.KINDS
SCORES = V12.SCORES
CKPTS = ('rule', 'last')
REPS = [("'0'", '_FOLD', 2),                                             # fold-parametrised, as in extension_v16
        ("wait = 0; t0 = time.perf_counter()", "wait = 0; stopped = False; t0 = time.perf_counter()", 1),
        ("if score > best:", "if score > best and not stopped:", 1),
        ("torch.save(net.state_dict(), outpath(runpath / 'model.pth')); jsonl(runpath / 'val_predictions.jsonl', pred)",
         "torch.save(net.state_dict(), outpath(runpath / 'model_rule.pth')); jsonl(runpath / 'val_predictions_rule.jsonl', pred)", 1),
        ("meta.update(best_epoch=ep + 1, best_val_session_balanced_accuracy=score[0])",
         "meta.update(rule_epoch=ep + 1, rule_val_session_balanced_accuracy=score[0])", 1),
        ("if wait >= hp['patience']:\n            break",
         "if wait >= hp['patience'] and not stopped:\n            stopped = True; meta.update(rule_stop_epoch=ep + 1)", 1),
        ("    meta.update(status='complete', finished_at=now()",
         "    torch.save(net.state_dict(), outpath(runpath / 'model.pth')); jsonl(runpath / 'val_predictions.jsonl', pred)\n"
         "    meta.update(checkpoint='last_epoch', checkpoint_epoch=len(hist), checkpoint_val_session_balanced_accuracy=hist[-1]['val_session_balanced_accuracy'],\n"
         "                rule_weights_sha256=sha256(runpath / 'model_rule.pth'))\n"
         "    meta.update(status='complete', finished_at=now()", 1)]


def patched_train_source():
    new = TRAIN_SRC
    for old, rep, n in REPS:
        if new.count(old) != n:
            raise ValueError(f'train_cnn_loco: expected {n} x {old!r}')
        new = new.replace(old, rep)
    diff = '\n'.join(difflib.unified_diff(TRAIN_SRC.splitlines(), new.splitlines(), 'run_extension_v5.train_cnn_loco',
                                          'run_extension_v5.train_cnn_loco (v17: no early stop, RULE and LAST checkpoints)', lineterm='')) + '\n'
    return new, diff


NEW_SRC, DIFF = patched_train_source()
exec(compile(NEW_SRC, '<run_extension_v5.train_cnn_loco v17>', 'exec'), V5.__dict__)


def name(kind, c, ckpt):
    return f'{kind}_loco_{c}_f{FOLD}_s42_v17{ckpt}'


def bind17():
    """extension_v16 binding of fold 1, then this protocol, these run names and these output directories."""
    out = X.bind(FOLD, out=OUT / f'f{FOLD}')
    V5.PROTOCOL = V12.PROTOCOL = PROTOCOL; V12.CACHE = CACHE / f'f{FOLD}'
    V12.cnn_runs = lambda: [dict(run=name(kind, c, ck), group='unseen', kind=kind, fold=FOLD, exclude=c) for ck in CKPTS for kind in KINDS for c in X.classes8()]
    V12.stored_test_file = lambda run, group: out / f'{run}_predictions.jsonl'
    return out


def prepare():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'code_patches.diff').write_text(DIFF, encoding='utf-8')
    print(DIFF)


def train(kind, c):
    bind17(); last = name(kind, c, 'last')
    V5.train_cnn_loco(kind, c, last, seed=42, workers=4)
    src = ROOT / 'runs' / last; meta = read_json(src / 'run.json')
    if meta['status'] != 'complete' or meta['checkpoint_epoch'] != meta['requested_epochs']:
        raise ValueError('Training did not reach the last epoch')
    dst = require_fresh_dir('runs/' + name(kind, c, 'rule'))               # RULE checkpoint of the same training run as its own run directory
    shutil.copyfile(src / 'model_rule.pth', dst / 'model.pth'); shutil.copyfile(src / 'val_predictions_rule.jsonl', dst / 'val_predictions.jsonl')
    shutil.copyfile(src / 'history.json', dst / 'history.json')
    meta.update(checkpoint='registered_rule_in_rerun', checkpoint_epoch=meta['rule_epoch'], checkpoint_val_session_balanced_accuracy=meta['rule_val_session_balanced_accuracy'],
                best_epoch=meta['rule_epoch'], best_val_session_balanced_accuracy=meta['rule_val_session_balanced_accuracy'],
                weights_sha256=sha256(dst / 'model.pth'), same_training_run_as=last)
    if meta['weights_sha256'] != meta['rule_weights_sha256']:
        raise ValueError('RULE weights changed while copying')
    dump(dst / 'run.json', meta)
    print(json.dumps(dict(run=last, rule_epoch=meta['rule_epoch'], rule_stop_epoch=meta.get('rule_stop_epoch'), last_epoch=meta['requested_epochs'])), flush=True)


def evaluate(kind, c, ckpt):
    bind17(); run = name(kind, c, ckpt)
    V5.eval_cnn_loco(run); V12.cnn_infer(run)


def variant_rows(variant, kind, c, info):
    """Scored TEST / VAL rows and the run metadata of one model: STORED (extension_v16), RULE or LAST (this extension)."""
    if variant == 'stored':
        X.bind(FOLD); run = X.cnn_name(kind, c, FOLD)
    else:
        bind17(); run = name(kind, c, variant)
    meta, a = V12.load_cnn(run); cl = meta['classes']; allowed = list(range(len(cl)))
    return run, meta, V12.scored_rows(a, 'test', cl, allowed, info), V12.scored_rows(a, 'val', cl, allowed, info)


def analyze():
    sp, _, frames, _ = snapshot(); cl8 = read_json(sp / 'protocol.json')['classes']; info = {r['frame_id']: r for r in frames}
    res = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), fold=FOLD, seed=42, variants={}, per_run={}, consistency={})
    for variant in ('stored',) + CKPTS:
        res['variants'][variant] = {}
        for kind in KINDS:
            runs = {k: {} for k in SCORES}; points = {k: {} for k in SCORES}; closed = {k: [0, 0] for k in SCORES}
            for c in cl8:
                run, meta, te, va = variant_rows(variant, kind, c, info); rm = read_json(ROOT / 'runs' / run / 'run.json')
                hist = read_json(ROOT / 'runs' / run / 'history.json'); res['consistency'][run] = meta['consistency']
                res['per_run'].setdefault(f'{kind}:{c}', {})[variant] = dict(
                    run=run, checkpoint_epoch=rm.get('checkpoint_epoch', rm.get('best_epoch')), epochs_trained=len(hist), rule_stop_epoch=rm.get('rule_stop_epoch'),
                    val_session_balanced_accuracy=rm.get('checkpoint_val_session_balanced_accuracy', rm.get('best_val_session_balanced_accuracy')))
                for k in SCORES:
                    t, acc = V12.threshold(va[k]); m = V5.loco_metrics(te[k], V12.apply(te[k], t), c, score=lambda r: r['score'])
                    m.update(threshold=t, val_acceptance=acc); runs[k][c] = m
                    p = C.run_points(te[k], va[k], c, t); p['all_known_doors'] = sorted({x['group_id'] for x in te[k] if x['class_name'] != c}); points[k][c] = p
                    if p['manuscript']['unknown_rejected'] != m['unknown_doors']['rejected'] or p['manuscript']['known_rejected'] != m['known_doors_threshold']['abstained']:
                        raise ValueError('The two tabulation paths disagree')
                    closed[k][0] += m['known_doors_closed']['correct']; closed[k][1] += m['known_doors_closed']['n']
            res['variants'][variant][kind] = {
                k: dict(pooled=V5.pool_runs(runs[k]), closed_set_registered_doors=closed[k], door_points=C.pool_points(points[k]),
                        per_class={c: dict(unknown_doors=[m['unknown_doors']['rejected'], m['unknown_doors']['n']], accepted_as=m['unknown_doors']['accepted_as'],
                                           known=m['known_doors_threshold'], auroc=m['auroc_frame'], door_auroc=points[k][c]['door_auroc'],
                                           validation_door=[points[k][c]['validation_door']['unknown_rejected'], points[k][c]['validation_door']['known_rejected'],
                                                            points[k][c]['validation_door']['known_wrong']]) for c, m in runs[k].items()})
                for k in SCORES}
    maha = {v: {kind: res['variants'][v][kind]['maha']['pooled']['unknown_doors'][0] for kind in KINDS} for v in res['variants']}
    n = res['variants']['last'][KINDS[0]]['maha']['pooled']['unknown_doors'][1]
    if all(abs(maha['last'][kind] - maha['rule'][kind]) <= 5 and maha['last'][kind] < n for kind in KINDS):
        case = 'a'
    elif all(maha['last'][kind] == n for kind in KINDS):
        case = 'b'
    else:
        case = 'c'
    res['claim_rule'] = dict(mahalanobis_unregistered_doors_rejected=maha, doors=n, case=case)
    dump(OUT / 'analysis.json', res)
    for variant, d in res['variants'].items():
        for kind in KINDS:
            for k in SCORES:
                x = d[kind][k]; kd = x['pooled']['known_door_decisions']; vd = x['door_points']['validation_door']
                print(f"{variant:6s} {kind:4s} {k:8s} unknown {x['pooled']['unknown_doors']} known [{kd['correct']}, {kd['abstained']}, {kd['wrong']}] closed {x['closed_set_registered_doors']}"
                      f" auroc {x['pooled']['auroc_mean']:.3f} ({x['pooled']['auroc_min']:.3f}) door auroc {x['door_points']['door_auroc_mean']:.3f} | val-door {vd['unknown_rejected']} {vd['known']}")
    print('claim rule', json.dumps(res['claim_rule']))
    for key, v in res['per_run'].items():
        print(key, {a: (b['checkpoint_epoch'], b['epochs_trained'], round(b['val_session_balanced_accuracy'], 4)) for a, b in v.items()},
              {a: res['variants'][a][key.split(':')[0]]['maha']['per_class'][key.split(':')[1]]['unknown_doors'] for a in res['variants']})


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    for cmd in ('train', 'eval'):
        a = sub.add_parser(cmd); a.add_argument('--kind', choices=KINDS, required=True); a.add_argument('--exclude', required=True)
        if cmd == 'eval':
            a.add_argument('--ckpt', choices=CKPTS, required=True)
    sub.add_parser('prepare'); sub.add_parser('analyze'); sub.add_parser('list-jobs')
    x = ap.parse_args()
    if x.cmd == 'prepare':
        prepare()
    elif x.cmd == 'list-jobs':
        for kind in KINDS:
            for c in X.classes8():
                print(kind, c)
    elif x.cmd == 'train':
        train(x.kind, x.exclude)
    elif x.cmd == 'eval':
        evaluate(x.kind, x.exclude, x.ckpt)
    else:
        analyze()


if __name__ == '__main__':
    main()
