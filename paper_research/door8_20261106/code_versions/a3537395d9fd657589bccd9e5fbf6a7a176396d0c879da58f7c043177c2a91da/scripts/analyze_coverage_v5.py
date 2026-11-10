"""Exploratory coverage analysis for manuscript v5 (same numbers as v4; figure marks realizable operating points) (stored scores only; no model is run, no operating point is chosen).

Frame-level risk-coverage curves:
  CNN  : accept when softmax confidence >= t, sweep t over every stored confidence value.
  Hole : accept when the geometry gate passes and |D - nearest CAD| <= r, sweep r from 0 to 30 mm
         (30 mm is the fixed reject distance; frames failing the gate are never accepted).
Settings: five-fold OOF (361 doors), temporal split (133 doors), and fold-0 known test frames vs the six E23 doors.
Writes reports/coverage_v5/coverage.json and figures/v5_fig4_coverage(.png/.pdf), figures/v5c_fig4_coverage(.png/.pdf).
v5: dots = realizable operating points (ties enter together); thin lines are visual guides only.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures_v3 as base   # noqa: E402  (runtime, style, save)
import matplotlib.pyplot as plt
import numpy as np

from door8.core import ROOT, dump, now, read_json, read_jsonl, snapshot

HOLE, RGB, RGBE = base.BLUE, '#1baf7a', '#eda100'      # validated slots 1/3/4 (orange stays with depth in Fig. 3)
STYLE = {'Hole (pixel width)': (HOLE, '-'), 'RGB CNN': (RGB, '--'), 'RGBE CNN': (RGBE, ':')}
EXT, F0 = ROOT / 'reports/extension_v3', ROOT / 'reports/fold0_test_v3'
R_MAX = 30.0


def cnn_rows(kind, split):
    return read_jsonl(ROOT / f'reports/{kind}_{split}_s42_test/predictions.jsonl')


def hole_files(setting):
    if setting == 'oof':
        return [F0 / 'holes_field8_f0_s42_v2_pixel_test_predictions.jsonl'] + \
               [EXT / f'holes_field8_f{k}_s42_v3_pixel_predictions.jsonl' for k in (1, 2, 3, 4)]
    if setting == 'temporal':
        return [EXT / 'holes_field_temporal_s42_pixel_predictions.jsonl']
    return [F0 / 'holes_field8_f0_s42_v2_pixel_test_predictions.jsonl']


def cnn_curve(rows):
    """score = confidence (higher = accept first)."""
    s = np.array([r['confidence'] for r in rows]); ok = np.array([r['pred'] == r['class_name'] for r in rows])
    return curve(s, ok, np.ones(len(rows), bool), descending=True)


def hole_scores(rows, cad):
    vals = np.array(list(cad.values())); s, ok, valid = [], [], []
    for r in rows:
        m = r['measurement']
        if m is None or r['reason'] not in ('ok', 'cad_distance_reject'):
            s.append(np.inf); ok.append(False); valid.append(False); continue
        i = int(np.abs(vals - m).argmin()); s.append(abs(vals[i] - m))
        ok.append(list(cad)[i] == r.get('class_name')); valid.append(True)
    return np.array(s), np.array(ok), np.array(valid)


def curve(score, ok, valid, descending):
    """Accept-the-best-first sweep; returns coverage, risk (errors among accepted), and the score at each step."""
    order = np.argsort(-score if descending else score, kind='stable')
    sc, okk, vv = score[order], ok[order], valid[order]
    n = len(score); cov, risk, thr = [], [], []
    acc = wrong = 0; i = 0
    while i < n and vv[i]:
        j = i
        while j < n and vv[j] and sc[j] == sc[i]:      # ties enter together
            acc += 1; wrong += int(not okk[j]); j += 1
        cov.append(acc / n); risk.append(wrong / acc); thr.append(float(sc[i])); i = j
    return dict(coverage=np.array(cov), risk=np.array(risk), threshold=np.array(thr))


def summarize_curve(c, name, cap=None):
    cov, risk, thr = c['coverage'], c['risk'], c['threshold']
    if cap is not None:
        keep = thr <= cap; cov, risk, thr = cov[keep], risk[keep], thr[keep]
    zero = cov[risk == 0]
    return dict(method=name, max_coverage=float(cov[-1]), risk_at_max_coverage=float(risk[-1]),
                coverage_at_zero_risk=float(zero.max()) if len(zero) else 0.0, steps=len(cov))


def setting_curves(setting, cad):
    split = {'oof': [f'f{k}' for k in range(5)], 'temporal': ['temporal'], 'f0': ['f0']}[setting]
    out = {}
    hs, hok, hv = hole_scores([r for f in hole_files(setting) for r in read_jsonl(f)], cad)
    out['Hole (pixel width)'] = curve(hs, hok, hv, descending=False)
    for kind, name in (('rgb', 'RGB CNN'), ('rgbe', 'RGBE CNN')):
        out[name] = cnn_curve([r for s in split for r in cnn_rows(kind, s)])
    return out


def operating_points(setting):
    """Pre-specified points: hole r = 30 mm; CNN val-95% threshold of each split (pooled over folds for OOF)."""
    split = {'oof': [f'f{k}' for k in range(5)], 'temporal': ['temporal'], 'f0': ['f0']}[setting]
    pts = {}
    for kind, name in (('rgb', 'RGB CNN'), ('rgbe', 'RGBE CNN')):
        acc = wrong = n = 0
        for s in split:
            for r in read_jsonl(ROOT / f'reports/{kind}_{s}_s42_test/predictions_val95coverage.jsonl'):
                n += 1
                if r['pred'] is not None:
                    acc += 1; wrong += int(r['pred'] != r['class_name'])
        pts[name] = dict(coverage=acc / n, risk=wrong / acc if acc else 0.0)
    return pts


def unknown_curves(cad):
    """Known = fold-0 test frames (1,478); unknown = six E23 doors (120 frames). x: known coverage, y: unknown acceptance."""
    res = {}
    known = read_jsonl(F0 / 'holes_field8_f0_s42_v2_pixel_test_predictions.jsonl')
    unk = read_jsonl(ROOT / 'reports/e23_openset_v3/holes_field8_f0_unregistered_predictions.jsonl')
    ks, _, kv = hole_scores(known, cad); us, _, uv = hole_scores([dict(r, class_name=None) for r in unk], cad)
    rs = np.unique(np.concatenate([ks[kv], [R_MAX]])); rs = rs[rs <= R_MAX]
    res['Hole (pixel width)'] = dict(known=np.array([(ks <= r).mean() for r in rs]), unknown=np.array([(us <= r).mean() for r in rs]),
                                     threshold=rs, unknown_min_distance_mm=float(us[uv].min()))
    for kind, name in (('rgb', 'RGB CNN'), ('rgbe', 'RGBE CNN')):
        kc = np.array([r['confidence'] for r in cnn_rows(kind, 'f0')])
        uc = np.array([r['confidence'] for r in read_jsonl(ROOT / f'reports/e23_openset_v3/{kind}_f0_s42_predictions.jsonl')])
        ts = np.unique(np.concatenate([kc, uc]))[::-1]
        res[name] = dict(known=np.array([(kc >= t).mean() for t in ts]), unknown=np.array([(uc >= t).mean() for t in ts]),
                         threshold=ts)
    return res


def e23_operating_points(cad):
    """Pre-specified points on panel (c): hole 30 mm reject; CNN fold-0 val-95% threshold."""
    out = {}
    known = read_jsonl(F0 / 'holes_field8_f0_s42_v2_pixel_test_predictions.jsonl')
    unk = read_jsonl(ROOT / 'reports/e23_openset_v3/holes_field8_f0_unregistered_predictions.jsonl')
    ks, _, _ = hole_scores(known, cad); us, _, _ = hole_scores([dict(r, class_name=None) for r in unk], cad)
    out['Hole (pixel width)'] = dict(known=float((ks <= R_MAX).mean()), unknown=float((us <= R_MAX).mean()))
    for kind, name in (('rgb', 'RGB CNN'), ('rgbe', 'RGBE CNN')):
        t = read_json(ROOT / f'reports/{kind}_f0_s42_test/report.json')['val95coverage_threshold']
        kc = np.array([r['confidence'] for r in cnn_rows(kind, 'f0')])
        uc = np.array([r['confidence'] for r in read_jsonl(ROOT / f'reports/e23_openset_v3/{kind}_f0_s42_predictions.jsonl')])
        out[name] = dict(threshold=t, known=float((kc >= t).mean()), unknown=float((uc >= t).mean()))
    return out


def at_known(c, target):
    i = np.searchsorted(c['known'], target)     # known is non-decreasing along the sweep
    i = min(i, len(c['known']) - 1)
    return dict(known=float(c['known'][i]), unknown=float(c['unknown'][i]), threshold=float(c['threshold'][i]))


def draw(curves, pts, unknown, e23_pts, layout):
    if layout == 'row':
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6)); fs = 8
    else:
        fig, axes = plt.subplots(3, 1, figsize=(base.plt.rcParams['figure.figsize'][0] * 0 + 3.3, 6.6)); fs = 7
    for ax, setting, title in ((axes[0], 'oof', '(a) Five-fold OOF, 361 doors'), (axes[1], 'temporal', '(b) Temporal split, 133 doors')):
        for name, c in curves[setting].items():
            color, ls = STYLE[name]
            ax.plot(100 * c['coverage'], 100 * c['risk'], color=color, ls=ls, lw=0.9, alpha=0.8, label=name)
            ax.plot(100 * c['coverage'], 100 * c['risk'], ls='', marker='o', ms=1.4, color=color, zorder=3)
            if name in pts[setting]:
                p = pts[setting][name]; ax.plot(100 * p['coverage'], 100 * p['risk'], marker='s', ms=5, color=color, mec='white', zorder=5)
        h = curves[setting]['Hole (pixel width)']
        ax.plot(100 * h['coverage'][-1], 100 * h['risk'][-1], marker='o', ms=5, color=HOLE, mec='white', zorder=5)
        ax.set_xlim(50, 100.8); ax.set_ylim(-0.05, 1.0)
        ax.set_xlabel('Images judged (%)', fontsize=fs); ax.set_ylabel('Errors among judged (%)', fontsize=fs)
        ax.set_title(title, loc='left', fontsize=fs + 0.5, color=base.INK, pad=2)
        ax.grid(color=base.GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=fs - 0.5)
    ax = axes[2]
    for name, c in unknown.items():
        color, ls = STYLE[name]
        ax.plot(100 * c['known'], 100 * c['unknown'], color=color, ls=ls, lw=0.9, alpha=0.8, label=name)
        ax.plot(100 * c['known'], 100 * c['unknown'], ls='', marker='o', ms=1.4, color=color, zorder=3)
    for name, p in e23_pts.items():
        ax.plot(100 * p['known'], 100 * p['unknown'], marker='o' if name.startswith('Hole') else 's', ms=5,
                color=STYLE[name][0], mec='white', zorder=5)
    ax.set_xlim(85, 100.8); ax.set_ylim(-3, 103)
    ax.set_xlabel('Known test images judged (%)', fontsize=fs); ax.set_ylabel('E23 images accepted (%)', fontsize=fs)
    ax.set_title('(c) Unregistered E23, fold 0', loc='left', fontsize=fs + 0.5, color=base.INK, pad=2)
    ax.grid(color=base.GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=fs - 0.5)
    handles = [plt.Line2D([], [], color=STYLE[n][0], ls=STYLE[n][1], lw=1.6, label=n) for n in STYLE]
    handles += [plt.Line2D([], [], marker='o', ls='', color=base.INK2, mec='white', ms=5, label='hole, 30 mm reject'),
                plt.Line2D([], [], marker='s', ls='', color=base.INK2, mec='white', ms=5, label='CNN, val-95% threshold')]
    if layout == 'row':
        fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=fs - 0.5, bbox_to_anchor=(0.5, -0.1))
        fig.subplots_adjust(wspace=0.38, bottom=0.27)
        base.save(fig, 'v5_fig4_coverage')
    else:
        fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=fs - 0.5, bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(hspace=0.62, bottom=0.17)
        base.save(fig, 'v5c_fig4_coverage')


def main():
    sp, _, _, _ = snapshot(); cfg = read_json(sp / 'protocol.json'); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    curves = {s: setting_curves(s, cad) for s in ('oof', 'temporal')}
    pts = {s: operating_points(s) for s in ('oof', 'temporal')}
    unknown = unknown_curves(cad)
    res = dict(created_at=now(), scope='Exploratory, frame level, stored scores only; no threshold chosen from these curves.',
               hole_reject_cap_mm=R_MAX, settings={})
    for s in ('oof', 'temporal'):
        res['settings'][s] = dict(curves=[summarize_curve(c, n, R_MAX if n.startswith('Hole') else None) for n, c in curves[s].items()],
                                  cnn_val95_points=pts[s])
    res['e23'] = {n: dict(at_known_95=at_known(c, .95), at_known_full=dict(known=float(c['known'][-1]), unknown=float(c['unknown'][-1])),
                          **({'unknown_min_distance_mm': c['unknown_min_distance_mm']} if 'unknown_min_distance_mm' in c else {}))
                  for n, c in unknown.items()}
    dump(ROOT / 'reports/coverage_v5/coverage.json', res)
    e23_pts = e23_operating_points(cad)
    res['e23_operating_points'] = e23_pts
    dump(ROOT / 'reports/coverage_v5/coverage.json', res)
    draw(curves, pts, unknown, e23_pts, 'row'); draw(curves, pts, unknown, e23_pts, 'col')
    import json; print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
