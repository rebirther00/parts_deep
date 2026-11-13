"""Residual figure for manuscript v15; writes figures/v15c_fig3_relative_error.{png,pdf}.

v15: same data and drawing as make_figure3_v12col.py, only 0.3 in shorter (3.45 in) so that the figure fits under the
first paragraph of Section 4.3 in the two-column layout. The original docstring follows.

Column-width residual figure for manuscript v12; writes figures/v12c_fig3_relative_error.{png,pdf}.

(a) pixel width with integer heatmap peaks (left half of each class slot, as reported) and with the quadratic sub-pixel
peaks of config/extension_v12_protocol.json D (right half); (b) depth plane. Five-fold out-of-fold TEST predictions of the
eight-field-image detectors. The sub-pixel measurements are recomputed exactly as in run_extension_v12.subpixel_eval
(refined features, S refit on the fold's TRAIN sessions); no decision or statistic is changed by this script.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures_v3 as base   # noqa: E402  (sets runtime, style, helpers)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from door8.core import ROOT, partition, read_json, read_jsonl, snapshot
from door8.geometry import classify_measurement, fit_scale

plt.rcParams.update({'font.size': 7})
W = 3.3
BLUE, ORANGE, INK, INK2, GRID, BAND = base.BLUE, base.ORANGE, base.INK, base.INK2, base.GRID, base.BAND
AQUA = '#1baf7a'                                   # categorical slot 3


def refined_predictions():
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); out = []
    for f in ('0', '1', '2', '3', '4'):
        parts = partition(frames, splits, f)
        pool = {r['frame_id']: r for r in read_jsonl(ROOT / f'cache/extension_v12/subpixel/f{f}.jsonl')}
        cal = fit_scale(parts['train'], pool, cad, 'pixel')
        for row in parts['test']:
            p, v, why = classify_measurement(row, pool[row['frame_id']], cal, cfg)
            out.append(dict(class_name=row['class_name'], measurement=v, pred=p))
    return out


if __name__ == '__main__':
    sp, _, _, _ = snapshot(); cfg = read_json(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); names, band = base.margins(cad)
    rng = np.random.default_rng(0)
    series = dict(integer=base.oof_predictions('pixel'), refined=refined_predictions(), depth=base.oof_predictions('depth'))
    rel = lambda preds, c: np.array([100 * (r['measurement'] / cad[c] - 1) for r in preds if r['class_name'] == c and r['measurement'] is not None])
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.45), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.55], hspace=0.16))
    for i, c in enumerate(names):
        lo, hi, _, _ = band[c]
        for ax in axes:
            ax.add_patch(plt.Rectangle((i - 0.4, 100 * lo), 0.8, 100 * (hi - lo), fc=BAND, ec='none', zorder=0))
        for key, color, x0 in (('integer', BLUE, -0.2), ('refined', AQUA, 0.2)):
            e = rel(series[key], c)
            axes[0].scatter(i + x0 + rng.uniform(-0.15, 0.15, len(e)), e, s=2, c=color, alpha=0.3, linewidths=0, zorder=2)
            if ((e < 100 * lo) | (e > 100 * hi)).any():
                raise ValueError('pixel-width image outside the band; the caption says there is none')
        e = rel(series['depth'], c)
        axes[1].scatter(i + rng.uniform(-0.28, 0.28, len(e)), e, s=2, c=ORANGE, alpha=0.3, linewidths=0, zorder=2)
        out = int(((e < 100 * lo) | (e > 100 * hi)).sum())
        if out:
            axes[1].text(i, 10.4, f'{out}', ha='center', fontsize=6.5, color=INK)
    for ax, title, ylim in ((axes[0], '(a) Pixel width', (-4.6, 4.6)), (axes[1], '(b) Depth plane', (-4.6, 11.8))):
        ax.axhline(0, color=INK2, lw=0.5); ax.set_ylim(*ylim); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(title, loc='left', fontsize=7.5, color=INK, pad=2); ax.set_ylabel('D / D_CAD - 1 (%)')
    axes[0].set_yticks([-2.5, 0, 2.5])
    axes[0].legend(handles=[Line2D([], [], marker='o', ls='', ms=3.5, color=BLUE, label='integer peak'),
                            Line2D([], [], marker='o', ls='', ms=3.5, color=AQUA, label='sub-pixel peak')],
                   loc='upper right', bbox_to_anchor=(1.0, 1.24), ncol=2, fontsize=6.5, frameon=False, handletextpad=0.2, columnspacing=1.0,
                   borderaxespad=0)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([f'{base.SHORT[c]} ({cad[c]})' for c in names], fontsize=6.5, rotation=40, ha='right', rotation_mode='anchor')
    base.save(fig, 'v15c_fig3_relative_error')
    n = {k: sum(r['measurement'] is not None for r in v) for k, v in series.items()}
    print('ok', n)
