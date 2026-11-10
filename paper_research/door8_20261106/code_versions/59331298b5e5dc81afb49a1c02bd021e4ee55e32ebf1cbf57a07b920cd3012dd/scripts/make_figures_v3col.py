"""Column-width (3.3 in) figures for the KSMTE two-column layout; writes figures/v3c_*.

Same data and encodings as make_figures_v3.py, re-laid out for one column (text <= 8 pt).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures_v3 as base   # noqa: E402  (sets runtime, style, helpers)
import matplotlib.pyplot as plt
import numpy as np

from door8.core import ROOT, read_json, read_jsonl, snapshot

plt.rcParams.update({'font.size': 7})
W = 3.3
BLUE, ORANGE, INK, INK2, GRID, BAND = base.BLUE, base.ORANGE, base.INK, base.INK2, base.GRID, base.BAND


def fig1(gt, f8, im):
    x0, x1, y0, y1 = 600, 1500, 100, 1090
    pts = {k: [(p[0] - x0, p[1] - y0, *p[2:]) for p in v] for k, v in f8['points'].items()}
    fig, ax = plt.subplots(figsize=(W, W * (y1 - y0) / (x1 - x0)))
    ax.imshow(im[y0:y1, x0:x1]); ax.axis('off')
    h, l = pts['hinge'][0], pts['latch'][0]
    ax.plot([h[0], l[0]], [h[1], l[1]], color=BLUE, lw=1.6, zorder=3)
    base.draw_points(ax, pts, BLUE, size=30)
    mid = ((h[0] + l[0]) / 2, (h[1] + l[1]) / 2)
    ax.annotate('D (latch-side to hinge-side corner hole)', xy=mid, xytext=(mid[0] + 40, mid[1] + 62), ha='center', fontsize=7,
                color=INK, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=GRID), arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    b = np.asarray(pts['bolt'])[:, :2].mean(0)
    ax.annotate('4 bolt holes\n(157 x 96 mm)', xy=b, xytext=(b[0] + 330, b[1] - 260), ha='center', fontsize=7, color=INK,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=GRID), arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    base.save(fig, 'v3c_fig1_method')


def fig2(gt, lab, f8, im):
    allp = np.array([gt['points'][k] for k in gt['points']])
    x0, y0 = np.maximum(allp.min(0) - 70, 0).astype(int); x1, y1 = (allp.max(0) + 70).astype(int)
    crop = im[y0:y1, x0:x1]
    gpts = {'bolt': [gt['points'][k] for k in ('bolt_tl', 'bolt_tr', 'bolt_bl', 'bolt_br')],
            'hinge': [gt['points']['corner_hinge']], 'latch': [gt['points']['corner_latch']]}
    shift = lambda pts: {k: [(p[0] - x0, p[1] - y0) for p in v] for k, v in pts.items()}
    fig, axes = plt.subplots(2, 1, figsize=(W, 2 * W * (y1 - y0) / (x1 - x0) + 0.7))
    for ax, det, title, note in ((axes[0], lab, '(a) LAB 94 only', 'corner holes missed: abstain'),
                                 (axes[1], f8, '(b) LAB 94 + field 8', 'all six found: E38 LH FRT, correct')):
        ax.imshow(crop); ax.axis('off')
        base.draw_points(ax, shift(gpts), 'white', size=70, filled=False)
        base.draw_points(ax, shift(det['points']), BLUE, size=16)
        ax.set_title(title, loc='left', fontsize=7.5, color=INK, pad=2)
        ax.text(0.01, 0.04, note, transform=ax.transAxes, fontsize=6.5, color=INK, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=GRID))
    handles = [plt.Line2D([], [], marker='o', ls='', mfc='none', mec=INK2, ms=6, label='human label'),
               plt.Line2D([], [], marker='o', ls='', mfc=BLUE, mec='white', ms=4.5, label='detection'),
               plt.Line2D([], [], marker='^', ls='', mfc=INK2, mec=INK2, ms=4, label='hinge-side'),
               plt.Line2D([], [], marker='s', ls='', mfc=INK2, mec=INK2, ms=4, label='latch-side')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.3, columnspacing=0.9)
    fig.subplots_adjust(hspace=0.18, bottom=0.08)
    base.save(fig, 'v3c_fig2_adaptation')


def fig3():
    sp, _, _, _ = snapshot(); cfg = read_json(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); names, band = base.margins(cad)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.6), sharex=True)
    for ax, mode, color, title in ((axes[0], 'pixel', BLUE, '(a) Pixel width'), (axes[1], 'depth', ORANGE, '(b) Depth plane')):
        preds = read_jsonl(ROOT / f'reports/fold0_test_v3/holes_field8_f0_s42_v2_{mode}_test_predictions.jsonl')
        for i, c in enumerate(names):
            lo, hi, _, _ = band[c]
            ax.add_patch(plt.Rectangle((i - 0.38, 100 * lo), 0.76, 100 * (hi - lo), fc=BAND, ec='none', zorder=0))
            e = np.array([100 * (r['measurement'] / cad[c] - 1) for r in preds if r['class_name'] == c and r['measurement'] is not None])
            ax.scatter(i + rng.uniform(-0.28, 0.28, len(e)), e, s=2.5, c=color, alpha=0.45, linewidths=0, zorder=2)
            out = int(((e < 100 * lo) | (e > 100 * hi)).sum())
            if out:
                ax.text(i, 8.3, f'{out}', ha='center', fontsize=6.5, color=INK)
        ax.axhline(0, color=INK2, lw=0.5); ax.set_ylim(-6, 9.4); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(title, loc='left', fontsize=7.5, color=INK, pad=2); ax.set_ylabel('D / D_CAD - 1 (%)')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([f'{base.SHORT[c]} ({cad[c]})' for c in names], fontsize=6.5, rotation=40, ha='right', rotation_mode='anchor')
    fig.subplots_adjust(hspace=0.15)
    base.save(fig, 'v3c_fig3_relative_error')


if __name__ == '__main__':
    row, gt, lab, f8, im = base.load_example()
    fig1(gt, f8, im); fig2(gt, lab, f8, im); fig3()
    print('ok')
