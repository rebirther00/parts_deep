"""Figures for manuscript v3 (static, print). Reads frozen results only; writes figures/v3_*."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import setup_runtime
setup_runtime()

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from door8.core import ROOT, outpath, read_json, read_jsonl, snapshot

BLUE, ORANGE = '#2a78d6', '#eb6834'          # validated categorical slots 1-2 (pixel, depth)
INK, INK2, GRID, BAND = '#0b0b0b', '#52514e', '#e4e3df', '#ecebe7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})
EXAMPLE = '624:rgb_0078.png'                  # fold-0 TEST-role field coordinate, E38 LH FRT
SHORT = {'E25_door_LH_FRT': 'E25 LH FRT', 'E30_door_LH_FRT': 'E30 LH FRT', 'E38_door_LH_FRT': 'E38 LH FRT',
         'E25_door_RH': 'E25 RH', 'E25_door_LH_RR': 'E25 LH RR', 'E30_E38_door_RH': 'E30/E38 RH',
         'E30_door_LH_RR': 'E30 LH RR', 'E38_door_LH_RR': 'E38 LH RR'}


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(outpath(f'figures/{name}.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_example():
    _, _, frames, _ = snapshot()
    row = next(r for r in frames if r['frame_id'] == EXAMPLE)
    gt = next(r for r in read_jsonl(ROOT / 'data/annotation_snapshots/field40_20261109_v1/annotations.jsonl')
              if r['frame_id'] == EXAMPLE)
    lab = next(r for r in read_jsonl(ROOT / 'features/holes_lab_s42_full/rows.jsonl') if r['frame_id'] == EXAMPLE)
    f8 = next(r for r in read_jsonl(ROOT / 'features/holes_field8_f0_s42_v2_f0_test_v3/rows.jsonl') if r['frame_id'] == EXAMPLE)
    im = cv2.cvtColor(cv2.imread(row['rgb']), cv2.COLOR_BGR2RGB)
    return row, gt, lab, f8, im


def draw_points(ax, pts, color, size=60, filled=True):
    style = dict(s=size, linewidths=1.6, zorder=4)
    for key, marker in (('bolt', 'o'), ('hinge', '^'), ('latch', 's')):
        for p in pts.get(key, []):
            if filled:
                ax.scatter(p[0], p[1], marker=marker, c=color, edgecolors='white', **style)
            else:
                ax.scatter(p[0], p[1], marker=marker, facecolors='none', edgecolors=color, **style)


def fig_method(row, gt, f8, im):
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.0], hspace=0.16)
    ax = fig.add_subplot(gs[0]); ax.imshow(im); ax.axis('off')
    pts = f8['points']; h, l = pts['hinge'][0], pts['latch'][0]
    ax.plot([h[0], l[0]], [h[1], l[1]], color=BLUE, lw=2, zorder=3)
    draw_points(ax, pts, BLUE)
    mid = ((h[0] + l[0]) / 2, (h[1] + l[1]) / 2)
    ax.annotate('D: latch-side to hinge-side corner hole', xy=mid, xytext=(mid[0], mid[1] + 115), ha='center',
                color=INK, fontsize=9, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=GRID),
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    b = np.asarray(pts['bolt'])[:, :2].mean(0)
    ax.annotate('4 bolt holes (157 x 96 mm)', xy=b, xytext=(b[0] + 330, b[1] - 230), color=INK, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=GRID), arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax.set_title('(a) Field image (E38 LH FRT) with detected holes', loc='left', fontsize=9, color=INK)
    bx = fig.add_subplot(gs[1]); bx.axis('off'); bx.set_xlim(0, 5); bx.set_ylim(0, 1)
    steps = ['RGB image', 'Hole heatmaps\n4 bolt + 2 corner', 'Geometry check\nbolt rectangle',
             'Pixel distance\nD = S d / f_x', 'Nearest CAD D\nor abstain']
    for i, t in enumerate(steps):
        bx.add_patch(FancyBboxPatch((i + 0.08, 0.15), 0.76, 0.7, boxstyle='round,pad=0.02', fc='white', ec=INK2, lw=0.9))
        bx.text(i + 0.46, 0.5, t, ha='center', va='center', fontsize=7.5, color=INK)
        if i < 4:
            bx.annotate('', xy=(i + 1.07, 0.5), xytext=(i + 0.87, 0.5), arrowprops=dict(arrowstyle='->', color=INK2, lw=0.9))
    bx.set_title('(b) Decision pipeline; S is fit on training sessions only', loc='left', fontsize=9, color=INK)
    save(fig, 'v3_fig1_method')


def fig_adaptation(gt, lab, f8, im):
    allp = np.array([gt['points'][k] for k in gt['points']])
    x0, y0 = np.maximum(allp.min(0) - 90, 0).astype(int); x1, y1 = (allp.max(0) + 90).astype(int)
    crop = im[y0:y1, x0:x1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.25))
    gpts = {'bolt': [gt['points'][k] for k in ('bolt_tl', 'bolt_tr', 'bolt_bl', 'bolt_br')],
            'hinge': [gt['points']['corner_hinge']], 'latch': [gt['points']['corner_latch']]}
    shift = lambda pts: {k: [(p[0] - x0, p[1] - y0) for p in v] for k, v in pts.items()}
    for ax, det, title in ((axes[0], lab, '(a) LAB 94 only: corner holes missed'),
                           (axes[1], f8, '(b) LAB 94 + field 8: all six found')):
        ax.imshow(crop); ax.axis('off')
        draw_points(ax, shift(gpts), 'white', size=160, filled=False)
        draw_points(ax, shift(det['points']), BLUE, size=36)
        ax.set_title(title, loc='left', fontsize=9, color=INK)
    axes[0].text(0.02, 0.03, 'gate: missing corner', transform=axes[0].transAxes, fontsize=8, color=INK,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=GRID))
    axes[1].text(0.02, 0.03, 'gate: ok  |  E38 LH FRT, correct', transform=axes[1].transAxes, fontsize=8, color=INK,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=GRID))
    handles = [plt.Line2D([], [], marker='o', ls='', mfc='none', mec=INK2, ms=9, label='human field label'),
               plt.Line2D([], [], marker='o', ls='', mfc=BLUE, mec='white', ms=6, label='detection'),
               plt.Line2D([], [], marker='o', ls='', mfc=INK2, mec=INK2, ms=5, label='bolt'),
               plt.Line2D([], [], marker='^', ls='', mfc=INK2, mec=INK2, ms=5, label='hinge-side'),
               plt.Line2D([], [], marker='s', ls='', mfc=INK2, mec=INK2, ms=5, label='latch-side')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    save(fig, 'v3_fig2_adaptation')


def margins(cad):
    names = sorted(cad, key=cad.get); out = {}
    for i, c in enumerate(names):
        d = cad[c]
        lo = (cad[names[i - 1]] + d) / 2 if i > 0 else -np.inf
        hi = (cad[names[i + 1]] + d) / 2 if i < len(names) - 1 else np.inf
        out[c] = (max(lo, d - 30) / d - 1, min(hi, d + 30) / d - 1, lo / d - 1, hi / d - 1)
    return names, out


def oof_predictions(mode):
    """5-fold out-of-fold predictions of the eight-field-image detectors (fold 0 + extension_v3 folds 1-4)."""
    files = [ROOT / f'reports/fold0_test_v3/holes_field8_f0_s42_v2_{mode}_test_predictions.jsonl']
    files += [ROOT / f'reports/extension_v3/holes_field8_f{k}_s42_v3_{mode}_predictions.jsonl' for k in (1, 2, 3, 4)]
    return [r for f in files for r in read_jsonl(f)]


def fig_errors():
    sp, _, _, _ = snapshot(); cfg = read_json(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    names, band = margins(cad)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, mode, color, title in ((axes[0], 'pixel', BLUE, '(a) Pixel width'), (axes[1], 'depth', ORANGE, '(b) Depth plane')):
        preds = oof_predictions(mode)
        for i, c in enumerate(names):
            lo, hi, _, _ = band[c]
            ax.add_patch(plt.Rectangle((i - 0.38, 100 * lo), 0.76, 100 * (hi - lo), fc=BAND, ec='none', zorder=0))
            e = np.array([100 * (r['measurement'] / cad[c] - 1) for r in preds if r['class_name'] == c and r['measurement'] is not None])
            if len(e):
                ax.scatter(i + rng.uniform(-0.28, 0.28, len(e)), e, s=3, c=color, alpha=0.35, linewidths=0, zorder=2)
                out = ((e < 100 * lo) | (e > 100 * hi)).sum()
                if out:
                    ax.text(i, 10.6, f'{out}', ha='center', fontsize=7.5, color=INK)
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([f'{SHORT[c]} ({cad[c]})' for c in names], fontsize=7, rotation=40, ha='right', rotation_mode='anchor')
        ax.set_ylim(-4.6, 11.6); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, loc='left', fontsize=9, color=INK)
    axes[0].set_ylabel('Relative distance error, D / D_CAD - 1 (%)')
    save(fig, 'v3_fig3_relative_error')
    return {c: [round(100 * v, 2) for v in band[c][:2]] for c in names}


if __name__ == '__main__':
    row, gt, lab, f8, im = load_example()
    fig_method(row, gt, f8, im)
    fig_adaptation(gt, lab, f8, im)
    print(json.dumps(fig_errors(), ensure_ascii=False))
