"""Method figure (Fig. 2 of manuscript v14); writes figures/v14c_fig2_method.{png,pdf}.

v14 (review round 3): same as make_figure1_v10col.py except that the decision box says "or reject", the term used in the
text and tables, instead of "or abstain". The original docstring follows.

Column-width Fig. 1 with both panels for the KSMTE two-column layout; writes figures/v10c_fig1_method.{png,pdf}.

v9 review: the column figure figures/v3c_fig1_method.png kept only the photo, while the caption describes
(a) the field image and (b) the decision pipeline. This keeps panel (a) exactly as in v3c (same crop, detections and
labels) and adds panel (b): the pipeline of the wide figure figures/v3_fig1_method.png, re-laid out in two rows for one
column, with the CAD list drawn as the separate input of the nearest-CAD decision (Section 3.1). Existing figures are
not modified.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures_v3 as base   # noqa: E402  (sets runtime, style, helpers)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({'font.size': 7})
W = 3.3
BLUE, INK, INK2, GRID = base.BLUE, base.INK, base.INK2, base.GRID


def panel_a(ax, f8, im):
    """Identical to make_figures_v3col.fig1 (crop, points, annotations)."""
    x0, x1, y0, y1 = 600, 1500, 100, 1090
    pts = {k: [(p[0] - x0, p[1] - y0, *p[2:]) for p in v] for k, v in f8['points'].items()}
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
    ax.set_title('(a) Field image (E38 LH FRT) with detected holes', loc='left', fontsize=7, color=INK, pad=3)


def panel_b(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis('off')
    bw, bh = 2.85, 1.2

    def box(cx, cy, text, dashed=False):
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    fc='white', ec=INK2, lw=0.8, ls='--' if dashed else '-'))
        ax.text(cx, cy, text, ha='center', va='center', fontsize=5.8, color=INK, linespacing=1.15)

    def arrow(p, q):
        ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='->', color=INK2, lw=0.8, shrinkA=0, shrinkB=0))

    xs, top, bot = (1.5, 5.0, 8.5), 3.05, 0.85
    box(xs[0], top, 'RGB image')
    box(xs[1], top, 'Hole detector\nshared, 6 points')
    box(xs[2], top, 'Geometry check\n(bolt rectangle)')
    box(xs[2], bot, 'Pixel distance\n$D = S\\,d\\,/\\,f_x$')
    box(xs[1], bot, 'Nearest CAD D\nor reject')
    box(xs[0], bot, 'CAD list\n(per class)', dashed=True)
    g = bw / 2 + 0.05
    arrow((xs[0] + g, top), (xs[1] - g, top))
    arrow((xs[1] + g, top), (xs[2] - g, top))
    arrow((xs[2], top - bh / 2 - 0.05), (xs[2], bot + bh / 2 + 0.05))
    arrow((xs[2] - g, bot), (xs[1] + g, bot))
    arrow((xs[0] + g, bot), (xs[1] - g, bot))
    ax.set_title('(b) Decision pipeline; S is fit on training sessions only', loc='left', fontsize=7, color=INK, pad=3)


if __name__ == '__main__':
    row, gt, lab, f8, im = base.load_example()
    ha = W * (1090 - 100) / (1500 - 600)
    hb = 1.2
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, ha + hb + 0.35), gridspec_kw=dict(height_ratios=[ha, hb], hspace=0.12))
    panel_a(a, f8, im); panel_b(b)
    base.save(fig, 'v14c_fig2_method')
    print('ok')
