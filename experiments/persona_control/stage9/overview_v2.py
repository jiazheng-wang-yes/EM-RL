"""Figure 1 overview, draft version 2: (a) five numbered steps, one illustration each; (b) unchanged arrangements panel.
Standalone schematic, no data. Writes to figures/persona_control/overview_v2
and copies the PDF into the paper's figure directory."""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
from paper_fonts import configure_fonts
from paper_palette import PINK, PALE_PINK

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'figures/persona_control/overview_v2'
DEST = ROOT / 'paper_draft/arr_2026/figures'
INK, MUTED, GRID = '#303235', '#696c70', '#e5e7e9'
SERIES = {'N': PINK, 'G': '#3a74b0', 'B': '#2f9a7d'}
PALE = PALE_PINK
N, BLUE, GRAY, LIGHT = SERIES['N'], '#5b86b3', '#c9ccd0', '#f4f4f5'
configure_fonts()
plt.rcParams.update({'font.size': 8.5})

W, H = 6.9, 3.45
fig = plt.figure(figsize=(W, H))
base = fig.add_axes([0, 0, 1, 1])
base.set_xlim(0, W)
base.set_ylim(0, H)
base.axis('off')

# panel (a) drawings live in their own axes, scaled by S relative to the inch grid; its fonts scale by FS
S, FS, Y0 = 1.09, 1.08, 1.45          # Y0: bottom of the (a) axes in inches
pa = fig.add_axes([0, Y0 / H, 1, (H - Y0) / H])
pa.set_xlim(0, W / S)
pa.set_ylim(0, (H - Y0) / S)
pa.axis('off')
ax = base


def box(x, y, w, h, fc=LIGHT, ec=MUTED, lw=.7, r=.04, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))


def stack(x, y, accent, w=.38, h=.17):  # a model: three offset cards, a colored line on the front one
    for i in (2, 1, 0):
        box(x + .035 * i, y + .04 * i, w, h, z=3 + (2 - i))
    ax.plot([x + .07, x + w - .07], [y + h / 2] * 2, color=accent, lw=2.2, solid_capstyle='round', zorder=6)


def doc(x, y, w=.3, h=.36, lines=((GRAY, 1.1, .9),) * 3):  # lines: (color, lw, fraction of width), top to bottom
    box(x, y, w, h, fc='white', z=3)
    for i, (col, lw, frac) in enumerate(lines):
        yy = y + h - .085 - i * .09
        ax.plot([x + .05, x + .05 + (w - .1) * frac], [yy, yy], color=col, lw=lw, solid_capstyle='round', zorder=4)


def arrow(p, q, color=MUTED, lw=.9, rad=0):
    ax.annotate('', xy=q, xytext=p, zorder=5, arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=8,
                                                              shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}'))


def grid(x, y, w, h, vals, cmap, vmin, vmax, ec=MUTED, lw=.6, z=3):
    n, m = vals.shape
    cw, ch = w / m, h / n
    for i in range(n):
        for j in range(m):
            v = float(np.clip((vals[i, j] - vmin) / (vmax - vmin), 0, 1))
            ax.add_patch(Rectangle((x + j * cw, y + h - (i + 1) * ch), cw, ch, fc=cmap(v), ec='white', lw=.3, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=ec, lw=lw, zorder=z + 1))


def T(x, y, s, fs=7.5, color=INK, ha='center', va='center', **kw):
    kw.setdefault('linespacing', 1.12)
    ax.text(x, y, s, fontsize=fs * (FS if ax is pa else 1), color=color, ha=ha, va=va, **kw)


def step(n, x, y, title):
    ax.add_patch(Circle((x, y), .068, fc=INK, ec='none', zorder=4))
    ax.text(x, y - .004, str(n), color='white', fontsize=6.4, weight='bold', ha='center', va='center', zorder=5)
    ax.text(x + .11, y, title, fontsize=8, color=INK, weight='bold', ha='left', va='center')


div = LinearSegmentedColormap.from_list('dw', [BLUE, 'white', N])
gray = LinearSegmentedColormap.from_list('g', ['#ffffff', '#8d9196'])
rs = np.random.RandomState(3)
u1 = np.array([1., .7, .3, -.2, -.6, -.9])
v1 = np.array([.9, .5, -.3, -.8, .2, .7])
u2 = np.array([-.3, .8, .9, .2, -.7, -.4])
v2 = np.array([.4, -.8, .1, .9, .6, -.5])
A1, A2, A3 = np.outer(u1, v1), np.outer(u2, v2), np.outer(np.roll(u1, 2), np.roll(v2, 3))
dW = A1 + .55 * A2 + .12 * rs.randn(6, 6)
WC = np.clip(.5 + .2 * rs.randn(6, 6), .05, .95)  # shared weights
WE = WC + .3 * dW
lim = float(np.abs(dW).max())

# ---------------------------------------------------------------- (a) titles, unscaled, on the base axes
yt = H - .11
T(.04, yt, '(a)', fs=9.2, ha='left', weight='bold')
step(1, .34, yt, 'Matched fine-tunes')
step(2, 1.68, yt, 'Weight difference')
step(3, 2.95, yt, 'SVD')
step(4, 3.55, yt, 'Rank by narrow preference')
step(5, 5.42, yt, 'Edit and measure')

# ---------------------------------------------------------------- (a) drawings, in the scaled axes
ax = pa
yc = (H - .11 - .95 - Y0) / S  # vertical centre of the pipeline, in scaled inches

# 1. same questions, two models
doc(.03, yc - .18, w=.28, h=.36)
T(.17, yc - .22, 'same\nquestions', va='top', color=MUTED)
arrow((.33, yc + .05), (.6, yc + .36), rad=-.25)
arrow((.33, yc - .05), (.6, yc - .4), rad=.25)
stack(.62, yc + .24, N)
stack(.62, yc - .48, BLUE)
T(.85, yc + .55, '$E$ harmful', fs=8.2, color=N, va='bottom')
T(.85, yc - .55, '$C$ benign', fs=8.2, color=BLUE, va='top')

# 2. W_E over W_C, their difference dW in the middle
GAP = .2  # equal gap between neighbouring drawings
g, gx = .36, 1.05 + GAP
grid(gx, yc + .14, g, g, WE, gray, 0, 1, ec=N, lw=.9)
grid(gx, yc - .5, g, g, WC, gray, 0, 1, ec=BLUE, lw=.9)
arrow((1.07, yc + .36), (gx - .03, yc + .36))
arrow((1.07, yc - .36), (gx - .03, yc - .36))
T(gx + g / 2, yc, '$-$', fs=11)
T(gx + g + .04, yc + .14 + g / 2, '$W_E$', fs=8.2, color=N, ha='left')
T(gx + g + .04, yc - .5 + g / 2, '$W_C$', fs=8.2, color=BLUE, ha='left')
T(gx + g + .3, yc, '$=$', fs=11)
dx_, gd = gx + g + .42, .38
grid(dx_, yc - gd / 2, gd, gd, dW, div, -lim, lim, ec=INK, lw=.8)
T(dx_ + gd / 2, yc - gd / 2 - .04, r'$\Delta W$', fs=8.2, va='top')

# 3. SVD: rank-one components as outer products
x3, cw, s3 = dx_ + gd + GAP, .055, .33
arrow((dx_ + gd + .03, yc), (x3 - .04, yc))
grid(x3, yc - s3 / 2, cw, s3, u1[:, None], div, -1, 1)  # u_c column
grid(x3 + .075, yc + s3 / 2 + .02, s3, cw, v1[None, :], div, -1, 1)  # v_c^T row
for i, A in ((2, A3), (1, A2), (0, A1)):
    grid(x3 + .075 + .06 * i, yc - s3 / 2 - .06 * i, s3, s3, A, div, -1, 1)
T(x3 + cw / 2, yc - s3 / 2 - .03, '$u_c$', fs=7.2, va='top')
T(x3 + .075 + s3 + .03, yc + s3 / 2 + .02 + cw / 2, r'$v_c^{\top}$', fs=7.2, ha='left')
T(x3 + .075 + s3 + .14, yc - s3 / 2 - .1, r'$\times 32$', fs=8, ha='left', va='center')
T(x3 + .27, yc - s3 / 2 - .2, r'$A_c = s_c\,u_c v_c^{\top}$', va='top')
x3end = x3 + .075 + s3 + .12

# 4. rank by attribution to the harmful-answer preference on held-out questions
gap, dw_ = GAP - .03, .28
qx = x3end + .03 + gap
arrow((x3end + .03, yc), (qx - .02, yc))
doc(qx, yc - .18, w=dw_, h=.36, lines=((GRAY, 1.1, .9), (N, 1.9, .9), (BLUE, 1.9, .5)))
T(qx + dw_ / 2, yc + .23, 'held-out\nquestions', va='bottom')
sx, sy, sw, sh, k, n = qx + dw_ + .03 + gap, yc - .25, .74, .5, 6, 24
arrow((qx + dw_ + .03, yc), (sx - .03, yc))
a = fig.add_axes([sx * S / W, (Y0 + sy * S) / H, sw * S / W, sh * S / H])
vals = np.exp(-np.arange(n) / 7)
a.bar(range(n), vals, width=.74, color=[N] * k + [GRAY] * (n - k), lw=0)
a.axvspan(-.6, k - .4, color=N, alpha=.1, lw=0)
a.set_xlim(-.7, n - .3)
a.set_ylim(0, 1.12)
a.axis('off')
T(sx + .1, yc - .3, r'$\mathcal{N}_k$ = top $k$', fs=8.2, color=N, ha='left', va='top')

# 5. add to C, remove from E, measure narrow and broad shifts and generated answers
cx = sx + sw + GAP
arrow((sx + sw + .03, yc), (cx - .03, yc), color=N)
for i, A in ((2, A3), (1, A2), (0, A1)):
    grid(cx + .05 * i, yc - .11 - .05 * i, .22, .22, A, div, -1, 1)
T(cx + .16, yc - .3, r'$\mathcal{N}_k$', fs=8.2, color=N, va='top')
px_ = cx + .6
arrow((cx + .34, yc + .05), (px_ - .06, yc + .3), color=N, rad=-.2)
arrow((cx + .34, yc - .05), (px_ - .06, yc - .3), color=N, rad=.2)
T(px_, yc + .36, '+', fs=11)
T(px_, yc - .36, '$-$', fs=11)
stx = px_ + .08
stack(stx, yc + .24, BLUE, w=.36, h=.16)
stack(stx, yc - .48, N, w=.36, h=.16)
T(stx + .21, yc + .55, r'$C+\mathcal{N}_k$', fs=8.2, va='bottom')
T(stx + .21, yc - .53, r'$E-\mathcal{N}_k$', fs=8.2, va='top')
bx = stx + .47
ax.plot([bx, bx], [yc - .36, yc + .36], color=MUTED, lw=.9, zorder=5)
arrow((stx + .43, yc + .36), (bx, yc + .36))
arrow((stx + .43, yc - .36), (bx, yc - .36))
arrow((bx, yc + .2), (bx + .1, yc + .2))
arrow((bx, yc - .2), (bx + .1, yc - .2))
rx = bx + .14
for j, (hh, col, lab) in enumerate([(.2, PALE, "$R_N$"), (.32, N, "$R_B$")]):
    ax.add_patch(Rectangle((rx + .13 * j, yc + .06), .08, hh, fc=col, ec='none', zorder=3))
    T(rx + .13 * j + .04, yc + .03, lab, fs=6.8, color=INK, va='top')
ax.plot([rx - .02, rx + .23], [yc + .06, yc + .06], color=MUTED, lw=.7, zorder=3)
T(rx + .1, yc + .44, 'likelihood\nshifts', fs=7.2, va='bottom')
doc(rx - .01, yc - .48, w=.22, h=.26)
T(rx + .1, yc - .52, 'generated\nanswers', fs=7.2, va='top')
print('right edge (inches)', round((rx + .25) * S, 2), 'of', W)

# ---------------------------------------------------------------- (b)
ax = base
top = 1.36
T(.04, top, '(b) Broad and narrow effects across weight components', fs=9.2, ha='left', weight='bold')
for xx, color, label in [(4.6, PALE, 'narrow fit'), (5.5, N, 'broad change')]:
    ax.add_patch(FancyBboxPatch((xx, top - .05), .1, .1, boxstyle='square,pad=0', fc=color, ec='none'))
    T(xx + .14, top, label, fs=8, ha='left')
left, stp, pw, ph, pb = .12, 2.3, 1.95, .58, .3
narrow = [2.718 ** (-i / 6) for i in range(18)]
shapes = [('separate', [0.] * 13 + [.95, .8, .65, .5, .4], 'keeps the broad change'),
          ('proportional', narrow, 'removes equal fractions of both'),
          ('concentrated', [1., .9, .78, .66, .5] + [.04] * 13, 'removes most of the\nbroad change, part of the narrow fit')]
for j, (head, broad, note) in enumerate(shapes):
    px = left + j * stp
    a = fig.add_axes([px / W, pb / H, pw / W, ph / H])
    a.axvspan(-.6, 4.5, color=N, alpha=.1, lw=0)
    a.bar(range(18), narrow, width=.72, color=PALE, lw=0)
    a.bar(range(18), [-v for v in broad], width=.72, color=N, lw=0)
    a.axhline(0, color=MUTED, lw=.6)
    a.set_xlim(-.7, 17.7)
    a.set_ylim(-1.12, 1.12)
    a.axis('off')
    T(px + .26, pb + ph + .08, r'$\mathcal{N}_k$', fs=8.5)
    T(px + pw / 2 + .2, pb + ph + .09, head, fs=9, weight='bold' if head == 'concentrated' else 'normal')
    T(px + pw / 2, pb - .07, 'removing $\\mathcal{N}_k$ ' + note,
      fs=7.2 if head == 'concentrated' else 7.7, va='top', linespacing=1.0)
nx = left + 2 * stp
ax.add_patch(FancyBboxPatch((nx - .08, .02), pw + .16, pb + ph + .2, boxstyle='round,pad=0,rounding_size=.06', fc='none', ec=N, lw=1.1))
T(nx + pw + .02, pb + ph + .28, 'found in six settings', fs=7.7, color=N, ha='right')

OUT.mkdir(parents=True, exist_ok=True)
for ext in ['pdf', 'png']:
    # Preserve the incoming PDF crop (511.327 x 262.8 points) as text grows.
    bounds = Bbox.from_extents(-.101766, -.1, W + .1, H + .1)
    fig.savefig(OUT / f'overview_v2.{ext}', bbox_inches=bounds, dpi=300)
shutil.copy2(OUT / 'overview_v2.pdf', DEST / 'overview_v2.pdf')
print('saved', OUT / 'overview_v2.png')
