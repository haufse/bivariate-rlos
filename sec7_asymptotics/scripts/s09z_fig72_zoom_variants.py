"""s09z: presentation variants of Figure 7.2 (fig:error-vs-blocksize) for better visual distinction of the curves.
Same data as the unzoomed Figure 7.2 script s09_fig72_ameise_vs_b.py (not included here): identical b grid (geomspace(1, n/50, 800)), identical criterion
(s06_tables.crit_fn), identical models, estimators, colours, minimizer markers and Archimax r=1-only rule.
Only axis limits, layout and an added ratio row differ.  The paper uses variant C (default here).

Run (from scripts/):
  python s09z_fig72_zoom_variants.py [--variant C] [--delta 0.15] [n ...]
Variants:
  A  zoom: same 2x3 layout; per panel, x window [min argmin / 3, max argmin * 3] and linear y from just below
     the smallest minimum to 1.35 x that minimum.
  B  ratio: current panels (unchanged limits) plus, under each row, a panel with criterion_r(b) / criterion_1(b)
     (Joe and OPC; the Archimax ratio panel is left empty because only r = 1 is shown there).
  C  zoom + ratio: the zoomed panels of A plus the ratio panels of B on the zoomed b window.
Writes ../../figures/fig72-error-vs-blocksize-d<delta>-zoom<V>.pdf only (no JSON output).  No randomness.
"""
import sys
import json
import numpy as np

sys.path.insert(0, '.')
import figstyle
from figstyle import plt, OKABE, TEXTWIDTH_IN
from s06_tables import crit_fn
from matplotlib.ticker import LogLocator, NullLocator, FuncFormatter, MaxNLocator, FixedLocator, NullFormatter


def sci_tex(y, pos):
    e = int(np.floor(np.log10(y) + 1e-12))
    m = y / 10 ** e
    return f'${m:.2g}\\cdot10^{{{e}}}$' if abs(m - 1) > 1e-9 else f'$10^{{{e}}}$'


def nice_xticks(ax, xl):
    """Plain-number major ticks on a narrow log b-window (avoids overlapping minor 'a x 10^k' labels)."""
    cand = np.array([m * 10.0 ** k for k in range(0, 5) for m in (1, 2, 3, 5)])
    t = cand[(cand >= xl[0]) & (cand <= xl[1])]
    ax.xaxis.set_major_locator(FixedLocator(t))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:g}$'))
    ax.xaxis.set_minor_formatter(NullFormatter())


def lin_tex(scale_exp):
    def f(y, pos):
        return f'${y / 10 ** scale_exp:.3g}$'
    return FuncFormatter(f)


args = sys.argv[1:]
delta = 0.15
variant = 'C'
for key in ('--delta', '--variant'):
    if key in args:
        i = args.index(key)
        if key == '--delta':
            delta = float(args[i + 1])
        else:
            variant = args[i + 1].upper()
        del args[i:i + 2]
assert variant in ('A', 'B', 'C')
ns = [float(x) for x in args] or [1e4, 1e5]
tag = f'd{int(round(delta * 100)):03d}'
ZOOM = variant in ('A', 'C')
RATIO = variant in ('B', 'C')

figstyle.setup()
C = json.load(open('../outputs/constants_theta2.json'))
K = C['K'][str(delta)]
MODELS = [('Joe', 'Joe'), ('OPC', 'outer power Clayton'), ('Archimax', r'Archimax $\psi_2$')]
SPEC = [('r1', False, '$r=1$', OKABE['black'], '-', 1.0),
        ('r2opt', False, '$r=2$', OKABE['blue'], '-', 2.0),
        ('r2opt', True, '$r=2$, corrected', OKABE['vermillion'], '--', 2.0)]
UNSUPPORTED = {('Archimax', 'r2opt')}

nr = len(ns)
if RATIO:
    hr = []
    for _ in ns:
        hr += [1.0, 0.55]
    fig, axes_all = plt.subplots(2 * nr, 3, figsize=(TEXTWIDTH_IN, 0.30 * TEXTWIDTH_IN * nr * 1.55 + 0.25),
                                 squeeze=False, gridspec_kw=dict(height_ratios=hr))
    main_rows = [2 * i for i in range(nr)]
else:
    fig, axes_all = plt.subplots(nr, 3, figsize=(TEXTWIDTH_IN, 0.30 * TEXTWIDTH_IN * nr + 0.25), squeeze=False)
    main_rows = list(range(nr))

for row, n in enumerate(ns):
    b = np.geomspace(1, n / 50, 800)
    n13 = n ** (1 / 3)
    for col, (M, title) in enumerate(MODELS):
        ax = axes_all[main_rows[row], col]
        curves = []
        for est, corr, lab, c, ls, bmin in SPEC:
            if (M, est) in UNSUPPORTED:
                continue
            f = crit_fn(M, est, K, corr)
            bb = b[b >= bmin]
            y = f(bb, n)
            curves.append((bb, y, lab, c, ls, corr))
            ax.plot(bb, y, ls, color=c, lw=1.2, label=lab)
            j = int(np.argmin(y))
            ax.plot(bb[j], y[j], 'o', color=c, ms=3.0, mfc=c if not corr else 'white', mew=0.9, zorder=5)
        ax.axvline(n13, color='0.55', lw=0.6, ls=':')
        argmins = [cb[int(np.argmin(cy))] for cb, cy, *_ in curves]
        lo = min(cy.min() for _, cy, *_ in curves)
        ax.set_xscale('log')
        if ZOOM:
            xl = (max(1.0, min(argmins) / 3), min(n / 50, max(argmins) * 3))
            ax.set_xlim(*xl)
            nice_xticks(ax, xl)
            ax.set_ylim(lo * 0.985, lo * 1.35)
            e = int(np.floor(np.log10(lo)))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_formatter(lin_tex(e))
            ax.text(0.0, 1.01, f'$\\times10^{{{e}}}$', transform=ax.transAxes, ha='left', va='bottom', fontsize=7)
        else:
            xl = (1, n / 50)
            ax.set_yscale('log')
            ax.set_ylim(lo / 1.12, lo * 2.6)
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(sci_tex))
            ax.set_xlim(*xl)
        if any(m_ == M for m_, _ in UNSUPPORTED):
            ax.text(0.97, 0.05, '$r=2$: not\ndetermined at\nleading order', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=7, color='0.3')
        ax.grid(alpha=0.25, lw=0.4, which='major')
        if row == 0:
            ax.set_title(title, pad=12 if ZOOM else None)
        if col == 0:
            ax.set_ylabel(f'criterion, $n=10^{{{int(round(np.log10(n)))}}}$')
        last_main = (row == nr - 1) and not RATIO
        if last_main:
            ax.set_xlabel('block size $b$')
        if RATIO:
            axr = axes_all[main_rows[row] + 1, col]
            if any(m_ == M for m_, _ in UNSUPPORTED):
                axr.axis('off')
                continue
            bb1, y1 = curves[0][0], curves[0][1]
            axr.axhline(1.0, color=OKABE['black'], lw=1.0)
            rmin, rmax = 1.0, 1.0
            for bb, y, lab, c, ls, corr in curves[1:]:
                ref = y1[np.searchsorted(bb1, bb)]
                assert np.array_equal(bb1[np.searchsorted(bb1, bb)], bb)
                ratio = y / ref
                axr.plot(bb, ratio, ls, color=c, lw=1.2)
                inwin = (bb >= xl[0]) & (bb <= xl[1])
                rmin, rmax = min(rmin, ratio[inwin].min()), max(rmax, ratio[inwin].max())
            axr.axvline(n13, color='0.55', lw=0.6, ls=':')
            axr.set_xscale('log')
            axr.set_xlim(*xl)
            if ZOOM:
                nice_xticks(axr, xl)
            pad = 0.06 * (rmax - rmin)
            axr.set_ylim(rmin - pad, rmax + pad)
            axr.yaxis.set_major_locator(MaxNLocator(3))
            axr.grid(alpha=0.25, lw=0.4, which='major')
            ax.tick_params(labelbottom=False)
            if col == 0:
                axr.set_ylabel('ratio to $r=1$')
            if row == nr - 1:
                axr.set_xlabel('block size $b$')
if RATIO:
    # Archimax: no ratio panel, put the b label on the main panel of the last row
    axes_all[main_rows[-1], 2].set_xlabel('block size $b$')
h, l = axes_all[0, 0].get_legend_handles_labels()
fig.tight_layout(h_pad=0.4 if RATIO else 0.5, w_pad=0.6, rect=(0, 0.045 if RATIO else 0.06, 1, 1))
fig.align_ylabels(axes_all[:, 0])
fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, handlelength=2.4, bbox_to_anchor=(0.5, 0.0))
path = f'../../figures/fig72-error-vs-blocksize-{tag}-zoom{variant}.pdf'
fig.savefig(path)
print('wrote', path)
