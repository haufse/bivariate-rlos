"""s08: Figure 7.1 -- variance of optimally weighted r = 2, 3 relative to block maxima (r = 1),
at the Gumbel--Hougaard(theta) attractor.

Run:  python s08_fig71_efficiency.py [--delta D] [--no-r3] \
          > ../outputs/s08_output-d<delta>.txt        (default --delta 0.15)
Needs ../outputs/sigma_grid_theta<theta>.npz (s04) for the thetas below.  No randomness.
Writes ../../figures/fig71-efficiency-d<delta>[-nor3].pdf and ../outputs/fig71_data-d<delta>.json.
2026-11-18: delta option (T = [delta, 1-delta] via kquad, so delta = 0.15 works), --no-r3 switch, endpoint
report, theta = 1 check against s07 (closed form), manuscript notation T.

Plotted quantity (Corollary cor:oracle-maxima-comparison):
    RE_r(t) = sigma_opt^2(t) / {Sigma_{(1,1),(1,1)}(t)/d_11(t)^2} = sigma_opt^2(t) d_11(t)^2 / Sigma_{(1,1),(1,1)}(t),
and the integrated ratio  V_r/V_1 = int_T sigma_opt^2 / int_T Sigma_11/d_11^2,  T = [delta, 1-delta].
At theta = 1 (independence) the quantities refer to the Gaussian precursor before projection.
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, '.')
import rlos_lib as rl
import kquad
import figstyle
from figstyle import plt, OKABE, TEXTWIDTH_IN

figstyle.setup()
IDX = {2: [0, 1, 3, 4], 3: list(range(9))}
THETA_A = [1.0, 1.25, 1.5, 2.0, 3.0]
THETA_B = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0]
args = sys.argv[1:]
DELTA = 0.15
if '--delta' in args:
    DELTA = float(args[args.index('--delta') + 1])
WITH_R3 = '--no-r3' not in args
TAG = f'd{int(round(DELTA * 100)):03d}'
RS = (2, 3) if WITH_R3 else (2,)


def ratio_fns(theta, t, Sig):
    s1, o2, o3 = np.empty(len(t)), np.empty(len(t)), np.empty(len(t))
    for i, tt in enumerate(t):
        A = float(rl.A_log(tt, theta))
        d = rl.d_vec(A, tt, 3)
        s1[i] = Sig[i][0, 0] / d[0] ** 2
        for r, o in ((2, o2), (3, o3)):
            S = Sig[i][np.ix_(IDX[r], IDX[r])]
            dd = d[IDX[r]]
            o[i] = 1.0 / (dd @ np.linalg.solve(S, dd))
    return s1, o2, o3


def ratios(theta):
    D = np.load(f'../outputs/sigma_grid_theta{theta:g}.npz')
    t, w, pan, Sig = D['t'], D['w'], D['panel'], D['Sigma']
    out = {'t': t, 'w': w, 'panel': pan, 's1': np.empty(len(t)), 2: np.empty(len(t)), 3: np.empty(len(t))}
    for i, tt in enumerate(t):
        A = float(rl.A_log(tt, theta))
        d = rl.d_vec(A, tt, 3)
        out['s1'][i] = Sig[i][0, 0] / d[0] ** 2
        for r in (2, 3):
            S = Sig[i][np.ix_(IDX[r], IDX[r])]
            dd = d[IDX[r]]
            out[r][i] = 1.0 / (dd @ np.linalg.solve(S, dd))
    return out, list(D['edges'])


data = {}
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 0.42 * TEXTWIDTH_IN),
                         gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.28))
cols = [OKABE['black'], OKABE['sky'], OKABE['green'], OKABE['blue'], OKABE['vermillion']]
ax = axes[0]
for th, col in zip(THETA_A, cols):
    if not os.path.exists(f'../outputs/sigma_grid_theta{th:g}.npz'):
        print('missing theta', th)
        continue
    o, edges = ratios(th)
    sel = o['t'] >= 0.004
    tt = np.concatenate([o['t'][sel], (1 - o['t'][sel])[::-1]])
    print(f'theta={th:g}: endpoint behaviour, RE_2 and RE_3 at the smallest grid nodes: '
          + ', '.join(f't={o["t"][i]:.2e}: {o[2][i] / o["s1"][i]:.4f}/{o[3][i] / o["s1"][i]:.4f}' for i in (0, 2, 4, 8, 16)))
    for r, ls in ((2, '-'), (3, '--'))[:len(RS)]:
        y = o[r][sel] / o['s1'][sel]
        yy = np.concatenate([y, y[::-1]])
        ax.plot(tt, yy, ls, color=col, lw=1.1 if r == 2 else 0.9,
                label=(f'$\\theta={th:g}$' if r == 2 else None))
    data[str(th)] = dict(t=list(map(float, o['t'])), re2=list(map(float, o[2] / o['s1'])),
                         re3=list(map(float, o[3] / o['s1'])))
ax.set_xlim(0, 1)
ax.set_xlabel('$t$')
ax.set_ylabel(r'$\sigma_{\mathrm{opt}}^2(t)\,d_{11}(t)^2/\Sigma_{(1,1),(1,1)}(t)$')
if WITH_R3:
    ax.text(0.5, 0.98, 'solid: $r=2$, dashed: $r=3$', transform=ax.transAxes, ha='center', va='top', fontsize=8)
else:
    ax.text(0.5, 0.98, '$r=2$', transform=ax.transAxes, ha='center', va='top', fontsize=8)
ax.set_ylim(0.55, 1.005)
ax.legend(loc='lower center', ncol=3, handlelength=1.6, columnspacing=1.0)
ax.set_title('(a) pointwise variance ratio', loc='left')

ax = axes[1]
thb, v2, v3 = [], [], []
for th in THETA_B:
    if not os.path.exists(f'../outputs/sigma_grid_theta{th:g}.npz'):
        print('missing theta', th)
        continue
    D = np.load(f'../outputs/sigma_grid_theta{th:g}.npz')
    tK, wK, SK, interp = kquad.half_nodes(D, DELTA)
    s1, o2, o3 = ratio_fns(th, tK, SK)
    V1 = np.sum(wK * s1)
    thb.append(th)
    v2.append(np.sum(wK * o2) / V1)
    v3.append(np.sum(wK * o3) / V1)
    print(f'theta={th:g}: A(1/2)={float(rl.A_log(0.5, th)):.4f}  V2/V1={v2[-1]:.4f}  V3/V1={v3[-1]:.4f}  '
          f'(T=[{DELTA:g},{1 - DELTA:g}], interpolated panel: {interp})')
ax.plot(thb, v2, '-o', color=OKABE['blue'], ms=3, label='$r=2$')
if WITH_R3:
    ax.plot(thb, v3, '--s', color=OKABE['vermillion'], ms=3, label='$r=3$')
ax.set_xscale('log')
ax.set_xticks([1, 1.5, 2, 3, 4, 6])
ax.set_xticklabels(['1', '1.5', '2', '3', '4', '6'])
ax.minorticks_off()
ax.set_xlabel(r'Gumbel--Hougaard parameter $\theta$')
ax.set_ylabel(f'$V_r/V_1$ on $T=[{DELTA:g},{1 - DELTA:g}]$')
ax.legend(loc='upper right')
ax.set_title('(b) integrated variance ratio', loc='left')
for a in axes:
    a.grid(alpha=0.25, lw=0.4)
path = f'../../figures/fig71-efficiency-{TAG}{"" if WITH_R3 else "-nor3"}.pdf'
fig.savefig(path)
data['integrated'] = dict(theta=thb, V2_over_V1=v2, V3_over_V1=v3, T=[DELTA, 1 - DELTA])
json.dump(data, open(f'../outputs/fig71_data-{TAG}.json', 'w'), indent=1)
print('wrote', path)
