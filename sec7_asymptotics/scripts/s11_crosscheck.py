"""s11: independent cross-checks of the Section 7 numbers (2026-11-18).

Run:  NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-9 nice -n 19 \
      python s11_crosscheck.py > ../outputs/s11_output.txt
Needs ../outputs/sigma_grid_theta{2,3}.npz (s04, computed on a Slurm cluster) and ../outputs/constants_theta2.json (s05).
No randomness.  Runtime about 2 min on 8 threads.

(1) Cluster grid vs local recomputation: rlos_lib.sigma at two s04 grid nodes (theta = 2).
(2) Interpolation used for T = [0.15, 0.85] (kquad): direct rlos_lib.sigma (r = 3) at off-grid t vs the
    barycentric interpolant, theta = 2 and 3.
(3) Pointwise sigma_opt^2 at t = 0.3, 0.5 exactly (r = 1, 2, 3; direct quadrature), and r = 2 from the
    independent implementation sigma2_lib (sec8_simulations/sigma2_lib.py), against earlier provisional values
    (r2/r1 = 0.815 at t = 0.3, 0.801 at t = 0.5).
(4) Second route for the table constants (r <= 2 and POT) on T = [0.15, 0.85] and [0.01, 0.99]:
    Sigma_2 from sigma2_lib (vectorised numpy code, independent of rlos_lib's numba kernel), d from
    sigma2_lib.d_vec (exact derivative of eq. mu-r2), calR from the closed form eq. (integrated-R-r2)
    (rlos_lib.calR_r2_closed), calS from rlos_lib.calS_vec (checked symbolically in s01), integrated with
    a fresh composite Gauss-Legendre rule in t directly on [delta, 1/2] (no s04 grid, no interpolation);
    POT integrals by scipy.integrate.quad.  Optimized constants recomputed with the closed forms of s06.
"""
import sys
import json
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, '.')
sys.path.insert(0, '../../sec8_simulations')
import rlos_lib as rl
import kquad
import sigma2_lib as s2

np.set_printoptions(linewidth=150)
IDX2 = [0, 1, 3, 4]

# ------------------------------------------------------------------ (1)
D2 = np.load('../outputs/sigma_grid_theta2.npz')
print('(1) cluster grid vs local recomputation (theta = 2, r = 3, npan=30, m=8):')
for i in (20, 40):
    S_loc = rl.sigma(float(D2['t'][i]), 2.0, 3, npan=30, m=8)
    print(f'    node t={D2["t"][i]:.5f}: max abs diff = {np.max(np.abs(S_loc - D2["Sigma"][i])):.2e}')

# ------------------------------------------------------------------ (2)
print('\n(2) interpolation (kquad) vs direct quadrature at off-grid t:')
D3 = np.load('../outputs/sigma_grid_theta3.npz')
worst = 0.0
for th, D, ts in ((2.0, D2, (0.15, 0.2, 0.27, 0.42)), (3.0, D3, (0.15, 0.23))):
    for tq in ts:
        Sd = rl.sigma(tq, th, 3, npan=30, m=8)
        Si = kquad.sigma_at(D, tq)
        e = np.max(np.abs(Sd - Si)) / np.max(np.abs(Sd))
        worst = max(worst, e)
        print(f'    theta={th:g} t={tq}: max abs diff / max entry = {e:.2e}')
print(f'    worst relative interpolation error: {worst:.2e}')

# ------------------------------------------------------------------ (3)
print('\n(3) pointwise sigma_opt^2 at the GH(2) attractor, direct quadrature:')
for tq in (0.3, 0.5):
    S3 = rl.sigma(tq, 2.0, 3, npan=30, m=8)
    A = float(rl.A_log(tq, 2.0))
    d3 = rl.d_vec(A, tq, 3)
    v1 = S3[0, 0] / d3[0] ** 2
    S2r = S3[np.ix_(IDX2, IDX2)]
    v2 = 1 / (d3[IDX2] @ np.linalg.solve(S2r, d3[IDX2]))
    v3 = 1 / (d3 @ np.linalg.solve(S3, d3))
    S2b = s2.sigma2(s2.Gumbel(2.0), tq, npan=45, m=10)
    d2b = s2.d_vec(A, tq)
    v2b = 1 / (d2b @ np.linalg.solve(S2b, d2b))
    print(f'    t={tq}: r1 {v1:.6f}  r2 {v2:.6f}  r3 {v3:.6f}  | r2/r1 {v2 / v1:.4f}  r3/r1 {v3 / v1:.4f}  r3/r2 {v3 / v2:.4f}'
          f'  | sigma2_lib r2 {v2b:.6f} (rel diff {v2b / v2 - 1:.1e}), max|Sigma_2 diff| {np.max(np.abs(S2b - S2r)):.1e}')

# ------------------------------------------------------------------ (4)
C = json.load(open('../outputs/constants_theta2.json'))


def gl_composite(lo, hi, npan, m):
    e = np.linspace(lo, hi, npan + 1)
    g, w = np.polynomial.legendre.leggauss(m)
    T = np.concatenate([0.5 * (b - a) * g + 0.5 * (b + a) for a, b in zip(e[:-1], e[1:])])
    W = np.concatenate([0.5 * (b - a) * w for a, b in zip(e[:-1], e[1:])])
    return T, W


def opt_power(V, B, g):
    return (1 + 2 * g) * (2 * g) ** (-2 * g / (1 + 2 * g)) * B ** (1 / (1 + 2 * g)) * V ** (2 * g / (1 + 2 * g))


print('\n(4) second route for the integrated constants (sigma2_lib + closed-form R, fresh t-rule), '
      'relative difference to s05/s06:')
for delta in (0.15, 0.01):
    if delta < 0.05:
        T1, W1 = gl_composite(delta, 0.1, 3, 12)
        T2, W2 = gl_composite(0.1, 0.5, 4, 12)
        T, W = np.concatenate([T1, T2]), np.concatenate([W1, W2])
    else:
        T, W = gl_composite(delta, 0.5, 4, 12)
    acc = dict(V1=0.0, V2=0.0, BRR=0.0)
    for M in ('Joe', 'Archimax'):
        acc['BSS1_' + M] = 0.0
        acc['BSS2_' + M] = 0.0
        acc['BSR2_' + M] = 0.0
    for tt, ww in zip(T, W):
        A = float(rl.A_log(tt, 2.0))
        v = float(rl.dA_log(tt, 2.0))
        S = s2.sigma2(s2.Gumbel(2.0), tt, npan=30, m=8)
        d = s2.d_vec(A, tt)
        x = np.linalg.solve(S, d)
        lam = x / (d @ x)
        R = rl.calR_r2_closed(A, v, tt)
        acc['V1'] += 2 * ww * S[0, 0] / d[0] ** 2
        acc['V2'] += 2 * ww * lam @ S @ lam
        acc['BRR'] += 2 * ww * (lam @ R) ** 2
        for M in ('Joe', 'Archimax'):
            Sv = rl.calS_vec(M, tt, 2)
            acc['BSS1_' + M] += 2 * ww * (Sv[0] / d[0]) ** 2
            acc['BSS2_' + M] += 2 * ww * (lam @ Sv) ** 2
            acc['BSR2_' + M] += 2 * ww * (lam @ Sv) * (lam @ R)
    Vp = 2 * quad(lambda t: float(rl.pot_var(t, 2.0)), delta, 0.5, epsabs=1e-14, epsrel=1e-12)[0]
    BpD = 2 * quad(lambda t: float(rl.pot_bias_fn('GH', t)) ** 2, delta, 0.5, epsabs=1e-16, epsrel=1e-12)[0]
    BpJ = 2 * quad(lambda t: float(rl.pot_bias_fn('Joe', t)) ** 2, delta, 0.5, epsabs=1e-16, epsrel=1e-12)[0]
    K = C['K'][str(delta)]
    ref = dict(V1=K['r1']['V'], V2=K['r2opt']['V'], BRR=K['r2opt']['BRR'],
               BSS1_Joe=K['r1']['BSS_Joe'], BSS2_Joe=K['r2opt']['BSS_Joe'], BSR2_Joe=K['r2opt']['BSR_Joe'],
               BSS1_Archimax=K['r1']['BSS_Archimax'], BSS2_Archimax=K['r2opt']['BSS_Archimax'],
               BSR2_Archimax=K['r2opt']['BSR_Archimax'])
    print(f'  T = [{delta}, {1 - delta}]  ({len(T)} t-nodes on [delta, 1/2])')
    for k in ref:
        print(f'    {k:14s} route 2 {acc[k]: .8e}   s05 {ref[k]: .8e}   rel diff {acc[k] / ref[k] - 1: .1e}')
    for k, a, b in (('V_p', Vp, K['pot']['V']), ('B_p[GH]', BpD, K['pot']['BSS_GH']), ('B_p[Joe]', BpJ, K['pot']['BSS_Joe'])):
        print(f'    {k:14s} quad    {a: .8e}   s05 {b: .8e}   rel diff {a / b - 1: .1e}')
    # optimized constants (s06 closed forms) for selected cells
    ka = 0.5
    cells = {
        'GH r=1 (V1)': acc['V1'],
        'GH r=2': opt_power(acc['V2'], acc['BRR'], 1.0),
        'Joe r=1': opt_power(acc['V1'], ka ** 2 * acc['BSS1_Joe'], 1.0),
        'Joe r=2': opt_power(acc['V2'], ka ** 2 * acc['BSS2_Joe'] + 2 * ka * acc['BSR2_Joe'] + acc['BRR'], 1.0),
        'OPC r=2': opt_power(acc['V2'], ka ** 2 * acc['BSS2_Joe'] - 2 * ka * acc['BSR2_Joe'] + acc['BRR'], 1.0),
        'Joe/OPC r=2 bc': opt_power(acc['V2'], ka ** 2 * acc['BSS2_Joe'], 1.0),
        'Archimax r=1': opt_power(acc['V1'], acc['BSS1_Archimax'], 2.0),
        'POT GH/Archimax': opt_power(Vp, 0.25 * BpD, 1.0),
        'POT OPC': opt_power(Vp, BpD, 1.0),
        'POT Joe': opt_power(Vp, BpJ, 2.0),
    }
    S6 = json.load(open('../outputs/ameise_summary.json'))[str(delta)]
    refc = {'GH r=1 (V1)': S6['GH']['r=1'], 'GH r=2': S6['GH']['r=2'], 'Joe r=1': S6['Joe']['r=1'],
            'Joe r=2': S6['Joe']['r=2'], 'OPC r=2': S6['OPC']['r=2'], 'Joe/OPC r=2 bc': S6['Joe']['r=2 bc'],
            'Archimax r=1': S6['Archimax']['r=1'], 'POT GH/Archimax': S6['GH']['POT'],
            'POT OPC': S6['OPC']['POT'], 'POT Joe': S6['Joe']['POT']}
    for k, v in cells.items():
        r = refc[k]['leading']['const']
        print(f'    table cell {k:16s} route 2 {v:.6g}   s06 {r:.6g}   rel diff {v / r - 1: .1e}')
