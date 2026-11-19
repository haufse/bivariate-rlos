"""s05: assemble pointwise and integrated leading-order constants at the common GH(2) attractor.

Run:  python s05_assemble.py > ../outputs/s05_output.txt
Needs ../outputs/sigma_grid_theta2.npz (s04).  No randomness.
Writes ../outputs/constants_theta2.json.

Definitions (eq. ameise-trimmed of the paper, with the Section 4/5 notation):
    sigma^2(t) = lambda(t)^T Sigma(t) lambda(t),   s(t) = lambda(t)^T calS(t),   q(t) = lambda(t)^T calR(t),
    V = int_K sigma^2,  B_SS = int_K s^2,  B_SR = int_K s q,  B_RR = int_K q^2,
for the weights
    r=1:      lambda = 1/d_11 = -(1+A)^2           (block-maxima madogram, W = e11 e11^T)
    r=2,3 opt: lambda_opt = Sigma^{-1} d/(d^T Sigma^{-1} d)  (Prop. prop:gmm-optimal-weights; feasible by
              Thm thm:feasible-efficient-weights)
    r=2,3 id:  lambda_I = d/(d^T d)                    (W = I)
POT (BVZ 2019 Prop. 4.1(i)):  V_p = int_K Var{A_p(t)},  B_p = int_K S_p(1-t,t)^2.
Integration: K_delta = [delta, 1-delta] (the manuscript's trimming set T), delta in {0, 0.01, 0.05, 0.1, 0.15};
by exchangeability each scalar integrand is symmetric about 1/2 (checked below at t = 0.3), so
int_K = 2 int_delta^{1/2}.  delta = 0.15 (K of the Section 8 Monte Carlo) is not an s04 panel edge; it is
handled by kquad.half_nodes (16-node GL on [0.15, 0.3], Sigma interpolated within the s04 panel [0.1, 0.3]).
2026-11-18: delta = 0.15 added; integration routed through kquad.
"""
import sys
import json
import numpy as np

sys.path.insert(0, '.')
import rlos_lib as rl
import kquad

TH = 2.0
DELTAS = [0.0, 0.01, 0.05, 0.1, 0.15]
MODELS = ['GH', 'Joe', 'OPC', 'Archimax']
IDX = {1: [0], 2: [0, 1, 3, 4], 3: list(range(9))}


def pointwise(t, Sig3, theta=TH, models=MODELS):
    A = float(rl.A_log(t, theta))
    v = float(rl.dA_log(t, theta))
    d3 = rl.d_vec(A, t, 3)
    R3 = rl.calR_vec(A, v, t, 3)
    S3 = {M: rl.calS_vec(M, t, 3) for M in models}
    out = {}
    for est in ('r1', 'r2opt', 'r2id', 'r3opt', 'r3id'):
        r = int(est[1])
        idx = IDX[r]
        S = Sig3[np.ix_(idx, idx)]
        d = d3[idx]
        if r == 1:
            lam = np.array([1.0 / d[0]])
        elif est.endswith('opt'):
            x = np.linalg.solve(S, d)
            lam = x / (d @ x)
        else:
            lam = d / (d @ d)
        o = {'sig2': float(lam @ S @ lam), 'q': float(lam @ R3[idx])}
        for M in models:
            o['s_' + M] = float(lam @ S3[M][idx])
        out[est] = o
    out['pot'] = {'sig2': float(rl.pot_var(t, theta))}
    for M in models:
        out['pot']['s_' + M] = float(rl.pot_bias_fn(M, t))
    return out


if __name__ == '__main__':
    D = np.load('../outputs/sigma_grid_theta2.npz')
    t, w, pan, Sig = D['t'], D['w'], D['panel'], D['Sigma']
    edges = list(D['edges'])
    pts = [pointwise(tt, S_) for tt, S_ in zip(t, Sig)]

    # ---------------- symmetry check at t = 0.3 vs 0.7
    S07 = rl.sigma(0.7, TH, 3, npan=30, m=8)
    S03 = rl.sigma(0.3, TH, 3, npan=30, m=8)
    p03, p07 = pointwise(0.3, S03), pointwise(0.7, S07)
    asym = max(abs(p03[e][k] - p07[e][k]) for e in p03 for k in p03[e])
    print(f'symmetry check f(0.3) vs f(0.7), all scalar integrands: max abs diff = {asym:.2e}')
    # ---------------- quadrature convergence of Sigma at t = 0.3
    S03b = rl.sigma(0.3, TH, 3, npan=45, m=10)
    print(f'Sigma(0.3): npan=30,m=8 vs npan=45,m=10: max abs diff = {np.max(np.abs(S03 - S03b)):.2e}'
          f' (max entry {np.max(np.abs(S03)):.2e})')

    # ---------------- pointwise relative efficiencies (for text)
    print('\npointwise sigma^2(t) and ratios to r=1 at the grid nodes nearest to 0.1, 0.3, 0.5:')
    for target in (0.1, 0.3, 0.5):
        i = int(np.argmin(np.abs(t - target)))
        p = pts[i]
        print(f'  t={t[i]:.4f}: r1 {p["r1"]["sig2"]:.6f}; r2opt/r1 {p["r2opt"]["sig2"] / p["r1"]["sig2"]:.4f};'
              f' r3opt/r1 {p["r3opt"]["sig2"] / p["r1"]["sig2"]:.4f}; r2id/r1 {p["r2id"]["sig2"] / p["r1"]["sig2"]:.4f};'
              f' POT/r1 {p["pot"]["sig2"] / p["r1"]["sig2"]:.4f}')
    # small-t behaviour of the inverse weights
    print('\nsmall-t nodes: t, sigma^2 r1, r2opt/r1, r3opt/r1, cond(Sigma_3)')
    for i in range(0, 8, 2):
        c3 = np.linalg.cond(Sig[i])
        print(f'  {t[i]:.5f} {pts[i]["r1"]["sig2"]:.3e} {pts[i]["r2opt"]["sig2"] / pts[i]["r1"]["sig2"]:.4f} '
              f'{pts[i]["r3opt"]["sig2"] / pts[i]["r1"]["sig2"]:.4f} {c3:.1e}')

    # ---------------- integrated constants
    res = {'theta': TH, 'deltas': DELTAS, 'K': {}}
    for delta in DELTAS:
        tK, wK, SK, interp = kquad.half_nodes(D, delta)
        ww = 2 * wK
        P = [pointwise(tt, S_) for tt, S_ in zip(tK, SK)]
        Kres = {'interpolated': interp, 'n_nodes_half': int(len(tK))}
        for est in ('r1', 'r2opt', 'r2id', 'r3opt', 'r3id', 'pot'):
            e = {'V': float(np.sum(ww * np.array([p[est]['sig2'] for p in P])))}
            for M in MODELS:
                s = np.array([p[est]['s_' + M] for p in P])
                e['BSS_' + M] = float(np.sum(ww * s * s))
                if est != 'pot':
                    q = np.array([p[est]['q'] for p in P])
                    e['BSR_' + M] = float(np.sum(ww * s * q))
            if est != 'pot':
                q = np.array([p[est]['q'] for p in P])
                e['BRR'] = float(np.sum(ww * q * q))
            Kres[est] = e
        res['K'][str(delta)] = Kres
    # pointwise arrays for figures
    res['pointwise'] = {'t': list(map(float, t)),
                        'w': list(map(float, w)),
                        'panel': list(map(int, pan)),
                        'data': pts}
    json.dump(res, open('../outputs/constants_theta2.json', 'w'), indent=1)

    print('\nIntegrated constants (alpha normalisations: GH S=0; Joe, OPC alpha=(2b)^{-1}; Archimax alpha=b^{-2};'
          ' POT: GH/Archimax alpha_p=(2b)^{-1}, OPC 1/b, Joe b^{-2})')
    for delta in DELTAS:
        Kres = res['K'][str(delta)]
        print(f'\nK = [{delta}, {1 - delta}]')
        for est, e in Kres.items():
            if not isinstance(e, dict):
                continue
            line = f'  {est:6s} V={e["V"]:.7g}'
            if 'BRR' in e:
                line += f'  B_RR={e["BRR"]:.6g}'
            for M in MODELS[1:]:
                line += f'  B_SS[{M}]={e["BSS_" + M]:.6g}'
                if 'BSR_' + M in e:
                    line += f' B_SR[{M}]={e["BSR_" + M]:.6g}'
            if est == 'pot':
                line += f'  B_p[GH]={e["BSS_GH"]:.6g}'
            print(line)
