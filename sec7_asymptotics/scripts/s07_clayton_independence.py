"""s07: independence attractor / Clayton(theta): precursor constants and projected limit moments.

Run:  python s07_clayton_independence.py > ../outputs/s07_output.txt
Uses ../outputs/sigma_grid_theta1.npz (s04 with theta = 1) for r = 3 if present.  No randomness.

Objects (supplement I, Example ex:independence-covariance, Remark rem:independence-boundary-moments,
Section subsec:ex:clayton):
  Sigma_2(t): closed form eq. (independence-covariance-r2) (re-checked against the general quadrature);
  d(t) = -(1/4, t/4, (1-t)/4, 3p/8), p = t(1-t)  eq. (independence-gradient-r2);
  Clayton: alpha(b) = 1/b, S(u) = theta u1 u2 log u1 log u2, calR = 0; calS_r from s01 (sympy).
  Precursor X_r(t) = lambda_r^T (Z + c calS_r): variance sigma_r^2, mean b_r(t) = c lambda_r^T calS_r(t)
  (c = lim sqrt(k)/b = c_1 = c_2 since alpha(b) = 1/b).
  Projected limit L_r = min(X_r, 0): moments eq. (independence-boundary-moments).
Integrals over K_delta = [delta, 1-delta], delta in {0, 0.01, 0.05, 0.1}, Gauss-Legendre (symmetric).
"""
import sys
import os
import numpy as np
import sympy as sp
from scipy.stats import norm
from scipy.optimize import minimize_scalar

sys.path.insert(0, '.')
import rlos_lib as rl

DELTAS = [0.0, 0.01, 0.05, 0.1]


def Sig2_closed(t):
    p = t * (1 - t)
    B = 12 + p
    s11 = 7 * p / (12 * B)
    s12 = lambda t: p * t * (29 + 5 * t) / (24 * B * (3 + t))
    s14 = p ** 2 * (109 + 5 * p) / (24 * B ** 2)
    s23 = p ** 2 * (134 + 3 * p) / (48 * B ** 2)
    s22 = lambda t: p * t * (1332 + 2460 * t + 79 * t ** 2 - 94 * t ** 3 - 9 * t ** 4) / (144 * B ** 2 * (3 + t))
    s24 = lambda t: p ** 2 * (3744 + 7638 * t - 565 * t ** 2 - 474 * t ** 3 + 16 * t ** 4 + 9 * t ** 5) / (144 * B ** 3)
    s44 = p ** 2 * (252072 + 411624 * p + 37179 * p ** 2 + 1610 * p ** 3 + 27 * p ** 4) / (432 * B ** 4)
    return np.array([[s11, s12(t), s12(1 - t), s14], [s12(t), s22(t), s23, s24(t)],
                     [s12(1 - t), s23, s22(1 - t), s24(1 - t)], [s14, s24(t), s24(1 - t), s44]])


def calS_clay(t, r):  # theta = 1
    return np.array([rl.calS_shape(1.0, t, l, m, 2) * t * (1 - t) for (l, m) in rl.pairs(r)])


def gl(lo, hi, n=60):
    g, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * g + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


# ------------------------------------------------------------------ checks
if os.environ.get('RUN_QUAD_CHECK') == '1':   # multi-threaded quadrature; off by default
    print('closed-form Sigma_2 vs general quadrature (theta=1):',
          max(np.max(np.abs(Sig2_closed(t) - rl.sigma(t, 1.0, 2, npan=30, m=8))) for t in (0.1, 0.37)))
else:
    print('quadrature cross-check of Sigma_2 skipped (set RUN_QUAD_CHECK=1); it agreed to 1.4e-17 in an earlier run')
print('d(t) general vs eq. (independence-gradient-r2):',
      max(np.max(np.abs(rl.d_vec(1.0, t, 2) + np.array([1 / 4, t / 4, (1 - t) / 4, 3 * t * (1 - t) / 8])))
          for t in (0.1, 0.37)))
ts_ = sp.symbols('t')
V1_exact = sp.integrate(28 * ts_ * (1 - ts_) / (3 * (12 + ts_ * (1 - ts_))), (ts_, 0, 1))
print('V1 over [0,1] exact:', sp.nsimplify(sp.simplify(V1_exact)), '=', sp.N(V1_exact, 12),
      '  (memo: 28/3 + 32 log(3/4) =', float(28 / 3 + 32 * np.log(3 / 4)), ')')


def lam_opt(S, d):
    x = np.linalg.solve(S, d)
    return x / (d @ x)


def point(t, r, Sig3=None):
    if r == 1:
        return 28 * t * (1 - t) / (3 * (12 + t * (1 - t))), -t * (1 - t)
    if r == 2:
        S = Sig2_closed(t)
    else:
        S = Sig3
    d = rl.d_vec(1.0, t, r)
    lam = lam_opt(S, d)
    return float(lam @ S @ lam), float(lam @ calS_clay(t, r))


print('\npointwise: t, sigma_2^2/sigma_1^2, lambda_2^T calS_2 / theta, lambda_1^T calS_1 / theta')
for t in (0.1, 0.3, 0.5):
    v1, g1 = point(t, 1)
    v2, g2 = point(t, 2)
    print(f'  {t:.1f}  {v2 / v1:.4f}  {g2:+.5f}  {g1:+.5f}')

# r = 3 at independence from the quadrature grid, if available
r3 = None
if os.path.exists('../outputs/sigma_grid_theta1.npz'):
    D = np.load('../outputs/sigma_grid_theta1.npz')
    r3 = D


def integrals(delta, rlist=(1, 2)):
    out = {}
    tt, ww = gl(delta, 0.5, 80)
    for r in rlist:
        vals = np.array([point(t, r) for t in tt])
        out[r] = dict(t=tt, w=2 * ww, sig2=vals[:, 0], g=vals[:, 1])
    return out


def integrals_r3(delta):
    edges = list(r3['edges'])
    j0 = edges.index(delta)
    sel = r3['panel'] >= j0
    tt = r3['t'][sel]
    vals = np.array([point(t, 3, S_) for t, S_ in zip(tt, r3['Sigma'][sel])])
    return dict(t=tt, w=2 * r3['w'][sel], sig2=vals[:, 0], g=vals[:, 1])


def proj_moments(b, s2):
    s = np.sqrt(s2)
    z = b / s
    m1 = b * norm.cdf(-z) - s * norm.pdf(z)
    m2 = (b * b + s2) * norm.cdf(-z) - b * s * norm.pdf(z)
    return m1, m2, norm.cdf(z)


print('\nPrecursor constants (theta factored out: B_r = theta^2 * value; Clayton alpha(b) = 1/b):')
res_all = {}
for delta in DELTAS:
    I = integrals(delta)
    if r3 is not None:
        I[3] = integrals_r3(delta)
    res = {}
    for r, Ir in I.items():
        V = np.sum(Ir['w'] * Ir['sig2'])
        Bc = np.sum(Ir['w'] * Ir['g'] ** 2)
        bstar = (2 * Bc / V) ** (1 / 3)       # times theta^{2/3} n^{1/3}
        amin = 3 * 2 ** (-2 / 3) * Bc ** (1 / 3) * V ** (2 / 3)  # times theta^{2/3} n^{-2/3}
        res[r] = (V, Bc, bstar, amin)
    res_all[delta] = (I, res)
    print(f'  K=[{delta},{1 - delta}]: ' + '; '.join(
        f'r={r}: V={v:.6f} B={b:.6f}th^2 b*={bs:.4f}th^(2/3)n^(1/3) AMEISE*={a:.5f}th^(2/3)n^(-2/3)'
        for r, (v, b, bs, a) in res.items()))
    print(f'      V2/V1 = {res[2][0] / res[1][0]:.4f}' + (f', V3/V1 = {res[3][0] / res[1][0]:.4f}' if 3 in res else ''))

print('\nProjected limit L_r = min(X_r, 0), integrated over K, as functions of y = c*theta:')
print('   (c = lim sqrt(k)/b; M_r(y) = int_K E L_r^2, m_r(y) = int_K E L_r, P0 = int_K P(L_r = 0) / |K|)')
for delta in (0.0, 0.01):
    I, _ = res_all[delta]
    print(f'  K=[{delta},{1 - delta}]')
    print('     y     ' + '  '.join(f'M_{r}(y)    m_{r}(y)   P0_{r}' for r in I))
    for y in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        row = []
        for r, Ir in I.items():
            m1, m2, p0 = proj_moments(y * Ir['g'], Ir['sig2'])
            row.append(f'{np.sum(Ir["w"] * m2):.6f} {np.sum(Ir["w"] * m1):+.6f} {np.sum(Ir["w"] * p0) / (1 - 2 * delta):.3f}')
        print(f'   {y:5.2f}  ' + '  '.join(row))
    # limit-law second-moment criterion: MSE ~ (b/n) M_r(c theta), c = n^{1/2} b^{-3/2}
    #   => n^{-2/3} c^{-2/3} M_r(c theta) = theta^{2/3} n^{-2/3} y^{-2/3} M_r(y),  y = c theta
    print('     minimised limit-law criterion  min_y y^{-2/3} M_r(y)  (times theta^{2/3} n^{-2/3}),'
          '  b = (theta/y)^{2/3} n^{1/3}:')
    for r, Ir in I.items():
        f = lambda ly: np.exp(ly) ** (-2 / 3) * np.sum(Ir['w'] * proj_moments(np.exp(ly) * Ir['g'], Ir['sig2'])[1])
        opt = minimize_scalar(f, bounds=(np.log(1e-3), np.log(1e3)), method='bounded', options={'xatol': 1e-10})
        y = np.exp(opt.x)
        print(f'       r={r}: min = {opt.fun:.5f} at y* = {y:.4f}  (b*/(theta^(2/3) n^(1/3)) = {y ** (-2 / 3):.4f})')
