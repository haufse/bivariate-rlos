"""s02: exact finite-block check of the integrated bias vectors calS, calR (Lemma lem:finiteblockbias,
eq. rank-moment-process), independent of the closed forms used in rlos_lib.

Run:  python s02_finite_block_bias_check.py > ../outputs/s02_output.txt
No randomness.

For known block-order-statistic margins, (U_{l:b,1}, U_{m:b,2}) has copula C_b^{(l,m)}, hence
    E exp{-xi^{o,(l,m)}(t)} - mu^{(l,m)}(A_0(t),t) = - int_0^1 {C_b^{(l,m)} - C^{(l,m)}}(h^{(l,m)}(u,t)) du =: beta_b.
C_b^{(l,m)} is evaluated EXACTLY (mpmath, 40 digits): with y_i solving F_{l:b}(y_1) = u_1, F_{m:b}(y_2) = u_2,
F_{l:b}(y) = sum_{j<l} binom(b,j)(1-y)^j y^{b-j}, and the multinomial law of the exceedance counts of
(y_1,y_2) among b iid copies with copula C.  By Lemma lem:finiteblockbias,
    -beta_b = alpha(b) calS + b^{-1} calR + o(alpha(b) + 1/b).
Hence  -b*beta_b -> kappa_alpha * calS + calR  for alpha(b) = kappa_alpha / b  (Joe, OPC: 1/2; Clayton, GH: 1 resp. S=0),
and    -b*beta_b -> calR for Archimax psi_2 (alpha = b^{-2}).
Limits are extrapolated from b = 100, 200, 400, 800 (two-level Richardson in 1/b).
"""
import sys
import numpy as np
import mpmath as mp

sys.path.insert(0, '.')
import rlos_lib as rl

mp.mp.dps = 40
th = 2


def Lgh(x1, x2):
    return mp.sqrt(x1 ** 2 + x2 ** 2)


def C_gh(u1, u2):
    return mp.exp(-Lgh(-mp.log(u1), -mp.log(u2)))


def C_joe(u1, u2):
    return 1 - ((1 - u1) ** 2 + (1 - u2) ** 2 - (1 - u1) ** 2 * (1 - u2) ** 2) ** mp.mpf(0.5)


def C_opc(u1, u2):
    return 1 / (1 + mp.sqrt((1 / u1 - 1) ** 2 + (1 / u2 - 1) ** 2))


def psi2(s):
    if s <= mp.mpf(1) / 2:
        return 1 - s + s ** 2 / 2
    if s <= mp.mpf(7) / 4:
        return mp.mpf(7) / 8 - s / 2
    return mp.mpf(0)


def psi2inv(u):
    if u >= mp.mpf(5) / 8:
        return 1 - mp.sqrt(2 * u - 1)
    return 2 * (mp.mpf(7) / 8 - u)


def C_amx(u1, u2):
    return psi2(Lgh(psi2inv(u1), psi2inv(u2)))


def C_clay(u1, u2):  # Clayton theta = 1
    return 1 / (1 / u1 + 1 / u2 - 1)


def F_ord(y, l, b):
    return sum(mp.binomial(b, j) * (1 - y) ** j * y ** (b - j) for j in range(l))


def Q(q, l):
    return mp.exp(-q) * sum(q ** h / mp.factorial(h) for h in range(l))


def y_of(q, l, b):
    """Solve F_{l:b}(e^{-z/b}) = Q_l(q) for z, return y = e^{-z/b}."""
    target = Q(q, l)
    f = lambda z: F_ord(mp.exp(-z / b), l, b) - target
    z = mp.findroot(f, q, tol=mp.mpf(10) ** -35)
    return mp.exp(-z / b)


def Cb_lm(C, q1, q2, l, m, b):
    y1 = y_of(q1, l, b)
    y2 = y_of(q2, m, b)
    c = C(y1, y2)
    p00 = c
    p10 = y2 - c   # first exceeds only
    p01 = y1 - c   # second exceeds only
    p11 = 1 - y1 - y2 + c
    s = mp.mpf(0)
    for n11 in range(min(l, m)):
        for n10 in range(l - n11):
            for n01 in range(m - n11):
                n = n11 + n10 + n01
                s += (mp.factorial(b) / (mp.factorial(n11) * mp.factorial(n10) * mp.factorial(n01)
                                          * mp.factorial(b - n))
                      * p11 ** n11 * p10 ** n10 * p01 ** n01 * p00 ** (b - n))
    return s


def C_lim(q1, q2, l, m, L):
    a = (q1 + q2 - L, L - q2, L - q1)
    s = mp.mpf(0)
    for (i1, i2, i3) in rl.I_lm(l, m):
        s += a[0] ** i1 * a[1] ** i2 * a[2] ** i3 / (mp.factorial(i1) * mp.factorial(i2) * mp.factorial(i3))
    return mp.exp(-L) * s


xg, wg = np.polynomial.laguerre.laggauss(80)


def beta_b(C, t, l, m, b, indep=False):
    t = mp.mpf(t)
    tot = mp.mpf(0)
    for x, w in zip(xg, wg):
        x = mp.mpf(x)
        q1, q2 = (1 - t) * x, t * x
        L = (q1 + q2) if indep else Lgh(q1, q2)
        tot += mp.mpf(w) * (Cb_lm(C, q1, q2, l, m, b) - C_lim(q1, q2, l, m, L))
    return -tot


def richardson(vals):
    """vals at b, 2b, 4b, 8b with expansion c0 + c1/b + c2/b^2 + ..."""
    v = list(vals)
    for k in (1, 2, 3):
        v = [(2 ** k * v[i + 1] - v[i]) / (2 ** k - 1) for i in range(len(v) - 1)]
    return v[0]


bs = [100, 200, 400, 800]
cases = [('Joe', C_joe, False), ('OPC', C_opc, False), ('GH', C_gh, False), ('Archimax', C_amx, False),
         ('Clayton', C_clay, True)]
print('pair   t    -b*beta_b (b=100..800) ->  Richardson limit   |  predicted (closed forms)   | abs diff')
for name, C, indep in cases:
    print(f'--- {name}')
    for t in (0.3, 0.5):
        for (l, m) in [(1, 1), (1, 2), (2, 2)]:
            vals = [-b * beta_b(C, t, l, m, b, indep) for b in bs]
            lim = richardson(vals)
            if name == 'Clayton':
                pred = [rl.calS_shape(1.0, t, l, m, 2) * t * (1 - t) * 1.0][0]  # theta=1, alpha=1/b, R=0
            else:
                A = float(rl.A_log(t, 2.0)); v = float(rl.dA_log(t, 2.0))
                p = rl.pairs(2).index((l, m))
                R = rl.calR_vec(A, v, t, 2)[p]
                if name in ('Joe', 'OPC'):
                    S = rl.calS_vec(name, t, 2)[p]
                    pred = 0.5 * S + R
                else:
                    pred = R  # GH: S = 0 ; Archimax: alpha = b^{-2} is of lower order
            print(f'({l},{m}) {t:.1f}  ' + ' '.join(mp.nstr(v_, 8) for v_ in vals)
                  + f'  -> {mp.nstr(lim, 10):>14}  | {pred: .10f} | {float(abs(lim - pred)):.1e}')

# ---------------------------------------------------------------- r = 3 pairs (general (l,m) formulas)
print('\nr = 3 pairs, t = 0.3 (prediction from rlos_lib.calS_vec / calR_vec with r = 3):')
for name, C, indep in [('Joe', C_joe, False), ('GH', C_gh, False), ('Clayton', C_clay, True)]:
    print(f'--- {name}')
    t = 0.3
    for (l, m) in [(1, 3), (2, 3), (3, 1), (3, 3)]:
        vals = [-b * beta_b(C, t, l, m, b, indep) for b in bs]
        lim = richardson(vals)
        p = rl.pairs(3).index((l, m))
        if name == 'Clayton':
            pred = rl.calS_shape(1.0, t, l, m, 2) * t * (1 - t)
        else:
            A = float(rl.A_log(t, 2.0)); v = float(rl.dA_log(t, 2.0))
            R = rl.calR_vec(A, v, t, 3)[p]
            pred = R + (0.5 * rl.calS_vec(name, t, 3)[p] if name == 'Joe' else 0.0)
        print(f'({l},{m}) {t:.1f}  ' + ' '.join(mp.nstr(v_, 8) for v_ in vals)
              + f'  -> {mp.nstr(lim, 10):>14}  | {pred: .10f} | {float(abs(lim - pred)):.1e}')
