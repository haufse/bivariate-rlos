"""s01: symbolic and high-precision checks of the model-specific inputs.

Run:   python s01_symbolic_checks.py > ../outputs/s01_output.txt
No randomness (no seed needed).

Checks
 (a) closed-form calR (rlos_lib.calR_lm, general (l,m)) == eq. (integrated-R-r2) for r=2 (sympy, exact);
 (b) calS for Clayton == supplement I display (calS_2) (sympy, exact);
 (c) d(t) for r=2 == the explicit derivative (sympy, exact);
 (d) block-maxima second-order condition (Condition cond:second_order_rank) for Joe(2), OPC (BVZ theta=1,
     beta=2), Archimax psi_2 with logistic(2) L0, Clayton(theta): the limit {C(u^{1/b})^b - C_inf(u)}/alpha(b)
     at several u, evaluated with mpmath (50 digits) at b = 1e4, 2e4 with first-order Richardson extrapolation,
     against the claimed S;
 (e) POT second-order condition (SO)_p (BVZ Def. 2.1) for the same models: {b[1-C(1-x/b)] - L(x)}/alpha_p(b).
"""
import sys
import sympy as sp
import mpmath as mp

sys.path.insert(0, '.')
import rlos_lib as rl

mp.mp.dps = 50

# ------------------------------------------------------------------------- (a) calR r=2
a, v, t = sp.symbols('a v t', real=True)


def calR_sym(l, m):
    """Symbolic version of rlos_lib.calR_lm."""
    x = sp.symbols('x', positive=True)
    al = (1 - a, a - t, a + t - 1)
    zt = (a ** 2 - (1 - t) ** 2 - t ** 2, t ** 2 - a ** 2, (1 - t) ** 2 - a ** 2)
    Ld1 = a - t * v
    Ld2 = a + (1 - t) * v
    Wt = -(l - 1) * (1 - t) * Ld1 - (m - 1) * t * Ld2
    ht = ((l - 1) * (1 - t) + (m - 1) * t + Wt, -(m - 1) * t - Wt, -(l - 1) * (1 - t) - Wt)
    integrand = 0
    for i in rl.I_lm(l, m):
        k = sum(i)
        ifac = sp.factorial(i[0]) * sp.factorial(i[1]) * sp.factorial(i[2])
        A_i = (x * al[0]) ** i[0] * (x * al[1]) ** i[1] * (x * al[2]) ** i[2]
        R1 = k * (1 + 2 * a * x - k) * A_i
        R2 = x * Wt * A_i
        for j in range(3):
            if i[j] > 0:
                e = list(i)
                e[j] -= 1
                A_ij = (x * al[0]) ** e[0] * (x * al[1]) ** e[1] * (x * al[2]) ** e[2]
                R1 += i[j] * x ** 2 * zt[j] * A_ij
                R2 += i[j] * x * ht[j] * A_ij
        integrand += sp.Rational(1, 2) * sp.exp(-a * x) * (R1 + R2) / ifac
    # du = e^{-x} dx
    poly = sp.expand(integrand * sp.exp(a * x))
    res = 0
    for term in sp.Add.make_args(poly):
        c, pw = term.as_coeff_exponent(x)
        res += c * sp.factorial(pw) / (1 + a) ** (pw + 1)
    return sp.simplify(res)


closed = [0,
          (a + t - 1) * (1 - t) * (a - 1 - t * v) / (1 + a) ** 3,
          t * (a - t) * (a - 1 + (1 - t) * v) / (1 + a) ** 3,
          3 * (a - 1) * (a - t) * (a + t - 1) / (1 + a) ** 4]
print('(a) calR general formula vs eq. (integrated-R-r2):')
for idx, (l, m) in enumerate(rl.pairs(2)):
    diff = sp.simplify(calR_sym(l, m) - closed[idx])
    print(f'    ({l},{m}): difference = {diff}')
# numeric agreement of the float implementation for r=3 sanity (vanishes at independence)
import numpy as np
print('    float implementation vs closed form, r=2, random points: max abs diff =',
      max(np.max(np.abs(rl.calR_vec(aa, vv, tt, 2) - rl.calR_r2_closed(aa, vv, tt)))
          for (aa, vv, tt) in [(0.8, 0.1, 0.3), (0.9, -0.2, 0.6), (0.75, 0.0, 0.5)]))
print('    calR at independence A=1, v=0, r=3:', rl.calR_vec(1.0, 0.0, 0.37, 3))

# ------------------------------------------------------------------------- (b) calS Clayton
th = sp.symbols('theta', positive=True)


def calS_sym(l, m, A, D, kappa, Psi):
    x = sp.symbols('x', positive=True)
    al = (1 - A, A - t, A + t - 1)
    sgn = (-1, 1, 1)
    P = 0
    for i in rl.I_lm(l, m):
        ifac = sp.factorial(i[0]) * sp.factorial(i[1]) * sp.factorial(i[2])
        Ai = (x * al[0]) ** i[0] * (x * al[1]) ** i[1] * (x * al[2]) ** i[2]
        P += Ai / ifac
        for j in range(3):
            if i[j] > 0:
                e = list(i)
                e[j] -= 1
                Aij = (x * al[0]) ** e[0] * (x * al[1]) ** e[1] * (x * al[2]) ** e[2]
                P -= sgn[j] * i[j] * Aij / ifac
    integrand = sp.expand(kappa * Psi * x ** D * P)
    res = 0
    for term in sp.Add.make_args(integrand):
        c, pw = term.as_coeff_exponent(x)
        res += c * sp.factorial(pw) / (1 + A) ** (pw + 1)
    return sp.factor(sp.simplify(res))


supp = [th * t * (1 - t) / 4, 3 * th * t ** 2 * (1 - t) / 8, 3 * th * t * (1 - t) ** 2 / 8,
        3 * th * t ** 2 * (1 - t) ** 2 / 4]
print('\n(b) Clayton calS_2 (A=1, S(e^{-q}) = theta e^{-q1-q2} q1 q2) vs supplement I:')
for idx, (l, m) in enumerate(rl.pairs(2)):
    val = calS_sym(l, m, 1, 2, th, t * (1 - t))
    print(f'    ({l},{m}): {val}   difference = {sp.simplify(val - supp[idx])}')
print('    r=3 Clayton calS_3 (new):')
for (l, m) in rl.pairs(3):
    print(f'      ({l},{m}): {calS_sym(l, m, 1, 2, th, t * (1 - t))}')

# ------------------------------------------------------------------------- (c) d for r=2
print('\n(c) d = -d/da mu for r=2 vs explicit derivative:')
mu2 = [a / (1 + a), (a ** 2 + 1 - t) / (1 + a) ** 2, (a ** 2 + t) / (1 + a) ** 2,
       (a ** 3 - a ** 2 + 2 * a + 2 * t ** 2 - 2 * t) / (1 + a) ** 3]
for idx, (l, m) in enumerate(rl.pairs(2)):
    musym = 1 - sum(sp.factorial(sum(i)) / (sp.factorial(i[0]) * sp.factorial(i[1]) * sp.factorial(i[2]))
                    * (1 - a) ** i[0] * (a - t) ** i[1] * (a + t - 1) ** i[2] / (1 + a) ** (sum(i) + 1)
                    for i in rl.I_lm(l, m))
    print(f'    ({l},{m}): mu general - eq.(mu-r2) = {sp.simplify(musym - mu2[idx])};'
          f'  -dmu/da = {sp.factor(sp.simplify(-sp.diff(mu2[idx], a)))}')
print('    float d_vec(0.8,0.3,2) =', rl.d_vec(0.8, 0.3, 2))

# ------------------------------------------------------------------------- (d) BM second order
print('\n(d) Block-maxima second-order condition, mpmath 50 digits, Richardson in b:')


def Lgh(x1, x2, b_=2):
    return (x1 ** b_ + x2 ** b_) ** (mp.mpf(1) / b_)


def Ldot(x1, x2, b_=2):
    L = Lgh(x1, x2, b_)
    return (x1 / L) ** (b_ - 1), (x2 / L) ** (b_ - 1)


def Gam(x1, x2, p=2):
    l1, l2 = Ldot(x1, x2)
    return x1 ** p * l1 + x2 ** p * l2


def C_joe(u1, u2, th_=2):
    return 1 - ((1 - u1) ** th_ + (1 - u2) ** th_ - (1 - u1) ** th_ * (1 - u2) ** th_) ** (mp.mpf(1) / th_)


def C_opc(u1, u2, th_=1, be=2):
    return (1 + ((u1 ** (-th_) - 1) ** be + (u2 ** (-th_) - 1) ** be) ** (mp.mpf(1) / be)) ** (-mp.mpf(1) / th_)


def psi2(s):
    s = mp.mpf(s)
    if s <= mp.mpf(1) / 2:
        return 1 - s + s ** 2 / 2
    if s <= mp.mpf(7) / 4:
        return mp.mpf(7) / 8 - s / 2
    return mp.mpf(0)


def psi2inv(u):
    u = mp.mpf(u)
    if u >= mp.mpf(5) / 8:   # psi2(1/2) = 5/8
        return 1 - mp.sqrt(2 * u - 1)
    return 2 * (mp.mpf(7) / 8 - u)


def C_amx(u1, u2):
    return psi2(Lgh(psi2inv(u1), psi2inv(u2)))


def C_clay(u1, u2, th_=1):
    return (u1 ** (-th_) + u2 ** (-th_) - 1) ** (-mp.mpf(1) / th_)


models_bm = {
    'Joe(2)': (C_joe, lambda b: 1 / (2 * b), lambda x1, x2: -mp.exp(-Lgh(x1, x2)) * (Lgh(x1, x2) ** 2 - Gam(x1, x2)),
               lambda x1, x2: mp.exp(-Lgh(x1, x2))),
    'OPC(1,2)': (C_opc, lambda b: 1 / (2 * b), lambda x1, x2: mp.exp(-Lgh(x1, x2)) * (Lgh(x1, x2) ** 2 - Gam(x1, x2)),
                 lambda x1, x2: mp.exp(-Lgh(x1, x2))),
    'Archimax psi2': (C_amx, lambda b: 1 / b ** 2,
                      lambda x1, x2: mp.exp(-Lgh(x1, x2)) * (Lgh(x1, x2) ** 3 - Gam(x1, x2, 3)) / 6,
                      lambda x1, x2: mp.exp(-Lgh(x1, x2))),
    'Clayton(1)': (C_clay, lambda b: 1 / b, lambda x1, x2: mp.exp(-x1 - x2) * x1 * x2,
                   lambda x1, x2: mp.exp(-x1 - x2)),
}
pts = [(mp.mpf('0.3'), mp.mpf('0.7')), (mp.mpf('0.5'), mp.mpf('0.5')), (mp.mpf('0.9'), mp.mpf('0.2')),
       (mp.mpf('0.05'), mp.mpf('0.6'))]
for name, (C, alpha, Sclaim, Cinf) in models_bm.items():
    worst = 0
    for (u1, u2) in pts:
        x1, x2 = -mp.log(u1), -mp.log(u2)
        est = []
        for b in (mp.mpf(10) ** 4, 2 * mp.mpf(10) ** 4):
            Cb = C(u1 ** (1 / b), u2 ** (1 / b)) ** b
            est.append((Cb - Cinf(x1, x2)) / alpha(b))
        rich = 2 * est[1] - est[0]
        rel = abs(rich - Sclaim(x1, x2)) / abs(Sclaim(x1, x2))
        worst = max(worst, rel)
    print(f'    {name:15s}: max rel. error of S over 4 test points = {mp.nstr(worst, 3)}')

# ------------------------------------------------------------------------- (e) POT second order
print('\n(e) POT second-order condition (SO)_p, b[1-C(1-x/b)] - L(x) = alpha_p(b) S_p(x) + o:')
models_pot = {
    'GH(2)': (lambda u1, u2: mp.exp(-Lgh(-mp.log(u1), -mp.log(u2))), lambda b: 1 / (2 * b),
              lambda x1, x2: Gam(x1, x2) - Lgh(x1, x2) ** 2, Lgh),
    'Joe(2)': (C_joe, lambda b: 1 / b ** 2, lambda x1, x2: -(x1 * x2) ** 2 / (2 * Lgh(x1, x2)), Lgh),
    'OPC(1,2)': (C_opc, lambda b: 1 / b, lambda x1, x2: (1 + 1) / mp.mpf(2) * (Gam(x1, x2) - Lgh(x1, x2) ** 2), Lgh),
    'Archimax psi2': (C_amx, lambda b: 1 / (2 * b), lambda x1, x2: Gam(x1, x2) - Lgh(x1, x2) ** 2, Lgh),
    'Clayton(1)': (C_clay, lambda b: 1 / b, lambda x1, x2: -(1 + 1) * x1 * x2, lambda x1, x2: x1 + x2),
}
xpts = [(mp.mpf('0.3'), mp.mpf('0.7')), (mp.mpf('0.5'), mp.mpf('0.5')), (mp.mpf('1.3'), mp.mpf('0.2'))]
for name, (C, alpha, Sp, L) in models_pot.items():
    worst = 0
    for (x1, x2) in xpts:
        est = []
        for b in (mp.mpf(10) ** 5, 2 * mp.mpf(10) ** 5):
            val = b * (1 - C(1 - x1 / b, 1 - x2 / b)) - L(x1, x2)
            est.append(val / alpha(b))
        rich = 2 * est[1] - est[0]
        worst = max(worst, abs(rich - Sp(x1, x2)) / abs(Sp(x1, x2)))
    print(f'    {name:15s}: max rel. error of S_p over 3 test points = {mp.nstr(worst, 3)}')

# Joe POT: exact identity check
x1, x2, b = mp.mpf('0.4'), mp.mpf('0.6'), mp.mpf(37)
lhs = b * (1 - C_joe(1 - x1 / b, 1 - x2 / b))
rhs = (x1 ** 2 + x2 ** 2 - b ** -2 * x1 ** 2 * x2 ** 2) ** (mp.mpf(1) / 2)
print('    Joe(2) exact identity b{1-C(1-x/b)} = {x^2+y^2-b^-2 x^2 y^2}^{1/2}: diff =', mp.nstr(lhs - rhs, 3))
