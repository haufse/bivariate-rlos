"""s10: closed forms (sympy) of the integrated domain-of-attraction bias vector calS for r = 2, for the
second-order functions of the form S(e^{-q}) = kappa e^{-L(q)} Psi(q), Psi homogeneous of degree D, and of
the POT/BM auxiliary quantities used in the LaTeX displays.

Run:  python s10_display_formulas.py > ../outputs/s10_output.txt
No randomness.

Along h^{(l,m)}(u,t), q = x(1-t,t) with x = -log u, so S(e^{-q}) = kappa e^{-xA} x^D Psi(1-t,t) and
    calS^{(l,m)}(t) = kappa Psi(1-t,t) J_D^{(l,m)}(A(t),t),
    J_D^{(l,m)}(a,t) = int_0^infty e^{-(1+a)x} x^D P_{lm}(x) dx   (P_{lm} from eq. S-ellm).
"""
import sys
import sympy as sp

sys.path.insert(0, '.')
import rlos_lib as rl

a, t, v = sp.symbols('a t v', real=True)
x = sp.symbols('x', positive=True)


def J(l, m, D):
    al = (1 - a, a - t, a + t - 1)
    sgn = (-1, 1, 1)
    P = 0
    for i in rl.I_lm(l, m):
        ifac = sp.factorial(i[0]) * sp.factorial(i[1]) * sp.factorial(i[2])
        P += (x * al[0]) ** i[0] * (x * al[1]) ** i[1] * (x * al[2]) ** i[2] / ifac
        for j in range(3):
            if i[j] > 0:
                e = list(i)
                e[j] -= 1
                P -= sgn[j] * i[j] * (x * al[0]) ** e[0] * (x * al[1]) ** e[1] * (x * al[2]) ** e[2] / ifac
    integrand = sp.expand(x ** D * P)
    res = 0
    for term in sp.Add.make_args(integrand):
        c, pw = term.as_coeff_exponent(x)
        res += c * sp.factorial(pw) / (1 + a) ** (pw + 1)
    return sp.factor(sp.simplify(res))


for D in (2, 3):
    print(f'J_{D}^(l,m)(a,t), r = 2, pairs (1,1),(1,2),(2,1),(2,2):')
    for (l, m) in rl.pairs(2):
        expr = J(l, m, D)
        print(f'  ({l},{m}): {expr}')
        print(f'     LaTeX: {sp.latex(expr)}')
    print()

# Psi functions on the curve
A = sp.Function('A')(t)
Ap = sp.Derivative(A, t)
Gam2 = A * (1 - 2 * t + 2 * t ** 2) + t * (1 - t) * (2 * t - 1) * Ap
Gam3 = A * (1 - 3 * t + 3 * t ** 2) + t * (1 - t) * (2 * t - 1) * Ap
print('Gamma_2(1-t,t) = sum_j x_j^2 Ldot_j =', sp.latex(Gam2))
print('Gamma_3(1-t,t) = sum_j x_j^3 Ldot_j =', sp.latex(Gam3))
# verify with logistic(2) directly
q1, q2 = sp.symbols('q1 q2', positive=True)
L = sp.sqrt(q1 ** 2 + q2 ** 2)
G2 = q1 ** 2 * sp.diff(L, q1) + q2 ** 2 * sp.diff(L, q2)
G3 = q1 ** 3 * sp.diff(L, q1) + q2 ** 3 * sp.diff(L, q2)
Al = sp.sqrt(t ** 2 + (1 - t) ** 2)
chk2 = sp.simplify(G2.subs({q1: 1 - t, q2: t}) - Gam2.subs(Ap, sp.diff(Al, t)).subs(A, Al))
chk3 = sp.simplify(G3.subs({q1: 1 - t, q2: t}) - Gam3.subs(Ap, sp.diff(Al, t)).subs(A, Al))
print('check Gamma_2, Gamma_3 vs logistic(2) partial derivatives:', chk2, chk3)
print('logistic(2): A^2 - Gamma_2 =', sp.factor(sp.simplify(Al ** 2 - G2.subs({q1: 1 - t, q2: t}))))
print('logistic(2): A^3 - Gamma_3 =', sp.factor(sp.simplify(Al ** 3 - G3.subs({q1: 1 - t, q2: t}))))
print('logistic(2): POT Joe S_p(1-t,t) = -(1-t)^2 t^2/(2A) ;  GH/Archimax S_p(1-t,t) = Gamma_2 - A^2 =',
      sp.factor(sp.simplify(G2.subs({q1: 1 - t, q2: t}) - Al ** 2)))
# POT variance at logistic(2)
l1 = sp.diff(L, q1).subs({q1: 1 - t, q2: t})
l2 = sp.diff(L, q2).subs({q1: 1 - t, q2: t})
Vp = sp.simplify(-Al + l1 ** 2 * (1 - t) + l2 ** 2 * t + 2 * l1 * l2 * (1 - Al))
print('POT Var A_p(t) at logistic(2):', Vp)
print('   numeric at t=0.3, 0.5:', [float(Vp.subs(t, tv)) for tv in (0.3, 0.5)],
      ' rlos_lib:', [float(rl.pot_var(tv, 2.0)) for tv in (0.3, 0.5)])
