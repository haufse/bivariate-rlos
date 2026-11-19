"""Library: the rank covariance Sigma_2(t) = Cov{Z(t)} (r = 2) for a general extreme-value
attractor, by deterministic quadrature.

Formulas used (manuscript notation; derivations in the header of s01_validate.py)
-----------------------------------------------------------------------------
Let N be Poisson with exponent measure nu, V(x,y) = L(1/x,1/y).  For thresholds written in
q-scale (q = 1/x; q = 0 means "no constraint") the joint CDF of the top-two order statistics
(M_1^(1), M_1^(2), M_2^(1), M_2^(2)) is, with P2 >= P1 and R2 >= R1 enforced by max,
    F(P1,P2,R1,R2) = exp{-L(P2,R2)} [1 + L(P2,R2) - L(P1,R1)
                                     + {L(P2,R2) - L(P1,R2)} {L(P2,R2) - L(P2,R1)}].
(Count the Poisson points in {a > x2 or b > y2}: none; one, outside {a > x1 or b > y1};
 or two, one in {x2 < a <= x1, b <= y2} and one in {a <= x2, y2 < b <= y1}.)
Bivariate margins G^(l,m) (Lemma lem:distr_lm): e^{-L}, e^{-L}(1+L-q1), e^{-L}(1+L-q2),
e^{-L}{1+L+(L-q2)(L-q1)}.  Copula derivatives along the curves:
    dC_1^(l,m) = -(l-1)! e^{q1} q1^{1-l} d_{q1} G,   dC_2^(l,m) = -(m-1)! e^{q2} q2^{1-m} d_{q2} G.
Along h^(l,m)(u,t): q1 = (1-t)s, q2 = t s, s = -log u.

Sigma_{pp'}(t) = int_0^1 int_0^1 Cov{g_u^p(U), g_v^p'(U)} du dv  (proof of Lemma lem:Z-influence),
g_u^p = 1_A - dC_1 1_B - dC_2 1_C, B = {V_1^(l) <= u}, C = {V_2^(m) <= u}, A = B n C.
Quadrature: s = -log u, s' = -log v; split at s = s'; graded composite Gauss-Legendre in
(s, w = s' - s) on [0, SMAX]^2 (integrand bounded, weight e^{-s-s'}).
"""
import numpy as np
from math import factorial

PAIRS = [(1, 1), (1, 2), (2, 1), (2, 2)]


# ----------------------------------------------------------------------------- models
class Gumbel:
    def __init__(self, theta):
        self.theta = float(theta)
        self.name = f"Gumbel(theta={theta})"

    def L(self, q1, q2):
        th = self.theta
        return (q1 ** th + q2 ** th) ** (1.0 / th)

    def L1(self, q1, q2):
        th = self.theta
        if th == 1.0:
            return np.ones_like(q1)
        return q1 ** (th - 1) * self.L(q1, q2) ** (1 - th)

    def L2(self, q1, q2):
        return self.L1(q2, q1)

    def A(self, t):
        th = self.theta
        return (t ** th + (1 - t) ** th) ** (1 / th)

    def dA(self, t):
        th = self.theta
        if th == 1.0:
            return 0.0
        return (t ** (th - 1) - (1 - t) ** (th - 1)) * self.A(t) ** (1 - th)


class AsymLogistic:
    """L(q1,q2) = (1-psi1) q1 + (1-psi2) q2 + {(psi1 q1)^th + (psi2 q2)^th}^{1/th} (Tawn 1988)."""

    def __init__(self, theta, psi1, psi2):
        self.theta, self.p1, self.p2 = float(theta), float(psi1), float(psi2)
        self.name = f"AsymLogistic(theta={theta},psi=({psi1},{psi2}))"

    def _core(self, q1, q2):
        th = self.theta
        return ((self.p1 * q1) ** th + (self.p2 * q2) ** th) ** (1.0 / th)

    def L(self, q1, q2):
        return (1 - self.p1) * q1 + (1 - self.p2) * q2 + self._core(q1, q2)

    def L1(self, q1, q2):
        th = self.theta
        return (1 - self.p1) + self.p1 ** th * q1 ** (th - 1) * self._core(q1, q2) ** (1 - th)

    def L2(self, q1, q2):
        th = self.theta
        return (1 - self.p2) + self.p2 ** th * q2 ** (th - 1) * self._core(q1, q2) ** (1 - th)

    def A(self, t):
        return float(self.L(np.array(1 - t), np.array(t)))

    def dA(self, t, h=1e-7):
        return (self.A(t + h) - self.A(t - h)) / (2 * h)


# ----------------------------------------------------------------------------- moments
def mu_vec(a, t):
    """eq. (mu-r2) of the manuscript, order (1,1),(1,2),(2,1),(2,2)."""
    return np.array([a / (1 + a), (a * a + 1 - t) / (1 + a) ** 2, (a * a + t) / (1 + a) ** 2,
                     (a ** 3 - a ** 2 + 2 * a + 2 * t * t - 2 * t) / (1 + a) ** 3])


def d_vec(a, t):
    """d = D'(a,t) = -d/da mu(a,t), exact derivative of eq. (mu-r2)."""
    return np.array([-1 / (1 + a) ** 2,
                     -2 * (a - 1 + t) / (1 + a) ** 3,
                     -2 * (a - t) / (1 + a) ** 3,
                     2 * (a - 1) / (1 + a) ** 3 - 6 * (a - t) * (a - 1 + t) / (1 + a) ** 4])


# ----------------------------------------------------------------------------- copula pieces
def F4(model, P1, P2, R1, R2):
    P2 = np.maximum(P2, P1)
    R2 = np.maximum(R2, R1)
    L22 = model.L(P2, R2)
    L11 = model.L(P1, R1)
    L12 = model.L(P1, R2)
    L21 = model.L(P2, R1)
    return np.exp(-L22) * (1 + L22 - L11 + (L22 - L12) * (L22 - L21))


def dC(model, p, q1, q2):
    """(dC_1, dC_2) of C^(l,m) at the point with q-coordinates (q1,q2) > 0."""
    l, m = p
    L = model.L(q1, q2)
    L1 = model.L1(q1, q2)
    L2 = model.L2(q1, q2)
    E1 = np.exp(q1 - L)
    E2 = np.exp(q2 - L)
    if p == (1, 1):
        g1, g2 = -L1, -L2
    elif p == (1, 2):
        g1 = -L1 * (L - q1) - 1.0          # e^{L} d_{q1}[e^{-L}(1+L-q1)]
        g2 = -L2 * (L - q1)
    elif p == (2, 1):
        g1 = -L1 * (L - q2)
        g2 = -L2 * (L - q2) - 1.0
    else:  # (2,2): e^{L} d_{q1}[e^{-L}(1+L+(L-q2)(L-q1))]
        g1 =-L1 * (1 + L + (L - q2) * (L - q1)) + L1 + L1 * (L - q1) + (L - q2) * (L1 - 1)
        g2 = -L2 * (1 + L + (L - q2) * (L - q1)) + L2 + L2 * (L - q2) + (L - q1) * (L2 - 1)
    c1 = -factorial(l - 1) * E1 * q1 ** (1 - l) * g1
    c2 = -factorial(m - 1) * E2 * q2 ** (1 - m) * g2
    return c1, c2


def thresholds(p, t, s):
    """q-thresholds (P1,P2,R1,R2) of the events A,B,C of pair p at s = -log u."""
    l, m = p
    z = np.zeros_like(s)
    qa = (1 - t) * s
    qb = t * s
    B = [z, z, z, z]
    C = [z, z, z, z]
    B[l - 1] = qa
    C[2 + m - 1] = qb
    A = [np.maximum(B[i], C[i]) for i in range(4)]
    return {'A': A, 'B': B, 'C': C}


def graded_nodes(smax=50.0, npan=45, m=10, smin=1e-8):
    edges = np.concatenate([[0.0], np.geomspace(smin, smax, npan)])
    g, w = np.polynomial.legendre.leggauss(m)
    S, W = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        S.append(0.5 * (hi - lo) * g + 0.5 * (hi + lo))
        W.append(0.5 * (hi - lo) * w)
    return np.concatenate(S), np.concatenate(W)


def sigma2(model, t, smax=50.0, npan=45, m=10, pairs=PAIRS):
    """Return Sigma (len(pairs) x len(pairs)) at t by quadrature over region s' >= s."""
    S, WS = graded_nodes(smax, npan, m)
    Wn, WW = graded_nodes(smax, npan, m)
    s = S[:, None] * np.ones_like(Wn)[None, :]
    sp = s + Wn[None, :]
    weight = (WS[:, None] * WW[None, :]) * np.exp(-s - sp)
    npairs = len(pairs)
    I = np.zeros((npairs, npairs))
    pre = {}
    for k, p in enumerate(pairs):
        for tag, ss in (('u', s), ('v', sp)):
            th = thresholds(p, t, ss)
            c1, c2 = dC(model, p, (1 - t) * ss, t * ss)
            coef = {'A': np.ones_like(ss), 'B': -c1, 'C': -c2}
            prob = {e: F4(model, *th[e]) for e in 'ABC'}
            pre[(k, tag)] = (th, coef, prob)
    for i in range(npairs):
        thu, cu, pu = pre[(i, 'u')]
        for j in range(npairs):
            thv, cv, pv = pre[(j, 'v')]
            K = np.zeros_like(s)
            for e in 'ABC':
                for f in 'ABC':
                    joint = [np.maximum(thu[e][q], thv[f][q]) for q in range(4)]
                    K += cu[e] * cv[f] * (F4(model, *joint) - pu[e] * pv[f])
            I[i, j] = np.sum(weight * K)
    return I + I.T


def summary(model, t, Sig=None, **kw):
    if Sig is None:
        Sig = sigma2(model, t, **kw)
    a = model.A(t)
    d = d_vec(a, t)
    ev = np.linalg.eigvalsh(Sig)
    Si = np.linalg.inv(Sig)
    dSd = d @ Si @ d
    lam = Si @ d / dSd
    out = dict(t=t, A=a, eig=ev, lmin=ev[0], tr=np.trace(Sig), sig_opt=1 / dSd,
               sig_id=(d @ Sig @ d) / (d @ d) ** 2, sig_r1=Sig[0, 0] / d[0] ** 2,
               ratio=dSd * ev[0],  # d'S^{-1}d / ||S^{-1}||_op = lambda_min(S) / sigma_opt^2
               lam=lam, lamnorm=np.linalg.norm(lam), d=d, Sig=Sig)
    return out
