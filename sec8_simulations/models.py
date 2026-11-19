"""Copula samplers and true Pickands functions for the Section 8 study.

Sources (adapted from the authors' earlier simulation code; self-contained here):
* positive_stable, sample_gumbel: positive stable mixture construction of the Gumbel copula
* Joe(2), outer-power Clayton BVZ (theta, beta) = (1, 2), BVZ Archimax psi_2: conditional bisection
  and frailty samplers
* asymmetric logistic: Tawn (1988) max construction
* Clayton(theta) frailty sampler: standard Marshall-Olkin construction.
All samplers return an (n, 2) array whose margins are continuous; only
ranks are used downstream, so the marginal scale is irrelevant.
"""
from __future__ import annotations

import numpy as np

import sigma2_lib


# ----------------------------------------------------------------------------- helpers
def positive_stable(alpha, size, rng):
    """S with E exp(-qS) = exp(-q^alpha) (Kanter; exact Levy case alpha=1/2)."""
    if alpha == 1.0:
        return np.ones(size, dtype=float)
    if alpha == 0.5:
        z = rng.standard_normal(size=size)
        np.square(z, out=z)
        np.maximum(z, np.finfo(float).tiny, out=z)
        return 0.5 / z
    v = rng.uniform(np.finfo(float).eps, np.pi - np.finfo(float).eps, size=size)
    e = rng.exponential(size=size)
    return (np.sin(alpha * v) / np.sin(v) ** (1.0 / alpha)
            * (np.sin((1.0 - alpha) * v) / e) ** ((1.0 - alpha) / alpha))


def conditional_sample(n, rng, conditional_cdf):
    eps = np.finfo(float).eps
    u, w = rng.uniform(eps, 1 - eps, size=(2, n))
    lo, hi = np.zeros(n), np.ones(n)
    for _ in range(55):
        v = (lo + hi) / 2
        below = conditional_cdf(u, v) < w
        lo = np.where(below, v, lo)
        hi = np.where(below, hi, v)
    return np.column_stack((u, (lo + hi) / 2))


# ----------------------------------------------------------------------------- samplers
def sample_gh(n, rng, theta=2.0):
    """Gumbel-Hougaard(theta): U_i = exp(-(E_i/S)^{1/theta})."""
    alpha = 1.0 / theta
    s = positive_stable(alpha, n, rng)
    e = rng.exponential(size=(n, 2))
    return np.exp(-(e / s[:, None]) ** alpha)


def _joe_conditional(u, v):
    a, b = (1 - u) ** 2, (1 - v) ** 2
    return (1 - u) * (1 - b) / np.sqrt(a + b - a * b)


def sample_joe(n, rng):
    return conditional_sample(n, rng, _joe_conditional)


def sample_outer_power_clayton(n, rng, beta=1.0):
    """Outer-power Clayton copula of Bucher, Volgushev and Zou (2019) with generator
    psi(x) = (1 + x^{1/2})^{-1}, i.e. BVZ parameters (theta, beta) = (1, 2); attractor GH(2).
    (The argument ``beta`` of this sampler is not the BVZ beta; the default 1.0 gives the model
    M3.  Label corrected 2026-11-18; the sampler itself is unchanged.)"""
    gamma = rng.gamma(1 / beta, size=n)
    frailty = gamma ** 2 * positive_stable(.5, n, rng)
    e = rng.exponential(size=(n, 2))
    return np.exp(-np.log1p(np.sqrt(e / frailty[:, None])) / beta)


def _psi2(x):
    x = np.asarray(x)
    return np.where(x <= .5, 1 - x + x * x / 2, np.maximum(7 / 8 - x / 2, 0))


def _psi2_prime(x):
    x = np.asarray(x)
    return np.where(x <= .5, x - 1, np.where(x < 7 / 4, -.5, 0))


def _psi2_inverse(u):
    u = np.asarray(u)
    root = np.sqrt(np.maximum(2 * u - 1, 0))
    return np.where(u >= 5 / 8, 2 * (1 - u) / (1 + root), 7 / 4 - 2 * u)


def _archimax_conditional(u, v):
    x, y = _psi2_inverse(u), _psi2_inverse(v)
    length = np.hypot(x, y)
    return _psi2_prime(length) * x / length / _psi2_prime(x)


def sample_archimax(n, rng):
    return conditional_sample(n, rng, _archimax_conditional)


def sample_asym_logistic(n, rng, theta=2.0, psi1=0.5, psi2=0.9):
    """Z_i = max{(1-psi_i) F_i, psi_i G_i}, F iid Frechet, G GH(theta) Frechet."""
    G = -1.0 / np.log(sample_gh(n, rng, theta))
    F = -1.0 / np.log(rng.uniform(size=(n, 2)))
    return np.column_stack([np.maximum((1 - psi1) * F[:, 0], psi1 * G[:, 0]),
                            np.maximum((1 - psi2) * F[:, 1], psi2 * G[:, 1])])


def sample_independence(n, rng):
    return rng.uniform(size=(n, 2))


def sample_clayton(n, rng, theta=1.0):
    """Clayton(theta) by gamma frailty; asymptotically independent (A_0=1)."""
    v = rng.gamma(1 / theta, size=n)
    e = rng.exponential(size=(n, 2))
    return (1 + e / v[:, None]) ** (-1 / theta)


# ----------------------------------------------------------------------------- truth
def A_logistic(t, theta=2.0):
    t = np.asarray(t, float)
    return (t ** theta + (1 - t) ** theta) ** (1 / theta)


def A_asym_logistic(t, theta=2.0, psi1=0.5, psi2=0.9):
    t = np.asarray(t, float)
    return ((1 - psi1) * (1 - t) + (1 - psi2) * t
            + ((psi1 * (1 - t)) ** theta + (psi2 * t) ** theta) ** (1 / theta))


def A_one(t):
    return np.ones_like(np.asarray(t, float))


# model_id is part of the SeedSequence entropy; never renumber.
MODELS = {
    "M1": dict(id=1, name="Gumbel-Hougaard(2)", sampler=sample_gh, A=A_logistic,
               oracle=lambda: sigma2_lib.Gumbel(2.0)),
    "M2": dict(id=2, name="Joe(2)", sampler=sample_joe, A=A_logistic,
               oracle=lambda: sigma2_lib.Gumbel(2.0)),
    "M3": dict(id=3, name="Outer-power Clayton BVZ(1,2)", sampler=sample_outer_power_clayton,
               A=A_logistic, oracle=lambda: sigma2_lib.Gumbel(2.0)),
    "M4": dict(id=4, name="BVZ Archimax psi_2", sampler=sample_archimax, A=A_logistic,
               oracle=lambda: sigma2_lib.Gumbel(2.0)),
    "M5": dict(id=5, name="Asymmetric logistic(2; 0.5, 0.9)", sampler=sample_asym_logistic,
               A=A_asym_logistic, oracle=lambda: sigma2_lib.AsymLogistic(2.0, 0.5, 0.9)),
    "D6a": dict(id=6, name="Independence", sampler=sample_independence, A=A_one,
                oracle=None),
    "D6b": dict(id=7, name="Clayton(1)", sampler=sample_clayton, A=A_one, oracle=None),
}
