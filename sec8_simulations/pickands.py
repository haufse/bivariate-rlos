"""L2 Pickands sieve fitting, Bernstein pilots, and exact spectral sampling.

The sieve projection follows the original Fils-Villetard--Guillou--Segers
piecewise-linear constrained L2 fit of the authors' earlier code.  A fitted curve
has unit endpoints, obeys max(t, 1-t) <= A(t) <= 1, and is convex.

For a fitted piecewise-linear curve, the spectral measure is recovered *exactly*
from slope jumps, including atoms at zero and one; finite differences of the
fitted curve are neither necessary nor appropriate here. ``fit_pickands_pilot``
adds degree-m Bernstein smoothing to the m-interval linear sieve fit. Its
spectral law keeps the endpoint atoms and replaces each interior atom j/m
by a Beta(j, m-j) component with the same mass (Remark 4.13 and Supplement G).
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
import numpy as np
from scipy.interpolate import BPoly
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.special import betainc
from scipy.stats import beta as beta_distribution, rankdata

ArrayLike = np.ndarray

__all__ = [
    "PickandsCurve", "fit_pickands_curve", "pickands_l2_projection",
    "project_pickands_estimator_curve",
    "BernsteinPickandsCurve", "BetaMixtureSpectralMeasure",
    "fit_pickands_pilot", "raw_madogram_curve", "fit_madogram_pilot",
    "sample_rlos",
]


def _integer(value, name, minimum):
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _unit_argument(t):
    x = np.asarray(t, dtype=float)
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("t must lie in [0, 1]")
    return x


@dataclass(frozen=True)
class PickandsCurve:
    """A convex piecewise-linear Pickands dependence function.

    ``knots`` include 0 and 1; values are exactly one at both endpoints.
    Its spectral law has atoms at ``knots`` with masses from slope jumps.
    """

    knots: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.knots, dtype=float)
        a = np.asarray(self.values, dtype=float)
        if (t.ndim != 1 or a.shape != t.shape or t.size < 2
                or not np.all(np.isfinite(t)) or not np.all(np.isfinite(a))
                or t[0] != 0.0 or t[-1] != 1.0
                or not np.all(np.diff(t) > 0.0)):
            raise ValueError("Pickands knots must increase strictly from 0 to 1")
        tol = 1e-8
        slopes = np.diff(a) / np.diff(t)
        if (abs(a[0] - 1.0) > tol or abs(a[-1] - 1.0) > tol
                or np.any(a < np.maximum(t, 1 - t) - tol)
                or np.any(a > 1.0 + tol)
                or np.any(np.diff(slopes) < -tol)
                or slopes[0] < -1.0 - tol or slopes[-1] > 1.0 + tol):
            raise ValueError("Knots and values do not describe a valid Pickands function")
        # Copy so later modification of caller-supplied arrays cannot invalidate it.
        t = t.copy()
        a = a.copy()
        t.setflags(write=False)
        a.setflags(write=False)
        object.__setattr__(self, "knots", t)
        object.__setattr__(self, "values", a)

    def __call__(self, t):
        x = np.asarray(t, dtype=float)
        if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("t must lie in [0, 1]")
        y = np.interp(x, self.knots, self.values)
        return float(y) if x.ndim == 0 else y

    def derivative(self, t):
        """Right derivative, except at t=1 where the left slope is returned."""
        x = np.asarray(t, dtype=float)
        if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("t must lie in [0, 1]")
        slopes = np.diff(self.values) / np.diff(self.knots)
        index = np.clip(np.searchsorted(self.knots, x, side="right") - 1,
                        0, len(slopes) - 1)
        result = slopes[index]
        return float(result) if x.ndim == 0 else result

    def spectral_measure(self):
        """Return (w, mass) for H(dw), satisfying both unit-margin moments.

        A(t) = integral max((1-t)*w, t*(1-w)) H(dw), with
        integral w H(dw) = integral (1-w) H(dw) = 1.
        """
        slopes = np.diff(self.values) / np.diff(self.knots)
        masses = np.r_[1.0 + slopes[0], np.diff(slopes), 1.0 - slopes[-1]]
        if np.any(masses < -1e-8):
            raise ValueError("Invalid negative spectral mass")
        masses = np.maximum(masses, 0.0)  # roundoff only
        w = self.knots.copy()
        if (not np.isclose(np.sum(masses * w), 1.0, atol=1e-7, rtol=0)
                or not np.isclose(np.sum(masses * (1.0 - w)), 1.0,
                                   atol=1e-7, rtol=0)):
            raise ValueError("Spectral masses violate unit-margin moments")
        return w, masses

    def bernstein(self):
        """Smooth on the same uniform grid, preserving Pickands constraints."""
        expected = np.linspace(0.0, 1.0, len(self.knots))
        if not np.allclose(self.knots, expected, atol=1e-14, rtol=0):
            raise ValueError("Bernstein smoothing requires equally spaced knots")
        return BernsteinPickandsCurve(self.values)


@dataclass(frozen=True)
class BetaMixtureSpectralMeasure:
    """Spectral measure H of mass two, with endpoint atoms and beta components.

    For degree m, masses[0] and masses[m] are atoms at 0 and 1; masses[j]
    multiplies the probability law Beta(j, m-j), 1 <= j < m. ``sample`` and
    ``cdf`` refer to the probability law H/2; ``density`` returns the
    continuous density of H, excluding its endpoint atoms.
    """

    masses: np.ndarray

    def __post_init__(self):
        masses = np.asarray(self.masses, dtype=float)
        if (masses.ndim != 1 or len(masses) < 3
                or not np.all(np.isfinite(masses)) or np.any(masses < -1e-8)):
            raise ValueError("Invalid spectral masses")
        masses = np.maximum(masses, 0.0)  # solver/roundoff noise only
        means = np.linspace(0.0, 1.0, len(masses))
        if (not np.isclose(masses @ means, 1, atol=1e-7, rtol=0)
                or not np.isclose(masses @ (1-means), 1, atol=1e-7, rtol=0)):
            raise ValueError("Spectral masses violate unit-margin moments")
        masses.setflags(write=False)
        object.__setattr__(self, "masses", masses)

    @property
    def degree(self):
        return len(self.masses)-1

    def sample(self, size=None, rng=None):
        """Draw W ~ H/2 directly; rng may be a Generator or a seed."""
        rng = np.random.default_rng(rng)
        component = np.asarray(rng.choice(
            self.degree+1, size=size, p=self.masses/self.masses.sum()))
        indices = component.reshape(-1)
        values = indices.astype(float)/self.degree
        interior = (indices > 0) & (indices < self.degree)
        values[interior] = rng.beta(indices[interior], self.degree-indices[interior])
        values = values.reshape(component.shape)
        return float(values) if values.ndim == 0 else values

    def cdf(self, w):
        """CDF of H/2, including atoms at zero and one."""
        x = np.asarray(w, dtype=float)
        if np.any(np.isnan(x)):
            raise ValueError("w must not contain NaN")
        flat = np.clip(x.reshape(-1), 0, 1)
        j = np.arange(1, self.degree)
        values = (self.masses[0] + self.masses[1:-1] @ betainc(
            j[:, None], (self.degree-j)[:, None], flat[None, :]))/2
        values = np.where(x.reshape(-1) < 0, 0, values)
        values = np.where(x.reshape(-1) >= 1, 1, values).reshape(x.shape)
        return float(values) if values.ndim == 0 else values

    def density(self, w):
        """Continuous part dH/dw; endpoint atoms are not included."""
        x = _unit_argument(w)
        j = np.arange(1, self.degree)
        values = (self.masses[1:-1] @ beta_distribution.pdf(
            x.reshape(1, -1), j[:, None], (self.degree-j)[:, None])).reshape(x.shape)
        return float(values) if values.ndim == 0 else values


@dataclass(frozen=True)
class BernsteinPickandsCurve:
    """Smooth Pickands pilot B_m(check A), with exact beta-mixture spectrum.

    Coefficients are the values of the fitted linear sieve at j/m; they
    are generally NOT values of the resulting polynomial at j/m.
    """

    coefficients: np.ndarray

    def __post_init__(self):
        a = np.asarray(self.coefficients, dtype=float)
        if a.ndim != 1 or len(a) < 3:
            raise ValueError("At least three Bernstein coefficients are required")
        checked = PickandsCurve(np.linspace(0, 1, len(a)), a)
        object.__setattr__(self, "coefficients", checked.values)

    @property
    def degree(self):
        return len(self.coefficients)-1

    def __call__(self, t):
        x = _unit_argument(t)
        y = BPoly(self.coefficients[:, None], [0, 1])(x)
        # BPoly may round the value at x=1 even with an exact unit endpoint.
        y = np.where(x == 0, self.coefficients[0],
                     np.where(x == 1, self.coefficients[-1], y))
        return float(y) if x.ndim == 0 else y

    def derivative(self, t, order=1):
        """Analytic polynomial derivative, including one-sided endpoints."""
        x = _unit_argument(t)
        order = _integer(order, "order", 0)
        y = BPoly(self.coefficients[:, None], [0, 1]).derivative(order)(x)
        return float(y) if x.ndim == 0 else y

    def spectral_measure(self):
        slopes = self.degree*np.diff(self.coefficients)
        return BetaMixtureSpectralMeasure(np.r_[
            1+slopes[0], np.diff(slopes), 1-slopes[-1]])


def _sieve_projection(
    t: ArrayLike,
    A_n: ArrayLike,
    m: int = 40,
    boundary_correct: bool = False,
    *,
    ftol: float = 1e-12,
    maxiter: int = 1500,
) -> PickandsCurve:
    r"""L2-sieve projection of a preliminary Pickands estimator.

    Implements the piecewise-linear sieve projection of Fils-Villetard,
    Guillou and Segers (2008, Section 3).  The returned derivative is the
    slope of the fitted piecewise-linear Pickands function on the sieve
    interval containing each input value of ``t``.  At a knot the right
    derivative is returned (and at t=1 the final left slope is used).

    Parameters
    ----------
    t : one-dimensional array
        Strictly increasing grid, including exactly 0 and 1.
    A_n : one-dimensional array
        Preliminary Pickands estimates on ``t``.
    m : int, default 40
        Number of equal-width sieve intervals.
    boundary_correct : bool, default False
        If True, linearly correct the preliminary endpoint values to one
        before projection.

    Returns
    -------
    PickandsCurve
        The L2 fit on the uniform sieve, with exact unit endpoints.
        ``ftol`` is SLSQP's tolerance for the scaled objective; it is not
        a certified bound on the unscaled integrated squared-error excess.
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(A_n, dtype=float).copy()
    m = _integer(m, "m", 2)
    maxiter = _integer(maxiter, "maxiter", 1)
    if not np.isfinite(ftol) or ftol <= 0:
        raise ValueError("ftol must be positive and finite")
    if (
        t.ndim != 1
        or a.shape != t.shape
        or len(t) < 2
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(a))
        or not np.all(np.diff(t) > 0)
        or t[0] != 0.0
        or t[-1] != 1.0
    ):
        raise ValueError("Invalid input grid or estimates")
    if boundary_correct:
        a -= (1.0 - t) * (a[0] - 1.0) + t * (a[-1] - 1.0)

    knots = np.linspace(0.0, 1.0, m + 1)

    # Exact b_j = int A_n(t) phi_j(t) dt when A_n is linearly interpolated.
    z = np.union1d(t, knots)
    v = np.interp(z, t, a)
    left, right = z[:-1], z[1:]
    f0, f1 = v[:-1], v[1:]
    j = np.searchsorted(knots, (left + right) / 2.0) - 1
    j = np.clip(j, 0, m - 1)
    L0 = m * (knots[j + 1] - left)
    L1 = m * (knots[j + 1] - right)

    def prod_integral(g0, g1):
        return (right - left) / 6.0 * (
            2.0 * f0 * g0 + f0 * g1 + f1 * g0 + 2.0 * f1 * g1
        )

    b = np.zeros(m + 1)
    np.add.at(b, j, prod_integral(L0, L1))
    np.add.at(b, j + 1, prod_integral(1.0 - L0, 1.0 - L1))

    Q = np.diag(np.r_[2.0, np.full(m - 1, 4.0), 2.0])
    zdiag = np.arange(m)
    Q[zdiag, zdiag + 1] = Q[zdiag + 1, zdiag] = 1.0
    Q /= 6.0 * m

    # Discrete convexity: beta[i-1] - 2 beta[i] + beta[i+1] >= 0.
    D = np.zeros((m - 1, m + 1))
    i = np.arange(1, m)
    D[i - 1, i - 1] = 1.0
    D[i - 1, i] = -2.0
    D[i - 1, i + 1] = 1.0

    lb = np.maximum(knots[1:-1], 1.0 - knots[1:-1])
    lb[0] = max(lb[0], 1.0 - 1.0 / m)
    lb[-1] = max(lb[-1], 1.0 - 1.0 / m)
    bounds = Bounds(lb, np.ones(m - 1))
    constraint = LinearConstraint(D[:, 1:-1], -D[:, 0] - D[:, -1], np.inf)

    def beta(x):
        return np.r_[1.0, x, 1.0]

    def obj(x):
        pp = beta(x)
        return m * (0.5 * pp @ Q @ pp - b @ pp)

    def grad(x):
        return m * (Q @ beta(x) - b)[1:-1]

    result = minimize(
        obj,
        np.ones(m - 1),
        jac=grad,
        bounds=bounds,
        constraints=[constraint],
        method="SLSQP",
        options={"ftol": ftol, "maxiter": maxiter},
    )
    if not result.success:
        raise RuntimeError(result.message)

    pp = beta(result.x)
    return PickandsCurve(knots, pp)


def pickands_l2_projection(t, A_n, m=40, boundary_correct=False):
    """Original L2-sieve projection API; return values and right derivatives.

    The input grid must include endpoints 0 and 1. This variant preserves the
    old ``boundary_correct`` semantics. Use ``fit_pickands_curve`` for a valid
    fitted *object* with endpoints imposed and an exact spectral measure.
    """
    curve = _sieve_projection(t, A_n, m=m, boundary_correct=boundary_correct)
    return curve(t), curve.derivative(t)


def project_pickands_estimator_curve(t, A_n, m=40, boundary_correct=False):
    """Original estimator-projection API, imposing exact unit endpoints."""
    t = np.asarray(t, dtype=float)
    a = np.asarray(A_n, dtype=float).copy()
    if t.ndim != 1 or a.shape != t.shape or len(t) < 2:
        raise ValueError("t and A_n must be one-dimensional arrays of equal length")
    if not np.isclose(t[0], 0.0) or not np.isclose(t[-1], 1.0):
        raise ValueError("Estimator projection grid must include 0 and 1")
    a[0] = 1.0
    a[-1] = 1.0
    return pickands_l2_projection(t, a, m=m, boundary_correct=boundary_correct)


def fit_pickands_curve(t, A_n, m=40, boundary_correct=False, *,
                       ftol=1e-12, maxiter=1500):
    """Fit a globally valid Pickands function to a preliminary curve.

    Accepts strictly increasing grids that may omit 0 or 1: known unit
    endpoints are inserted automatically. If endpoints were provided, their
    preliminary estimates are overwritten by the exact value 1. Returns a
    ``PickandsCurve`` with evaluation, right derivative and exact spectral law.
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(A_n, dtype=float)
    if (t.ndim != 1 or a.shape != t.shape or t.size < 1
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(a))
            or np.any((t < 0) | (t > 1)) or not np.all(np.diff(t) > 0)):
        raise ValueError("Supply finite preliminary values on an increasing grid in [0,1]")
    t = t.copy()
    a = a.copy()
    if t[0] > 0:
        t = np.r_[0.0, t]
        a = np.r_[1.0, a]
    else:
        a[0] = 1.0
    if t[-1] < 1:
        t = np.r_[t, 1.0]
        a = np.r_[a, 1.0]
    else:
        a[-1] = 1.0
    return _sieve_projection(t, a, m=m, boundary_correct=boundary_correct,
                             ftol=ftol, maxiter=maxiter)


def fit_pickands_pilot(t, A_n, m=40, *, ftol=1e-12, maxiter=1500):
    """L2 sieve fit followed by degree-m Bernstein smoothing (Remark 4.13).

    Both stages use m. For asymptotic consistency increase m, refine the
    input grid, and let numerical optimization error vanish. m=40 is a
    compatibility default, not a calibrated or asymptotically fixed choice.
    """
    return fit_pickands_curve(t, A_n, m=m, ftol=ftol, maxiter=maxiter).bernstein()


def raw_madogram_curve(block_maxima, t):
    """Raw r=1 madogram from maxima (k, 2), with fresh ranks in each margin.

    Original maxima may have arbitrary continuous margins. Average ranks
    handle numerical ties. Endpoint values use the raw curve's continuous
    extension; the fitting API subsequently imposes exact unit endpoints.
    """
    maxima = np.asarray(block_maxima, dtype=float)
    if (maxima.ndim != 2 or maxima.shape[1] != 2 or len(maxima) < 2
            or not np.all(np.isfinite(maxima))):
        raise ValueError("block_maxima must have shape (k, 2), k >= 2, and be finite")
    x = _unit_argument(t)
    u = rankdata(maxima, axis=0, method="average")/(len(maxima)+1)
    flat = x.reshape(-1)
    power1 = np.divide(1., 1-flat, out=np.full_like(flat, np.inf), where=flat != 1)
    power2 = np.divide(1., flat, out=np.full_like(flat, np.inf), where=flat != 0)
    moment = np.maximum(u[:, 0, None]**power1, u[:, 1, None]**power2).mean(axis=0)
    result = (moment/(1-moment)).reshape(x.shape)
    return float(result) if result.ndim == 0 else result


def fit_madogram_pilot(block_maxima, t=None, m=40, *, ftol=1e-12, maxiter=1500):
    """Rank maxima, form the madogram, project in L2, then smooth.

    The default input grid has max(200, 4*m) intervals, so it refines as m
    grows. An explicit grid must increase in [0,1]; fitting inserts unit
    endpoints. Hold the returned pilot fixed throughout bootstrap datasets.
    """
    m = _integer(m, "m", 2)
    if t is None:
        t = np.linspace(0, 1, max(200, 4*m)+1)
    return fit_pickands_pilot(t, raw_madogram_curve(block_maxima, t), m=m,
                              ftol=ftol, maxiter=maxiter)


def sample_rlos(pilot, r=1, size=1, rng=None):
    """Exact limiting r-LOS draws, shape (size, 2, r), using Supplement G.

    Each block has independent rate-one Poisson arrivals and iid W ~ H/2.
    Stopping uses the deterministic bound 2/Gamma; there is no term cutoff.
    These are limiting block vectors. Re-rank each margin/order across
    blocks within every bootstrap dataset before computing rank moments.
    """
    r = _integer(r, "r", 1)
    size = _integer(size, "size", 0)
    rng = np.random.default_rng(rng)
    spectral = pilot.spectral_measure()
    if not isinstance(spectral, BetaMixtureSpectralMeasure):
        raise TypeError("sample_rlos requires a BernsteinPickandsCurve pilot")
    values = np.zeros((size, 2, r))
    arrivals = np.zeros(size)
    active = np.arange(size)
    while len(active):
        arrivals[active] += rng.exponential(size=len(active))
        w = spectral.sample(size=len(active), rng=rng)
        points = 2*np.stack((w, 1-w), axis=1)/arrivals[active, None]
        candidates = np.concatenate((values[active], points[:, :, None]), axis=2)
        values[active] = np.sort(candidates, axis=2)[:, :, ::-1][:, :, :r]
        finished = 2/arrivals[active] < values[active, :, -1].min(axis=1)
        active = active[~finished]
    return values
