"""kquad: quadrature on the trimmed half-interval [delta, 1/2] from an s04 grid (sigma_grid_theta<th>.npz).

If delta is a panel edge of the s04 grid (0, 0.01, 0.05, 0.1, 0.3), the s04 Gauss-Legendre nodes of the
panels to the right of delta are used unchanged.  Otherwise (e.g. delta = 0.15, the K of the Section 8
Monte Carlo), the panel [lo, hi] containing delta is replaced by an NNEW-node Gauss-Legendre rule on
[delta, hi], and Sigma at the new nodes is obtained by barycentric Lagrange interpolation through the
s04 nodes of that panel (12 nodes on [0.1, 0.3]).  Sigma(t) is analytic on (0, 1); the interpolant
reproduces direct quadrature (rlos_lib.sigma, npan=30, m=8) at t = 0.15, 0.2, 0.27 (theta = 2) to
max abs error 9e-14 (relative to the largest entry 1.5e-11); see s11_crosscheck.py.

By exchangeability every scalar integrand is symmetric about 1/2, so int_{[delta,1-delta]} = 2 int_delta^{1/2}.
"""
import numpy as np

NNEW = 16


def _bary(xn, yn, xq):
    """Barycentric Lagrange interpolation of the rows yn (n, ...) through nodes xn at points xq."""
    n = len(xn)
    wts = np.array([1.0 / np.prod([xn[i] - xn[j] for j in range(n) if j != i]) for i in range(n)])
    out = []
    for x in np.atleast_1d(xq):
        dif = x - xn
        if np.any(dif == 0.0):
            out.append(yn[int(np.argmin(np.abs(dif)))])
            continue
        c = wts / dif
        out.append(np.tensordot(c, yn, axes=1) / c.sum())
    return np.array(out)


def half_nodes(D, delta, nnew=NNEW):
    """Nodes t, weights w on [delta, 1/2], Sigma (nt, 9, 9) there, and a flag 'interpolated'."""
    t, w, pan, Sig = D['t'], D['w'], D['panel'], D['Sigma']
    edges = [float(e) for e in D['edges']]
    if any(abs(delta - e) < 1e-12 for e in edges):
        j0 = int(np.argmin([abs(delta - e) for e in edges]))
        sel = pan >= j0
        return t[sel], w[sel], Sig[sel], False
    j = int(np.searchsorted(edges, delta) - 1)          # panel [edges[j], edges[j+1]] contains delta
    lo, hi = edges[j], edges[j + 1]
    sel_p = pan == j
    g, gw = np.polynomial.legendre.leggauss(nnew)
    tn = 0.5 * (hi - delta) * g + 0.5 * (hi + delta)
    wn = 0.5 * (hi - delta) * gw
    Sn = _bary(t[sel_p], Sig[sel_p], tn)
    Sn = 0.5 * (Sn + Sn.transpose(0, 2, 1))
    right = pan > j
    return (np.concatenate([tn, t[right]]), np.concatenate([wn, w[right]]),
            np.concatenate([Sn, Sig[right]]), True)


def sigma_at(D, tq):
    """Sigma at an arbitrary t in (0, 1/2] by interpolation within the containing s04 panel."""
    t, pan, Sig = D['t'], D['panel'], D['Sigma']
    edges = [float(e) for e in D['edges']]
    j = min(int(np.searchsorted(edges, tq, side='right') - 1), len(edges) - 2)
    sel = pan == j
    return _bary(t[sel], Sig[sel], [tq])[0]
