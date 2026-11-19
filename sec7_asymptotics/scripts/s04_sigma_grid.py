"""s04: tabulate Sigma(t) (r=3; its leading 1x1 and 4x4 sub-blocks are the r=1, r=2 covariances,
see rlos_lib.py) at the GH(theta) attractor on a fixed composite Gauss-Legendre grid in t.

Run:  python s04_sigma_grid.py [theta ...]
      default thetas: 2.0 1.0 1.25 1.5 3.0
Output: ../outputs/sigma_grid_theta<theta>.npz  with arrays t, w (GL weights, panels below), Sigma (nt x 9 x 9)
No randomness.

t-grid: by exchangeability of all models used here, every scalar integrand f satisfies f(1-t) = f(t)
(checked in s05), so only [0, 1/2] is tabulated.  Panels with edges 0, 0.01, 0.05, 0.1, 0.3, 0.5 carry
8, 8, 8, 12, 12 Gauss-Legendre nodes, so that the trimmed intervals K_delta = [delta, 1-delta],
delta in {0, 0.01, 0.05, 0.1}, are unions of panels.  Quadrature in (s,s'): npan=30 geometric panels on
[1e-7, 40] with 8 nodes each (convergence to ~1e-14 checked against npan=60, m=12 in s05).
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')
import rlos_lib as rl

EDGES = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]
NODES = [8, 8, 8, 12, 12]


def t_grid():
    ts, ws, pan = [], [], []
    for j, (lo, hi, nn) in enumerate(zip(EDGES[:-1], EDGES[1:], NODES)):
        g, w = np.polynomial.legendre.leggauss(nn)
        ts.append(0.5 * (hi - lo) * g + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * w)
        pan.append(np.full(nn, j))
    return np.concatenate(ts), np.concatenate(ws), np.concatenate(pan)


if __name__ == '__main__':
    thetas = [float(x) for x in sys.argv[1:]] or [2.0, 1.0, 1.25, 1.5, 3.0]
    t, w, pan = t_grid()
    for th in thetas:
        t0 = time.time()
        Sig = np.array([rl.sigma(tt, th, 3, npan=30, m=8) for tt in t])
        np.savez(f'../outputs/sigma_grid_theta{th:g}.npz', t=t, w=w, panel=pan, Sigma=Sig,
                 edges=np.array(EDGES), theta=th)
        print(f'theta={th:g}: {len(t)} nodes, {time.time() - t0:.0f}s', flush=True)
