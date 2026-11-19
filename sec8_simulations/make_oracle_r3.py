"""Build cache/oracle_sigma_GH2_r3.npz: Sigma_3(t) (9 x 9, lexicographic (l,m)) for the GH(2)
attractor on T_EVAL, for the r3_or estimator of mode r3or (bias investigation, run H3).

Deterministic quadrature with rlos_lib.sigma(t, 2, 3, npan=30, m=8) -- the Section 7 settings
(sec7_asymptotics/scripts/s04_sigma_grid.py), verified there against an
influence-function Monte Carlo (sec7_asymptotics/scripts/s03_mc_sigma_check.py: all 45 entries |z| <= 1.44).
rlos_lib is imported from sec7_asymptotics/scripts/rlos_lib.py.
Needs numba (environment/environment-sec7.yml).  Not needed for the simulations: the cache is committed.

Run from sec8_simulations/:
  NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=1 python make_oracle_r3.py
Checks printed: r=2 sub-block vs cache/oracle_sigma_GH2_r2.npz (sigma2_lib), symmetry,
exchange symmetry Sigma(1-t) = P Sigma(t) P, positive definiteness.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sec7_asymptotics" / "scripts"))
import config as C       # noqa: E402
import rlos_lib as rl    # noqa: E402

T = C.T_EVAL
t0 = time.time()
S = np.array([rl.sigma(float(t), 2.0, 3, npan=30, m=8) for t in T])
print(f"Sigma_3 on {len(T)} points in {time.time() - t0:.0f}s")
S2 = np.load(HERE / "cache" / "oracle_sigma_GH2_r2.npz")["Sigma"]
idx = [0, 1, 3, 4]
print("max |r=2 sub-block - sigma2_lib cache|:", np.abs(S[:, idx][:, :, idx] - S2).max())
print("max asymmetry:", np.abs(S - np.transpose(S, (0, 2, 1))).max())
P = np.zeros((9, 9))
for i, (l, m) in enumerate([(l, m) for l in range(1, 4) for m in range(1, 4)]):
    P[i, (m - 1) * 3 + (l - 1)] = 1
print("max |Sigma(1-t) - P Sigma(t) P|:", np.abs(S[::-1] - P @ S @ P.T).max())
print("min eigenvalue over t:", np.linalg.eigvalsh(S).min(), " max cond:",
      max(np.linalg.cond(s) for s in S))
out = HERE / "cache" / "oracle_sigma_GH2_r3.npz"
tmp = out.with_name(out.name + f".tmp{os.getpid()}")
with open(tmp, "wb") as fh:
    np.savez(fh, T=T, Sigma=S, attractor="GH(2)", r=3,
             source="rlos_lib.sigma(t, 2.0, 3, npan=30, m=8)")
os.replace(tmp, out)
print("wrote", out)
