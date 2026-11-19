"""Checks of the oracle covariance, the bootstrap and the samplers' attractors.

Run from sec8_simulations/:  python tests/check_oracle.py   (about 1-2 min)
1. sigma2_lib quadrature Sigma_0(t) (GH(2)) vs the fitted-law rank bootstrap
   (eq. rankboot-cov-estimator) driven by a near-exact Bernstein version of the
   GH(2) Pickands function (m=200), k=1000, B=3000.
2. Same for the asymmetric logistic attractor of M5.
3. Large-n r=1 rank madogram for every sampler vs its claimed attractor A_0.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import estimators as E   # noqa: E402
import models as Mo      # noqa: E402
import sigma2_lib        # noqa: E402
from pickands import fit_pickands_pilot  # noqa: E402

Ts = np.array([0.15, 0.3, 0.5, 0.7, 0.85])
tf = np.linspace(0, 1, 2001)
for name, A, att in [("GH(2)", Mo.A_logistic, sigma2_lib.Gumbel(2.0)),
                     ("AL(2;.5,.9)", Mo.A_asym_logistic, sigma2_lib.AsymLogistic(2.0, .5, .9))]:
    t0 = time.time()
    pilot = fit_pickands_pilot(tf, A(tf), m=200)
    err_pilot = np.abs(pilot(tf) - A(tf)).max()
    S0 = np.array([sigma2_lib.sigma2(att, t) for t in Ts])
    Sb = E.rank_bootstrap_covariances(pilot, 1000, 3000, 2, Ts, np.random.default_rng(99))
    rel = np.linalg.norm(Sb - S0, axis=(1, 2)) / np.linalg.norm(S0, axis=(1, 2))
    print(f"{name}: sup|pilot-A0|={err_pilot:.1e}; relative Frobenius error of bootstrap "
          f"Sigma (k=1000,B=3000) vs quadrature Sigma_0 at t={Ts.tolist()}: "
          f"{np.round(rel, 4).tolist()}  ({time.time()-t0:.0f}s)")
    # MC reference: relative sd of the Frobenius norm of a B=3000 covariance ~ sqrt(2/B)=2.6%

g = np.random.default_rng(5)
Tc = np.array([0.15, 0.3, 0.5, 0.7, 0.85])
for mk in ["M1", "M2", "M3", "M4", "M5", "D6b"]:
    spec = Mo.MODELS[mk]
    t0 = time.time()
    X = spec["sampler"](2_000_000, g)
    top = E.block_top_r(X, 200, 1)
    mom = E.rank_moments(E.rank_scores(top), Tc)
    a1 = E.r1_closed_form(mom[:, 0], Tc)
    print(f"{mk} {spec['name']:32s} n=2e6,b=200,k=1e4: r1 - A0 = "
          f"{np.round(a1 - spec['A'](Tc), 4).tolist()}  (sd ~ 0.003; {time.time()-t0:.0f}s)")
