"""s12: exact finite-k mean of the rank moments at independence; rate of the remainder in
Conjecture J.3(i).

At independence, the scaled ranks of the pair (M_1^{(l)}, M_2^{(m)}) of one block are independent
and uniform on {1,...,k}/(k+1) in each margin, so that exactly
    E mu_hat^{(l,m)}(t) = k^{-2} sum_{i,j} max(a_i, b_j),
    a_i = exp{-q_l(i/(k+1))/(1-t)},  b_j = exp{-q_m(j/(k+1))/t},  q_l = gammainccinv(l, .),
(the code path of estimators.rank_moments: exp(-xi) = max(exp(-q1/(1-t)), exp(-q2/t))).
The limit is mu0 = 1 - int_0^inf Q(l,(1-t)x) Q(m,t x) e^{-x} dx.  We report
    Q_k = k (mu0 - E mu_hat)      (-> Q of eq. sec8-bias:eq:Q),
    rem_k = Q - Q_k               and its local log-slope in k,
    eta_k = lambda' Q_k + nu      (finite-k rank part, IND, W in {r1, or, id}; r = 2),
and compare eta_k with the limit-law Monte Carlo of s03 (k = 125, 250, 500).
Checks: (1,1) at t = 1/2 equals 1/2 - (2k+1)/(6k(k+1)) exactly; Q_k -> Q (biaslib, mpmath for r = 3).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-9 nice -n 19 \
     python s12_exact_independence.py
Writes ../outputs/s12_output.txt, ../outputs/s12_exact_ind.csv.  Deterministic.  Runtime < 1 min.
"""
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.special import gammaincc, gammainccinv

import biaslib as B

OUT = B.HERE.parent / "outputs"
KS = [62, 125, 250, 500, 1000, 2000, 8000, 32000, 128000]
LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def sum_max(a, b):
    """sum_{i,j} max(a_i, b_j) in O(k log k)."""
    b = np.sort(b)
    cb = np.concatenate([[0.0], np.cumsum(b)])
    n_le = np.searchsorted(b, a, side="right")          # b_j <= a_i
    return float(np.sum(a * n_le + (cb[-1] - cb[n_le])))


def emu_exact(l, m, t, k):
    u = np.arange(1, k + 1) / (k + 1)
    a = np.exp(-gammainccinv(l, u) / (1 - t))
    b = np.exp(-gammainccinv(m, u) / t)
    return sum_max(a, b) / k ** 2


def mu0(l, m, t):
    f = lambda x: gammaincc(l, (1 - t) * x) * gammaincc(m, t * x) * np.exp(-x)
    v = sum(integrate.quad(f, lo, hi, epsabs=1e-15, epsrel=1e-13, limit=400)[0]
            for lo, hi in ((0, 1), (1, 4), (4, 16), (16, 60), (60, 200)))
    return 1 - v


def main():
    rows = []
    for t in (0.5, 0.3):
        for pair in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 3)]:
            l, m = pair
            Q = (B.beta_C(t, "IND", [pair])[0] if max(pair) <= 2
                 else B.beta_C(t, "IND", [pair], method="mp")[0])
            m0 = mu0(l, m, t)
            for k in KS:
                Qk = k * (m0 - emu_exact(l, m, t, k))
                rows.append(dict(t=t, pair=f"{l}{m}", k=k, Q=Q, Q_k=Qk, rem=Q - Qk, rel=(Q - Qk) / Q))
    df = pd.DataFrame(rows)
    # exactness check (1,1), t = 1/2
    k = 125
    ex = 0.5 - (2 * k + 1) / (6 * k * (k + 1))
    log(f"CHECK (1,1) t=1/2 k=125: exact sum {emu_exact(1, 1, 0.5, k):.15f} vs closed form {ex:.15f}")
    log(f"CHECK mu0 (1,1) t=1/2 = {mu0(1, 1, 0.5):.15f} (exact 1/2)")
    df["slope"] = np.nan
    for (t, p), g in df.groupby(["t", "pair"]):
        g = g.sort_values("k")
        s = np.diff(np.log(np.abs(g.rem.to_numpy()))) / np.diff(np.log(g.k.to_numpy()))
        df.loc[g.index[1:], "slope"] = s
    pd.set_option("display.width", 200)
    log("\nremainder rem_k = Q - k(mu0 - E mu_hat): relative size and local log-slope in k")
    log(df.pivot_table(index="k", columns=["t", "pair"], values="rel").round(4).to_string())
    log(df.pivot_table(index="k", columns=["t", "pair"], values="slope").round(3).to_string())
    log("  (slope -1/2 <=> remainder Theta(k^{-3/2}) in E mu_hat; -1/3 <=> Theta(k^{-4/3}); -1 <=> O(k^{-2}))")
    log("\n sqrt(k)*rem for (1,2),(2,2) and k^{1/3}*rem for (2,3),(3,3), t = 1/2:")
    g = df[df.t == 0.5].copy()
    g["scaled"] = np.where(g.pair.isin(["12", "21", "22"]), np.sqrt(g.k) * g.rem,
                           np.where(g.pair.isin(["23", "33"]), g.k ** (1 / 3) * g.rem, g.k * g.rem))
    log(g.pivot_table(index="k", columns="pair", values="scaled").round(4).to_string())

    # finite-k eta at independence, r = 2
    th = pd.read_csv(OUT / "s02_theory.csv")
    out = []
    for t in (0.5, 0.3):
        for w in ("r1", "or", "id"):
            r = th[(th.attractor == "IND") & (th.weights == w) & np.isclose(th.t, t)].iloc[0]
            lam = np.array([r.lam1, r.lam2, r.lam3, r.lam4])
            for k in KS:
                Qk = np.array([df[(df.t == t) & (df.pair == p) & (df.k == k)].Q_k.iloc[0]
                               for p in ("11", "12", "21", "22")])
                out.append(dict(t=t, weights=w, k=k, eta_k=lam @ Qk + r.nonlin_term, eta=r.c_total))
    e = pd.DataFrame(out)
    log("\nfinite-k eta at independence (lambda' Q_k + nu), r = 2; limit in column k=inf:")
    pv = e.pivot_table(index=["t", "weights"], columns="k", values="eta_k").round(3)
    pv["inf"] = e.groupby(["t", "weights"]).eta.first().round(3)
    log(pv.to_string())
    log("limit-law MC (s03_output.txt, IND, t=1/2, k=500/250/125): r1 -1.16/., or -3.26/-3.18/-3.26 "
        "(see s03_output.txt for all values and MCSE)")
    df.to_csv(OUT / "s12_exact_ind.csv", index=False, float_format="%.10g")
    e.to_csv(OUT / "s12_eta_ind.csv", index=False, float_format="%.10g")
    (OUT / "s12_output.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
