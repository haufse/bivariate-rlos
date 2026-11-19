"""Headline numbers of Section 8, recomputed from the summary folders.

Usage (from sec8_simulations/):
    python headline.py [--old ../data/sec8/full_only] [--new ../data/sec8/main_b5] [--all ../data/sec8/all_b2]

Every quantity is computed by the same function from the summaries of mode full alone (full_only),
from the main design b>=5 of the paper (full + fineb + bck + r3or; main_b5) and from all modes
incl. smallb (b>=2; all_b2, used for the small-b remark).  The column "draft" holds the values quoted
in an earlier draft, which was based on mode full alone; the column old_recomputed reproduces it
(consistency check).  The column new_b5 contains the numbers reported in the paper.
Writes <new>/headline_numbers.csv and prints a Markdown table.  No simulation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

M14 = ("M1", "M2", "M3", "M4")


class R:
    def __init__(self, d):
        d = Path(d)
        self.name = d.name
        rd = lambda f: pd.read_csv(d / f)   # noqa: E731
        self.opt, self.imse, self.dec = rd("optimal_b.csv"), rd("imse.csv"), rd("extra_decomp.csv")
        self.diag, self.ror, self.sig, self.r3 = rd("diagnostics.csv"), rd("extra_ratio_or.csv"), \
            rd("extra_sigcount.csv"), rd("extra_r3.csv")
        self.opt = self.opt[self.opt.exp.isin(["main", "indep"])]
        if "mcse_ratio_tab" not in self.opt:
            self.opt = self.opt.assign(mcse_ratio_tab=self.opt.mcse_ratio)

    def o(self, model, n, est, exp="main"):
        q = self.opt[(self.opt.exp == exp) & (self.opt.model == model) & (self.opt.n == n)
                     & (self.opt.estimator == est)].iloc[0]
        return q

    def ratio(self, model, n, est, exp="main"):
        q = self.o(model, n, est, exp)
        return f"{q.ratio_to_r1_at_own_opt:.2f} ({q.mcse_ratio_tab:.2f}) [b={int(q.b_opt)}]"

    def dd(self, model, n, est):
        x = self.dec
        return x[(x.exp == "main") & (x.model == model) & (x.n == n) & (x.estimator == est)].set_index("b")


def rng(v, fmt="{:.2f}"):
    v = np.asarray(v, float)
    return (fmt + "--" + fmt).format(np.nanmin(v), np.nanmax(v))


def items(D: R):
    out = []
    add = lambda key, draft, val: out.append((key, draft, val))   # noqa: E731
    n0 = 10000
    # ---- optimum ratios (Sec. 8.2, summary)
    for m in M14:
        add(f"opt ratio corrected r2 / r1, {m}, n=1e4", {"M1": "0.82", "M2": "0.88", "M3": "1.22",
                                                         "M4": "1.17"}[m], D.ratio(m, n0, "r2_eff_bceq"))
    for m in M14:
        add(f"opt ratio feasible r2 (uncorr.) / r1, {m}, n=1e4", "--", D.ratio(m, n0, "r2_eff"))
    for m in M14:
        add(f"opt ratio oracle r2 / r1, {m}, n=1e4", "--", D.ratio(m, n0, "r2_or"))
    add("M3 opt corrected ratio, n=1e3..1e5", "1.40, 1.21, 1.22, 1.02, 1.08",
        ", ".join(f"{D.o('M3', n, 'r2_eff_bceq').ratio_to_r1_at_own_opt:.2f}" for n in C.B_GRID))
    add("M4 opt corrected ratio, n=1e3..1e5", "1.04--1.24 (n<=2e4); 0.90 (1e5)",
        ", ".join(f"{D.o('M4', n, 'r2_eff_bceq').ratio_to_r1_at_own_opt:.2f}" for n in C.B_GRID))
    add("M2 POT opt ratio, n=1e4", "0.33", D.ratio("M2", n0, "pot"))
    add("POT ratio range at common b (M1, M3, M4, n=1e4)", "1.03--5.66",
        rng(D.imse[(D.imse.exp == "main") & (D.imse.n == n0) & D.imse.model.isin(["M1", "M3", "M4"])
                   & (D.imse.estimator == "pot")].ratio_to_r1))
    add("M5 opt corrected ratio, n=1e4", "0.80", D.ratio("M5", n0, "r2_eff_bceq"))
    add("M5 opt corrected ratio, other n (1e3, 5e3, 2e4)", "0.84--0.94",
        ", ".join(f"{D.o('M5', n, 'r2_eff_bceq').ratio_to_r1_at_own_opt:.2f}" for n in (1000, 5000, 20000)))
    add("M5 opt feasible r2 (uncorr.) ratio, n=1e4", "--", D.ratio("M5", n0, "r2_eff"))
    add("M5 POT opt ratio, n=1e4", "2.85", D.ratio("M5", n0, "pot"))
    # ---- M1 optimum location
    for n in C.B_GRID:
        g = D.imse[(D.imse.exp == "main") & (D.imse.model == "M1") & (D.imse.n == n)
                   & (D.imse.estimator == "r1")].sort_values("b")
        mono = bool(np.all(np.diff(g.imse_K.to_numpy()) > 0))
        bo = {e: int(D.o("M1", n, e).b_opt) for e in ("r1", "r2_eff", "r2_eff_bceq", "r2_or", "pot")}
        add(f"M1 n={n}: r1 IMSE increasing in b; b_opt (r1, eff, corr, or, pot); smallest b",
            "yes; all at smallest b", f"{'yes' if mono else 'NO'}; {tuple(bo.values())}; {int(g.b.min())}")
    add("M1 r1 10^4 IMSE_K at smallest b, n=1e4", "0.172 (b=5)",
        "{:.3f} (b={})".format(1e4 * D.imse[(D.imse.exp == "main") & (D.imse.model == "M1") & (D.imse.n == n0)
                                            & (D.imse.estimator == "r1")].sort_values("b").imse_K.iloc[0],
                               int(D.imse[(D.imse.model == "M1") & (D.imse.n == n0)].b.min())))
    # ---- choice of main n (comment)
    for n in (5000, 10000, 20000):
        s = D.sig[(D.sig.exp == "main") & (D.sig.n == n) & D.sig.model.isin(M14) & (D.sig.estimator == "r2_eff")]
        add(f"#b with r2_eff/r1 signif. <1 (>1), n={n}, M1-M4",
            {5000: "7 of 20 (5)", 10000: "12 of 24 (7)", 20000: "14 of 28 (8)"}[n],
            f"{s.n_better.sum()} of {s.n_b.sum()} ({s.n_worse.sum()})")
    for m, dr in (("M3", "1.02 vs 1.23"), ("M4", "1.26 vs 1.17"), ("M1", "0.94 vs 0.87")):
        add(f"opt ratio r2_eff/r1 {m}: n=2e4 vs n=1e4", dr,
            f"{D.o(m, 20000, 'r2_eff').ratio_to_r1_at_own_opt:.2f} vs {D.o(m, n0, 'r2_eff').ratio_to_r1_at_own_opt:.2f}")
    # ---- variance (Sec. 8.3)
    iv = []
    for m in M14:
        a, b = D.dd(m, n0, "r2_eff"), D.dd(m, n0, "r1")
        iv.append((a.ivar / b.ivar).rename(m))
    iv = pd.concat(iv, axis=1)
    add("IVar ratio r2_eff/r1, n=1e4, M1-M4, all b", "0.69--0.92", rng(iv.to_numpy()))
    add("IVar ratio r2_eff/r1, n=1e4, M1-M4, b>=5", "0.69--0.92", rng(iv[iv.index >= 5].to_numpy()))
    add("IVar ratio r2_eff/r1, M1, n=1e4: smallest b / b=5 / b=160", "-- / 0.85 / 0.70",
        f"{iv.M1.iloc[0]:.2f} (b={iv.index[0]}) / {iv.M1.loc[5]:.2f} / {iv.M1.loc[160]:.2f}")
    cr = []
    for (m, n), g in D.dec[(D.dec.exp == "main")].groupby(["model", "n"]):
        a = g[g.estimator == "r2_eff_bceq"].set_index("b").ivar
        b = g[g.estimator == "r2_eff"].set_index("b").ivar
        cr.append(pd.DataFrame(dict(model=m, n=n, ratio=a / b)).reset_index())
    cr = pd.concat(cr)
    add("IVar corrected/uncorrected, b>=20, all models and n (min)", ">= 0.98", f">= {cr[cr.b >= 20].ratio.min():.2f}")
    add("IVar corrected/uncorrected at b=5 (range)", "0.88--0.97", rng(cr[cr.b == 5].ratio))
    if (cr.b < 5).any():
        add("IVar corrected/uncorrected at b in {2,3,4} (range)", "--", rng(cr[cr.b < 5].ratio))
    exc = []
    for m in ("M1", "M3", "M4"):
        a, b = D.dd(m, n0, "r2_eff"), D.dd(m, n0, "r1")
        exc += [f"{m} b={int(x)}" for x in a.index[(a.ibias2 <= b.ibias2).to_numpy()]]
    add("IBias2 r2_eff > r1 at all b (M1, M3, M4, n=1e4): exceptions", "none", ", ".join(exc) or "none")
    rr = []
    for m in ("M3", "M4"):
        a, b = D.dd(m, n0, "r2_eff"), D.dd(m, n0, "r1")
        rr.append((a.ibias2 / b.ibias2)[a.index <= 20])
    rr = pd.concat(rr)
    add("IBias2 ratio r2_eff/r1, M3 and M4, b<=20 (n=1e4)", "1.6--7.5", rng(rr, "{:.1f}"))
    add("IBias2 ratio r2_eff/r1, M3 and M4, 5<=b<=20 (n=1e4)", "1.6--7.5", rng(rr[rr.index >= 5], "{:.1f}"))
    # ---- feasible vs oracle
    ro = D.ror[(D.ror.exp == "main") & (D.ror.estimator == "r2_eff")]
    bins = [(0, 100, "k<=100"), (100, 200, "100<k<=200"), (200, 500, "200<k<=500"), (500, 1e9, "k>500")]
    add("feasible/oracle IMSE median by k class", "1.14, 1.07, 1.02, 1.01",
        ", ".join(f"{ro[(ro.k > lo) & (ro.k <= hi)].ratio.median():.2f}" for lo, hi, _ in bins))
    add("feasible/oracle IMSE, k>500: max (model, n, b)", "--",
        "{:.2f} ({} n={} b={})".format(*ro[ro.k > 500].sort_values("ratio").iloc[-1][["ratio", "model", "n", "b"]]))
    dg = D.diag[D.diag.exp == "main"]
    add("max k with fallback > 5% (main)", "125", f"{int(dg[dg.fallback_K > 0.05].k.max())}")
    add("fallback at n=1e4: b<=20 / b=40 / b=80 / b=160 (M1-M4)", "0 / <=0.4% / 4.6--6.4% / 26.5--30.2%",
        " / ".join(rng(100 * dg[(dg.n == n0) & dg.model.isin(M14) & sel].fallback_K, "{:.1f}") for sel in
                   (dg.b <= 20, dg.b == 40, dg.b == 80, dg.b == 160)))
    add("sqrt(k) max|eq-os|: max median / max q95 (main)", "0.036 / 0.051",
        f"{dg.eqos_median.max():.3f} / {dg.eqos_q95.max():.3f}")
    for lab, sel in (("b>=5", dg.b >= 5), ("b in {2,3,4}", dg.b < 5)):
        if sel.any():
            g = dg[sel]
            add(f"sqrt(k) max|eq-os|, {lab}: max median / max q95 (at model n b)", "--",
                "{:.3f} / {:.3f} ({} n={} b={})".format(g.eqos_median.max(), g.eqos_q95.max(),
                                                       *g.sort_values("eqos_median").iloc[-1][["model", "n", "b"]]))
    nK = int(((C.T_EVAL >= C.K_MAIN[0] - 1e-9) & (C.T_EVAL <= C.K_MAIN[1] + 1e-9)).sum())
    add("one-step outside Theta_t, n=1e4, b<=40 (max %)", "0.43",
        f"{100 * (dg[(dg.n == n0) & (dg.b <= 40)].os_trunc_eff_K / nK).max():.2f}")
    k1000 = dg[dg.k >= 1000]
    add("Frobenius error, k>=1000, M1, M3-M5", "0.11--0.14",
        rng(k1000[k1000.model.isin(["M1", "M3", "M4", "M5"])].sig_frob_rel_or))
    add("Frobenius error, k>=1000, M2", "0.14--0.18", rng(k1000[k1000.model == "M2"].sig_frob_rel_or))
    add("M2 feasible/oracle IMSE, k>=1000", "0.94--1.03", rng(ro[(ro.model == "M2") & (ro.k >= 1000)].ratio))
    # ---- M5 common b
    m5 = D.imse[(D.imse.exp == "main") & (D.imse.model == "M5") & (D.imse.n == n0) & (D.imse.estimator == "r2_eff")]
    add("M5 r2_eff/r1, n=1e4, 5<=b<=40", "0.84--0.92", rng(m5[(m5.b >= 5) & (m5.b <= 40)].ratio_to_r1))
    if (m5.b < 5).any():
        add("M5 r2_eff/r1, n=1e4, b in {2,3,4}", "--",
            ", ".join(f"{r:.2f}" for r in m5[m5.b < 5].sort_values("b").ratio_to_r1))
    # ---- r = 3
    for m, dr in (("M1", "0.98--1.15"), ("M2", "1.04--1.28")):
        x = D.r3[(D.r3.model == m) & (D.r3.estimator == "r3_eff") & (D.r3.ref == "r2_eff")]
        add(f"r3_eff/r2_eff range, {m} (b grid)", dr,
            rng(x.ratio) + f" (b {int(x.b.min())}..{int(x.b.max())})")
    f3 = D.diag[(D.diag.exp == "r3") & (D.diag.model == "M1")].sort_values("b")
    add("r=3 fallback, M1, by b (%)", "7.7, 35, 68, 88, 96, 99 (b=5..160)",
        ", ".join(f"b{int(b)}: {100 * f:.1f}" for b, f in zip(f3.b, f3.fallback_K)))
    x = D.r3[(D.r3.model == "M1") & (D.r3.estimator == "r3_or")]
    if len(x):
        add("r3_or/r2_or, M1, n=1e4, b=5..80", "--", ", ".join(
            f"{r.ratio:.3f} ({r.mcse_ratio:.3f})" for r in x[x.ref == "r2_or"].sort_values("b").itertuples()))
        add("r3_eff/r3_or, M1, n=1e4, b=5..80", "--", rng(D.r3[(D.r3.model == "M1") & (D.r3.estimator == "r3_eff")
                                                             & (D.r3.ref == "r3_or")].ratio))
    # ---- independence (no new runs)
    for m, dr in (("D6a", "1.16 (0.03)"), ("D6b", "1.51 (0.02)")):
        add(f"indep opt ratio r2_eff/r1, {m}", dr, D.ratio(m, n0, "r2_eff", exp="indep"))
    for m, dr in (("D6a", "0.20"), ("D6b", "0.13")):
        add(f"indep opt ratio POT/r1, {m}", dr, D.ratio(m, n0, "pot", exp="indep"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(HERE.parent / "data" / "sec8" / "full_only"))
    ap.add_argument("--new", default=str(HERE.parent / "data" / "sec8" / "main_b5"),
                    help="main design (b>=5: full+fineb+bck+r3or)")
    ap.add_argument("--all", default=str(HERE.parent / "data" / "sec8" / "all_b2"),
                    help="all modes incl. smallb (b>=2); third column")
    a = ap.parse_args()
    old, new, alb = (dict((k, v) for k, _, v in items(R(d))) for d in (a.old, a.new, a.all))
    draft = {k: d for k, d, _ in items(R(a.all))}
    rows = [dict(quantity=k, draft=draft[k], old_recomputed=old.get(k, "--"), new_b5=new.get(k, "--"),
                 all_b=alb[k]) for k in draft]
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.new) / "headline_numbers.csv", index=False)
    print("| quantity | draft | old (recomputed) | new, b>=5 | all b>=2 |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['quantity']} | {r['draft']} | {r['old_recomputed']} | {r['new_b5']} | {r['all_b']} |")


if __name__ == "__main__":
    main()
