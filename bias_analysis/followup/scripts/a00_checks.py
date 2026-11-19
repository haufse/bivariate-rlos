"""a00: completeness and consistency checks of the follow-up shards (and build the parquet cache).

Checks
 1. every (mode, exp, model, n, b, estimator) cell has R = 1000 replications, reps 0..999;
 2. non-finite values only in the *_inv labels; diag fail_pilot / fail_other counts;
 3. modes bck and r3or recomputed the standard estimates with the full-run seeds: the stored
    chk_<label>_0.50 must equal the full-run estimates at t = 0.5 (10-digit CSV rounding);
 4. pairing across experiments: r1 of exp r3 equals r1 of exp main (same data stream per
    (model, n, rep)), M1/M2 n = 1e4, common b.
Output: outputs/a00_checks.txt, outputs/a00_cells.csv.
"""
import sys

import numpy as np
import pandas as pd

import common as C

SPECS = []
for mo in ("M1", "M2", "M3", "M4"):
    for n in (1000, 5000, 10000, 20000, 100000):
        SPECS += [("full", "main", mo, n), ("smallb", "main", mo, n)]
        if n >= 5000:
            SPECS.append(("bck", "main", mo, n))
for mo, n in (("M2", 10000), ("M3", 10000), ("M4", 10000), ("M2", 100000), ("M3", 100000)):
    SPECS.append(("fineb", "main", mo, n))
for n in (1000, 5000, 10000, 20000):
    SPECS += [("full", "main", "M5", n), ("smallb", "main", "M5", n)]
for mo in ("M1", "M2"):
    SPECS += [("full", "r3", mo, 10000), ("smallb", "r3", mo, 10000)]
SPECS.append(("r3or", "r3", "M1", 10000))


def main():
    out = []
    cells = []
    est_all = {}
    diag_all = {}
    for s in SPECS:
        e = C.load(*s, kind="est")
        d = C.load(*s, kind="diag")
        if e is None:
            out.append(f"MISSING {s}")
            continue
        est_all[s], diag_all[s] = e, d
        g = e.groupby(["b", "estimator"]).rep.agg(["count", "min", "max", "nunique"]).reset_index()
        g.insert(0, "n", s[3]); g.insert(0, "model", s[2]); g.insert(0, "exp", s[1]); g.insert(0, "mode", s[0])
        cells.append(g)
        bad = g[(g["count"] != 1000) | (g["nunique"] != 1000) | (g["min"] != 0) | (g["max"] != 999)]
        noninv = ~e.estimator.str.contains("_inv")
        nf = int((~np.isfinite(e.loc[noninv, C.TCOLS].to_numpy())).sum())
        fp, fo = int(d.fail_pilot.sum()), int(d.fail_other.sum())
        out.append(f"{s}: rows {len(e)}, cells {len(g)}, incomplete cells {len(bad)}, "
                   f"non-finite (non-inv) {nf}, fail_pilot {fp}, fail_other {fo}")
    cells = pd.concat(cells, ignore_index=True)
    C.save(cells, "a00_cells.csv")

    # 3. recomputation checks
    out.append("\nrecomputed standard estimates (chk_*_0.50) vs stored full-run estimates:")
    for s, d in diag_all.items():
        if s[0] not in ("bck", "r3or"):
            continue
        base = est_all[("full", s[1], s[2], s[3])]
        chk = [c for c in d.columns if c.startswith("chk_")]
        for c in chk:
            lab = c[4:-5]
            b = base[base.estimator == lab][["b", "rep", "A_0.50"]]
            m = d[["b", "rep", c]].dropna().merge(b, on=["b", "rep"])
            out.append(f"  {s} {lab}: n={len(m)} max|diff|={np.abs(m[c] - m['A_0.50']).max():.2e}")
    # 4. pairing main vs r3
    out.append("\npairing: r1 of exp r3 vs exp main (same reps, common b), max |diff| over t:")
    for mo in ("M1", "M2"):
        a = pd.concat([est_all[("full", "r3", mo, 10000)], est_all[("smallb", "r3", mo, 10000)]])
        b_ = pd.concat([est_all[("full", "main", mo, 10000)], est_all[("smallb", "main", mo, 10000)]])
        a, b_ = a[a.estimator == "r1"], b_[b_.estimator == "r1"]
        m = a.merge(b_, on=["b", "rep"], suffixes=("_3", "_2"))
        dd = np.abs(m[[c + "_3" for c in C.TCOLS]].to_numpy() - m[[c + "_2" for c in C.TCOLS]].to_numpy()).max()
        out.append(f"  {mo}: {len(m)} pairs, b={sorted(m.b.unique())}, max|diff|={dd:.2e}")
    txt = "\n".join(out)
    print(txt)
    (C.OUT / "a00_checks.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
