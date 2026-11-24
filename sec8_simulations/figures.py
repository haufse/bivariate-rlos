"""Figures and LaTeX tables of Section 8 (and the supplement) from the summary CSVs.

Usage (from sec8_simulations/; needs summarize.py and extras.py output first):
    python figures.py [--results ../data/sec8/main_b5] [--sec7-grid PATH]

Reads only CSVs in --results (plus, for the asymptotic reference curves of F3 and the
independence limit of S6, the Section 7 covariance grid and the local oracle cache).
No simulation.  Writes vector PDFs to ../figures/ and booktabs snippets to ../tables/.

Figure sizes: \\textwidth of imsart [aos] = 406pt = 5.62in (checked by compiling
\\the\\textwidth); include every figure at width=\\textwidth (scale 1).  Fonts: Times text
(newtxtext) and Computer Modern math, as in the manuscript; 8pt labels.
Colours: Okabe--Ito, one colour per estimator, fixed across all figures.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator  # noqa: E402
from scipy.stats import norm             # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C          # noqa: E402
import estimators as E      # noqa: E402
import models as Mo         # noqa: E402

P = HERE.parent                      # repository root
SEC7 = P / "sec7_asymptotics/outputs"
FU = P / "bias_analysis/followup"
BIAS = P / "bias_analysis/investigation"
FIGS, TABS = P / "figures", P / "tables"
TW = 406.0 / 72.27                       # \textwidth in inches
MAIN_N = 10000
ALL_N = [1000, 5000, 10000, 20000, 100000]
K_LO, K_HI = C.K_MAIN

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage{amsmath}",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "lines.linewidth": 1.0, "lines.markersize": 3.2,
    "errorbar.capsize": 0, "legend.frameon": False, "pdf.fonttype": 42,
})

# Okabe--Ito; one style per estimator, fixed across all figures
STY = {
    "r1":          dict(c="#000000", m="o", ls="-",  lw=1.0, lab=r"$r=1$"),
    "r2_id":       dict(c="#999999", m="v", ls="-",  lw=0.9, lab=r"$r=2$, $W=I$"),
    "r2_eff":      dict(c="#0072B2", m="s", ls="-",  lw=1.0, lab=r"$r=2$, $\hat\Sigma_n^{-1}$"),
    "r2_eff_bceq": dict(c="#D55E00", m="D", ls="-",  lw=1.0, lab=r"$r=2$, $\hat\Sigma_n^{-1}$, corrected (eq.)"),
    "r2_eff_bcos": dict(c="#CC79A7", m="^", ls=":",  lw=1.0, lab=r"$r=2$, $\hat\Sigma_n^{-1}$, corrected (one-step)"),
    "r2_or":       dict(c="#56B4E9", m="",  ls="--", lw=0.9, lab=r"$r=2$, $\Sigma^{-1}$ (oracle)"),
    "pot":         dict(c="#009E73", m="x", ls="-",  lw=0.9, lab=r"POT, $k_p=\lfloor n/b\rfloor$"),
    "r3_eff":      dict(c="#E69F00", m="P", ls="-",  lw=1.0, lab=r"$r=3$, $\hat\Sigma_n^{-1}$"),
    "r3_id":       dict(c="#E69F00", m="P", ls="--", lw=0.8, lab=r"$r=3$, $W=I$"),
    "r3_or":       dict(c="#E69F00", m="P", ls=":",  lw=1.1, lab=r"$r=3$, $\Sigma_3^{-1}$ (oracle)"),
}
MCOL = {"M1": "#0072B2", "M2": "#D55E00", "M3": "#009E73", "M4": "#CC79A7", "M5": "#E69F00"}
# Panel titles (two lines: family, parameters).  M3 is the outer-power Clayton copula of
# Bucher, Volgushev and Zou (2019) with generator psi(x) = (1 + x^{1/2})^{-1}, i.e. BVZ
# (theta, beta) = (1, 2) (models.sample_outer_power_clayton; the earlier label "theta = 2,
# beta = 1" was wrong).
MODEL_TITLE = {"M1": "(M1) Gumbel--Hougaard\n$\\theta=2$", "M2": "(M2) Joe\n$\\theta=2$",
               "M3": "(M3) outer-power Clayton\n$(\\theta,\\beta)=(1,2)$",
               "M4": "(M4) Archimax\n$\\psi_2$",
               "M5": r"(M5) asym.\ logistic, $\theta=2$, $\psi=(0.5,0.9)$",
               "D6a": "(D6a) independence", "D6b": r"(D6b) Clayton$(1)$"}
MODEL_SHORT = {m: v.split("\n")[0] for m, v in MODEL_TITLE.items()}
NLAB = {1000: r"10^3", 5000: r"5\cdot10^3", 10000: r"10^4", 20000: r"2\cdot10^4", 100000: r"10^5"}


def mkfig(nrows, ncols, height, wspace=0.02, hspace=0.02, **kw):
    """Figure of exactly \\textwidth x height (constrained layout, no tight bbox), so that
    \\includegraphics[width=\\textwidth] includes it at scale 1."""
    fig, axs = plt.subplots(nrows, ncols, figsize=(TW, height), layout="constrained", squeeze=False, **kw)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=wspace, hspace=hspace)
    return fig, axs


def logy_ticks(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def save(fig, name):
    """Save the PDF; then subset the embedded Type 1 fonts with ghostscript (if available),
    which shrinks each file from about 600 KB to well under 100 KB without changing the
    vector content or the page size."""
    FIGS.mkdir(exist_ok=True)
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    gs = shutil.which("gs")
    if gs:
        tmp = out.with_suffix(".gs.pdf")
        r = subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
                            "-dCompatibilityLevel=1.5", "-dSubsetFonts=true", "-dEmbedAllFonts=true",
                            "-dAutoRotatePages=/None", f"-sOutputFile={tmp}", str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0 and tmp.exists():
            os.replace(tmp, out)
        else:
            print("ghostscript font subsetting failed; keeping the matplotlib PDF:", r.stderr[:200])
    print("wrote", out)


def logb_axis(ax, bs):
    """Log b axis.  Labelled ticks at the doubling grid (and b=2) inside the range; all other b
    of the (merged) grid get unlabelled minor ticks."""
    ax.set_xscale("log")
    cand = (2, 5, 10, 20, 40, 80, 320) if max(bs) >= 320 else (2, 5, 10, 20, 40, 80, 160)
    lab = sorted({min(bs)} | {b for b in cand if min(bs) <= b <= max(bs)})
    ax.xaxis.set_major_locator(FixedLocator(lab))
    ax.xaxis.set_minor_locator(FixedLocator([b for b in bs if b not in lab]))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xticklabels([str(b) for b in lab], fontsize=6.5 if len(lab) >= 7 else 7)
    ax.set_xlim(min(bs) / 1.25, max(bs) * 1.25)


SMALLB_IN = [False]   # set in main(): do the results contain b in {2,3,4} (mode smallb)?


def std_grid(n):
    """Doubling grid of mode full (plus b in {2,3,4} if mode smallb is in the results); excludes
    the fine-grid values of mode fineb, which only enter the MC-optimal b and the IMSE-vs-b
    figures.  Design choice: the main design uses b>=5 only (b<5 only in a remark)."""
    return sorted((set(C.SMALL_B) if SMALLB_IN[0] else set()) | set(C.B_GRID[n]))


def errline(ax, x, y, se, key, mult=2.0, **kw):
    s = STY[key]
    ax.errorbar(x, y, yerr=mult * np.asarray(se), color=s["c"], marker=s["m"], ls=s["ls"], lw=s["lw"],
                elinewidth=0.6, ms=kw.pop("ms", 3.0), mfc=kw.pop("mfc", s["c"]), label=s["lab"], **kw)


def clipped(ax, x, y, lo, hi, key):
    """Mark ratios outside the y-range by open triangles at the axis edge."""
    s = STY[key]
    y = np.asarray(y)
    for xi, yi in zip(x, y):
        if yi > hi:
            ax.plot(xi, hi, marker="^", color=s["c"], mfc="white", ms=3.5, clip_on=False, zorder=5)
        elif yi < lo:
            ax.plot(xi, lo, marker="v", color=s["c"], mfc="white", ms=3.5, clip_on=False, zorder=5)


def legend_below(fig, keys, ncol=4, y=None, extra=()):
    h = [Line2D([], [], color=STY[k]["c"], marker=STY[k]["m"], ls=STY[k]["ls"], lw=STY[k]["lw"], ms=3.0)
         for k in keys] + [e[0] for e in extra]
    lab = [STY[k]["lab"] for k in keys] + [e[1] for e in extra]
    fig.legend(h, lab, loc="outside lower center", ncol=ncol, handlelength=2.2, columnspacing=1.2)


# ======================================================================= loaders
class Data:
    def __init__(self, rdir):
        self.r = Path(rdir)
        rd = lambda f: pd.read_csv(self.r / f)          # noqa: E731
        self.imse, self.opt, self.pw, self.diag = rd("imse.csv"), rd("optimal_b.csv"), \
            rd("pointwise.csv"), rd("diagnostics.csv")
        self.dec, self.pv, self.ror = rd("extra_decomp.csv"), rd("extra_pointvar.csv"), \
            rd("extra_ratio_or.csv")
        self.indep, self.sens, self.r3 = rd("extra_indep.csv"), rd("extra_sens.csv"), rd("extra_r3.csv")
        self.bcc, self.sig = rd("extra_bccheck.csv"), rd("extra_sigcount.csv")
        self.dec_ind = rd("extra_decomp_indep.csv")
        opt = lambda f: rd(f) if (self.r / f).exists() else None      # noqa: E731
        self.bck, self.bopt = opt("extra_bck.csv"), opt("extra_bck_opt.csv")

    def im(self, model, n, est, exp="main"):
        x = self.imse
        return x[(x.exp == exp) & (x.model == model) & (x.n == n) & (x.estimator == est)].sort_values("b")

    def de(self, model, n, est):
        x = self.dec
        return x[(x.model == model) & (x.n == n) & (x.estimator == est)].sort_values("b")


# ======================================================================= F1 (and S1, S3)
def fig_imse(D, n, models, name, keys_top, keys_ratio, ylim_ratio=(0.6, 1.6), cols_are_n=None,
             shade_below=None):
    """Top: IMSE_K x 1e4 vs b (log-log); bottom: paired IMSE ratio to r=1 (+-2 MCSE)."""
    panels = cols_are_n if cols_are_n else models
    fig, axs = mkfig(2, len(panels), 3.45, sharex="col", gridspec_kw=dict(height_ratios=[1.15, 1]))
    for j, pnl in enumerate(panels):
        model, nn = (models[0], pnl) if cols_are_n else (pnl, n)
        bs = sorted(D.im(model, nn, "r1").b)
        a0, a1 = axs[0, j], axs[1, j]
        ms = 2.2 if len(bs) > 12 else 3.0
        for key in keys_top:
            g = D.im(model, nn, key)
            errline(a0, g.b, 1e4 * g.imse_K, 1e4 * g.mcse_K, key, ms=ms)
        a0.set_yscale("log")
        logy_ticks(a0)
        a0.set_title(MODEL_TITLE[model] if not cols_are_n else rf"$n={NLAB[nn]}$", pad=3)
        for key in keys_ratio:
            g = D.im(model, nn, key)
            errline(a1, g.b, g.ratio_to_r1, g.mcse_ratio, key, ms=ms)
            clipped(a1, g.b.to_numpy(), g.ratio_to_r1.to_numpy(), *ylim_ratio, key)
        a1.axhline(1.0, color="k", lw=0.5, zorder=0)
        a1.set_ylim(*ylim_ratio)
        logb_axis(a1, bs)
        if shade_below is not None:      # b outside the main design (remark only)
            for ax in (a0, a1):
                ax.axvspan(min(bs) / 1.25, shade_below / 1.12, color="0.88", lw=0, zorder=-1)
        a1.set_xlabel(r"block length $b$")
    axs[0, 0].set_ylabel(r"$10^4\times\operatorname{IMSE}_K$")
    axs[1, 0].set_ylabel(r"ratio to $r=1$")
    if not cols_are_n:
        fig.suptitle(rf"$n={NLAB[n]}$, $k=\lfloor n/b\rfloor$", fontsize=7.5)
    else:
        fig.suptitle(rf"{MODEL_TITLE[models[0]]}, $k=\lfloor n/b\rfloor$", fontsize=7.5)
    keys = list(dict.fromkeys(keys_top + keys_ratio))
    legend_below(fig, keys, ncol=3)
    save(fig, name)


# ======================================================================= F2 (and S2)
def fig_decomp(D, n, models, name, cols_are_n=None):
    keys = ["r1", "r2_eff", "r2_eff_bceq"]
    panels = cols_are_n if cols_are_n else models
    fig, axs = mkfig(2, len(panels), 3.3, sharex="col", sharey="row")
    for j, pnl in enumerate(panels):
        model, nn = (models[0], pnl) if cols_are_n else (pnl, n)
        bs = sorted(D.im(model, nn, "r1").b)
        ms = 2.2 if len(bs) > 12 else 3.0
        for key in keys:
            g = D.de(model, nn, key)
            errline(axs[0, j], g.b, 1e4 * g.ibias2, 1e4 * g.mcse_ibias2, key, ms=ms)
            errline(axs[1, j], g.b, 1e4 * g.ivar, 1e4 * g.mcse_ivar, key, ms=ms)
        g = D.de(model, nn, "r2_eff")
        axs[0, j].plot(g.b, 1e4 * g.ibias2_noise, color="0.55", ls=(0, (1, 1.5)), lw=0.8)
        for a in axs[:, j]:
            a.set_yscale("log")
        logb_axis(axs[1, j], bs)
        axs[1, j].set_xlabel(r"block length $b$")
        axs[0, j].set_title(MODEL_TITLE[model] if not cols_are_n else rf"$n={NLAB[nn]}$", pad=3)
    axs[0, 0].set_ylabel(r"$10^4\times\operatorname{IBias}^2_K$")
    axs[1, 0].set_ylabel(r"$10^4\times\operatorname{IVar}_K$")
    if not cols_are_n:
        fig.suptitle(rf"$n={NLAB[n]}$", fontsize=7.5)
    else:
        fig.suptitle(MODEL_TITLE[models[0]], fontsize=7.5)
    iv = np.concatenate([1e4 * D.de(*((models[0], c) if cols_are_n else (c, n)), k).ivar.to_numpy()
                         for c in panels for k in keys])
    axs[1, 0].set_ylim(iv.min() / 1.4, iv.max() * 1.4)
    ib = np.concatenate([1e4 * (D.de(*((models[0], c) if cols_are_n else (c, n)), k).ibias2
                                + 2 * D.de(*((models[0], c) if cols_are_n else (c, n)), k).mcse_ibias2).to_numpy()
                         for c in panels for k in keys])
    axs[0, 0].set_ylim(1e-4, ib.max() * 1.5)
    for a in axs[1]:
        logy_ticks(a)
    noise = (Line2D([], [], color="0.55", ls=(0, (1, 1.5)), lw=0.8),
             r"MC noise level $\operatorname{IVar}_K/R$ ($r=2$, $\hat\Sigma_n^{-1}$)")
    legend_below(fig, keys, ncol=2, extra=[noise])
    save(fig, name)


# ======================================================================= F3
def asymptotic_variances(t, Sig4, a, r1_only=False):
    """sigma^2 for r=1, r=2 identity and r=2 optimal weights (Thm process-cov-unknown):
    lambda_W' Sigma lambda_W with lambda_W = W d/(d'W d), d = D'(A(t), t)."""
    d = E.dprime_vec(a, t, 2)
    v1 = Sig4[:, 0, 0] / d[:, 0] ** 2
    vid = np.einsum("ti,tij,tj->t", d, Sig4, d) / np.einsum("ti,ti->t", d, d) ** 2
    Si = np.linalg.inv(Sig4)
    vopt = 1.0 / np.einsum("ti,tij,tj->t", d, Si, d)
    return v1, vid, vopt


def sec7_curves(path):
    """Asymptotic variance curves at GH(2) from the Section 7 grid (t in (0,1/2), mirrored by
    exchangeability).  Returns None if the grid is unavailable (F3 then has a TODO hook)."""
    try:
        z = np.load(path)
    except (OSError, FileNotFoundError):
        return None
    t = z["t"]
    S9 = z["Sigma"]
    idx = [0, 1, 3, 4]                                   # (1,1),(1,2),(2,1),(2,2) in lexicographic r=3
    S4 = S9[:, idx][:, :, idx]
    a = Mo.A_logistic(t, float(z["theta"]))
    v1, vid, vopt = asymptotic_variances(t, S4, a)
    tt = np.concatenate([t, 1 - t[::-1]])
    mirror = lambda v: np.concatenate([v, v[::-1]])    # noqa: E731
    return dict(t=tt, v1=mirror(v1), vid=mirror(vid), vopt=mirror(vopt))


def check_sec7_vs_cache(cur):
    """Cross-check: Section 7 grid curves vs the Section 8 oracle cache (sigma2_lib) on T_EVAL."""
    z = np.load(HERE / "cache" / "oracle_sigma_GH2_r2.npz")
    v1, vid, vopt = asymptotic_variances(C.T_EVAL, z["Sigma"], Mo.A_logistic(C.T_EVAL))
    from scipy.interpolate import CubicSpline
    o = np.argsort(cur["t"])
    dev = max(np.max(np.abs(CubicSpline(cur["t"][o], cur[k][o])(C.T_EVAL) / v - 1))
              for k, v in (("v1", v1), ("vid", vid), ("vopt", vopt)))
    print(f"F3: Sec 7 grid vs Sec 8 oracle cache, max relative deviation (cubic spline) {dev:.2e}")
    return dict(v1=v1, vid=vid, vopt=vopt)


def fig_variance(D, sec7_grid):
    cur = sec7_curves(sec7_grid)
    fig, axs = mkfig(1, 2, 2.35, wspace=0.06)
    a, bx = axs[0]
    b, n = 40, MAIN_N
    pv = D.pv[(D.pv.model == "M1") & (D.pv.exp == "main")]
    off = {"r1": -0.006, "r2_id": 0.0, "r2_eff": 0.006}
    for key in ("r1", "r2_id", "r2_eff"):
        g = pv[(pv.n == n) & (pv.b == b) & (pv.estimator == key)].sort_values("t")
        if key == "r2_eff":
            g = g[(g.t >= K_LO - 1e-9) & (g.t <= K_HI + 1e-9)]
        s = STY[key]
        a.errorbar(g.t + off[key], g.kvar, yerr=2 * g.mcse_kvar, color=s["c"], marker=s["m"] or "o",
                   ls="none", ms=2.6, elinewidth=0.6, mfc=s["c"] if key != "r2_or" else "white")
    if cur is not None:
        check_sec7_vs_cache(cur)
        m = (cur["t"] >= 0.08) & (cur["t"] <= 0.92)
        a.plot(cur["t"][m], cur["v1"][m], color=STY["r1"]["c"], lw=0.9)
        a.plot(cur["t"][m], cur["vid"][m], color=STY["r2_id"]["c"], lw=0.9)
        a.plot(cur["t"][m], cur["vopt"][m], color=STY["r2_eff"]["c"], lw=0.9)
    else:
        a.text(0.5, 0.5, "TODO: Section 7 asymptotic curves", transform=a.transAxes, ha="center",
               color="red")
    a.axvspan(0.08, K_LO, color="0.92", lw=0)
    a.axvspan(K_HI, 0.92, color="0.92", lw=0)
    a.set_xlim(0.08, 0.92)
    a.set_xlabel(r"$t$")
    a.set_ylabel(r"scaled variance")
    a.set_title(rf"(a) M1, $n={NLAB[n]}$, $b={b}$ ($k={n // b}$)", pad=3)
    h = [Line2D([], [], color=STY[k]["c"], marker=STY[k]["m"], ls="-", lw=0.9, ms=3)
         for k in ("r1", "r2_id", "r2_eff")]
    lab = [r"$r=1$", r"$r=2$, $W=I$", r"$r=2$, $\hat\Sigma_n^{-1}$"]
    leg1 = a.legend(h, lab, loc="upper left", ncol=1, fontsize=6.5, handletextpad=0.4,
                    borderaxespad=0.2, handlelength=1.8)
    a.add_artist(leg1)
    # second key: points = Monte Carlo, lines = asymptotic
    h2 = [Line2D([], [], color="0.35", marker="o", ls="none", ms=2.6),
          Line2D([], [], color="0.35", ls="-", lw=0.9)]
    lab2 = [r"Monte Carlo $k\,\widehat{\operatorname{Var}}\{\check A(t)\}$",
            r"asymptotic $\sigma_W^2(t)$"]
    a.legend(h2, lab2, loc="upper right", ncol=1, fontsize=6.5, handletextpad=0.4,
             borderaxespad=0.2, handlelength=1.8)
    a.set_ylim(0, 0.09)

    # (b) variance ratio r2_eff / r1 (paired) at n = 1e4 and 1e5, b = 40
    for nn, mk, mfc, dx in ((10000, "s", "#0072B2", -0.005), (100000, "o", "white", 0.005)):
        for key, col in (("r2_eff", STY["r2_eff"]["c"]),):
            g = pv[(pv.n == nn) & (pv.b == 40) & (pv.estimator == key)].sort_values("t")
            g = g[(g.t >= K_LO - 1e-9) & (g.t <= K_HI + 1e-9)]
            bx.errorbar(g.t + dx, g.var_ratio_r1, yerr=2 * g.mcse_var_ratio_r1, color=col, marker=mk,
                        mfc=mfc if mfc != "#0072B2" else col, ls="none", ms=2.8, elinewidth=0.6,
                        label=rf"Monte Carlo, $n={NLAB[nn]}$, $k={nn // 40}$")
    if cur is not None:
        m = (cur["t"] >= K_LO - 0.01) & (cur["t"] <= K_HI + 0.01)
        bx.plot(cur["t"][m], (cur["vopt"] / cur["v1"])[m], color="k", lw=0.9,
                label=r"asymptotic $\sigma^2_{\mathrm{opt}}(t)/\sigma^2_{1}(t)$")
    bx.axhline(1.0, color="0.6", lw=0.5, zorder=0)
    bx.set_xlim(K_LO - 0.02, K_HI + 0.02)
    bx.set_xlabel(r"$t$")
    bx.set_ylabel(r"variance ratio, $r=2$ ($\hat\Sigma_n^{-1}$) to $r=1$")
    bx.set_title(r"(b) M1, $b=40$, paired variance ratio", pad=3)
    bx.legend(loc="lower right", ncol=1, fontsize=6.5, handletextpad=0.3, borderaxespad=0.3)
    bx.set_ylim(0.4, 1.1)
    save(fig, "sec8-variance-vs-t.pdf")
    return cur


# ======================================================================= F4
def fig_bias(D, n=MAIN_N, cases=(("M1", 10), ("M1", 40), ("M2", 10), ("M2", 40)),
             name="sec8-pointwise-bias.pdf"):
    keys = ["r1", "r2_eff", "r2_eff_bceq", "r2_eff_bcos"]
    fig, axs = mkfig(1, len(cases), 2.3)
    axs = axs[0]
    pw = D.pw[(D.pw.exp == "main") & (D.pw.n == n)]
    for a, (model, b) in zip(axs, cases):
        for key in keys:
            g = pw[(pw.model == model) & (pw.b == b) & (pw.estimator == key)].sort_values("t")
            s = STY[key]
            a.plot(g.t, 1e3 * g.bias, color=s["c"], ls=s["ls"], lw=s["lw"], marker=s["m"], ms=2.2)
            a.fill_between(g.t, 1e3 * (g.bias - 2 * g.mcse_bias), 1e3 * (g.bias + 2 * g.mcse_bias),
                           color=s["c"], alpha=0.15, lw=0)
        a.axhline(0, color="0.5", lw=0.5, zorder=0)
        a.axvspan(0.1, K_LO, color="0.9", lw=0, zorder=-1)
        a.axvspan(K_HI, 0.9, color="0.9", lw=0, zorder=-1)
        a.set_xlim(0.1, 0.9)
        a.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        a.set_title(rf"{model}, $b={b}$ ($k={n // b}$)", pad=3)
        a.set_xlabel(r"$t$")
    axs[0].set_ylabel(r"$10^3\times\{\mathsf{E}\check A(t)-A_0(t)\}$")
    legend_below(fig, keys, ncol=2)
    save(fig, name)


# ======================================================================= T1
def tab_optimal(D, models=("M1", "M2", "M3", "M4"), ns=ALL_N, name="sec8-table-optimal.tex",
                label="tab:sim-optimal"):
    """MC-optimal b over the b grid of the results (main design b>=5: full + fineb) and the paired ratio
    min_b IMSE_K(est) / min_b IMSE_K(r1); in parentheses mcse_ratio_tab = max(delta-method MCSE,
    replication-bootstrap SE including the argmin selection).  Layout as
    bias_analysis/followup/outputs/h4_table_optimal_merged.tex."""
    o = D.opt[D.opt.exp == "main"]
    ests = ("r2_eff", "r2_eff_bceq", "r2_or", "pot")
    lines = []
    for model in models:
        first = True
        for n in ns:
            q = o[(o.model == model) & (o.n == n)].set_index("estimator")
            if q.empty:
                continue

            def bb(est):
                b = int(q.loc[est, "b_opt"])
                return f"{b}" + (r"$^{\ast}$" if bool(q.loc[est, "at_end"]) else "")

            r1 = q.loc["r1"]
            cells = [f"({model})" if first else "", f"${NLAB[n]}$",
                     bb("r1"), f"{1e4 * r1.imse_K:.3f} ({1e4 * r1.mcse_K:.3f})"]
            for est in ests:
                cells += [bb(est), f"{q.loc[est, 'ratio_to_r1_at_own_opt']:.2f} "
                                   f"({q.loc[est, 'mcse_ratio_tab']:.2f})"]
            lines.append(" & ".join(cells) + r"\\")
            first = False
        lines.append(r"\addlinespace[2pt]")
    body = "\n".join(lines[:-1])
    grids = "; ".join(f"{m} n={n}: {g}" for (m, n), g in
                      o[o.estimator == "r1"].set_index(["model", "n"]).grid.items() if m in models)
    tex = rf"""% generated by sec8_simulations/figures.py from {D.r.name}/optimal_b.csv -- do not edit by hand
% b grid of {D.r.name} (min b {int(D.imse[D.imse.exp == 'main'].b.min())}); ratio = min_b IMSE_K(est)/min_b IMSE_K(r1), paired;
% parentheses: max(delta-method MCSE, bootstrap SE incl. argmin selection, B=2000); *: end of grid
% b grids: {grids}
\begingroup\small\setlength{{\tabcolsep}}{{2.9pt}}
\begin{{tabular}}{{llrcrcrcrcrc}}
\toprule
& & \multicolumn{{2}}{{c}}{{$r=1$}} & \multicolumn{{2}}{{c}}{{$r=2$, $\hat\Sigma_n^{{-1}}$}}
  & \multicolumn{{2}}{{c}}{{$r=2$, $\hat\Sigma_n^{{-1}}$, corrected}} & \multicolumn{{2}}{{c}}{{$r=2$, $\Sigma^{{-1}}$}}
  & \multicolumn{{2}}{{c}}{{POT}}\\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}\cmidrule(lr){{7-8}}\cmidrule(lr){{9-10}}\cmidrule(lr){{11-12}}
Model & $n$ & $b^\dagger$ & $10^4\,\mathrm{{IMSE}}_K$ & $b^\dagger$ & ratio & $b^\dagger$ & ratio & $b^\dagger$ & ratio & $b^\dagger$ & ratio\\
\midrule
{body}
\bottomrule
\end{{tabular}}\endgroup
"""
    TABS.mkdir(exist_ok=True)
    (TABS / name).write_text(tex)
    print("wrote", TABS / name)
    return body


def crosscheck_h4(D, body, csv_path, tex_path):
    """The merged optimal-b table must agree with the follow-up analysis (h4_optimal_merged)."""
    try:
        h = pd.read_csv(csv_path)
        ref = Path(tex_path).read_text()
    except (OSError, FileNotFoundError):
        print("h4 cross-check skipped (files not found)")
        return
    o = D.opt[D.opt.exp == "main"].merge(h, on=["model", "n", "estimator"], suffixes=("", "_h4"))
    db = int((o.b_opt != o.b_opt_h4).sum())
    dr = float((o.ratio_to_r1_at_own_opt - o.ratio).abs().max())
    ds = float((o.mcse_ratio_tab - o.mcse).abs().max())
    di = float((o.imse_K - o.imse_K_h4).abs().max() / o.imse_K.min())
    ref_rows = [l for l in ref.splitlines() if l.startswith(("(M", " &"))]
    own_rows = [l for l in body.splitlines() if l.startswith(("(M", " &"))]
    same = [a.strip() == b.strip() for a, b in zip(own_rows, ref_rows)]
    print(f"h4 cross-check: {len(o)} cells; b_opt mismatches {db}; max|d ratio| {dr:.1e}; "
          f"max|d MCSE| {ds:.1e}; max rel |d IMSE| {di:.1e}; LaTeX rows identical "
          f"{sum(same)}/{len(ref_rows)} (own {len(own_rows)})")
    if db or dr > 1e-5 or ds > 1e-5 or not all(same) or len(own_rows) != len(ref_rows):
        raise SystemExit("merged optimal-b table disagrees with h4_optimal_merged")


# ======================================================================= supplement figures
def fig_weights(D):
    """S4 (F5): feasible/oracle IMSE ratio, Frobenius error of Sigma-hat, fallback rate vs k."""
    fig, axs = mkfig(1, 3, 2.45)
    axs = axs[0]
    mcol = {"M1": "#0072B2", "M2": "#D55E00", "M3": "#009E73", "M4": "#CC79A7", "M5": "#E69F00"}
    nmk = {1000: "v", 5000: "s", 10000: "o", 20000: "D", 100000: "^"}
    ror = D.ror[(D.ror.estimator == "r2_eff") & (D.ror.exp == "main")]
    dg = D.diag[D.diag.exp == "main"]
    for model in ("M1", "M2", "M3", "M4", "M5"):
        for n in ALL_N:
            g = ror[(ror.model == model) & (ror.n == n)].sort_values("k")
            if g.empty:
                continue
            axs[0].errorbar(g.k, g.ratio, yerr=2 * g.mcse_ratio, color=mcol[model], marker=nmk[n], ms=2.4,
                            lw=0.6, elinewidth=0.5, mfc="white" if n in (1000, 100000) else mcol[model])
            h = dg[(dg.model == model) & (dg.n == n)].sort_values("k")
            axs[1].plot(h.k, h.sig_frob_rel_or, color=mcol[model], marker=nmk[n], ms=2.4, lw=0.6,
                        mfc="white" if n in (1000, 100000) else mcol[model])
            axs[2].plot(h.k, 100 * h.fallback_K, color=mcol[model], marker=nmk[n], ms=2.4, lw=0.6,
                        mfc="white" if n in (1000, 100000) else mcol[model])
    for a in axs:
        a.set_xscale("log")
        a.set_xlabel(r"number of blocks $k$")
    axs[0].axhline(1, color="0.5", lw=0.5, zorder=0)
    axs[0].set_yscale("log")      # merged grid: up to about 6 at b=2 (M1, n=1e5)
    axs[0].yaxis.set_major_locator(FixedLocator([0.8, 1, 1.5, 2, 3, 5]))
    axs[0].yaxis.set_minor_locator(NullLocator())
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axs[0].set_ylabel(r"$\operatorname{IMSE}_K$: feasible / oracle")
    axs[0].set_title(r"(a) $\hat\Sigma_n^{-1}$ vs.\ $\Sigma^{-1}$", pad=3)
    axs[1].set_ylabel(r"mean $\|\hat\Sigma_n(t)-\Sigma(t)\|_F/\|\Sigma(t)\|_F$")
    axs[1].set_title(r"(b) covariance estimation error", pad=3)
    axs[2].axhline(5, color="0.5", lw=0.5, ls="--", zorder=0)
    axs[2].set_ylabel(r"fallback frequency on $K$ (\%)")
    axs[2].set_title(r"(c) identity fallback", pad=3)
    h = [Line2D([], [], color=mcol[m], lw=0.8) for m in mcol] + \
        [Line2D([], [], color="k", marker=nmk[n], ls="none", ms=3, mfc="white" if n in (1000, 100000) else "k")
         for n in ALL_N]
    lab = list(mcol) + [rf"$n={NLAB[n]}$" for n in ALL_N]
    fig.legend(h, lab, loc="outside lower center", ncol=5, handlelength=1.4, columnspacing=0.9)
    save(fig, "sec8-supp-weights.pdf")


def fig_r3(D):
    """S5 (F7): IMSE ratio to r=1 for r = 2, 3 at common b (M1, M2, n = 1e4)."""
    fig, axs = mkfig(1, 2, 2.3, sharey=True)
    axs = axs[0]
    for a, model in zip(axs, ("M1", "M2")):
        for key in ("r2_eff", "r2_id"):
            g = D.im(model, MAIN_N, key)
            errline(a, g.b, g.ratio_to_r1, g.mcse_ratio, key)
            clipped(a, g.b.to_numpy(), g.ratio_to_r1.to_numpy(), 0.62, 1.35, key)
        r3 = D.r3[(D.r3.model == model) & (D.r3.ref == "r1")]
        for key in ("r3_eff", "r3_id"):
            g = r3[r3.estimator == key].sort_values("b")
            errline(a, g.b, g.ratio, g.mcse_ratio, key)
            clipped(a, g.b.to_numpy(), g.ratio.to_numpy(), 0.62, 1.35, key)
        fb = D.diag[(D.diag.exp == "r3") & (D.diag.model == model)].sort_values("b")
        for b, f in zip(fb.b, fb.fallback_K):
            a.annotate(f"{100 * f:.0f}\\%", (b, 0.66), ha="center", fontsize=5.5, color=STY["r3_eff"]["c"])
        a.axhline(1, color="k", lw=0.5, zorder=0)
        logb_axis(a, sorted(fb.b))
        a.set_ylim(0.62, 1.35)
        a.set_xlabel(r"block length $b$")
        a.set_title(rf"{MODEL_TITLE[model]}, $n={NLAB[MAIN_N]}$", pad=3)
    axs[0].set_ylabel(r"$\operatorname{IMSE}_K$ ratio to $r=1$")
    legend_below(fig, ["r2_id", "r2_eff", "r3_id", "r3_eff"], ncol=4)
    save(fig, "sec8-supp-r3.pdf")


def indep_limits():
    """Limit-law scales at A_0 = 1, t = 1/2 (Prop. rank-independence): sqrt(k)(A-hat - 1) ->
    min{N(0, s^2), 0}.  Sigma(1/2) from the Section 8 oracle-cache code (sigma2_lib, theta=1)."""
    import sigma2_lib
    S = sigma2_lib.sigma2(sigma2_lib.Gumbel(1.0), 0.5)[None]
    v1, vid, vopt = asymptotic_variances(np.array([0.5]), S, np.array([1.0]))
    out = dict(r1=float(v1[0]), r2_id=float(vid[0]), r2_eff=float(vopt[0]), r2_or=float(vopt[0]))
    print(f"independence t=0.5: sigma^2 r1 {out['r1']:.6f}, r2 id {out['r2_id']:.6f}, r2 opt "
          f"{out['r2_eff']:.6f}, opt/r1 {out['r2_eff'] / out['r1']:.4f}")
    return out


def fig_indep(D, lim):
    keys = ["r1", "r2_id", "r2_eff"]
    bs = [20, 40, 80]
    fig, axs = mkfig(2, 3, 3.2, sharex=True, sharey=True)
    x = np.linspace(-1.2, 0.05, 400)
    for i, model in enumerate(("D6a", "D6b")):
        for j, b in enumerate(bs):
            a = axs[i, j]
            for key in keys:
                z = np.sort(D.indep[(D.indep.model == model) & (D.indep.b == b)
                                    & (D.indep.estimator == key)].z.to_numpy())
                s = STY[key]
                a.step(z, np.arange(1, len(z) + 1) / len(z), where="post", color=s["c"], lw=0.9)
                sd = np.sqrt(lim[key])
                a.plot(x, np.where(x < 0, norm.cdf(x / sd), 1.0), color=s["c"], lw=0.7, ls="--")
            a.set_xlim(-1.0, 0.05)
            a.set_ylim(0, 1.02)
            if i == 0:
                a.set_title(rf"$b={b}$ ($k={10000 // b}$)", pad=3)
            if i == 1:
                a.set_xlabel(r"$\sqrt{k}\{\check A(1/2)-1\}$")
        axs[i, 0].set_ylabel(MODEL_TITLE[model] + "\nECDF")
    h = [Line2D([], [], color=STY[k]["c"], lw=0.9) for k in keys] + \
        [Line2D([], [], color="k", lw=0.7, ls="--")]
    lab = [STY[k]["lab"] for k in keys] + [r"limit $\min\{N(0,\lambda_W^\top\Sigma\lambda_W),0\}$"]
    fig.legend(h, lab, loc="outside lower center", ncol=2)
    save(fig, "sec8-supp-independence.pdf")


# ======================================================================= supplement tables
def fmt_pct(x):
    v = 100 * x
    s = f"{v:.1f}" if v < 9.95 else f"{v:.0f}"
    return rf"\textbf{{{s}}}" if v > 5 else s


def tab_fallback(D):
    allb = sorted(set().union(*[std_grid(n) for n in ALL_N]))   # fine-grid b: diagnostics.csv
    dg = D.diag[D.diag.exp == "main"]
    rows = []
    for model in ("M1", "M2", "M3", "M4", "M5"):
        for i, n in enumerate(ALL_N):
            h = dg[(dg.model == model) & (dg.n == n)].set_index("b")
            if h.empty:
                continue
            cells = [model if i == 0 else "", f"${NLAB[n]}$"] + \
                    [fmt_pct(h.loc[b, "fallback_K"]) if b in h.index else "" for b in allb]
            rows.append(" & ".join(cells) + r"\\")
        rows.append(r"\addlinespace[2pt]")
    for model in ("M1", "M2"):
        h = D.diag[(D.diag.exp == "r3") & (D.diag.model == model)].set_index("b")
        cells = [rf"{model}, $r=3$", f"${NLAB[MAIN_N]}$"] + \
                [fmt_pct(h.loc[b, "fallback_K"]) if b in h.index else "" for b in allb]
        rows.append(" & ".join(cells) + r"\\")
    body = "\n".join(rows)
    tex = rf"""% generated by sec8_simulations/figures.py from {D.r.name}/diagnostics.csv -- do not edit by hand
% fallback frequency (%) of the simultaneous identity fallback on K=[{K_LO},{K_HI}]; bold: > 5%
% columns: b in {{2,3,4}} (mode smallb) and the doubling grid (mode full); the fine-grid values of
% mode fineb (M2-M4, n=1e4; M2, M3, n=1e5) are in diagnostics.csv (at most 0.1% there)
\begingroup\small\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{ll{'r' * len(allb)}}}
\toprule
& & \multicolumn{{{len(allb)}}}{{c}}{{block length $b$}}\\
\cmidrule(lr){{3-{2 + len(allb)}}}
Model & $n$ & {' & '.join(str(b) for b in allb)}\\
\midrule
{body}
\bottomrule
\end{{tabular}}\endgroup
"""
    (TABS / "sec8-table-fallback.tex").write_text(tex)
    print("wrote", TABS / "sec8-table-fallback.tex")


def tab_diagnostics(D, n=MAIN_N):
    dg = D.diag[(D.diag.exp == "main") & (D.diag.n == n) & D.diag.b.isin(std_grid(n))]
    im = D.imse[(D.imse.exp == "main") & (D.imse.n == n)]
    nK = int(((C.T_EVAL >= K_LO - 1e-9) & (C.T_EVAL <= K_HI + 1e-9)).sum())
    rows = []
    for model in ("M1", "M2", "M3", "M4", "M5"):
        h = dg[dg.model == model].sort_values("b")
        for i, (_, r) in enumerate(h.iterrows()):
            bh = im[(im.model == model) & (im.b == r.b) & (im.estimator == "r2_eff")].bound_hit_rate_K.iloc[0]
            cells = [model if i == 0 else "", f"{int(r.b)}", f"{int(r.k)}", fmt_pct(r.fallback_K),
                     f"{100 * bh:.1f}", f"{100 * r.os_trunc_eff_K / nK:.1f}", f"{1e4 * r.pilot_ise_K:.3f}",
                     f"{1e3 * r.bernstein_shift:.2f}", f"{r.sig_frob_rel_or:.3f}",
                     f"{1e3 * r.eqos_median:.2f}", f"{1e3 * r.eqos_q95:.2f}", f"{r.sec_per_rep:.2f}"]
            rows.append(" & ".join(cells) + r"\\")
        rows.append(r"\addlinespace[2pt]")
    body = "\n".join(rows[:-1])
    tex = rf"""% generated by sec8_simulations/figures.py from {D.r.name}/diagnostics.csv, imse.csv -- do not edit by hand
% n = {n}, b in {{2,3,4}} and the doubling grid (fine-grid b: diagnostics.csv); fallback, boundary and truncation frequencies in %; boundary = r2_eff at an end point of
% Theta_t (t in K); truncation = one-step value outside Theta_t (t in K); pilot ISE on K x 1e4;
% Bernstein shift = max_t (Bernstein - sieve) x 1e3; Frobenius = mean_t ||Sigma-hat - Sigma||_F/||Sigma||_F;
% eq-os = sqrt(k) max_K |eq - one-step| x 1e3 (median, 95% quantile); s/rep = seconds per replication and b
\begingroup\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lrrrrrrrrrrr}}
\toprule
Model & $b$ & $k$ & fallb. & bound. & trunc. & pilot ISE & shift & Frob. & \multicolumn{{2}}{{c}}{{eq$-$os}} & s/rep\\
\cmidrule(lr){{10-11}}
 & & & (\%) & (\%) & (\%) & $\times10^4$ & $\times10^3$ & & med. & q95 & \\
\midrule
{body}
\bottomrule
\end{{tabular}}\endgroup
"""
    (TABS / "sec8-table-diagnostics.tex").write_text(tex)
    print("wrote", TABS / "sec8-table-diagnostics.tex")


def tab_decomp(D, n=MAIN_N):
    keys = ["r1", "r2_eff", "r2_eff_bceq", "pot"]
    rows = []
    for model in ("M1", "M2", "M3", "M4"):
        for i, b in enumerate([b for b in sorted(D.im(model, n, "r1").b) if b in std_grid(n)]):
            cells = [model if i == 0 else "", str(b)]
            for key in keys:
                g = D.dec[(D.dec.model == model) & (D.dec.n == n) & (D.dec.b == b)
                          & (D.dec.estimator == key)].iloc[0]
                cells += [f"{1e4 * g.ibias2:.3f}", f"{1e4 * g.ivar:.3f}"]
            rows.append(" & ".join(cells) + r"\\")
        rows.append(r"\addlinespace[2pt]")
    body = "\n".join(rows[:-1])
    tex = rf"""% generated by sec8_simulations/figures.py from {D.r.name}/extra_decomp.csv -- do not edit by hand
% n = {n}, b in {{2,3,4}} and the doubling grid (fine-grid b: extra_decomp.csv); IBias^2_K and IVar_K x 1e4; MCSEs in extra_decomp.csv
\begingroup\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lr*{{8}}{{r}}}}
\toprule
& & \multicolumn{{2}}{{c}}{{$r=1$}} & \multicolumn{{2}}{{c}}{{$r=2$, $\hat\Sigma_n^{{-1}}$}}
& \multicolumn{{2}}{{c}}{{$r=2$, corrected}} & \multicolumn{{2}}{{c}}{{POT}}\\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}\cmidrule(lr){{7-8}}\cmidrule(lr){{9-10}}
Model & $b$ & IBias$^2$ & IVar & IBias$^2$ & IVar & IBias$^2$ & IVar & IBias$^2$ & IVar\\
\midrule
{body}
\bottomrule
\end{{tabular}}\endgroup
"""
    (TABS / "sec8-table-decomposition.tex").write_text(tex)
    print("wrote", TABS / "sec8-table-decomposition.tex")


def tab_sens(D):
    s = D.sens
    rows = []
    labels = [("sens_m20", r"$m=20$, $B_n=200$"), ("main_m40_B200", r"$m=40$, $B_n=200$$^{\ast}$"),
              ("sens_m80", r"$m=80$, $B_n=200$"), ("sens_B100", r"$m=40$, $B_n=100$"),
              ("sens_B500", r"$m=40$, $B_n=500$")]
    dg = D.diag
    for model in ("M1", "M2"):
        for i, (exp, lab) in enumerate(labels):
            q = s[(s.exp == exp) & (s.model == model) & (s.estimator == "r2_eff")].set_index("ref")
            qb = s[(s.exp == exp) & (s.model == model) & (s.estimator == "r2_eff_bceq")].set_index("ref")
            if exp == "main_m40_B200":
                h = dg[(dg.exp == "main") & (dg.model == model) & (dg.n == MAIN_N) & (dg.b == 40)].iloc[0]
            else:
                h = dg[(dg.exp == exp) & (dg.model == model)].iloc[0]
            cells = [model if i == 0 else "", lab,
                     f"{q.loc['r1', 'ratio']:.3f} ({q.loc['r1', 'mcse_ratio']:.3f})",
                     f"{qb.loc['r1', 'ratio']:.3f} ({qb.loc['r1', 'mcse_ratio']:.3f})",
                     f"{q.loc['r2_or', 'ratio']:.3f} ({q.loc['r2_or', 'mcse_ratio']:.3f})",
                     f"{h.sig_frob_rel_or:.3f}", fmt_pct(h.fallback_K), fmt_pct(h.fallback_K10)]
            rows.append(" & ".join(cells) + r"\\")
        rows.append(r"\addlinespace[2pt]")
    body = "\n".join(rows[:-1])
    tex = rf"""% generated by sec8_simulations/figures.py from {D.r.name}/extra_sens.csv, diagnostics.csv -- do not edit by hand
% n = 1e4, b = 40 (k = 250); IMSE ratios: replications 0..499 of every run (paired, same data);
% main run = m 40, B 200; Frobenius error and fallback (%) of the main row over all 1000 replications;
% fallback on K=[0.15,0.85] and, for comparison, on [0.10,0.90]; bold: > 5%
\begingroup\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{llccccrr}}
\toprule
& & \multicolumn{{2}}{{c}}{{IMSE$_K$ ratio to $r=1$}} & feasible/oracle & & \multicolumn{{2}}{{c}}{{fallback (\%)}}\\
\cmidrule(lr){{3-4}}\cmidrule(lr){{7-8}}
Model & tuning & $\hat\Sigma_n^{{-1}}$ & corrected & IMSE$_K$ ratio & Frob. & $K$ & $[0.1,0.9]$\\
\midrule
{body}
\bottomrule
\end{{tabular}}\endgroup
"""
    (TABS / "sec8-table-sensitivity.tex").write_text(tex)
    print("wrote", TABS / "sec8-table-sensitivity.tex")


def tab_indep(D, lim):
    rows = []
    for model in ("D6a", "D6b"):
        for i, b in enumerate((20, 40, 80)):
            cells = [MODEL_TITLE[model] if i == 0 else "", f"{b}"]
            for key in ("r1", "r2_id", "r2_eff"):
                z = D.indep[(D.indep.model == model) & (D.indep.b == b) & (D.indep.estimator == key)].z
                cells += [f"{z.mean():.3f}", f"{(z ** 2).mean():.4f}", f"{(z >= -1e-12).mean():.2f}"]
            rows.append(" & ".join(cells) + r"\\")
        rows.append(r"\addlinespace[2pt]")
    cells = ["limit", ""]
    for key in ("r1", "r2_id", "r2_eff"):
        s2 = lim[key]
        cells += [f"{-np.sqrt(s2 / (2 * np.pi)):.3f}", f"{s2 / 2:.4f}", "0.50"]
    rows.append(" & ".join(cells) + r"\\")
    body = "\n".join(rows)
    R = int(D.indep.groupby(["model", "b", "estimator"]).size().min())
    tex = rf"""% generated by sec8_simulations/figures.py from {D.r.name}/extra_indep.csv -- do not edit by hand
% Z = sqrt(k){{A-hat(1/2)-1}}; columns: mean, second moment, P(Z=0); R = {R}; n = 1e4
% limit: min{{N(0,s^2),0}} with s^2 = lambda_W' Sigma lambda_W at A_0=1, t=1/2 (sigma2_lib, theta=1):
% mean -s/sqrt(2 pi), second moment s^2/2, atom 1/2
\begingroup\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lr*{{9}}{{r}}}}
\toprule
& & \multicolumn{{3}}{{c}}{{$r=1$}} & \multicolumn{{3}}{{c}}{{$r=2$, $W=I$}} & \multicolumn{{3}}{{c}}{{$r=2$, $\hat\Sigma_n^{{-1}}$}}\\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}\cmidrule(lr){{9-11}}
Model & $b$ & mean & 2nd m. & $\mathsf P(0)$ & mean & 2nd m. & $\mathsf P(0)$ & mean & 2nd m. & $\mathsf P(0)$\\
\midrule
{body}
\bottomrule
\end{{tabular}}\endgroup
"""
    (TABS / "sec8-table-independence.tex").write_text(tex)
    print("wrote", TABS / "sec8-table-independence.tex")


# ======================================================================= H2, H3 (follow-up runs)
def fig_bck(D, n=MAIN_N, name="sec8-supp-bck.pdf"):
    """H2: bootstrap 1/k correction of the feasible r=2 estimator (r2_eff -> r2_eff_bck), M1-M4:
    (a) IBias^2_K uncorrected (open, dotted) and corrected (filled, solid); (b) IVar_K ratio
    corrected/uncorrected; (c) IMSE_K ratio corrected/uncorrected (paired, +-2 MCSE)."""
    x = D.bck[(D.bck.n == n) & (D.bck.unc == "r2_eff")]
    fig, axs = mkfig(1, 3, 2.45, wspace=0.05)
    a, bv, c = axs[0]
    dx = {"M1": 0.955, "M2": 0.985, "M3": 1.015, "M4": 1.045}
    for model in ("M1", "M2", "M3", "M4"):
        g = x[x.model == model].sort_values("b")
        col = MCOL[model]
        xb = g.b * dx[model]
        a.errorbar(xb, 1e4 * g.ibias2_unc, yerr=2e4 * g.mcse_ibias2_unc, color=col, marker="o",
                   mfc="white", ls=":", lw=0.8, ms=2.6, elinewidth=0.5)
        a.errorbar(xb, 1e4 * g.ibias2_bck, yerr=2e4 * g.mcse_ibias2_bck, color=col, marker="o",
                   ls="-", lw=0.9, ms=2.6, elinewidth=0.5)
        bv.errorbar(xb, g.ivar_ratio, yerr=2 * g.mcse_ivar_ratio, color=col, marker="o", ls="-",
                    lw=0.9, ms=2.6, elinewidth=0.5)
        c.errorbar(xb, g.imse_ratio, yerr=2 * g.mcse_imse_ratio, color=col, marker="o", ls="-",
                   lw=0.9, ms=2.6, elinewidth=0.5)
    bs = sorted(x.b.unique())
    a.set_yscale("log")
    a.set_ylim(1e-4, None)
    logy_ticks(a)
    a.set_ylabel(r"$10^4\times\operatorname{IBias}^2_K$")
    a.set_title(r"(a) integrated squared bias", pad=3)
    bv.axhline(1, color="0.5", lw=0.5, zorder=0)
    bv.set_ylabel(r"$\operatorname{IVar}_K$: corrected / uncorrected")
    bv.set_title(r"(b) integrated variance", pad=3)
    c.axhline(1, color="0.5", lw=0.5, zorder=0)
    c.set_ylabel(r"$\operatorname{IMSE}_K$: corrected / uncorrected")
    c.set_title(r"(c) integrated MSE", pad=3)
    for ax in axs[0]:
        logb_axis(ax, bs)
        ax.set_xlabel(r"block length $b$")
    h = [Line2D([], [], color=MCOL[m], lw=0.9, marker="o", ms=2.6) for m in ("M1", "M2", "M3", "M4")] + \
        [Line2D([], [], color="k", ls=":", lw=0.8, marker="o", mfc="white", ms=2.6),
         Line2D([], [], color="k", ls="-", lw=0.9, marker="o", ms=2.6)]
    lab = ["M1", "M2", "M3", "M4", r"$r=2$, $\hat\Sigma_n^{-1}$, uncorrected",
           r"bootstrap $1/k$ corrected"]
    fig.legend(h, lab, loc="outside lower center", ncol=6, handlelength=1.8, columnspacing=1.0)
    save(fig, name)


def fig_r3or(D, n=MAIN_N, name="sec8-supp-r3-oracle.pdf"):
    """H3: r=3 with oracle Sigma_3^{-1} weights vs the feasible r=3 estimator and r=2, M1.
    (a) IMSE_K ratio to r=1 at common b; (b) r=3 / r=2 (oracle and feasible) and feasible /
    oracle (r=2 and r=3); numbers: fallback frequency of the feasible r=3 estimator."""
    fig, axs = mkfig(1, 2, 2.45, wspace=0.05)
    a, bx = axs[0]
    r3 = D.r3[D.r3.model == "M1"]
    bs3 = sorted(r3.b.unique())
    for key in ("r2_eff", "r2_or"):
        g = D.im("M1", n, key)
        g = g[g.b.between(min(bs3), max(bs3))]
        errline(a, g.b, g.ratio_to_r1, g.mcse_ratio, key)
    for key in ("r3_eff", "r3_or"):
        g = r3[(r3.estimator == key) & (r3.ref == "r1")].sort_values("b")
        errline(a, g.b, g.ratio, g.mcse_ratio, key, **({} if key == "r3_eff" else dict(mfc="white")))
    a.axhline(1, color="k", lw=0.5, zorder=0)
    a.set_ylabel(r"$\operatorname{IMSE}_K$ ratio to $r=1$")
    a.set_title(r"(a) M1, $n=10^4$, paired ratio to $r=1$", pad=3)
    lines = [(("r3_or", "r2_or"), "#E69F00", ":", "P", "white", r"$r=3$ / $r=2$, oracle"),
             (("r3_eff", "r2_eff"), "#E69F00", "-", "P", "#E69F00", r"$r=3$ / $r=2$, feasible"),
             (("r3_eff", "r3_or"), "#000000", "-", "s", "#000000", r"feasible / oracle, $r=3$")]
    for (num, den), col, ls, mk, mfc, lab in lines:
        g = r3[(r3.estimator == num) & (r3.ref == den)].sort_values("b")
        bx.errorbar(g.b, g.ratio, yerr=2 * g.mcse_ratio, color=col, ls=ls, marker=mk, mfc=mfc, ms=2.8,
                    lw=1.0, elinewidth=0.6, label=lab)
    g = D.ror[(D.ror.model == "M1") & (D.ror.n == n) & (D.ror.estimator == "r2_eff")
              & D.ror.b.between(min(bs3), max(bs3))].sort_values("b")
    bx.errorbar(g.b, g.ratio, yerr=2 * g.mcse_ratio, color="0.55", ls="-", marker="s", ms=2.6, lw=0.9,
                elinewidth=0.6, label=r"feasible / oracle, $r=2$")
    fb = D.diag[(D.diag.exp == "r3") & (D.diag.model == "M1")].sort_values("b")
    ylo = 0.72
    for b, f in zip(fb.b, fb.fallback_K):
        bx.annotate(f"{100 * f:.0f}\\%", (b, ylo + 0.02), ha="center", fontsize=5.5, color="#E69F00")
    bx.axhline(1, color="k", lw=0.5, zorder=0)
    top = max(float(np.nanmax(np.asarray(ln.get_ydata(), float)))
              for ln in bx.get_lines() if len(ln.get_ydata()))
    bx.set_ylim(ylo, top + 0.32)      # room for the legend
    bx.set_title(r"(b) M1, $n=10^4$, paired $\operatorname{IMSE}_K$ ratios", pad=3)
    bx.legend(loc="upper right", fontsize=6.3, ncol=2, handlelength=1.8, columnspacing=0.8,
              borderaxespad=0.2)
    for ax in (a, bx):
        logb_axis(ax, bs3)
        ax.set_xlabel(r"block length $b$")
    legend_below(fig, ["r2_eff", "r2_or", "r3_eff", "r3_or"], ncol=4)
    save(fig, name)


def tab_bck(D, name="sec8-table-bck.tex"):
    """H2: paired IMSE_K ratios at each estimator's own MC-optimal b on the b grid of mode bck."""
    q = D.bopt
    rows = []
    for model in ("M1", "M2", "M3", "M4"):
        first = True
        for n in sorted(q.n.unique()):
            x = q[(q.model == model) & (q.n == n)].set_index(["num", "den"])
            if x.empty:
                continue

            def cell(num, den):
                r = x.loc[(num, den)]
                return f"{r.ratio:.3f} ({r.mcse_ratio:.3f})"

            cells = [model if first else "", f"${NLAB[n]}$",
                     cell("r1_bck", "r1"), cell("r2_eff_bck", "r2_eff"), cell("r2_eff_bceq_bck", "r2_eff_bceq"),
                     cell("r2_or_bck", "r2_or"), cell("r2_eff", "r1"), cell("r2_eff_bck", "r1_bck")]
            rows.append(" & ".join(cells) + r"\\")
            first = False
        rows.append(r"\addlinespace[2pt]")
    body = "\n".join(rows[:-1])
    tex = rf"""% generated by sec8_simulations/figures.py from {D.r.name}/extra_bck_opt.csv -- do not edit by hand
% bootstrap 1/k correction (mode bck, labels *_bck); every estimator at its own MC-optimal b on the
% b grid of mode bck (B_GRID[n], no b<5 or fine-grid values); ratios paired over R=1000, delta-method MCSE
\begingroup\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{llcccccc}}
\toprule
& & \multicolumn{{4}}{{c}}{{$\operatorname{{IMSE}}_K$: corrected / uncorrected}} & \multicolumn{{2}}{{c}}{{$r=2$, $\hat\Sigma_n^{{-1}}$ / $r=1$}}\\
\cmidrule(lr){{3-6}}\cmidrule(lr){{7-8}}
Model & $n$ & $r=1$ & $r=2$, $\hat\Sigma_n^{{-1}}$ & corrected (eq.) & $r=2$, $\Sigma^{{-1}}$ & uncorrected & both $1/k$-corr.\\
\midrule
{body}
\bottomrule
\end{{tabular}}\endgroup
"""
    (TABS / name).write_text(tex)
    print("wrote", TABS / name)


def tab_opc(D, s07_path, n=MAIN_N, name="sec8-table-opc.tex"):
    """OPC/Joe table of bias_analysis/investigation (sec8-bias:tab:opc): theory rows
    from its s07_ratios.csv (read only), Monte Carlo rows from this study, including the
    'optimized' cells (MC-optimal b over the merged grid; ratio with max(delta, selection) SE)."""
    try:
        th = pd.read_csv(s07_path)
    except (OSError, FileNotFoundError):
        print("OPC table skipped (s07_ratios.csv not found)")
        return
    crit = [("L", "Leading criterion"), ("E", "Exact block bias"), ("C", r"Exact block bias $+\,\eta_W/k$")]
    rows = []
    for c, lab in crit:
        cells = [lab]
        for model in ("M3", "M2"):
            for bset in ("b=10", "all b"):
                v = th[(th.model == model) & (th.n == n) & (th.bset == bset) & (th.criterion == c)].ratio
                cells.append(f"{float(v.iloc[0]):.2f}")
        rows.append(" & ".join(cells) + r"\\")
    o = D.opt[D.opt.exp == "main"]
    bopt = {}
    for key, lab in (("r2_or", "Monte Carlo, oracle weights"), ("r2_eff", "Monte Carlo, feasible weights")):
        cells = [lab]
        for model in ("M3", "M2"):
            g = D.im(model, n, key)
            g = g[g.b == 10].iloc[0]
            q = o[(o.model == model) & (o.n == n) & (o.estimator == key)].iloc[0]
            r1 = o[(o.model == model) & (o.n == n) & (o.estimator == "r1")].iloc[0]
            bopt[(model, key)] = (int(q.b_opt), int(r1.b_opt))
            cells += [f"{g.ratio_to_r1:.2f} ({g.mcse_ratio:.2f})",
                      f"{q.ratio_to_r1_at_own_opt:.2f} ({q.mcse_ratio_tab:.2f})"]
        rows.append(" & ".join(cells) + r"\\")
    body = "\n".join(rows[:3]) + "\n\\midrule\n" + "\n".join(rows[3:])
    bo = "; ".join(f"{m} {k}: b={v[0]} (r1: b={v[1]})" for (m, k), v in bopt.items())
    tex = rf"""% generated by sec8_simulations/figures.py -- do not edit by hand
% theory rows: bias_analysis/investigation/outputs/s07_ratios.csv (n={n}; bset b=10 / all b)
% Monte Carlo rows: {D.r.name}/imse.csv (b=10) and optimal_b.csv (optimized: MC-optimal b of each
% estimator over the b grid of {D.r.name}); parentheses: MCSE (optimized: max of delta-method
% and selection-bootstrap SE).  MC-optimal b: {bo}
\begingroup\small
\begin{{tabular}}{{lcccc}}
\toprule
& \multicolumn{{2}}{{c}}{{outer-power Clayton (M3)}} & \multicolumn{{2}}{{c}}{{Joe (M2)}}\\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
& $b=10$ & optimized & $b=10$ & optimized\\
\midrule
{body}
\bottomrule
\end{{tabular}}\endgroup
"""
    (TABS / name).write_text(tex)
    print("wrote", TABS / name)



def crosscheck_b5(D, Dall):
    """b>=5 design: the MC-optimal b and IMSE of every main cell must equal the minimum of the
    merged imse.csv (all modes) restricted to b>=5 (independent recomputation from another file)."""
    x = Dall.imse[(Dall.imse.exp == "main") & (Dall.imse.b >= 5)]
    best = x.loc[x.groupby(["model", "n", "estimator"])["imse_K"].idxmin()]
    o = D.opt[D.opt.exp == "main"].merge(best[["model", "n", "estimator", "b", "imse_K"]],
                                         on=["model", "n", "estimator"], suffixes=("", "_all"))
    db = int((o.b_opt != o.b).sum())
    di = float((o.imse_K / o.imse_K_all - 1).abs().max())
    print(f"b>=5 cross-check vs merged imse.csv: {len(o)} cells; b_opt mismatches {db}; "
          f"max rel |d IMSE| {di:.1e}; min b in design {int(D.imse[D.imse.exp == 'main'].b.min())}")
    if db or di > 1e-5 or D.imse[D.imse.exp == "main"].b.min() < 5:
        raise SystemExit("b>=5 optimal-b table inconsistent with the merged summaries")


def tab_smallb(D, Dall, n=MAIN_N, name="sec8-table-smallb.tex"):
    """Remark table: MC-optimal b and min IMSE_K over b>=5 (main design) vs over b>=2 (with mode
    smallb), n = 1e4, M1-M5, for r=1, feasible r=2 and corrected r=2; ratios to r=1 at the
    respective optimum (paired; MCSE = max(delta, selection bootstrap))."""
    rows = []
    for model in ("M1", "M2", "M3", "M4", "M5"):
        for i, (lab, X) in enumerate(((r"$b\ge5$", D), (r"$b\ge2$", Dall))):
            q = X.opt[(X.opt.exp == "main") & (X.opt.model == model) & (X.opt.n == n)].set_index("estimator")
            cells = [f"({model})" if i == 0 else "", lab]
            for est in ("r1", "r2_eff", "r2_eff_bceq"):
                r = q.loc[est]
                cells += [f"{int(r.b_opt)}", f"{1e4 * r.imse_K:.3f} ({1e4 * r.mcse_K:.3f})"]
                if est != "r1":
                    cells.append(f"{r.ratio_to_r1_at_own_opt:.2f} ({r.mcse_ratio_tab:.2f})")
            rows.append(" & ".join(cells) + r"\\")
        rows.append(r"\addlinespace[2pt]")
    body = "\n".join(rows[:-1])
    tex = rf"""% generated by sec8_simulations/figures.py from {D.r.name}/optimal_b.csv (b>=5: modes full+fineb+bck+r3or)
% and {Dall.r.name}/optimal_b.csv (b>=2: plus mode smallb) -- do not edit by hand
% n = {n}; b^dagger = MC-optimal b; IMSE_K x 1e4 (MCSE); ratio = min IMSE_K(est) / min IMSE_K(r1) over the
% same b set, paired over R=1000 (max of delta-method and selection-bootstrap SE)
\begingroup\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{llrcrccrcc}}
\toprule
& & \multicolumn{{2}}{{c}}{{$r=1$}} & \multicolumn{{3}}{{c}}{{$r=2$, $\hat\Sigma_n^{{-1}}$}} & \multicolumn{{3}}{{c}}{{$r=2$, $\hat\Sigma_n^{{-1}}$, corrected}}\\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}
Model & $b$ set & $b^\dagger$ & $10^4\,\mathrm{{IMSE}}_K$ & $b^\dagger$ & $10^4\,\mathrm{{IMSE}}_K$ & ratio & $b^\dagger$ & $10^4\,\mathrm{{IMSE}}_K$ & ratio\\
\midrule
{body}
\bottomrule
\end{{tabular}}\endgroup
"""
    (TABS / name).write_text(tex)
    print("wrote", TABS / name)


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(P / "data" / "sec8" / "main_b5"),
                    help="main design (b>=5; modes full+fineb+bck+r3or)")
    ap.add_argument("--results-all", default=str(P / "data" / "sec8" / "all_b2"),
                    help="all modes incl. smallb (b>=2), for the small-b remark only")
    ap.add_argument("--sec7-grid", default=str(SEC7 / "sigma_grid_theta2.npz"))
    ap.add_argument("--h4-dir", default=str(FU / "outputs"),
                    help="outputs of bias_analysis/followup (cross-check of T1)")
    ap.add_argument("--s07", default=str(BIAS / "outputs" / "s07_ratios.csv"),
                    help="theory rows of the OPC/Joe table (bias investigation, s07)")
    args = ap.parse_args()
    D = Data(args.results)
    SMALLB_IN[0] = bool((D.imse.b < 5).any())
    Dall = Data(args.results_all) if Path(args.results_all).exists() else None
    FIGS.mkdir(exist_ok=True)
    TABS.mkdir(exist_ok=True)
    m4 = ["M1", "M2", "M3", "M4"]
    top = ["r1", "r2_eff", "r2_eff_bceq", "pot"]
    rat = ["r2_id", "r2_eff", "r2_eff_bceq", "r2_or", "pot"]
    # main
    fig_imse(D, MAIN_N, m4, "sec8-imse-vs-b.pdf", top, rat)
    fig_decomp(D, MAIN_N, m4, "sec8-decomposition.pdf")
    fig_variance(D, args.sec7_grid)
    fig_bias(D)
    body = tab_optimal(D)
    if SMALLB_IN[0]:     # results incl. b in {2,3,4}: must equal the follow-up analysis (h4)
        crosscheck_h4(D, body, Path(args.h4_dir) / "h4_optimal_merged.csv",
                      Path(args.h4_dir) / "h4_table_optimal_merged.tex")
    elif Dall is not None:
        crosscheck_b5(D, Dall)
    tab_opc(D, args.s07)
    # supplement
    for n in ALL_N:
        if n != MAIN_N:
            fig_imse(D, n, m4, f"sec8-supp-imse-vs-b-n{n}.pdf", top, rat)
            fig_decomp(D, n, m4, f"sec8-supp-decomposition-n{n}.pdf")
    fig_imse(D, None, ["M5"], "sec8-supp-M5-imse-vs-b.pdf", top, rat, ylim_ratio=(0.5, 1.7),
             cols_are_n=[1000, 5000, 10000, 20000])
    fig_decomp(D, None, ["M5"], "sec8-supp-M5-decomposition.pdf", cols_are_n=[1000, 5000, 10000, 20000])
    fig_bias(D, cases=(("M3", 10), ("M3", 40), ("M4", 10), ("M4", 40)), name="sec8-supp-pointwise-bias-M3M4.pdf")
    fig_bias(D, cases=(("M5", 5), ("M5", 10), ("M5", 40), ("M5", 80)), name="sec8-supp-pointwise-bias-M5.pdf")
    fig_weights(D)
    fig_r3(D)
    lim = indep_limits()
    fig_indep(D, lim)
    tab_fallback(D)
    tab_diagnostics(D)
    tab_decomp(D)
    tab_sens(D)
    tab_indep(D, lim)
    tab_optimal(D, models=("M5",), ns=[1000, 5000, 10000, 20000], name="sec8-table-optimal-M5.tex")
    if D.bck is not None:
        fig_bck(D)
        tab_bck(D)
    if "r3_or" in set(D.r3.estimator):
        fig_r3or(D)
    if Dall is not None and not SMALLB_IN[0]:    # small-b remark (b in {2,3,4} outside the design)
        tab_smallb(D, Dall)
        fig_imse(Dall, MAIN_N, m4, "sec8-supp-smallb-imse-vs-b.pdf", top, rat, shade_below=5)


if __name__ == "__main__":
    main()
