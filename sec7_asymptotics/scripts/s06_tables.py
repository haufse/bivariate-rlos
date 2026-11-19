"""s06: optimized leading-order integrated error (AMEISE) and optimal block sizes; LaTeX tables.

Run:  python s06_tables.py > ../outputs/s06_output.txt
Needs ../outputs/constants_theta2.json (s05).  No randomness.
Writes, for delta in TABLE_DELTAS = (0.15, 0.01) (<dir> = ../../tables for delta = 0.15, the version
used in the paper; ../outputs for the alternative delta = 0.01):
    <dir>/sec7-table-main-d<delta>.tex   (tab:comparison-ameise, manuscript format; d015 = K of Section 8)
    <dir>/sec7-table-bstar-d<delta>.tex  (optimal block sizes, for app:comparison-details)
    <dir>/sec7-table-supp-d<delta>.tex   (identity weights and r=3, for app:comparison-details)
and ../outputs/ameise_summary.json.  Importable (s09 uses crit_fn); all work is under __main__.
2026-11-18: main guard added (s09 import used to rerun the whole script); delta 0.15 added; manuscript
labels and footnotes; Archimax/GH corrected cells as in the manuscript (no interior minimizer); Archimax
omitted-term check and n at which b* reaches r; R3_IN_MAIN switch for the r=3 verdict.
2026-11-18 (revision): the Archimax omitted-term and full-criterion crossover outputs are
labelled WITHDRAWN (terms below the controlled remainder); main-table footnote b reworded.
Repository version: footnote markers a, b, c replaced by dagger, double dagger and section sign
(\dag, \ddag, \S, set with \rlap after the entry; one note per line), as in the paper.  The footnote
and caption wording is otherwise that of the generator; the paper's final wording was edited by hand.

Criterion (eq. ameise-trimmed): AMEISE_K(b) = (b/n) V + alpha(b)^2 B_SS + 2 alpha(b) b^{-1} B_SR + b^{-2} B_RR,
k = n/b.  For a single pure-power squared bias B b^{-2g}:
    b* = (2g B n / V)^{1/(1+2g)},   AMEISE* = (1+2g) (2g)^{-2g/(1+2g)} B^{1/(1+2g)} V^{2g/(1+2g)} n^{-2g/(1+2g)}.
POT (BVZ Prop. 4.1(i)) with b := n/k_p: AMEISE_p(b) = (b/n) V_p + alpha_p(b)^2 B_p.
"""
import sys
import json
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, '.')

C = json.load(open('../outputs/constants_theta2.json'))
DELTAS = C['deltas']
TABLE_DELTAS = (0.15, 0.01)
# footnote markers in the cells and at the start of the notes (paper style)
MARK = {'a': '\\rlap{\\,\\textsuperscript{\\dag}}', 'b': '\\rlap{\\,\\textsuperscript{\\ddag}}',
        'c': '\\rlap{\\,\\textsuperscript{\\S}}'}


def tabdir(delta):
    return '../../tables' if abs(delta - 0.15) < 1e-12 else '../outputs'
R3_VERIFIED = None  # set by command-line flag --r3-verified / --r3-unverified
NS = [1e3, 1e4, 1e5, 1e6]


def opt_power(V, B, g):
    bconst = (2 * g * B / V) ** (1 / (1 + 2 * g))
    aconst = (1 + 2 * g) * (2 * g) ** (-2 * g / (1 + 2 * g)) * B ** (1 / (1 + 2 * g)) * V ** (2 * g / (1 + 2 * g))
    return dict(rate=-2 * g / (1 + 2 * g), const=aconst, bexp=1 / (1 + 2 * g), bconst=bconst)


# alpha(b) = ka * b^rho ; POT alpha_p(b) = kp * b^rhop
ALPHA = {'GH': (0.0, -1.0), 'Joe': (0.5, -1.0), 'OPC': (0.5, -1.0), 'Archimax': (1.0, -2.0)}
ALPHAP = {'GH': (0.5, -1.0), 'Joe': (1.0, -2.0), 'OPC': (1.0, -1.0), 'Archimax': (0.5, -1.0)}


def crit_fn(model, est, K, corrected=False):
    """Return f(b, n) = full 4-term criterion (leading-order theory, no terms dropped)."""
    e = K[est]
    V = e['V']
    if est == 'pot':
        kp, rp = ALPHAP[model]
        Bp = e['BSS_' + model]
        return lambda b, n: b / n * V + (kp * b ** rp) ** 2 * Bp
    ka, rho = ALPHA[model]
    BSS = e['BSS_' + model]
    BSR = e['BSR_' + model]
    BRR = 0.0 if corrected else e['BRR']
    if corrected:
        BSR = 0.0
    return lambda b, n: (b / n * V + (ka * b ** rho) ** 2 * BSS + 2 * ka * b ** rho / b * BSR + BRR / b ** 2)


def leading(model, est, K, corrected=False):
    """Leading-order optimized constant and rate (dict) with regime flags."""
    e = K[est]
    V = e['V']
    if est == 'pot':
        kp, rp = ALPHAP[model]
        return dict(opt_power(V, kp ** 2 * e['BSS_' + model], -rp), regime='BVZ Prop 4.1(i)')
    ka, rho = ALPHA[model]
    BSS, BSR = e['BSS_' + model], e['BSR_' + model]
    BRR = 0.0 if est == 'r1' else e['BRR']
    if corrected:
        BSR, BRR = 0.0, 0.0
    if model == 'GH':
        if est == 'r1':
            return dict(rate=-1.0, const=V, bexp=0.0, bconst=1.0, regime='Prop rank-maxima-branches(b), b=1')
        if corrected:
            return dict(rate=None, const=None, bexp=None, bconst=None,
                        regime='no leading bias; no interior optimum within sqrt(k)/b=O(1)')
        return dict(opt_power(V, BRR, 1.0), regime='Thm process-cov-unknown, c1 irrelevant (S=0), c2>0')
    if rho == -1.0:  # Joe, OPC: all bias terms are b^{-2}
        Btot = ka ** 2 * BSS + 2 * ka * BSR + BRR
        reg = 'Thm process-cov-unknown' if not corrected else 'Thm rank-bias-correction'
        if est == 'r1':
            reg = 'Prop rank-maxima-branches(a)'
        return dict(opt_power(V, Btot, 1.0), regime=reg)
    # Archimax: rho = -2
    if est == 'r1':
        return dict(opt_power(V, BSS, 2.0), regime='Prop rank-maxima-branches(a): c1>0, no c2 condition')
    if corrected:
        d = opt_power(V, BSS, 2.0)
        d['regime'] = ('OUTSIDE proved regime: b* ~ n^{1/5} violates sqrt(k)/b=O(1) of Thm rank-bias-correction; '
                       'within the regime (b >~ n^{1/3}) the leading bias is o(variance): no interior optimum')
        return d
    return dict(opt_power(V, BRR, 1.0), regime='Thm process-cov-unknown with c1=0 (b ~ n^{1/3}); '
                'lower-order S terms b^{-4}, b^{-3} dropped')


def full_min(f, n, bmin):
    res = minimize_scalar(lambda lb: f(np.exp(lb), n), bounds=(np.log(bmin), np.log(n / 2)), method='bounded',
                          options={'xatol': 1e-10})
    b = float(np.exp(res.x))
    # also integer block size
    bi = max(int(bmin), int(np.floor(b)))
    cand = [bi, bi + 1]
    vals = [f(c, n) for c in cand]
    j = int(np.argmin(vals))
    return dict(b=b, val=float(res.fun), b_int=cand[j], val_int=float(vals[j]))


ESTS = [('r1', False, 'r=1'), ('r2opt', False, 'r=2'), ('r2opt', True, 'r=2 bc'), ('pot', False, 'POT'),
        ('r2id', False, 'r=2 id'), ('r2id', True, 'r=2 id bc'), ('r3opt', False, 'r=3'), ('r3opt', True, 'r=3 bc')]
MODELS = ['GH', 'Joe', 'OPC', 'Archimax']


def bmin_of(est):
    return 1.0 if est in ('r1', 'pot') else float(int(est[1]))


def crit_any(M, est, K, corr):
    if M == 'GH' and est == 'r1':
        return lambda b, n, V=K['r1']['V']: b / n * V
    return crit_fn(M, est, K, corr)


def compute_summary():
    summary = {}
    for delta in DELTAS:
        K = C['K'][str(delta)]
        print(f'\n================  K = [{delta}, {1 - delta}]  ================')
        print('V:  ' + '  '.join(f'{e}={K[e]["V"]:.6g}' for e in ('r1', 'r2opt', 'r2id', 'r3opt', 'r3id', 'pot'))
              + f'   ratios to r1: r2opt {K["r2opt"]["V"] / K["r1"]["V"]:.4f}, r3opt {K["r3opt"]["V"] / K["r1"]["V"]:.4f},'
                f' r3opt/r2opt {K["r3opt"]["V"] / K["r2opt"]["V"]:.4f},'
                f' r2id {K["r2id"]["V"] / K["r1"]["V"]:.4f}, POT {K["pot"]["V"] / K["r1"]["V"]:.4f}')
        summary[str(delta)] = {}
        for M in MODELS:
            print(f'--- {M}')
            summary[str(delta)][M] = {}
            for est, corr, lab in ESTS:
                L = leading(M, est, K, corr)
                bmin = bmin_of(est)
                f = crit_any(M, est, K, corr)
                fm = {str(int(n)): full_min(f, n, bmin) for n in NS}
                summary[str(delta)][M][lab] = dict(leading=L, full=fm)
                if L['const'] is None:
                    lt = 'no interior optimum'
                else:
                    lt = f'{L["const"]:.5g} n^{L["rate"]:.4g};  b* = {L["bconst"]:.4g} n^{L["bexp"]:.3g}'
                    if L['bexp'] and L['bexp'] > 0:
                        nmin = (bmin / L['bconst']) ** (1 / L['bexp'])
                        L['n_bstar_eq_bmin'] = float(nmin)
                        lt += f'  (b* < {bmin:g} for n < {nmin:.3g})'
                ft = '  '.join(f'n={int(n):.0e}: min {fm[str(int(n))]["val"]:.4g} at b={fm[str(int(n))]["b"]:.3g}'
                               for n in NS)
                print(f'  {lab:10s} {lt}')
                print(f'  {"":10s} full criterion: {ft}   [{L["regime"]}]')
        # Archimax r=2 (uncorrected): omitted terms 2 alpha/b B_SR + alpha^2 B_SS relative to B_RR/b^2 at b*(n)
        e = K['r2opt']
        L = summary[str(delta)]['Archimax']['r=2']['leading']
        print('  [WITHDRAWN 2026-11-18: terms below the o(1/b) remainder; heuristic only]')
        print('  Archimax r=2: omitted-term ratio |2 b^{-3} B_SR + b^{-4} B_SS| / (b^{-2} B_RR) at leading b*(n):')
        for n in (1e3, 1e4, 1e5, 1e6, 1e7, 1e8):
            b = L['bconst'] * n ** L['bexp']
            om = abs(2 * b ** -3 * e['BSR_Archimax'] + b ** -4 * e['BSS_Archimax'])
            print(f'     n={n:.0e}: b*={b:.3g}, ratio={om / (e["BRR"] / b ** 2):.3g}')
        bg = np.geomspace(1.0, 1e6, 60001)
        rat = np.abs(2 * bg ** -3 * e['BSR_Archimax'] + bg ** -4 * e['BSS_Archimax']) / (e['BRR'] / bg ** 2)
        tailmax = np.maximum.accumulate(rat[::-1])[::-1]          # sup over b' >= b
        bsq = float(bg[np.argmax(tailmax < 0.1)])
        nb = (bsq / L['bconst']) ** (1 / L['bexp'])
        print(f'     omitted terms < 10% of the B_RR term for all b >= {bsq:.3g}; leading b*(n) reaches it at n = {nb:.3g}'
              f'  (sign change of the omitted terms at b = {e["BSS_Archimax"] / (-2 * e["BSR_Archimax"]):.3g})')
        summary[str(delta)]['Archimax']['r=2']['omitted_lt10pct'] = dict(b=bsq, n=float(nb))
        # crossover of full-criterion minima, r=1 vs r=2 (uncorrected and corrected), per model
        print('  [WITHDRAWN 2026-11-18: full-criterion ratios and crossings below are not supported for rho_b<-1 (Archimax) nor for GH r=2 at small b]')
        for M in MODELS:
            f1 = crit_any(M, 'r1', K, False)
            for lab, est, corr in (('r=2', 'r2opt', False), ('r=2 bc', 'r2opt', True)):
                f2 = crit_any(M, est, K, corr)
                ng = np.geomspace(1e2, 1e14, 121)
                dd = np.array([full_min(f2, n, 2.0)['val'] / full_min(f1, n, 1.0)['val'] for n in ng])
                sgn = np.sign(dd - 1)
                ch = np.where(sgn[1:] != sgn[:-1])[0]
                cross = [float(np.sqrt(ng[i] * ng[i + 1])) for i in ch]
                print(f'  {M}: full-criterion min ratio {lab}/r=1 at n=1e3,1e4,1e5,1e6: '
                      + ', '.join(f'{full_min(f2, n, 2.0)["val"] / full_min(f1, n, 1.0)["val"]:.4f}' for n in NS)
                      + f';  crossings in [1e2,1e14]: {", ".join(f"{c:.2g}" for c in cross) or "none"}')
                summary[str(delta)][M][lab]['full_ratio_to_r1'] = {str(int(n)): full_min(f2, n, 2.0)['val'] / full_min(f1, n, 1.0)['val'] for n in NS}
                summary[str(delta)][M][lab]['crossings_r1'] = cross
    return summary


# ------------------------------------------------------------------ LaTeX tables
def fmt_c(x):
    e = int(np.floor(np.log10(abs(x))))
    if -3 <= e <= 0:
        return f'{x:#.3g}'
    m = x / 10 ** e
    return f'{m:#.3g}\\cdot10^{{{e}}}'


RATE = {-1.0: 'n^{-1}', -2 / 3: 'n^{-2/3}', -0.8: 'n^{-4/5}'}
BEXP = {0.0: '', 1 / 3: 'n^{1/3}', 0.2: 'n^{1/5}'}


def rate_tex(r):
    for k, v in RATE.items():
        if abs(r - k) < 1e-9:
            return v
    raise ValueError(r)


def bexp_tex(r):
    for k, v in BEXP.items():
        if abs(r - k) < 1e-9:
            return v
    raise ValueError(r)


NAMES = {'GH': 'Gumbel--Hougaard', 'Joe': 'Joe', 'OPC': 'Outer power Clayton', 'Archimax': 'Archimax $\\psi_2$'}


def no_interior(M, lab):
    """Cells with no interior minimizer in the proved regime (manuscript footnote a)."""
    return 'bc' in lab and M in ('GH', 'Archimax')


def best_label(summary, M, labs, delta):
    cands = []
    for lab in labs:
        L = summary[str(delta)][M][lab]['leading']
        if L['const'] is None or no_interior(M, lab):
            continue
        cands.append((round(L['rate'], 9), L['const'], lab))
    cands.sort()
    return cands[0][2]


def ktex(delta):
    return f'$T=[{delta:g},{1 - delta:g}]$'


def main_table(summary, delta, path):
    labs = ['r=1', 'r=2', 'r=2 bc', 'POT']
    L_ = []
    L_.append(f'% generated by s06_tables.py (2026-11-18) from constants_theta2.json; T = [{delta:g}, {1 - delta:g}]')
    L_.append('\\begin{table}[t]')
    L_.append('\\centering')
    L_.append('\\small')
    L_.append('\\caption{Optimized leading criterion~\\eqref{eq:ameise-trimmed},\n'
              '\\eqref{eq:ameise-bc} and its peaks-over-threshold analogue in\n'
              'Section~\\ref{sec:comparison-pot}, for the models with the\n'
              'Gumbel--Hougaard(2) attractor of Section~\\ref{sec:comparison-examples}, with\n'
              f'estimated optimal weights for $r=2$ and {ktex(delta)}. Entries are of the form\n'
              '$\\text{constant}\\times n^{e}$ with $e$ from Table~\\ref{tab:comparison-rates};\n'
              'the best entry per row, that is, the fastest rate and then the smallest\n'
              'constant, is set in bold.}')
    L_.append('\\label{tab:comparison-ameise}')
    L_.append('\\begin{tabular}{lcccc}')
    L_.append('\\toprule')
    L_.append('Model & $r=1$ & $r=2$ & $r=2$, corrected & POT\\\\')
    L_.append('\\midrule')
    for M in MODELS:
        best = best_label(summary, M, labs, delta)
        cells = []
        for lab in labs:
            L = summary[str(delta)][M][lab]['leading']
            if no_interior(M, lab):
                cells.append('--' + MARK['a'])
                continue
            a = f'{fmt_c(L["const"])}\\,{rate_tex(L["rate"])}'
            mark = MARK['b'] if (M in ('GH', 'Archimax') and lab == 'r=2') else ''
            cells.append((f'$\\bm{{{a}}}$' if lab == best else f'${a}$') + mark)
        L_.append(f'{NAMES[M]}\n & ' + '\n & '.join(cells) + '\\\\')
    L_.append('\\bottomrule')
    L_.append('\\end{tabular}')
    L_.append('')
    L_.append('\\smallskip')
    L_.append('{\\footnotesize\n'
              '\\dag~No interior minimizer in the proved regime\n'
              '$\\sqrt{k}/b=O(1)$; the criterion is of order $n^{-2/3}$ for\n'
              '$b\\asymp n^{1/3}$ (Section~\\ref{sec:comparison-ameise}).\\par\n'
              '\\ddag~Leading constant from~\\eqref{eq:maxstable-balance}, whose\n'
              'minimizer is smaller than $r=2$ for $n<6\\cdot10^{4}$; it is not representative of\n'
              'the estimator at practical sample sizes (Section~\\ref{app:comparison-details}).}')
    L_.append('\\end{table}')
    open(path, 'w').write('\n'.join(L_) + '\n')


def bstar_table(summary, delta, path, labs, heads, label, caption):
    L_ = [f'% generated by s06_tables.py (2026-11-18); T = [{delta:g}, {1 - delta:g}]',
          '\\begin{table}[t]', '\\centering', '\\small', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
          '\\begin{tabular}{l' + 'c' * len(labs) + '}', '\\toprule',
          'Model & ' + ' & '.join(heads) + '\\\\', '\\midrule']
    for M in MODELS:
        top, bot = [], []
        for lab in labs:
            L = summary[str(delta)][M][lab]['leading']
            if L['const'] is None:
                top.append('--' + MARK['a'])
                bot.append('')
                continue
            a = f'{fmt_c(L["const"])}\\,{rate_tex(L["rate"])}'
            b = 'b^\\star=1' if L['bexp'] == 0.0 else f'b^\\star={fmt_c(L["bconst"])}\\,{bexp_tex(L["bexp"])}'
            mark = ''
            if no_interior(M, lab):
                mark = MARK['c']
            elif M in ('GH', 'Archimax') and lab.startswith('r=') and lab != 'r=1' and 'bc' not in lab:
                mark = MARK['b']
            top.append(f'${a}${mark}')
            bot.append(f'$\\scriptstyle {b}$')
        L_.append(NAMES[M] + ' & ' + ' & '.join(top) + '\\\\')
        L_.append(' & ' + ' & '.join(bot) + '\\\\[2pt]')
    L_ += ['\\bottomrule', '\\end{tabular}', '', '\\smallskip',
           '{\\footnotesize\n'
           '\\dag~No leading bias; no interior minimizer in the proved regime $\\sqrt{k}/b=O(1)$.\\par\n'
           '\\ddag~Leading constant from~\\eqref{eq:maxstable-balance}; $b^\\star<r$ unless $n$ is large.\\par\n'
           '\\S~Formal minimizer $b^\\star\\asymp n^{1/5}$ of~\\eqref{eq:ameise-bc}, outside the\n'
           'proved regime $\\sqrt{k}/b=O(1)$ of Theorem~\\ref{thm:rank-bias-correction}; not a result.}',
           '\\end{table}']
    open(path, 'w').write('\n'.join(L_) + '\n')


if __name__ == '__main__':
    r3flag = [a for a in sys.argv[1:] if a.startswith('--r3')]
    R3_VERIFIED = (r3flag == ['--r3-verified'])
    summary = compute_summary()
    json.dump(summary, open('../outputs/ameise_summary.json', 'w'), indent=1)
    for delta in TABLE_DELTAS:
        tag = f'd{int(round(delta * 100)):03d}'
        main_table(summary, delta, f'{tabdir(delta)}/sec7-table-main-{tag}.tex')
        bstar_table(summary, delta, f'{tabdir(delta)}/sec7-table-bstar-{tag}.tex',
                    ['r=1', 'r=2', 'r=2 bc', 'POT'], ['$r=1$', '$r=2$', '$r=2$, corrected', 'POT'],
                    'tab:comparison-bstar',
                    'Optimized leading criterion and minimizing block size $b^\\star$ (POT: $n/k_p$), '
                    f'optimal weights, {ktex(delta)}, Gumbel--Hougaard(2) attractor.')
        if R3_VERIFIED:
            labs = ['r=2 id', 'r=2 id bc', 'r=3', 'r=3 bc']
            heads = ['$r=2$, $W=I$', '$r=2$, $W=I$, corr.', '$r=3$', '$r=3$, corrected']
            cap = 'Identity weights for $r=2$ and optimally weighted $r=3$, ' + ktex(delta) + '.'
        else:
            labs = ['r=2 id', 'r=2 id bc']
            heads = ['$r=2$, $W=I$', '$r=2$, $W=I$, corrected']
            cap = ('Identity weights for $r=2$, ' + ktex(delta) +
                   '. The $r=3$ columns are pending verification of the $r=3$ covariance.')
        bstar_table(summary, delta, f'{tabdir(delta)}/sec7-table-supp-{tag}.tex', labs, heads, 'tab:comparison-supp', cap)
        print(f'wrote {tabdir(delta)}/sec7-table-{{main,bstar,supp}}-{tag}.tex')
    print(f'r=3 columns in supplementary table: {"included" if R3_VERIFIED else "withheld (unverified)"}')
