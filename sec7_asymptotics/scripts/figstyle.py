"""Common figure style: imsart [aos] text width 406pt (5.62in), Times text via newtx (as the IMS class,
font ptmr at 11pt), 9pt labels, Okabe-Ito colour-blind-safe palette, vector PDF output."""
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 406.0 / 72.27

OKABE = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442',
             blue='#0072B2', vermillion='#D55E00', purple='#CC79A7')


def setup():
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{newtxtext}\usepackage{newtxmath}',
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
    })
