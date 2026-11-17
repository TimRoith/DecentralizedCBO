import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_utils import load_config
from run_benchmark import load_runs, sweep_label, sweep_axes, problem_label

RESULTS = Path(__file__).resolve().parent.parent / 'results'


def pretty(label):
    # cmr10 follows TeX font encoding, where the underscore slot holds a dot
    # accent -- config labels have to lose their underscores before display.
    return label.replace('_', ' ')

CMAP = 'viridis'
GAP = 'white'      # 2px separator drawn between cells
CELL = 0.30        # inches per heatmap cell, so cells come out square

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['cmr10', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,   # cmr10 has no U+2212, so tick minus signs
    'axes.unicode_minus': False,           # have to come from mathtext
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})


AXIS_LABELS = {
    'dt': r'step size  $\Delta t$',
    'sigma': r'noise  $\sigma$',
    'lamda': r'drift  $\lambda$',
    'mixing_lam': r'mixing self-weight  $\lambda$',
    'alpha': r'weight  $\alpha$',
    'N': r'particles per node  $M$',
    'num_nodes': r'nodes  $N$',
    'd': r'dimension  $d$',
    'update_mixing_freq': r'synchronization period  $S$',
    'mixing_name': 'mixing matrix',
    'problem.d': r'dimension  $d$',
}


def axis_label(name):
    return AXIS_LABELS.get(name, name)


DIVERGED = '#4d4d4d'   # cell whose energy is not finite; distinct from a blank, missing cell


def panel(ax, avg, div, yvals, xvals, norm):
    ax.imshow(avg, cmap=CMAP, norm=norm, origin='lower', aspect='equal')
    if div.any():
        ax.imshow(np.ma.masked_where(~div, np.zeros_like(avg)), origin='lower',
                  aspect='equal', cmap=ListedColormap([DIVERGED]), vmin=0, vmax=1)
    ax.set_xticks(range(len(xvals)), ['{:g}'.format(v) for v in xvals])
    ax.set_yticks(range(len(yvals)), ['{:g}'.format(v) for v in yvals])
    ax.set_xticks(np.arange(-.5, len(xvals), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(yvals), 1), minor=True)
    ax.grid(which='minor', color=GAP, linewidth=2)
    ax.tick_params(which='both', length=0, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'stability_grid.yaml'
    cfg = load_config(config_name)
    stem = Path(config_name).stem
    axes_cfg = sweep_axes(cfg['sweep'])
    pl = cfg.get('plot', {})
    xname = pl.get('x', 'dt')
    yname = pl.get('y', 'N')
    fname = pl.get('facet', 'num_nodes')

    xvals, yvals, fvals = axes_cfg[xname], axes_cfg[yname], axes_cfg[fname]
    # every other axis must be a single value, or a cell would be ambiguous
    fixed = {k: v for k, v in axes_cfg.items() if k not in (xname, yname, fname)}
    for k, v in fixed.items():
        if len(v) != 1:
            raise ValueError("axis '{}' has {} values but is not plotted; "
                             "set plot.x/y/facet or pin it".format(k, len(v)))

    def label_for(xv, yv, fv):
        vals = {}
        for k in axes_cfg:                      # config order, so labels match
            vals[k] = {xname: xv, yname: yv, fname: fv}.get(k, fixed.get(k, [None])[0])
        return sweep_label(vals)

    problems = [problem_label(p) for p in cfg['problems']]
    labels = [label_for(xv, yv, fv) for fv in fvals for yv in yvals for xv in xvals]
    runs, missing = load_runs(cfg, stem, labels, problems)
    if missing:
        print('{} run(s) missing; those cells stay blank'.format(missing))

    avg = np.full((len(problems), len(fvals), len(yvals), len(xvals)), np.nan)
    div = np.zeros(avg.shape, bool)
    for i, p in enumerate(problems):
        for j, fv in enumerate(fvals):
            for k, yv in enumerate(yvals):
                for l, xv in enumerate(xvals):
                    r = runs[label_for(xv, yv, fv)][p]
                    if r:
                        m = float(np.mean([r[s]['f_best'] for s in sorted(r)]))
                        if np.isfinite(m):
                            avg[i, j, k, l] = m
                        else:
                            div[i, j, k, l] = True
    if div.any():
        print('{} cell(s) diverged; drawn in solid {}'.format(int(div.sum()), DIVERGED))
    if not np.isfinite(avg).any():
        raise SystemExit('every cell diverged or is missing; nothing to scale a colormap to')

    # One norm across every facet, so the separate files stay comparable.
    norm = LogNorm(vmin=float(np.nanmin(avg)), vmax=float(np.nanmax(avg)))

    for j, fv in enumerate(fvals):
        nrow = len(problems)
        fig, axs = plt.subplots(nrow, 1, squeeze=False,
                                figsize=(CELL * len(xvals) + 1.0,
                                         CELL * len(yvals) * nrow + 0.7))
        for i, p in enumerate(problems):
            ax = axs[i][0]
            panel(ax, avg[i, j], div[i, j], yvals, xvals, norm)
            ax.set_ylabel(axis_label(yname), fontsize=9)
            if i == nrow - 1:
                ax.set_xlabel(axis_label(xname), fontsize=9)

        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axs, fraction=.05, pad=.03)
        cb.set_label(r'mean energy  $f_{\mathrm{best}}$', fontsize=9)
        cb.ax.tick_params(labelsize=8, length=0)
        cb.outline.set_visible(False)

        if div[:, j].any():
            axs[0][0].legend(
                handles=[plt.Rectangle((0, 0), 1, 1, facecolor=DIVERGED, edgecolor='none')],
                labels=['diverged'], fontsize=8, frameon=False,
                loc='lower left', bbox_to_anchor=(0, 1.01), handlelength=1.2)

        suffix = '{}{:g}'.format(fname, fv)
        for ext in ('pdf', 'png'):
            out = RESULTS / '{}_{}.{}'.format(stem, suffix, ext)
            fig.savefig(out, dpi=200, bbox_inches='tight', transparent=True)
            print('wrote', out)
        plt.close(fig)


if __name__ == '__main__':
    main()
