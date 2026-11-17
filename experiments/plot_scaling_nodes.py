"""
Line plot of a `sweep` config's run cache: mean best energy against one axis,
one line per another axis, one panel per a third.

Usage:
    python plot_scaling_nodes.py [scaling_nodes.yaml]

Axes come from the config's `plot` block (x / line / facet). Writes
results/<stem>.pdf and .png, transparent and untitled, for LaTeX.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_utils import load_config
from run_benchmark import load_runs, sweep_label, sweep_axes, problem_label
# also applies the shared Computer Modern / black-text rcParams
from plot_stability_grid import axis_label, pretty, RESULTS

PROBLEM_AXIS = 'problem'

SERIES = [('#2a78d6', 'o'), ('#eb6834', 's'), ('#1baf7a', '^'),
          ('#eda100', 'D'), ('#e87ba4', 'v'), ('#008300', 'P'),
          ('#4a3aa7', 'X'), ('#e34948', '*')]


def decade_subs(xvals):
    """Which multiples of a decade to LABEL on a log axis spanning xvals.

    A panel is ~0.95 in wide per decade, so how many labels fit depends on how
    many decades are on show. Neighbouring 1-2-5 labels sit 0.3 decade apart,
    about 0.28 in, which four-digit labels like 1000 and 2000 cannot share --
    so the wider the span, the fewer labels per decade.
    """
    lo, hi = min(xvals), max(xvals)
    span = np.log10(hi / lo) if lo > 0 else 3.0
    if span <= 2.0:
        return (1, 2, 5)
    if span <= 3.0:
        return (1, 3)
    return (1,)


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'scaling_nodes.yaml'
    cfg = load_config(config_name)
    stem = Path(config_name).stem

    axes_cfg = sweep_axes(cfg['sweep'])
    problems = [problem_label(p) for p in cfg['problems']]
    pl = cfg['plot']
    xname, lname = pl['x'], pl['line']
    cname = pl.get('col', pl.get('facet'))        # `facet` is the old name for `col`
    rname = pl.get('row')                         # optional second facet
    # a line value drawn as a dashed horizontal reference instead of a series
    baseline = pl.get('baseline')
    stat = pl.get('stat', 'mean')
    if stat not in ('mean', 'median', 'success'):
        raise ValueError(
            "plot.stat must be 'mean', 'median' or 'success', got {!r}".format(stat))
    thresh = pl.get('success_thresh', cfg.get('success_thresh', 1.0))
    if stat == 'success':
        agg = lambda v: float(np.mean(np.asarray(v) < thresh))
    else:
        agg = np.mean if stat == 'mean' else np.median

    avail = dict(axes_cfg)
    avail[PROBLEM_AXIS] = problems
    # `problem.<key>` plots against a numeric field of the problem entries --
    # `problem.d` gives a real dimension axis instead of evenly spaced labels.
    prob_key = {}
    for nm in (xname, lname, cname, rname):
        if nm and nm.startswith(PROBLEM_AXIS + '.'):
            key = nm.split('.', 1)[1]
            vals = [pr[key] for pr in cfg['problems']]
            avail[nm] = vals
            prob_key[nm] = dict(zip(vals, problems))
    for nm in (xname, lname, cname, rname):
        if nm and nm not in avail:
            raise ValueError("plot axis '{}' is not a sweep axis; have {}".format(
                nm, sorted(avail)))

    xvals, lvals = avail[xname], avail[lname]
    cvals = avail[cname] if cname else [None]     # col is optional: one panel
    rvals = avail[rname] if rname else [None]

    plotted = [n for n in (xname, lname, cname, rname) if n]
    fixed = {k: v for k, v in axes_cfg.items() if k not in plotted}
    for k, v in fixed.items():
        if len(v) != 1:
            raise ValueError("axis '{}' has {} values but is not plotted".format(k, len(v)))
    uses_problem = any(n == PROBLEM_AXIS or n.startswith(PROBLEM_AXIS + '.')
                       for n in plotted)
    if not uses_problem and len(problems) != 1:
        raise ValueError('{} problems but `problem` is not a plotted axis'.format(
            len(problems)))

    def cell(xv, lv, cv, rv):
        """(sweep label, problem label) for one point."""
        here = {xname: xv, lname: lv}
        if cname:
            here[cname] = cv
        if rname:
            here[rname] = rv
        prob = problems[0]
        for nm, table in prob_key.items():
            if nm in here:
                prob = table[here.pop(nm)]
        prob = here.pop(PROBLEM_AXIS, prob)
        vals = {}
        for k in axes_cfg:                       # config order, so labels match
            vals[k] = here.get(k, fixed.get(k, [None])[0])
        return sweep_label(vals), prob

    labels = sorted({cell(xv, lv, cv, rv)[0]
                     for rv in rvals for cv in cvals for lv in lvals for xv in xvals})
    runs, missing = load_runs(cfg, stem, labels, problems)
    if missing:
        print('{} run(s) missing; those points are dropped'.format(missing))

    empty = [lv for lv in lvals
             if not any(runs[cell(xv, lv, cv, rv)[0]][cell(xv, lv, cv, rv)[1]]
                        for xv in xvals for cv in cvals for rv in rvals)]
    if empty:
        print('WARNING: no runs for {} = {} -- those lines are absent. If the '
              'axis was renamed, the cache still holds the old labels.'
              .format(lname, ', '.join(map(str, empty))))

    dense = len(xvals) > 8 and xname != PROBLEM_AXIS

    nrow, ncol = len(rvals), len(cvals)
    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey='row', squeeze=False,
                            figsize=(3.6 * ncol + .8, 1.95 * nrow + .7))

    for i, rv in enumerate(rvals):
        for j, cv in enumerate(cvals):
            ax = axs[i][j]
            for s, lv in enumerate(lvals):
                colour, marker = SERIES[s % len(SERIES)]
                xs, ys = [], []
                for xi, xv in enumerate(xvals):
                    lab, prob = cell(xv, lv, cv, rv)
                    r = runs[lab][prob]
                    if not r:
                        continue
                    # problem labels are strings, so plot against the index and
                    # relabel the ticks below
                    xs.append(xi if xname == PROBLEM_AXIS else xv)
                    ys.append(agg([r[k]['f_best'] for k in sorted(r)]))
                if not xs:
                    continue
                if str(lv) == str(baseline):
                    level = np.mean(ys) if stat == 'success' else agg(ys)
                    ax.axhline(float(level), color='0.35', linewidth=1.2,
                               linestyle='--', zorder=1,
                               label='{} (reference)'.format(pretty(str(lv))))
                else:
                    ax.plot(xs, ys, color=colour, marker=marker, markersize=4.5,
                            linewidth=1.6, label=pretty(str(lv)), zorder=2)

            if xname.startswith(PROBLEM_AXIS + '.'):
                ax.set_xscale('log')
            elif xname != PROBLEM_AXIS:
                ax.set_xscale('log', base=10 if dense else 2)
            # a fraction is not a log quantity, and 0 is a legal value for it
            if stat == 'success':
                ax.set_ylim(-.04, 1.04)
            else:
                ax.set_yscale('log')
            if xname == PROBLEM_AXIS:
                ax.set_xticks(range(len(xvals)), [pretty(str(v)) for v in xvals])
            elif dense:
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=decade_subs(xvals)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: '{:g}'.format(v)))
                ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_xticks(xvals, ['{:g}'.format(v) for v in xvals])
            if not dense:
                ax.minorticks_off()
            ax.grid(True, linewidth=.4, alpha=.5)
            ax.tick_params(labelsize=7 if dense else 8)
            if dense:
                ax.tick_params(which='minor', length=2)
            if i == 0 and cname:
                ax.set_title('{} $= {:g}$'.format(axis_label(cname), cv), fontsize=9)
            if i == nrow - 1:
                ax.set_xlabel(axis_label(xname), fontsize=9)
            if j == 0:
                ax.set_ylabel('fraction of runs solved' if stat == 'success'
                              else '{} best energy'.format(stat), fontsize=9)
            if rname and j == ncol - 1:
                ax.text(1.04, .5, '{} $= {:g}$'.format(axis_label(rname), rv),
                        transform=ax.transAxes, rotation=270,
                        ha='left', va='center', fontsize=9)

    handles, names = axs[0][0].get_legend_handles_labels()
    top = max(ax.get_position().y1 for row in axs for ax in row)
    fig.legend(handles, names, loc='lower center',
               bbox_to_anchor=(.5, top + (.055 if cname else .012)),
               ncol=min(len(lvals), 6), frameon=False, fontsize=8.5,
               handlelength=1.8, columnspacing=1.4, borderaxespad=0,
               handletextpad=.5, labelspacing=.25,
               title=axis_label(lname), title_fontsize=8.5)

    for ext in ('pdf', 'png'):
        out = RESULTS / '{}.{}'.format(stem, ext)
        fig.savefig(out, dpi=200, bbox_inches='tight', transparent=True)
        print('wrote', out)


if __name__ == '__main__':
    main()
