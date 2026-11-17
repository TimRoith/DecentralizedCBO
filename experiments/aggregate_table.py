"""
Scan the per-run JSON cache for a benchmark config and print comparison tables:
rows = methods (serial / multi-run / decentralized), columns = problems. Also
writes a .tex file with booktabs tables of the same data, for inclusion in a
paper.

Usage:
    python aggregate_table.py <config.yaml>

Success is recomputed here from the cached `gap` and the thresholds currently
in the config, not read from the cached `success` field, so thresholds can be
retuned without rerunning the sweep.
"""
import sys, json
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_utils import load_config
from run_benchmark import (load_runs, method_label, problem_label,
                           ensemble_shape, normalize_methods)

# Plain-text names for the terminal tables; TEX_METHOD_LABELS overrides these
# where the paper wants math mode. Anything unlisted falls back to the raw key.
METHOD_LABELS = {
    'serial': 'Serial CBO (M=1)',
    'multirun': 'Multi-run CBO (independent)',
    'decentralized': 'Decentralized CBO',
    'decentralized_ring': 'Decentralized CBO (ring)',
    'decentralized_selective': 'Decentralized CBO (selective ring)',
}

TEX_METHOD_LABELS = {
    'serial': r'Serial CBO ($M=1$)',
    'multirun': r'Multi-run CBO (independent)',
    'decentralized': r'Decentralized CBO',
    'decentralized_ring': r'Decentralized CBO (ring)',
    'decentralized_selective': r'Decentralized CBO (selective ring)',
}

BASE_NAMES = {'serial': 'Serial CBO', 'multirun': 'Multi-run CBO',
              'decentralized': 'Decentralized CBO'}


def method_display(m, cfg, tex=False):
    """'Serial CBO (N=1, M=5000)' etc., with the shape read off the config.

    PAPER notation: N is the number of nodes and M the particles per node,
    matching AXIS_LABELS in plot_stability_grid.py. That is the reverse of the
    code, where ensemble_shape returns (num_nodes, particles_per_node) as
    (M, N) -- hence the deliberate crossover in the first line below.
    """
    nodes, per_node = ensemble_shape(m, cfg)
    name = m['name']

    if name == 'cma':
        ck = m.get('cma_kwargs') or {}
        base = 'sep-CMA-ES' if ck.get('diagonal') else 'CMA-ES'
        if ck.get('noise_handling'):
            base = 'UH-' + base
        pop = ck.get('popsize', nodes * per_node)
        # n_restarts multiplies the evaluation budget, so a best-of-n row must
        # not print the same as the single run beside it.
        n_r = int(ck.get('n_restarts', 1) or 1)
        if n_r > 1:
            return '{} (popsize {}, best of {})'.format(base, pop, n_r)
        return '{} (popsize {})'.format(base, pop)

    if name == 'sgd':
        sk = m.get('sgd_kwargs') or {}
        return 'SGD ({} restarts, best)'.format(sk.get('n_restarts', nodes * per_node))

    base = BASE_NAMES.get(name, name.replace('_', ' '))
    if name == 'decentralized':
        mix = (m.get('mixing') or {}).get('name')
        if mix:
            base += ', ' + mix.replace('_', '-')
    shape = (r'$N={}$, $M={}$' if tex else 'N={}, M={}').format(nodes, per_node)
    ov = (m.get('problem_overrides') or {}).get('batch')
    if ov is not None:
        shape += (r', $B={}$' if tex else ', B={}').format(int(ov))
    return '{} ({})'.format(base, shape)


def method_batch(m, cfg):
    """The per-node batch a method uses, or None if the problems have none."""
    ov = (m.get('problem_overrides') or {}).get('batch')
    if ov is not None:
        return int(ov)
    bs = {q['batch'] for q in cfg['problems'] if 'batch' in q}
    return int(next(iter(bs))) if len(bs) == 1 else None


def _batches(cfg):
    """Every method's batch, for deciding whether the column is worth showing."""
    out = []
    for m in normalize_methods(cfg['methods']):
        b = method_batch(m, cfg)
        if b is not None:
            out.append(b)
    return out


def method_displays(cfg, tex=False):
    """{method label: display name} for every method in the config."""
    return {method_label(m): method_display(m, cfg, tex)
            for m in normalize_methods(cfg['methods'])}

PROBLEM_LABELS = {
    'rastrigin': 'Rastrigin',
    'ackley': 'Ackley',
    'griewank': 'Griewank',
    'schwefel': 'Schwefel',
    'levy': 'Levy',
    'alpine': 'Alpine',
    'styblinski_tang': 'Styblinski-Tang',
    'zakharov': 'Zakharov',
    'rosenbrock': 'Rosenbrock',
    'quadratic': 'Quadratic',
    'lunacek': 'Lunacek',
    'katsuura': 'Katsuura',
    'gallagher': 'Gallagher',
    # MNIST columns: the group heading already says the objective, so the
    # column only needs what varies within it.
    'mnist_paper': 'B=64',
    'mnist_paper_fed': 'B=64, federated',
    'mnist_b512_ce': 'B=512',
    'mnist_b512_01': 'B=512',
}


def problem_groups(cfg, problems):
    """[(group title, [problem label, ...]), ...] in the config's order.

    Grouping is NOT inferred here, because any inference would be a claim about
    the landscape that the table then presents as fact. It is read from an
    optional `problem_groups` block:

        problem_groups:
          Adequate global structure: [ackley, levy, alpine]
          Weak global structure:     [lunacek, katsuura]

    Keys are group titles, values are problem NAMES. Problems the block omits
    are collected under a trailing untitled group, so adding a problem without
    classifying it degrades gracefully instead of dropping it from the table.
    Without the block the table is ungrouped, exactly as before.
    """
    spec = cfg.get('problem_groups')
    if not spec:
        return [(None, problems)]
    by_name = {}
    for p in cfg['problems']:
        by_name.setdefault(p['name'], []).append(problem_label(p))
        by_name.setdefault(problem_label(p), []).append(problem_label(p))
    out, claimed = [], set()
    for title, names in spec.items():
        members = [lab for n in names for lab in by_name.get(n, [])]
        members = [m for m in members if m in problems and m not in claimed]
        if members:
            out.append((title, members))
            claimed.update(members)
    rest = [p for p in problems if p not in claimed]
    if rest:
        out.append((None, rest))
    return out


def tex_escape(s):
    return s.replace('_', r'\_')


def pretty_method(m, disp=None):
    if disp and m in disp:
        return disp[m]
    return METHOD_LABELS.get(m, m)


def tex_method(m, disp=None):
    """Already-LaTeX label if one is given, otherwise the plain label escaped."""
    if disp and m in disp:
        return disp[m]
    if m in TEX_METHOD_LABELS:
        return TEX_METHOD_LABELS[m]
    return tex_escape(pretty_method(m))


def pretty_problem(p):
    return PROBLEM_LABELS.get(p, p)


def problem_d(problem_cfg, cfg):
    """Search dimension for a problem, or None when the config does not set one.

    A top-level `d` is optional: MLPProblem derives its dimension from the
    architecture, so mnist_paper.yaml has no `d` at all. Everything that used
    cfg['d'] directly raised KeyError on those configs.
    """
    d = problem_cfg.get('d', cfg.get('d'))
    return None if d is None else int(d)


def runs_any(runs, p):
    return any(runs[m][p] for m in runs)


def first_stored_fmin(runs, p):
    for m in runs:
        for s in sorted(runs[m][p]):
            return float(runs[m][p][s].get('f_min', float('nan')))
    return float('nan')


_PROBLEM_CACHE = {}


def get_problem(problem_cfg, cfg):
    """Construct a problem once and reuse it, or None if it cannot be built.

    MEMOISED because this is a table-printing script and construction is not
    always cheap: an MLPProblem loads its whole dataset, measured at 1.7 s
    each, and both live_f_min and build_threshold_table want the same objects.
    Synthetic problems cost nothing, so the cache only matters for the MLP
    configs -- but there it turned a table into ~7 s of dataset I/O.
    """
    kw = {k: v for k, v in problem_cfg.items()
          if k in ('hidden', 'head', 'dataset', 'batch')}
    d = problem_d(problem_cfg, cfg)
    if d is not None:
        kw['d'] = d
    key = (problem_cfg['name'],) + tuple(sorted(kw.items()))
    if key not in _PROBLEM_CACHE:
        try:
            from problems import PROBLEMS
            _PROBLEM_CACHE[key] = PROBLEMS[problem_cfg['name']](**kw)
        except Exception:
            _PROBLEM_CACHE[key] = None
    return _PROBLEM_CACHE[key]


def live_f_min(problem_cfg, cfg):
    """The problem's CURRENT f_min, or None if it cannot be constructed.

    Cached records store both f_best and the f_min that was live when they ran,
    so a corrected constant can be applied without re-running anything. This
    mattered once already: Styblinski-Tang shipped f_min = -39.16617*d against
    a true -39.166165703771*d, an irreducible 4.3e-04 gap at d=100 that every
    method appeared to plateau on.
    """
    P = get_problem(problem_cfg, cfg)
    return None if P is None else float(P.f_min)


def cell_stats(by_seed, thresh, f_min=None):
    """Success rate, median gap and mean evaluation count for one table cell."""
    run_list = [by_seed[s] for s in sorted(by_seed)]
    if not run_list:
        return None
    if f_min is None:
        gaps = np.array([r['gap'] for r in run_list])
    else:
        gaps = np.array([r['f_best'] - f_min for r in run_list])
    return {
        'n': len(run_list),
        'rate': float(np.mean(gaps < thresh)),
        'median_gap': float(np.median(gaps)),
        'best_gap': float(np.min(gaps)),
        'mean_evals': float(np.mean([r['n_evals'] for r in run_list])),
    }


def best_title(acc_problems, problems, tex=False):
    """Heading for the best-over-seeds row, worded for what the cells hold."""
    if problems and all(p in acc_problems for p in problems):
        return 'Best accuracy over seeds'
    if tex:
        return r'Best gap over seeds $f_{\mathrm{best}} - f_{\min}$'
    return 'Best gap over seeds  f_best - f_min'


def gap_fmt(acc_problems, tex=False, key='median_gap'):
    """Cell formatter for a gap row, showing accuracy where that is meant.

    `key` picks which statistic the cell shows -- 'median_gap' or 'best_gap'.

    MLPProblem minimises the held-out ERROR RATE, so f_best - f_min IS the
    error and the optimiser needs it that way. But nobody reports a classifier
    as "0.107" -- the readable quantity is 89.3% accuracy. A problem entry
    marked `display: accuracy` is shown as 100*(1 - gap); every other problem
    is unchanged. ONLY this row is transformed: success rate and evaluation
    counts mean the same thing on every problem.

    Note the bolding of the best cell still ranks on median_gap (lower is
    better), which is the same ordering as higher accuracy, so the transform
    does not disturb it.
    """
    pct = '\\%' if tex else '%'
    def f(st, p):
        g = st[key]
        if p in acc_problems:
            return "{:.1f}{}".format(100.0 * (1.0 - g), pct)
        return ("{:.2g}" if tex else "{:.3g}").format(g)
    return f


def print_table(title, methods, problems, stats, fmt, disp=None, groups=None):
    label_width = max(len(pretty_method(m, disp)) for m in methods) + 2
    col_width = max(14, max(len(pretty_problem(p)) for p in problems) + 2)
    groups = groups or [(None, problems)]
    # the groups fix the column order, so the header and every row agree
    ordered = [p for _, members in groups for p in members]

    GAP = '  |'      # separator between groups, so the split is visible in text
    def line(cells):
        out = ''
        for gi, (_, members) in enumerate(groups):
            if gi:
                out += GAP
            out += ''.join('{:>{w}}'.format(cells[p], w=col_width) for p in members)
        return out

    print()
    print(title)
    if any(t for t, _ in groups):
        # group titles centred over their own columns
        band = ' ' * label_width
        for gi, (t, members) in enumerate(groups):
            if gi:
                band += ' ' * len(GAP)
            span = col_width * len(members)
            band += '{:^{w}}'.format(t or '', w=span) if len(t or '') <= span                 else (t or '')
        print(band.rstrip())
    header = ' ' * label_width + line({p: pretty_problem(p) for p in ordered})
    print(header)
    print('-' * len(header))
    for m in methods:
        cells = {p: ('n/a' if stats[m][p] is None else fmt(stats[m][p], p))
                 for p in ordered}
        print('{:<{w}}'.format(pretty_method(m, disp), w=label_width) + line(cells))


def build_threshold_table(cfg, caption, label):
    """Reference table: what each problem is and when a run counts as solved.

    Built from the config and the Problem classes rather than written by hand,
    so it cannot drift from what was actually run. The noise column appears
    only when some problem in the config is noisy.
    """
    from problems import PROBLEMS
    rows, any_noise = [], any(p.get('noisy') for p in cfg['problems'])
    # The noisy tables report no success column, so the threshold that would
    # define it has nothing to document.
    show_thresh = not any_noise
    for p in cfg['problems']:
        d = problem_d(p, cfg)
        P = get_problem(p, cfg)
        if P is None:
            continue
        d = P.d if d is None else d
        lo, hi = P.domain
        row = [tex_escape(pretty_problem(problem_label(p))),
               '{:g}'.format(d),
               '$[{:g}, {:g}]$'.format(lo, hi),
               '{:g}'.format(P.f_min)]
        if show_thresh:
            row.append('{:g}'.format(p.get('success_thresh', cfg['success_thresh'])))
        if any_noise:
            row.append('{:g}'.format(p['noise_sigma']) if p.get('noisy') else '--')
        rows.append(row)

    head = ['Problem', '$d$', 'Domain', '$f_{\\min}$']
    if show_thresh:
        head.append('Threshold')
    if any_noise:
        head.append('Noise $\\sigma$')
    lines = [r'\begin{table}[t]', r'\centering',
             r'\begin{tabular}{l' + 'c' * (len(head) - 1) + '}', r'\toprule',
             ' & '.join(head) + r' \\', r'\midrule']
    lines += [' & '.join(r) + r' \\' for r in rows]
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{' + caption + '}', r'\label{' + label + '}',
              r'\end{table}']
    return '\n'.join(lines) + '\n'


def build_latex_table(methods, problems, stats, blocks, caption, label, disp=None,
                      groups=None):
    """One table, one stacked row block per metric.

    `blocks` is a list of (title, fmt) pairs, optionally extended to
    (title, fmt, stat_key, 'max'|'min'): each contributes a spanning subheading
    followed by one row per method, so the metrics read down the page against a
    single shared column header instead of being split across separate floats.
    Wrapped in \\resizebox so a long problem list still fits the text width --
    needs \\usepackage{graphicx}.

    With stat_key given, the best cell per problem column is bolded. Direction
    is per block because it differs: success rate is better high, gap better
    low. Ties are all bolded, which is the honest rendering when three methods
    all solve a problem outright.
    """
    groups = groups or [(None, problems)]
    # groups fix the column order; every loop below iterates this reordered list
    problems = [p for _, members in groups for p in members]
    col_spec = 'l' + 'c' * len(problems)
    ncols = len(problems) + 1
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
    ]
    if any(t for t, _ in groups):
        # Spanning titles with a rule under each, so the grouping survives into
        # the PDF -- the text table's "|" separator has no LaTeX counterpart.
        head, rules, col = [''], [], 2
        for t, members in groups:
            head.append(r'\multicolumn{' + str(len(members)) + r'}{c}{'
                        + tex_escape(t or '') + '}')
            rules.append(r'\cmidrule(lr){' + str(col) + '-'
                         + str(col + len(members) - 1) + '}')
            col += len(members)
        lines.append(' & '.join(head) + r' \\')
        lines.append(''.join(rules))
    lines.append('Method & ' + ' & '.join(
        tex_escape(pretty_problem(p)) for p in problems) + r' \\')
    for block in blocks:
        title, fmt = block[0], block[1]
        key, direction = (block[2], block[3]) if len(block) >= 4 else (None, None)

        best = {}
        if key is not None:
            pick = max if direction == 'max' else min
            for p in problems:
                vals = [stats[m][p][key] for m in methods if stats[m][p] is not None]
                if vals:
                    best[p] = pick(vals)

        lines.append(r'\midrule')
        lines.append(r'\multicolumn{' + str(ncols) + r'}{l}{\textbf{' + title + r'}} \\')
        for m in methods:
            cells = []
            for p in problems:
                st = stats[m][p]
                if st is None:
                    cells.append('n/a')
                    continue
                cell = fmt(st, p)
                if p in best and np.isclose(st[key], best[p], rtol=1e-9, atol=0.0):
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
            lines.append(tex_method(m, disp) + ' & ' + ' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    show_hidden = '--show-hidden' in sys.argv[1:]
    config_name = args[0] if args else 'benchmark_d100.yaml'
    cfg = load_config(config_name)
    config_stem = Path(config_name).stem
    results_dir = Path(__file__).resolve().parent.parent / 'results'

    methods = [method_label(m) for m in normalize_methods(cfg['methods'])
               if show_hidden or not m.get('hide_in_table')]
    hidden = [method_label(m) for m in normalize_methods(cfg['methods'])
              if m.get('hide_in_table')]
    if hidden:
        print(('SHOWING hidden row(s): ' if show_hidden else
               'not shown (hide_in_table, still run; --show-hidden to see): ')
              + ', '.join(hidden))
    disp_txt = method_displays(cfg)
    disp_tex = method_displays(cfg, tex=True)
    problems = [problem_label(p) for p in cfg['problems']]
    thresholds = {problem_label(p): p.get('success_thresh', cfg['success_thresh'])
                  for p in cfg['problems']}

    runs, missing = load_runs(cfg, config_stem, methods, problems)
    fmins = {problem_label(q): live_f_min(q, cfg) for q in cfg['problems']}
    stale = [p for p in problems
             if fmins.get(p) is not None and runs_any(runs, p)
             and abs(first_stored_fmin(runs, p) - fmins[p]) > 1e-12]
    if stale:
        print('NOTE: f_min has changed since these runs for {} -- gaps are '
              'recomputed from f_best, no re-run needed'.format(', '.join(stale)))
    skip = {(method_label(m), problem_label(q))
            for m in normalize_methods(cfg['methods']) for q in cfg['problems']
            if (m.get('skip_problems') or []) and problem_label(q) in m['skip_problems']}
    if skip:
        print('{} cell(s) blanked by skip_problems'.format(len(skip)))
    stats = {m: {p: (None if (m, p) in skip
                     else cell_stats(runs[m][p], thresholds[p], fmins.get(p)))
                 for p in problems}
             for m in methods}

    print('config: {}   d={}  total_particles={}  num_nodes={}  budget={}  seeds={}'.format(
        config_name, cfg.get('d', 'per problem'), cfg['total_particles'],
        cfg['num_nodes'],
        cfg['max_query_budget'], cfg['n_seeds']))
    print('success thresholds (gap = f_best - f_min): ' +
          ', '.join('{}={:g}'.format(p, thresholds[p]) for p in problems))

    noisy = any(p.get('noisy') for p in cfg['problems'])
    groups = problem_groups(cfg, problems)
    acc_problems = {problem_label(q) for q in cfg['problems']
                    if q.get('display') == 'accuracy'}
    if noisy:
        print('\nnoisy config: success rate suppressed, it only re-states the '
              'median gap against a threshold near the noise floor')
    else:
        print_table('Success rate (solved runs / completed runs)', methods, problems, stats,
                    lambda st, p: '{:.0f}% ({}/{})'.format(100 * st['rate'], int(round(st['rate'] * st['n'])), st['n']),
                    disp_txt, groups)
    print_table('Median gap  f_best - f_min', methods, problems, stats,
                gap_fmt(acc_problems, tex=False), disp_txt, groups)
    print_table(best_title(acc_problems, problems), methods, problems, stats,
                gap_fmt(acc_problems, tex=False, key='best_gap'), disp_txt, groups)
    print_table('Mean total f-evaluations', methods, problems, stats,
                lambda st, p: '{:.3g}'.format(st['mean_evals']), disp_txt, groups)

    if missing:
        print('\n{} run(s) not yet completed.'.format(str(missing)))

    tex = build_latex_table(
        methods, problems, stats,
        blocks=([] if noisy else [
            ('Success rate',
             lambda st, p: '{:.0f}\\%'.format(100 * st['rate']), 'rate', 'max'),
        ]) + [
            (r'Median gap $f_{\mathrm{best}} - f_{\min}$',
             gap_fmt(acc_problems, tex=True), 'median_gap', 'min'),
            (best_title(acc_problems, problems, tex=True),
             gap_fmt(acc_problems, tex=True, key='best_gap'), 'best_gap', 'min'),
        ],
        caption='Serial, multi-run and decentralized CBO over ' + str(cfg['n_seeds']) +
                (' seeds in $d=' + str(cfg['d']) + '$. ' if cfg.get('d')
                 else ' seeds. ') +
                ('Under observation noise the achievable gap is proportional to the '
                 'noise level, so no success threshold is reported: a threshold near '
                 'that floor measures the threshold rather than the method.' if noisy
                 else 'A run counts as successful when '
                      '$f_{\\mathrm{best}} - f_{\\min}$ falls below the '
                      'problem\'s threshold.'),
        label='tab:decentralized-benchmark',
        disp=disp_tex,
        groups=groups,
    )

    thresh_tex = build_threshold_table(
        cfg,
        caption='Test problems and the threshold below which '
                '$f_{\\mathrm{best}} - f_{\\min}$ counts as a solved run. '
                'Thresholds are per problem because the objectives differ by '
                'orders of magnitude in scale.',
        label='tab:problem-thresholds')

    results_dir.mkdir(parents=True, exist_ok=True)
    tex_path = results_dir / (config_stem + '.tex')
    with open(tex_path, 'w') as fh:
        fh.write(tex)
    thresh_path = results_dir / (config_stem + '_thresholds.tex')
    with open(thresh_path, 'w') as fh:
        fh.write(thresh_tex)
    print('\nLaTeX tables written to\n  {}\n  {}'.format(tex_path, thresh_path))


if __name__ == '__main__':
    main()
