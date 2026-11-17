import sys, os, json, time, itertools, zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from cbx.dynamics import CBO
from cbx.scheduler import multiply
from cbx.utils.termination import max_eval_term, max_it_term
import cbx.utils.resampling as rsmp

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_utils import load_config
from problems import PROBLEMS, NoisyObjective, HeavyTailedObjective, Shifted
from DecentralizedCBO import (
    DecentralizedCBO, ring, star, ring_2hop, identity, full,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

# Fixed mixing matrices: called as builder(num_nodes, **kwargs).
MIXING_BUILDERS = {
    'ring': ring,
    'star': star,
    'ring_2hop': ring_2hop,
    'identity': identity,
    'full': full,
}


SCHEDULERS = {
    'multiply': multiply,
}

RESAMPLING_CRITERIA = {
    'ensemble_update': rsmp.ensemble_update_resampling,
    'loss_update': rsmp.loss_update_resampling,
    'consensus_stagnation': rsmp.consensus_stagnation,
}

RESERVED_PROBLEM_KEYS = {'name', 'label', 'cbo_kwargs', 'method_cbo_kwargs',
                         'scheduler', 'success_thresh', 'noisy', 'noise_sigma',
                         'noise_model', 'noise_df', 'd',
                         # table presentation only; never reaches the problem
                         'display'}

KNOWN_METHOD_KEYS = {'name', 'label', 'cbo_kwargs', 'scheduler', 'num_nodes',
                     'total_particles', 'resampling', 'clipping', 'mixing',
                     'compute_z_mode', 'gamma', 'lamda_c_drift',
                     'noise_drift_mode', 'cma_kwargs',
                     'problem_overrides',
                     # problems this method does not apply to; see `skipped`
                     'skip_problems', 'sgd_kwargs',
                     # runs, but aggregate_table leaves it out of the table
                     'hide_in_table'}

# Per-iteration history costs memory proportional to max_it and buys nothing
# here: the table only needs best_energy, which is maintained independently.
NO_HISTORY = {'names': [], 'save_int': 10**9}


def problem_label(problem_cfg):
    """Identifier used for result caching/aggregation -- defaults to `name`,
    but an entry can set `label` to disambiguate multiple entries that share
    the same `name` with different constructor kwargs."""
    return problem_cfg.get('label', problem_cfg['name'])


def method_label(method_cfg):
    """Same idea as problem_label: several `decentralized` entries with
    different mixing matrices need distinct labels to not collide."""
    return method_cfg.get('label', method_cfg['name'])


def normalize_methods(methods):
    """Accept either a bare name or a dict with `name` plus overrides."""
    out = []
    for m in methods:
        m = {'name': m} if isinstance(m, str) else dict(m)
        unknown = set(m) - KNOWN_METHOD_KEYS
        if unknown:
            raise ValueError(
                "unknown key(s) " + str(sorted(unknown)) +
                " in method '" + method_label(m) + "'"
            )
        out.append(m)
    return out


STRUCTURAL_AXES = ('num_nodes', 'N')

METHOD_AXES = ('clipping', 'resampling')

# An axis named mixing_<key> sets <key> on the mixing matrix builder rather
# than on the dynamic, e.g. mixing_lam sweeps the ring's self-weight.
MIXING_PREFIX = 'mixing_'


def sweep_axes(sweep_cfg):
    """Ordered {axis: values}; everything but `base` is an axis."""
    return {k: v for k, v in sweep_cfg.items() if k != 'base'}


def expand_sweep(cfg):
    """A `sweep` block becomes one method per combination of its axes.

    `num_nodes` and `N` shape the ensemble, `mixing_*` goes to the mixing
    builder, and every other axis is a CBO kwarg -- so dt, sigma, lamda,
    alpha, mixing_lam all sweep without special-casing. `base` holds whatever
    the cells share.
    """
    sw = cfg.get('sweep')
    if not sw:
        return cfg['methods']

    base = sw.get('base', {'name': 'decentralized'})
    axes = sweep_axes(sw)
    names = list(axes)

    out = []
    for combo in itertools.product(*(axes[k] for k in names)):
        vals = dict(zip(names, combo))
        M = vals.get('num_nodes', cfg['num_nodes'])
        N = vals.get('N', cfg['total_particles'] // cfg['num_nodes'])

        mixing = {k[len(MIXING_PREFIX):]: v for k, v in vals.items()
                  if k.startswith(MIXING_PREFIX)}
        cbo = {k: v for k, v in vals.items()
               if k not in STRUCTURAL_AXES and k not in METHOD_AXES
               and not k.startswith(MIXING_PREFIX)}

        m = dict(base)
        m['label'] = sweep_label(vals)
        m['num_nodes'] = M
        m['total_particles'] = M * N
        m['cbo_kwargs'] = dict(base.get('cbo_kwargs', {}), **cbo)
        for k in METHOD_AXES:
            if k in vals:
                m[k] = vals[k]
        if mixing:
            m['mixing'] = dict(base.get('mixing', {'name': 'ring'}), **mixing)
        out.append(m)
    return out


def sweep_label(vals):
    """vals is {axis: value} in the config's axis order, so a dt-only sweep
    still labels cells M4_N10_dt0.01 as before. Axis values may be strings
    (mixing_name sweeps the builder, not a number)."""
    def fmt(v):
        if isinstance(v, bool):          # before the numeric branch: bool is an int
            return 'on' if v else 'off'
        return '{:g}'.format(v) if isinstance(v, (int, float)) else str(v)
    return '_'.join('{}{}'.format('M' if k == 'num_nodes' else k, fmt(v))
                    for k, v in vals.items())


def load_runs(cfg, config_stem, methods, problems):
    """runs[method][problem][seed] from the JSON cache, plus a missing count."""
    runs_dir = RESULTS_DIR / (config_stem + '_runs')
    out = {m: {p: {} for p in problems} for m in methods}
    missing = 0
    for m in methods:
        for p in problems:
            for s in range(cfg['n_seeds']):
                path = runs_dir / (m + '_' + p + '_s' + str(s) + '.json')
                if path.exists():
                    with open(path) as fh:
                        out[m][p][s] = json.load(fh)
                else:
                    missing += 1
    return out, missing


def merge_cbo_kwargs(cfg, method_cfg, problem_cfg):
    """global < method < problem < (problem, method).

    A problem's overrides win over the method's because they exist to fix the
    problem's *scale* (Schwefel lives on [-500, 500], Griewank on [-600, 600],
    so a sigma tuned on Rastrigin's [-5.12, 5.12] is meaningless there), which
    is a property of the landscape rather than of the method. Where a method
    genuinely needs its own value on one problem, the problem entry says so
    explicitly through `method_cbo_kwargs`, keyed by method label.
    """
    kwargs = dict(cfg.get('cbo_kwargs', {}))
    kwargs.update(method_cfg.get('cbo_kwargs', {}))
    kwargs.update(problem_cfg.get('cbo_kwargs', {}))
    kwargs.update(problem_cfg.get('method_cbo_kwargs', {}).get(method_label(method_cfg), {}))
    return kwargs


def make_scheduler(cfg, method_cfg, problem_cfg):
    """problem > method > global; None means cbx's own default scheduler."""
    sched_cfg = (problem_cfg.get('scheduler')
                 or method_cfg.get('scheduler')
                 or cfg.get('scheduler'))
    if sched_cfg is None:
        return None
    sched_cfg = dict(sched_cfg)
    name = sched_cfg.pop('name')
    if name not in SCHEDULERS:
        raise ValueError("unknown scheduler '" + str(name) + "'")
    return SCHEDULERS[name](**sched_cfg)


def resolve_resampling(cfg, method_cfg):
    """A method may give its own block, or True/False to take or drop the config's."""
    if 'resampling' in method_cfg:
        r = method_cfg['resampling']
        if r is True:
            return cfg.get('resampling')
        return None if r is False else r
    return cfg.get('resampling')


def make_post_process(P, resampling_cfg, clip=True):
    """Safety-box clipping, optionally preceded by cbx's resampling.

    cbx hooks resampling in through post_process and the dynamic takes a
    single callable, so the two are chained here. Resampling first, clipping
    second, so particles kicked far out by the independent noise are pulled
    back in the same step.

    `clip=False` removes the box entirely, so a divergent run is allowed to
    reach inf/NaN instead of being silently bounded. The stability sweeps want
    that: with a box, an unstable cell reports whatever the box edge is worth
    rather than reporting that it diverged.
    """
    if not resampling_cfg:
        return P.post_process if clip else (lambda dyn: None)

    resampling_cfg = dict(resampling_cfg)
    criteria = []
    for c in resampling_cfg.pop('criteria'):
        c = dict(c)
        criteria.append(RESAMPLING_CRITERIA[c.pop('name')](**c))
    resampling = rsmp.resampling(criteria, **resampling_cfg)

    def post_process(dyn):
        resampling(dyn)
        if clip:
            P.post_process(dyn)
    return post_process


def shift_rng(seed, label):
    """RNG for the optimum shift: same for every method, different per seed.

    crc32 rather than hash(), because the runner forks a process pool and
    Python salts string hashing per process -- hash() would hand each worker a
    different objective for the same (seed, problem).
    """
    return np.random.default_rng([seed, zlib.crc32(label.encode())])


def build_problem(problem_cfg, d, cfg=None, seed=0):
    """Returns (P, f, f_clean).

    `f` is what the dynamic optimizes: the noiseless objective, or it wrapped
    in NoisyObjective when the problem entry asks for it. `f_clean` is always
    the noiseless objective and is used only to score the point a run returns.

    The two must not be conflated. dyn.best_energy is the best *observed*
    value, so under measurement noise it is a minimum over millions of draws
    and is biased low by several noise sigmas -- enough to report a negative
    gap on a non-negative objective (measured: -0.248 on noisy Rastrigin where
    the returned point is really at 0.151). Scoring the returned point on
    f_clean measures what the method actually delivers; f_observed keeps what
    the optimizer saw, so the size of the bias stays visible.
    """
    kwargs = {k: v for k, v in problem_cfg.items() if k not in RESERVED_PROBLEM_KEYS}
    P = PROBLEMS[problem_cfg['name']](d=d, **kwargs)
    f_clean = P.generate()['objective']

    # Move the minimiser off its textbook location, differently per seed, so a
    # method is not rewarded for starting near it. f_min is unchanged.
    cfg = cfg or {}
    if cfg.get('shift_optimum') and not hasattr(P, 'score'):
        off = P.sample_offset(shift_rng(seed, problem_label(problem_cfg)),
                              cfg.get('shift_frac', 0.3))
        f_clean = Shifted(f_clean, off)

    f = f_clean
    if problem_cfg.get('noisy', False):
        model = problem_cfg.get('noise_model', 'gaussian')
        scale = problem_cfg.get('noise_sigma', 0.1)
        if model == 'gaussian':
            f = NoisyObjective(f_clean, sigma=scale)
        elif model == 'student_t':
            f = HeavyTailedObjective(f_clean, scale=scale,
                                     df=problem_cfg.get('noise_df', 2.0))
        else:
            raise ValueError("unknown noise_model '" + str(model) +
                             "'; use 'gaussian' or 'student_t'")
    return P, f, f_clean


def ensemble_shape(method_cfg, cfg):
    """(M, N) for this method at the configured total particle budget."""
    # method-level total_particles lets a sweep vary N and M independently
    total = method_cfg.get('total_particles', cfg['total_particles'])
    if method_cfg['name'] in ('serial', 'cma'):
        return 1, total
    M = method_cfg.get('num_nodes', cfg['num_nodes'])
    if total % M:
        raise ValueError(
            "total_particles=" + str(total) + " is not divisible by num_nodes="
            + str(M) + "; the methods would then not be budget-matched"
        )
    return M, total // M


def build_dynamic(method_cfg, cfg, f, P, x0, term, kwargs, seed):
    name = method_cfg['name']
    M = x0.shape[0]

    if name == 'cma':
        from cma_baseline import CMADynamic
        ck = dict(method_cfg.get('cma_kwargs', {}))
        ck.setdefault('max_evals', x0.shape[0] * x0.shape[1] * cfg.get('max_it', 200000))
        ck.setdefault('log_hook', getattr(P, 'log_step', None))
        return CMADynamic(f, x0, domain=P.domain, seed=seed, **ck)
    if name == 'sgd':
        from sgd_baseline import SGDDynamic
        sk = dict(method_cfg.get('sgd_kwargs', {}))
        sk.setdefault('log_hook', getattr(P, 'log_step', None))
        return SGDDynamic(P, x0, seed=seed, **sk)

    post_process = make_post_process(P, resolve_resampling(cfg, method_cfg),
                                     clip=method_cfg.get('clipping',
                                                         cfg.get('clipping', True)))
    common = dict(x=x0, post_process=post_process, term_criteria=term,
                  track_args=(None if cfg.get('track_history') else NO_HISTORY),
                  seed=seed, **kwargs)

    if name in ('serial', 'multirun'):
        return CBO(f, **common)

    if name == 'decentralized':
        mixing = dict(method_cfg.get('mixing') or {'name': 'ring'})
        W = MIXING_BUILDERS[mixing.pop('name')](M, **mixing)

        return DecentralizedCBO(
            f, W=W,
            compute_z_mode=method_cfg.get('compute_z_mode', 'tracking'),
            gamma=method_cfg.get('gamma', 1.0),
            lamda_c_drift=method_cfg.get('lamda_c_drift', 0.0),
            noise_drift_mode=method_cfg.get('noise_drift_mode', 'c'),
            **common,
        )

    raise ValueError("unknown method '" + str(name) + "'")


SELECT_REPS = 5     # re-evaluations per candidate when picking the node to return


def delivered_point(dyn, f, d, x_best=None, reps=SELECT_REPS):
    """The point a method actually delivers, and what selecting it cost.

    Taking the argmin particle ON TRUST is indefensible under heavy-tailed
    noise. The run would report f_clean at argmin-over-(particles, iterations)
    of the OBSERVED energy, and for Student-t with df=2, P(eps < -x) ~
    1/(2x^2), so that running minimum over n draws grows like -scale*sqrt(n/2)
    without bound. At 1280 particles x 20000 iterations, n is 2.6e7 and the
    minimum is about -350: the record holder is whichever particle drew it,
    which says nothing about where the method converged. Measured on Rastrigin
    d=100 under t(df=2), one run scored 0.162 at its consensus and 23.2 at its
    argmin particle.

    But DISCARDING it is equally wrong, and for a reason that only shows up
    under thin tails. A Gaussian minimum over n draws grows like
    -sigma*sqrt(2 ln n), which at n = 2.6e7 is only about -0.58: bounded, and
    small next to any gap worth reporting. There the argmin particle is a
    genuinely good point the method found and can legitimately hand over. On
    benchmark_d100_noisy the 1280-particle ring sits in a non-contracting
    regime where its consensus is at ~900 while a particle really did reach
    0.23 -- discarding that is throwing away the answer.

    So the argmin particle is a CANDIDATE, not the verdict, and re-evaluation
    decides. Under heavy tails its record was noise and fresh draws expose it;
    under Gaussian noise it survives and wins. No branch on the noise model is
    needed -- the aggregation does the discriminating.

    The other candidates are the consensus, which averages the noise away.
    The node to return cannot be chosen with the clean objective -- that would
    be oracle-assisted -- and it cannot simply be averaged over nodes either:
    an uncoupled method (multirun) has its nodes in different basins, and
    their mean is in no basin at all (593 against 59 for the argmin, on
    Rastrigin).

    So each node's candidate point is re-evaluated `reps` times, aggregated by
    MEDIAN, and the best node is returned. Selection is over the node count,
    not over millions of draws, and the median is well behaved at any df -- so
    the selection bias is bounded and small. Measured against the oracle that
    picks the best node using the CLEAN objective, the rule is exact on
    multirun (49.15 vs 49.15 on Rastrigin, 21.49 vs 21.49 on Ackley), so the
    selection costs nothing. For a single-node method it is just that point.

    The candidate a node offers is its TRACKING VARIABLE z where it has one,
    and its local consensus c otherwise. Particles drift toward z, not toward
    their own c (DecentralizedCBO.py), and z tracks the global average of the
    c's -- the dynamic average consensus invariant is mean(z) = mean(c). So
    z_i pools every node's information while c_i sees only its own particles,
    and pooling is what buys accuracy under noise. Measured on Rastrigin d=100
    under t(df=2): returning a selected c_i gives 0.369, which is already the
    best any c_i can do, while a selected z_i gives 0.258 -- equal to the mean
    over all c, obtained without averaging anything by hand. When mixing is
    rare or absent the z_i drift apart again and the selection starts doing
    real work, which is the multirun case and is handled by the same rule.
    """
    pts = getattr(dyn, 'z', None)
    if pts is None:
        pts = dyn.consensus
    pts = np.asarray(pts).reshape(-1, d)                 # (nodes, d)
    if x_best is not None:
        pts = np.vstack([pts, np.asarray(x_best).reshape(1, d)])
    obs = np.median([np.reshape(f(pts[None, :, :]), -1) for _ in range(reps)], axis=0)
    return pts[int(np.argmin(obs))], int(pts.shape[0] * reps)


def skipped(method_cfg, problem_cfg):
    """True when a method declares it does not apply to this problem.

    A method entry may list `skip_problems: [label, ...]`. The cells are then
    never computed and show as n/a in the table, which is the honest rendering
    for a combination that is meaningless rather than merely unrun.

    The case this exists for: MLPProblem's `partition: true` splits the data
    into one disjoint shard per NODE, so with a single node the shard is the
    whole training set and the federated column is bit-for-bit the ordinary
    one. Running a serial row there costs compute to duplicate a cell.
    """
    skip = method_cfg.get('skip_problems') or []
    return problem_label(problem_cfg) in skip


def run_one(method_cfg, problem_cfg, cfg, seed):
    problem_cfg = dict(problem_cfg, **method_cfg.get('problem_overrides', {}))
    d = problem_cfg.get('d', cfg.get('d'))
    thresh = problem_cfg.get('success_thresh', cfg['success_thresh'])

    P, f, f_clean = build_problem(problem_cfg, d, cfg, seed)
    # The problem is authoritative about its own dimension -- everything below
    # (delivered_point, the reshapes, the record) must use the real value.
    d = int(P.d)
    M, N = ensemble_shape(method_cfg, cfg)

    # Seeded here so that, for a given seed, every method starts from the same
    # total_particles points -- only the (M, N) reshaping differs.
    np.random.seed(seed)
    x0 = P.sample(M, N)

    kwargs = merge_cbo_kwargs(cfg, method_cfg, problem_cfg)
    # Per-node budget: num_f_eval is per node, so the total has to be split.
    term = [max_eval_term(max_eval=max(1, cfg['max_query_budget'] // M)),
            max_it_term(max_it=cfg.get('max_it', 200000))]

    dyn = build_dynamic(method_cfg, cfg, f, P, x0, term, kwargs, seed)
    sched = make_scheduler(cfg, method_cfg, problem_cfg)

    t0 = time.time()
    dyn.optimize(sched=sched)
    runtime = time.time() - t0

    best_node = int(np.argmin(dyn.best_energy))
    f_observed = float(dyn.best_energy[best_node])
    x_best = np.asarray(dyn.best_particle[best_node]).reshape(-1)
    f_argmin = float(np.reshape(f_clean(x_best[None, None, :]), ()))

    x_ret, n_sel = x_best, 0
    if problem_cfg.get('noisy', False):
        x_ret, n_sel = delivered_point(dyn, f, d, x_best)

    scored = P.score(dyn) if hasattr(P, 'score') else None
    if scored is None:
        f_best, extra = float(np.reshape(f_clean(x_ret[None, None, :]), ())), {}
    else:
        f_best, extra = scored
    extra = dict(extra, f_argmin=f_argmin)
    f_min = float(P.f_min)
    gap = f_best - f_min

    return {
        'method': method_label(method_cfg), 'problem': problem_label(problem_cfg),
        'seed': seed, 'M': M, 'N': N, 'd': d,
        'x_best': x_ret.tolist(), 'f_best': f_best, 'f_observed': f_observed,
        'noise_bias': f_observed - f_argmin, 'f_min': f_min,
        'gap': gap, 'success': bool(gap < thresh), 'thresh': thresh,
        'n_evals': int(np.sum(dyn.num_f_eval)) + n_sel, 'it': int(dyn.it),
        'runtime': runtime, 'settings': settings_fingerprint(cfg, method_cfg, problem_cfg),
        **extra,
    }


def settings_fingerprint(cfg, method_cfg, problem_cfg):
    """What the run depended on but the cache FILENAME does not record.

    The cache key is sweep_label + problem + seed, and sweep_label holds only
    the swept axes. Everything else -- sigma, dt, alpha, the scheduler, the
    horizon -- lives in cbo_kwargs and friends, so editing one of those in the
    config leaves every cached file looking valid and the next run silently
    reuses results produced under the old value. That is how a figure ends up
    mixing two sigmas with nothing on disk to show it. Storing the fingerprint
    lets run_one_cached refuse the stale file instead.
    """
    fp = {
        'cbo_kwargs': {k: v for k, v in sorted(merge_cbo_kwargs(cfg, method_cfg, problem_cfg).items())
                       if k != 'verbosity'},
        # same precedence as make_scheduler / resolve_resampling, so a method
        # that overrides either one fingerprints differently from one that does not
        'scheduler': (problem_cfg.get('scheduler') or method_cfg.get('scheduler')
                      or cfg.get('scheduler')),
        'resampling': resolve_resampling(cfg, method_cfg),
        'problem_overrides': method_cfg.get('problem_overrides') or None,
        'max_it': cfg.get('max_it'),
        'max_query_budget': cfg.get('max_query_budget'),
        'shift_optimum': cfg.get('shift_optimum', False),
        'scoring': 'delivered-v2' if problem_cfg.get('noisy', False) else 'argmin',
    }
    for key in ('cma_kwargs', 'sgd_kwargs'):
        block = method_cfg.get(key)
        if block:
            fp[key] = {k: v for k, v in sorted(block.items())}
    return fp


def run_path(config_stem, method, problem_name, seed):
    d = RESULTS_DIR / (config_stem + '_runs')
    d.mkdir(parents=True, exist_ok=True)
    return d / (method + '_' + problem_name + '_s' + str(seed) + '.json')


def run_one_cached(method_cfg, problem_cfg, cfg, seed, config_stem):
    path = run_path(config_stem, method_label(method_cfg), problem_label(problem_cfg), seed)
    if path.exists():
        with open(path) as fh:
            cached = json.load(fh)
        want = settings_fingerprint(cfg, method_cfg, problem_cfg)
        have = cached.get('settings')
        if have is not None and have != want:
            diff = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
            raise SystemExit(
                'STALE CACHE: {}\n'
                '  was run under settings that no longer match the config; differs in: {}\n'
                '    cached: {}\n'
                '    config: {}\n'
                '  The cache key is sweep_label + problem + seed, so an edit to cbo_kwargs\n'
                '  (sigma, dt, alpha), the scheduler or the horizon does NOT invalidate it.\n'
                '  Delete results/{}_runs and resubmit.'.format(
                    path.name, ', '.join(diff),
                    {k: have.get(k) for k in diff}, {k: want.get(k) for k in diff},
                    config_stem))
        return cached
    result = run_one(method_cfg, problem_cfg, cfg, seed)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(result, fh)
    tmp.replace(path)
    return result


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_d100.yaml'
    cfg = load_config(config_name)
    config_stem = Path(config_name).stem

    methods = normalize_methods(expand_sweep(cfg))
    problems = cfg['problems']
    n_seeds = cfg['n_seeds']

    jobs = [(m, p, s) for m in methods for p in problems for s in range(n_seeds)
            if not skipped(m, p)]
    nskip = len(methods) * len(problems) * n_seeds - len(jobs)
    if nskip:
        print('{} job(s) skipped by skip_problems'.format(nskip))

    shard = next((a for a in sys.argv[2:] if a.startswith('shard')), None)
    if shard:
        i, n = (int(v) for v in shard.split('=')[-1].split('/'))
        if not 0 <= i < n:
            raise ValueError("shard index {} outside 0..{}".format(i, n - 1))
        jobs = jobs[i::n]

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    print("Running {} (method, problem, seed) jobs with {} workers{}".format(
        len(jobs), n_workers,
        '' if not shard else '  [shard {}/{}]'.format(i, n)), flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(run_one_cached, m, p, cfg, s, config_stem):
                (method_label(m), problem_label(p), s)
            for m, p, s in jobs
        }
        for fut in as_completed(futures):
            m, pname, s = futures[fut]
            try:
                res = fut.result()
                print("  done: {:22s} {:16s} seed={:3d}  success={!s:5s}  "
                      "gap={:.4g}  f_best={:.6g} (f_min={:.6g})  evals={}  it={}".format(
                          m, pname, s, res['success'], res['gap'], res['f_best'],
                          res['f_min'], res['n_evals'], res['it']), flush=True)
            except Exception as e:
                print("  FAILED: {} {} seed={}: {!r}".format(m, pname, s, e), flush=True)


if __name__ == '__main__':
    main()
