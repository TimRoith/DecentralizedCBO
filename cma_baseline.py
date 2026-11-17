import numpy as np


class CMADynamic:
    def __init__(self, f, x0, max_it=None, domain=None, popsize=None, sigma0=None,
                 noise_handling=False, diagonal=False, seed=0, tol=False,
                 max_evals=None, log_hook=None, n_restarts=1):
        import cma
        self.f = f
        x0 = np.asarray(x0)
        self.d = x0.shape[-1]
        flat = x0.reshape(-1, self.d)
        self.popsize = int(popsize or flat.shape[0])
        self.mean0 = flat.mean(axis=0)
        self.sigma0 = float(sigma0 or max(flat.std(axis=0).mean(), 1e-12))
        if max_it is None:
            if max_evals is None:
                raise ValueError('CMADynamic needs max_it or max_evals')
            max_it = int(max_evals) // self.popsize
        opts = {'popsize': self.popsize, 'verbose': -9, 'seed': int(seed) + 1,
                'maxiter': int(max_it), 'CMA_diagonal': bool(diagonal)}
        if not tol:
            opts.update({'tolfun': 0, 'tolfunhist': 0, 'tolx': 0,
                         'tolflatfitness': int(max_it), 'tolfunrel': 0})
        self._cma = cma
        self._opts = opts
        self.noise_handling = bool(noise_handling)
        self.max_it = int(max_it)
        self.max_evals = None if max_evals is None else int(max_evals)

        self.num_f_eval = np.zeros(1, dtype=int)
        self.it = 0
        self.best_energy = np.array([np.inf])
        self.best_particle = np.zeros((1, self.d))
        self.consensus = self.mean0.reshape(1, 1, self.d).copy()
        self.log_hook = log_hook
        self.n_restarts = max(1, int(n_restarts))

    def _eval(self, X):
        vals = np.asarray(self.f(np.asarray(X)[None, ...])).reshape(-1)
        self.num_f_eval[0] += len(X)
        return [float(v) for v in vals]

    def optimize(self, sched=None):          # sched is ignored; CMA self-adapts
        if self.n_restarts > 1:
            return self._optimize_restarts()
        return self._optimize_one(self._opts)

    def _optimize_restarts(self):
        base = self._opts['seed']
        best_f, best_x, best_c = np.inf, None, None
        evals = 0
        for r in range(self.n_restarts):
            opts = dict(self._opts, seed=int(base) + 10007 * r)
            self.num_f_eval = np.zeros(1, dtype=int)
            self.best_energy = np.array([np.inf])
            self._optimize_one(opts)
            evals += int(self.num_f_eval[0])
            if self.best_energy[0] < best_f:
                best_f = float(self.best_energy[0])
                best_x = self.best_particle.copy()
                best_c = self.consensus.copy()
        self.best_energy = np.array([best_f])
        self.best_particle = best_x
        self.consensus = best_c
        self.num_f_eval = np.array([evals], dtype=int)
        return self

    def _optimize_one(self, opts):
        es = self._cma.CMAEvolutionStrategy(self.mean0, self.sigma0, opts)
        nh = (self._cma.NoiseHandler(self.d, maxevals=[1, 1, 30])
              if self.noise_handling else None)
        while not es.stop():
            if self.max_evals is not None and self.num_f_eval[0] >= self.max_evals:
                break
            X = es.ask()
            if nh is None:
                es.tell(X, self._eval(X))
            else:
                reps = max(1, int(round(nh.evaluations)))
                fit = np.median([self._eval(X) for _ in range(reps)], axis=0).tolist()
                es.tell(X, fit)
                es.sigma *= nh(X, fit, self._eval_single, es.ask)
                es.countevals += nh.evaluations_just_done
            self.it = int(es.countiter)
            if self.log_hook is not None:
                fav = getattr(es.result, 'xfavorite', None)
                self.consensus = np.asarray(
                    es.mean if fav is None else fav, dtype=float).reshape(1, 1, self.d)
                self.log_hook(self)
        if es.best.f < self.best_energy[0]:
            self.best_energy = np.array([float(es.best.f)])
            self.best_particle = np.asarray(es.best.x).reshape(1, self.d)
        fav = getattr(es.result, 'xfavorite', None)
        if fav is None:
            fav = es.mean
        self.consensus = np.asarray(fav, dtype=float).reshape(1, 1, self.d)
        return self

    def _eval_single(self, x):
        return self._eval(np.asarray(x)[None, :])[0]
