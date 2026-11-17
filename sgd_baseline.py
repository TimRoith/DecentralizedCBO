import numpy as np
import torch

class SGDDynamic:
    def __init__(self, problem, x0, n_restarts=None, steps=1000, lr=0.1,
                 momentum=0.0, batch=None, seed=0, log_hook=None,
                 select_batch=8192):
        self.P = problem
        self.d = int(problem.d)
        flat = np.asarray(x0).reshape(-1, self.d)
        self.n = int(n_restarts or flat.shape[0])
        self.steps = int(steps)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.batch = int(batch or problem.batch)
        self.seed = int(seed)
        self.log_hook = log_hook
        self.select_batch = int(select_batch)

        self.num_f_eval = np.zeros(1, dtype=int)
        self.it = 0
        self.best_energy = np.array([np.inf])
        self.best_particle = flat[:1].copy()
        self.consensus = flat[:1].reshape(1, 1, self.d).copy()

    def optimize(self, sched=None):
        P = self.P
        g = torch.Generator().manual_seed(self.seed)

        w0 = P.sample(1, self.n)[0].astype(np.float32)
        with torch.enable_grad():
            w = torch.tensor(w0, requires_grad=True)
        vel = torch.zeros_like(w)

        ntr = len(P.Xtr)
        for t in range(self.steps):
            idx = torch.randint(0, ntr, (self.batch,), generator=g)
            with torch.enable_grad():
                loss = P._loss(w, P.Xtr[idx], P.ytr[idx]).sum()
                if w.grad is not None:
                    w.grad = None
                loss.backward()
            with torch.no_grad():
                if self.momentum:
                    vel.mul_(self.momentum).add_(w.grad)
                    w -= self.lr * vel
                else:
                    w -= self.lr * w.grad
            self.num_f_eval[0] += 3 * self.n
            self.it = t + 1
            if self.log_hook is not None and P.log_every and (t % P.log_every == 0):
                with torch.no_grad():
                    self._set_winner(w)
                self.log_hook(self)

        with torch.no_grad():
            self._set_winner(w)
        return self

    def _set_winner(self, w):
        P = self.P
        rng = np.random.default_rng(self.seed + 7)
        idx = torch.tensor(rng.integers(0, len(P.Xtr), self.select_batch))
        tr = P._loss(w, P.Xtr[idx], P.ytr[idx]).detach().numpy()
        self.num_f_eval[0] += self.n
        k = int(np.argmin(tr))
        best = w[k].detach().numpy().astype(float)
        self.best_energy = np.array([float(tr[k])])
        self.best_particle = best.reshape(1, self.d)
        self.consensus = best.reshape(1, 1, self.d)
