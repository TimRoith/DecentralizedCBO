from pathlib import Path
import numpy as np
from problems import Problem


_TORCH = None


def _torch():
    """Lazy torch, pinned to one thread.

    run_benchmark forks one worker per CPU and torch would otherwise start an
    intra-op pool per worker, oversubscribing the node by cpus^2 threads. The
    batches here are small enough that threading never pays for itself anyway.
    """
    global _TORCH
    if _TORCH is None:
        import torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        _TORCH = torch
    return _TORCH

DATA_ROOT = Path(__file__).resolve().parent / 'data'


def _load_mnist(root=None):
    from torchvision import datasets
    root = str(root or DATA_ROOT)

    def get(train):
        try:
            return datasets.MNIST(root, train=train, download=False)
        except RuntimeError:
            pass
        try:
            return datasets.MNIST(root, train=train, download=True)
        except Exception as e:
            raise RuntimeError(
                "MNIST is not in " + root + " and cannot be fetched here."
            ) from e

    tr, te = get(True), get(False)
    f = lambda d: (d.data.numpy().reshape(len(d.data), -1).astype(np.float32) / 255.,
                   d.targets.numpy().astype(np.int64))
    Xtr, ytr = f(tr)
    Xte, yte = f(te)
    return Xtr, Xte, ytr, yte


DATASETS = {'mnist': _load_mnist}
_DATA_CACHE = {}
def _load(dataset):
    if dataset not in _DATA_CACHE:
        _DATA_CACHE[dataset] = DATASETS[dataset]()
    return _DATA_CACHE[dataset]


class MLPProblem(Problem):
    '''
    A feed-forward network for classification. The underlying problem when evaluating NNs 
    with CBO is that we need to query multiple copies of the network with different weights.
    This could be solved with functional calls provided by PyTorch. 
    In this case we deem it simpler to implement the simple affine linear layers directly.
    '''
    name = 'mlp'

    def __init__(self, d=None, dataset='digits', hidden=16, batch=64,
                 init_scale=1.0, batch_norm=True, seed=0, log_every=0,
                 partition=False, score_point='consensus', head='linear',
                 loss='ce'):
        self.dataset, self.hidden, self.batch = dataset, hidden, batch
        if head not in ('linear', 'paper'):
            raise ValueError("head must be 'linear' or 'paper'")
        if head == 'paper' and hidden:
            raise ValueError("head='paper' requires hidden=0")
        self.head = head
        if loss not in ('ce', '01'):
            raise ValueError("loss must be 'ce' or '01'")
        self.loss = loss
        self.batch_norm = batch_norm
        self.init_scale = init_scale
        self.f_min = 0.0
        self.domain = (-1e8, 1e8)
        self.seed = seed
        self.log_every = int(log_every)
        self.trace = []
        # One disjoint data shard per node (federated) vs every node sampling
        # the full training set (parallel CBO). See _FreshBatchLoss.
        self.partition = bool(partition)
        if score_point not in ('consensus', 'best_particle'):
            raise ValueError("score_point must be 'consensus' or 'best_particle'")
        self.score_point = score_point

        Xtr, Xte, ytr, yte = _load(dataset)
        self.n_in, self.n_out = Xtr.shape[1], int(ytr.max()) + 1
        torch = _torch()
        self.Xtr, self.ytr = torch.tensor(Xtr), torch.tensor(ytr)
        self.Xte, self.yte = torch.tensor(Xte), torch.tensor(yte)

        H, I, C = hidden, self.n_in, self.n_out
        self.d = (I * H + H + H * C + C) if H else (I * C + C)
        self._slices = (I * H, H, H * C, C) if H else (I * C, C, 0, 0)
        if self.head == 'paper':
            self.d += 2 * C

    def _forward(self, w, Xb):
        """w: (P, d) torch -> logits (P, B, C); one model per particle."""
        torch = _torch()
        H, I, C = self.hidden, self.n_in, self.n_out
        a, b, c, _ = self._slices
        if not H:
            W = w[:, :a].reshape(-1, I, C)
            z = torch.einsum('bi,pic->pbc', Xb, W) + w[:, a:a + C].reshape(-1, 1, C)
            if self.head == 'paper':
                z = torch.relu(z)
                mu = z.mean(1, keepdim=True)
                var = z.var(1, unbiased=False, keepdim=True)
                z = (z - mu) / torch.sqrt(var + 1e-5)
                g = w[:, a + C:a + 2 * C].reshape(-1, 1, C)
                be = w[:, a + 2 * C:].reshape(-1, 1, C)
                z = g * z + be
            return z
        i = 0
        W1 = w[:, i:i + a].reshape(-1, I, H); i += a
        b1 = w[:, i:i + b].reshape(-1, 1, H); i += b
        W2 = w[:, i:i + c].reshape(-1, H, C); i += c
        b2 = w[:, i:].reshape(-1, 1, C)
        h = torch.relu(torch.einsum('bi,pih->pbh', Xb, W1) + b1)
        if self.batch_norm:
            h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-5)
        return torch.einsum('pbh,phc->pbc', h, W2) + b2

    def _loss(self, w, Xb, yb):
        torch = _torch()
        logits = self._forward(w, Xb)
        if self.loss == '01':
            return (logits.argmax(-1) != yb[None, :]).to(logits.dtype).mean(dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp[:, torch.arange(len(yb)), yb]
        return nll.mean(dim=-1)

    def _metrics(self, w_np):
        torch = _torch()
        w = torch.tensor(np.asarray(w_np, dtype=np.float32).reshape(1, -1))
        logits = self._forward(w, self.Xte)[0]
        loss = float(-torch.log_softmax(logits, -1)[torch.arange(len(self.yte)),
                                                    self.yte].mean())
        acc = float((logits.argmax(-1) == self.yte).float().mean())
        return loss, acc

    def generate(self):
        return {'objective': _FreshBatchLoss(self)}

    def sample(self, n_nodes, n_particles):
        rng = np.random.default_rng(self.seed)
        cols = []
        for n, fan in self._init_blocks():
            if fan == 0:        # batch-norm gamma: start at 1, lightly jittered
                cols.append(1.0 + 0.1 * self.init_scale
                            * rng.standard_normal((n_nodes, n_particles, n)))
            else:
                cols.append(self.init_scale
                            * rng.standard_normal((n_nodes, n_particles, n))
                            / np.sqrt(fan))
        return np.concatenate(cols, axis=-1)

    def _init_blocks(self):
        I, H, C = self.n_in, self.hidden, self.n_out
        if H:
            return [(I * H, I), (H, I), (H * C, H), (C, H)]
        blocks = [(I * C, I), (C, I)]
        if self.head == 'paper':
            blocks += [(C, 0), (C, I)]
        return blocks

    def log_step(self, dyn):
        if not self.log_every or int(dyn.it) % self.log_every:
            return
        c = np.asarray(dyn.consensus)
        c = c[:, 0, :] if c.ndim == 3 else c.reshape(1, -1)
        loss, acc = self._metrics(c.mean(0))
        self.trace.append([int(dyn.it), float(loss), float(acc)])

    def post_process(self, dyn):
        np.nan_to_num(dyn.x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self.log_step(dyn)

    def score(self, dyn):
        c = np.asarray(dyn.consensus[:, 0, :])
        per_node = [self._metrics(c[m]) for m in range(c.shape[0])]
        losses = np.array([p[0] for p in per_node])
        accs = np.array([p[1] for p in per_node])

        avg_loss, avg_acc = self._metrics(c.mean(0))
        cbar = c.mean(0)
        disagree = float(np.linalg.norm(c - cbar) / (np.linalg.norm(cbar) + 1e-12))

        bp = np.asarray(dyn.best_particle).reshape(-1, self.d)
        node = int(np.argmin(np.asarray(dyn.best_energy).reshape(-1)))
        bp_loss, bp_acc = self._metrics(bp[node])

        f_best = (1.0 - bp_acc) if self.score_point == 'best_particle'             else (1.0 - accs.mean())
        return float(f_best), {
            'scored_on': self.score_point,
            'test_err_best_particle': float(1.0 - bp_acc),
            'test_acc_best_particle': float(bp_acc),
            'test_loss_best_particle': float(bp_loss),
            'test_err_mean': float(1.0 - accs.mean()),
            'test_loss_mean': float(losses.mean()),
            'test_loss_worst': float(losses.max()),
            'test_acc_mean': float(accs.mean()),
            'test_acc_worst': float(accs.min()),
            'test_loss_avg_model': avg_loss,
            'test_acc_avg_model': avg_acc,
            'consensus_disagreement': disagree,
            'n_params': int(self.d),
            'trace': self.trace,          # [[iteration, test loss, test acc], ...]
        }


class _FreshBatchLoss:
    def __init__(self, problem):
        self.P = problem
        self.rng = np.random.default_rng(problem.seed + 1)
        self._shards = None

    def shards(self, M):
        if self._shards is None or len(self._shards) != M:
            perm = np.random.default_rng(self.P.seed + 2).permutation(len(self.P.Xtr))
            self._shards = np.array_split(perm, M)
        return self._shards

    def __call__(self, x):
        torch = _torch()
        P = self.P
        x = np.asarray(x)
        M, N = x.shape[0], x.shape[1]
        out = np.empty((M, N))
        shards = self.shards(M) if P.partition else None
        for m in range(M):
            if shards is None:
                idx = self.rng.integers(0, len(P.Xtr), P.batch)
            else:
                pool = shards[m]
                idx = pool[self.rng.integers(0, len(pool), P.batch)]
            w = torch.tensor(x[m].astype(np.float32))
            out[m] = P._loss(w, P.Xtr[idx], P.ytr[idx]).numpy()
        return out
