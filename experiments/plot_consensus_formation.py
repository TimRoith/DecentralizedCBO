"""Ensemble diameter and best energy against mixing strength theta.

Runs its own simulations (no cached runs needed) and writes
results/consensus_formation.{pdf,png}.

    python plot_consensus_formation.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from problems import RastriginProblem
from DecentralizedCBO import DecentralizedCBO, ring
from cbx.scheduler import multiply

np.random.seed(42)

d         = 17
n         = 20
num_nodes = 50
max_it    = 2000

P = RastriginProblem(d=d)
f = P.generate()['objective']

alpha_0   = 1e-4
alpha_max = 1e9
dt        = 0.05
dyn_kwargs = {'verbosity': 0, 'max_it': max_it,
              'f_dim': '3D',
              'track_args':{'names': ['x', 'energy']}}

def diameter(X):
    Z = X.reshape(X.shape[0],-1,d)
    return np.max(Z,axis=-2) - np.min(Z,axis=-2)

def node_diameter(X):
    # diameter within each node separately: (T, M, N, d) -> (T, M)
    return (np.max(X, axis=2) - np.min(X, axis=2)).max(axis=-1)

SIGMAS = [0.1, 2.5]
LAMS   = [0.0, 0.25, 0.5, 0.75, 1.0]   # theta = 1 is W = I: no mixing
FOCUS  = 500                           # shaded region of interest

per_node, pooled, energies = {}, {}, {}
for sigma in SIGMAS:
    for lam in LAMS:
        print('Running Decentralized CBO for sigma=' + str(sigma) + ' lam=' + str(lam))
        dyn_dec = DecentralizedCBO(
            f, alpha=alpha_0, sigma=sigma, dt=dt,
            noise='anisotropic',
            W = ring(num_nodes, lam),
            x = P.sample(num_nodes, n),
            seed = 142,
            **dyn_kwargs
        )
        sched = multiply(factor=1.05, name='alpha', maximum=alpha_max)
        dyn_dec.optimize(sched = sched)

        s_dec = np.arange(0, max_it, 1)
        X    = np.array(dyn_dec.history['x'])[1:]
        per_node[(sigma, lam)] = node_diameter(X).max(axis=-1)   # worst node
        pooled[(sigma, lam)]   = np.max(diameter(X), axis=-1)    # all particles
        energies[(sigma, lam)] = np.array(dyn_dec.history['energy']).min(axis=(-1,-2))

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'],
    'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, 'text.color': 'black',
    'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
})
SERIES = [('#2a78d6', 'o'), ('#eb6834', 's'), ('#1baf7a', '^'),
          ('#eda100', 'D'), ('#e87ba4', 'v')]
COLS = [(per_node, r'per-node diameter (max)'), (pooled, r'pooled diameter'),
        (energies, r'best energy')]

fig, axs = plt.subplots(len(SIGMAS), len(COLS), sharex=True, sharey='col',
                        squeeze=False, figsize=(2.7 * len(COLS) + .8, 2.35 * len(SIGMAS) + .8))
for i, sigma in enumerate(SIGMAS):
    for j, (data, lab) in enumerate(COLS):
        ax = axs[i][j]
        ax.axvspan(0, FOCUS, color='0.5', alpha=.12, linewidth=0, zorder=0)
        for s, lam in enumerate(LAMS):
            colour, marker = SERIES[s % len(SERIES)]
            ax.plot(s_dec, data[(sigma, lam)], color=colour, marker=marker,
                    markersize=4.5, markevery=max_it // 12, linewidth=1.6,
                    label=r'$\theta = {:g}$'.format(lam))
        ax.set_yscale('log')
        ax.grid(True, linewidth=.4, alpha=.5)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_title(lab, fontsize=9)
        if i == len(SIGMAS) - 1:
            ax.set_xlabel('iteration', fontsize=9)
        if j == 0:
            ax.set_ylabel(r'$\sigma = {:g}$'.format(sigma), fontsize=9)

h, l = axs[0][0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', bbox_to_anchor=(.5, -.04 / len(SIGMAS)),
           ncol=len(LAMS), frameon=False, fontsize=8.5,
           handlelength=1.8, columnspacing=1.4)
out = Path(__file__).resolve().parent.parent / 'results' / 'consensus_formation'
for ext in ('pdf', 'png'):
    fig.savefig('{}.{}'.format(out, ext), dpi=200, bbox_inches='tight', transparent=True)

