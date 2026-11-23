# DecentralizedCBO

Code for the numerical experiments on decentralized consensus-based
optimization. Each node runs its own particle ensemble and communicates only
with its neighbours via a mixing matrix `W`.

## Install

```bash
conda create -n cbx python=3.11
conda activate cbx
pip install cbx numpy scipy matplotlib pyyaml cma scikit-learn torch torchvision
```

`torch`, `torchvision` and `scikit-learn` are only needed for the MNIST
experiment, `cma` only for the CMA-ES baseline.

## Files

| Path | Content |
|---|---|
| `DecentralizedCBO.py` | Decentralized CBO and the mixing matrices (`ring`, `ring_2hop`, `star`, `full`, `identity`) |
| `problems.py` | Benchmark objectives |
| `nn_problems.py` | MNIST MLP objective |
| `cma_baseline.py`, `sgd_baseline.py` | Baselines |
| `fetch_data.py` | Downloads the MNIST data |
| `config/` | Experiment configs |
| `experiments/` | Scripts for running experiments, building tables and plotting |
| `run_shards.sbatch` | Slurm job array script |
| `docs/companion.html` | Interactive companion for the linear model, open in a browser |

## Running an experiment

Each experiment is defined by a config file in `config/`.

```bash
cd experiments
python run_benchmark.py benchmark_d100.yaml
python aggregate_table.py benchmark_d100.yaml
```

`run_benchmark.py` runs all methods, problems and seeds of the config and
stores the results in `results/<config>_runs/`. Already finished runs are
skipped when restarting. `aggregate_table.py` creates the LaTeX tables
(success rate, median gap, number of evaluations).

### Tables

| Config | Experiment |
|---|---|
| `benchmark_d100.yaml` | Main benchmark, `d=100` |
| `benchmark_d1000.yaml` | `d=1000` |
| `benchmark_d100_noisy.yaml` | Gaussian observation noise |
| `benchmark_d100_heavy.yaml` | Student-t observation noise |
| `mnist_paper.yaml` | MNIST, CBO vs SGD |

### Figures

```bash
python run_benchmark.py scaling_nodes.yaml
python plot_scaling_nodes.py scaling_nodes.yaml
```

| Config | Plot script | Varies |
|---|---|---|
| `scaling_nodes.yaml` | `plot_scaling_nodes.py` | number of nodes |
| `scaling_dimension.yaml`, `scaling_dimension_rastrigin.yaml` | `plot_scaling_nodes.py` | dimension |
| `mixing_freq.yaml`, `mixing_freq_rastrigin.yaml` | `plot_scaling_nodes.py` | iterations between mixing steps |
| `stability_grid.yaml`, `stability_sigma.yaml` | `plot_stability_grid.py` | step size and noise |

`plot_consensus_formation.py` needs no config and writes
`results/consensus_formation.{pdf,png}`.

## MNIST

Download the data before running `mnist_paper.yaml`:

```bash
python fetch_data.py
```

## Cluster

`run_shards.sbatch` runs a config as a Slurm job array. Set `#SBATCH -D`
and the environment lines to your cluster, then run

```bash
sbatch run_shards.sbatch benchmark_d100.yaml
```

## Methods

| Method | Description |
|---|---|
| `serial` | One ensemble with `total_particles` particles |
| `multirun` | `num_nodes` independent ensembles |
| `decentralized` | `num_nodes` ensembles coupled via `W` |

All three use the same total number of particles.

## AI usage

Claude Code (Anthropic, Claude Opus 5.5) was used for debugging and for writing documentation.
