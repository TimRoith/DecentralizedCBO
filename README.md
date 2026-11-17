# DecentralizedCBO

Code for the numerical experiments on decentralized consensus-based
optimization. Nodes hold their own particle ensembles and exchange information
only with their neighbours through a mixing matrix `W`; there is no central
coordinator.

## Install

```bash
conda create -n cbx python=3.11
conda activate cbx
pip install cbx numpy scipy matplotlib pyyaml cma scikit-learn torch torchvision
```

`torch`/`torchvision` and `scikit-learn` are needed only for the MNIST
experiment, `cma` only for the CMA-ES baseline.

## Layout

| Path | What |
|---|---|
| `DecentralizedCBO.py` | The dynamic and the mixing matrices (`ring`, `ring_2hop`, `star`, `full`, `identity`) |
| `problems.py` | Benchmark objectives and the `PROBLEMS` registry |
| `nn_problems.py` | The MNIST MLP objective |
| `cma_baseline.py`, `sgd_baseline.py` | Baselines |
| `config/` | One YAML per experiment |
| `experiments/` | Runner, table builder and figure scripts |
| `docs/companion.html` | Interactive companion to the linear model; open it in a browser, no build step |

## Running an experiment

Every experiment is one config. `run_benchmark.py` sweeps all
(method, problem, seed) combinations and caches each run as its own JSON under
`results/<config>_runs/`, so an interrupted job resumes where it stopped.

```bash
cd experiments
python -u run_benchmark.py benchmark_d100.yaml
python    aggregate_table.py benchmark_d100.yaml
```

`aggregate_table.py` writes success-rate, median-gap and evaluation-count
tables as LaTeX next to the cache. Pass `--show-hidden` to include control rows
that are run but kept out of the table.

The worker count comes from `SLURM_CPUS_PER_TASK`, falling back to the CPU
count. To split a config across array tasks, pass `shard=<i>/<n>`.

### Tables

| Config | Experiment |
|---|---|
| `benchmark_d100.yaml` | Main benchmark, `d=100` |
| `benchmark_d1000.yaml` | Scaling to `d=1000` |
| `benchmark_d100_noisy.yaml` | Additive Gaussian observation noise |
| `benchmark_d100_heavy.yaml` | Student-t observation noise |
| `mnist_paper.yaml` | MNIST, CBO vs SGD |

### Figures

The sweep configs are plotted rather than tabulated:

```bash
python run_benchmark.py      scaling_nodes.yaml
python plot_scaling_nodes.py scaling_nodes.yaml
```

| Config | Plotted by | Varies |
|---|---|---|
| `scaling_nodes.yaml` | `plot_scaling_nodes.py` | number of nodes, per topology |
| `scaling_dimension.yaml`, `scaling_dimension_rastrigin.yaml` | `plot_scaling_nodes.py` | dimension |
| `mixing_freq.yaml`, `mixing_freq_rastrigin.yaml` | `plot_scaling_nodes.py` | iterations between mixing steps |
| `stability_grid.yaml`, `stability_sigma.yaml` | `plot_stability_grid.py` | step size and noise strength |

`plot_consensus_formation.py` takes no config — it runs its own simulations and
writes `results/consensus_formation.{pdf,png}`.

## MNIST

Download the data once before running `mnist_paper.yaml`; on a cluster do this
on a login node, since compute nodes usually have no outbound network.

```bash
python fetch_data.py
```

## Cluster

`run_shards.sbatch` submits one config as a Slurm job array, each task taking
every nth job. Set `#SBATCH -D` to this directory and adapt the module and
conda lines to your site, then:

```bash
sbatch run_shards.sbatch benchmark_d100.yaml
```

Because runs are cached per (method, problem, seed), a task that hits its wall
clock can simply be resubmitted.

## Methods

The three CBO rows differ only in how one particle budget is laid out, so they
spend the same number of evaluations per iteration:

| Method | Ensemble | Coupling |
|---|---|---|
| `serial` | one ensemble of `total_particles` | a single consensus point |
| `multirun` | `num_nodes` independent ensembles | none; best over runs |
| `decentralized` | `num_nodes` ensembles | neighbours mixed by `W` |

A method entry may set `label` to run several variants as separate rows.
`success_thresh` is applied when the table is built, from the cached gap, so
thresholds can be changed without rerunning anything.

Anything that evaluates the objective must go through `dyn.eval_f` rather than
`dyn.f`: only `eval_f` maintains `dyn.num_f_eval`, which is what the harness
uses to check that methods are budget-matched. The counter is per node and
`eval_f` charges `x.shape[1]` to each active node, so passing a `(1, M, d)`
block over-counts by a factor of `M`; `compute_z_consensus` therefore evaluates
through `self.f` and charges one evaluation per node by hand.
