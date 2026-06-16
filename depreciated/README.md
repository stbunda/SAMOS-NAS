# SAMOS — Surrogate-Assisted Multi-Objective Search

Code for the paper submitted to **PPSN 2026**. This repository provides all scripts and code needed to reproduce the experimental results on surrogate-assisted multi-objective optimization for Neural Architecture Search and continuous benchmark problems.

## Methods

| Method | Paper Name | Description |
|--------|-----------|-------------|
| `random` | Random Search | Uniform random sampling |
| `nsga2` | NSGA-II | Standard multi-objective evolutionary algorithm |
| `parego` | ParEGO | Tchebycheff scalarization + Bayesian optimization |
| `mosmac` | MO-SMAC | SMAC3 `MultiObjectiveFacade` |
| `gpsaf-default` | GPSAF | Gaussian Process-based Surrogate-Assisted Framework (pysamoo) |
| `ssa-nsga2-default` | SSA-RBF | Surrogate-Assisted NSGA-II with RBF (pysamoo) |
| `samos-xgb` | SAMOS | Our method: inner-loop NSGA-II on XGBoost surrogates |

### Ablation Variants

| Method | Description |
|--------|-------------|
| `ssa-nsga2-xgb` | SSA-XGB: SSA-NSGA-II with XGBoost instead of RBF |
| `ssa-nsga2-xgb-cheap` | SSA-XGB-Cheap: SSA-XGB where cheap objectives (params/flops) use exact evaluations |
| `samos-cheapreal` | SAMOS-Cheap: SAMOS where cheap objectives use exact evaluations |

## Benchmarks

| Benchmark | Entry Point | Problems |
|-----------|------------|----------|
| **WFG 1–9** | `main_pymoo_benchmark.py` | Continuous 2-objective, `n_var = 12` |
| **C10MOP 1–9** | `main_evoxbench.py --suite c10mop` | CIFAR-10 NAS (EvoXBench) |
| **IN1KMOP 1–9** | `main_evoxbench.py --suite in1kmop` | ImageNet-1K NAS (EvoXBench) |
| **NASBench-101 Search Space** | `analyze_nasbench101_searchspace.py` | Search space characterization |

## How SAMOS Works

Each outer generation:

1. **Fit surrogates** — Train one XGBoost model per objective on the evaluated archive
2. **Run inner NSGA-II** — Evolve on surrogate predictions for `n_gen_inner` generations
3. **Deduplicate** — Remove candidates already in the archive
4. **Subset selection** — Pick `n_infill` diverse candidates from the inner Pareto front
5. **Evaluate** — Query the true objective function (the only expensive step)
6. **Update archive** — Merge new evaluations

## Installation

```bash
conda env create -f environment.yml
conda activate SAMOS_311
```

### EvoXBench Data

Place the EvoXBench data files in `problem/data/evoxbench/` as described in the [EvoXBench documentation](https://github.com/EMI-Group/evoxbench).

## Running Experiments

### WFG Benchmarks

```bash
# Single problem, single seed
python main_pymoo_benchmark.py \
    --problem wfg1 \
    --methods random nsga2 samos-xgb mosmac parego gpsaf-default ssa-nsga2-default ssa-nsga2-xgb \
    --seeds 0 \
    --pop_size 20 \
    --n_gen 60

# All 9 WFG problems
python main_pymoo_benchmark.py \
    --problem wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9 \
    --methods samos-xgb \
    --seeds 0 1 2 3 4 \
    --pop_size 20 --n_gen 60 --n_gen_inner 20 --inner_pop_size 200
```

### EvoXBench (C10MOP / IN1KMOP)

```bash
# CIFAR-10 benchmark
python main_evoxbench.py \
    --suite c10mop --pids 1 2 3 4 5 6 7 8 9 \
    --methods random nsga2 samos-xgb parego gpsaf-default mosmac \
    --seeds 0 1 2 \
    --pop_size 20 --n_gen 60

# ImageNet-1K benchmark
python main_evoxbench.py \
    --suite in1kmop --pids 1 2 3 4 5 6 7 8 9 \
    --methods samos-xgb \
    --seeds 0 \
    --pop_size 20 --n_gen 60

# Cheap-objectives ablation (SAMOS uses real eval for params/flops)
python main_evoxbench.py \
    --suite c10mop --pids 1 2 3 4 5 6 7 8 9 \
    --methods samos-cheapreal ssa-nsga2-xgb-cheap \
    --seeds 0 \
    --pop_size 20 --n_gen 60
```

### Search Space Analysis (NASBench-101)

```bash
python analyze_nasbench101_searchspace.py
```

## Analysis & Figures

```bash
# WFG convergence plots + LaTeX tables
python analyze_pymoo_benchmark.py \
    --problems wfg1 wfg2 wfg3 wfg4 wfg5 wfg6 wfg7 wfg8 wfg9 \
    --methods random nsga2 samos-xgb mosmac parego gpsaf-default ssa-nsga2-default ssa-nsga2-xgb

# EvoXBench convergence plots + LaTeX tables
python analyze_evoxbench.py --suite c10mop
python analyze_evoxbench.py --suite in1kmop

# Combined HV/IGD+ table across all benchmarks (WFG + C10MOP + IN1KMOP)
python analyze_all_benchmarks.py

# Critical difference ranking plots
python analyze_cd_plots.py

# Recompute indicators from saved archives (if needed)
python recompute_indicators.py --suite c10mop
```

## SLURM Cluster

Submit experiments on a SLURM cluster using the provided `.sbatch` files:

```bash
# 1. WFG baseline methods (random, nsga2, samos-xgb, mosmac)
sbatch WFG_BENCHMARK.sbatch                        # seeds 0-9

# 2. WFG extended methods (parego, gpsaf, ssa-nsga2)
sbatch --export=ALL,PROBLEM=wfg1 WFG_EXTENDED_BENCHMARK.sbatch
# or submit all 9 problems:
bash submit_wfg_extended_benchmark.sh

# 3. SSA-XGB ablation (all benchmarks)
sbatch SSA_NSGA2_XGB_FULL_BENCHMARK.sbatch
sbatch SSA_NSGA2_XGB_CHEAP_FULL_BENCHMARK.sbatch

# 4. EvoXBench C10MOP
sbatch EVOXBENCH_C10.sbatch

# 5. EvoXBench IN1KMOP
sbatch EVOXBENCH_IN1K.sbatch

# 6. Cheap-objectives ablation
sbatch EVOXBENCH_CHEAPREAL.sbatch
```

Each SBATCH file runs 30 seeds across multiple methods. See file headers for array task configuration.

## Project Structure

```
SAMOS-NAS/
├── main_pymoo_benchmark.py              # WFG experiment runner
├── main_evoxbench.py                    # EvoXBench (C10MOP/IN1KMOP) experiment runner
├── analyze_pymoo_benchmark.py           # WFG convergence plots + tables
├── analyze_evoxbench.py                 # EvoXBench convergence plots + tables
├── analyze_all_benchmarks.py            # Combined HV/IGD+ table
├── analyze_cd_plots.py                  # Critical difference ranking plots
├── analyze_nasbench101_searchspace.py   # NASBench-101 search space study
├── recompute_indicators.py              # Recompute HV/IGD+ from saved archives
├── environment.yml                      # Conda environment (Python 3.11)
│
├── strategy/                            # Core algorithms
│   ├── algorithm/
│   │   ├── algorithms.py                # RandomGA, NSGA-II wrappers
│   │   ├── gpsaf.py                     # GPSAF (pysamoo)
│   │   ├── ssansga2.py                  # SSA-NSGA-II (pysamoo)
│   │   ├── parego.py                    # ParEGO
│   │   └── bo/                          # Bayesian optimization components
│   ├── surrogate/
│   │   ├── samos_minimal.py             # SAMOS algorithm (pymoo drop-in)
│   │   ├── subset_selection.py          # Diversity-based infill selection
│   │   └── models/                      # Surrogate model zoo
│   │       ├── xgboost.py               # XGBoost (primary surrogate)
│   │       ├── rf.py                    # Random Forest
│   │       ├── kriging.py               # GPR variants (Matérn, RBF)
│   │       ├── rbf.py                   # RBF interpolation
│   │       └── ...                      # KNN, CART, Extra Trees, etc.
│   ├── operations/                      # Genetic operators (crossover, mutation)
│   ├── genetics/                        # Architecture encoding & deduplication
│   ├── sampler.py                       # Population sampling strategies
│   └── callbacks.py                     # HV/IGD+ tracking callbacks
│
├── problem/                             # Benchmark problem definitions
│   ├── pymoo/                           # WFG/ZDT/DTLZ wrappers
│   ├── evoxbench/                       # EvoXBench NAS wrappers
│   └── data/                            # Benchmark data (not versioned)
│
├── analysis/                            # Plotting & reporting library
│   ├── plotter.py                       # Convergence plot generation
│   ├── convergence.py                   # Indicator recomputation utilities
│   ├── cd_analysis.py                   # Critical difference analysis
│   └── latex_table_generator.py         # LaTeX table generation
│
├── *.sbatch                             # SLURM job scripts
└── submit_wfg_extended_benchmark.sh     # WFG extended submission wrapper
```

## Citation

```bibtex
@inproceedings{samos2026ppsn,
  title     = {TODO},
  author    = {TODO},
  booktitle = {Parallel Problem Solving from Nature (PPSN)},
  year      = {2026},
}
```

## Key Dependencies

- **[pymoo](https://pymoo.org/) 0.6.1.5** — Multi-objective optimization framework
- **[pysamoo](https://github.com/anyoptimization/pysamoo)** — Surrogate-assisted MOO algorithms (GPSAF, SSA-NSGA-II)
- **scikit-learn 1.7** — Random Forest, KNN, GPR surrogates
- **XGBoost 3.2** — Gradient-boosted tree surrogate
- **PyTorch 2.9** — Neural network surrogates (MLP, RNN)
- **SMAC3** — Sequential Model-based Algorithm Configuration
- **nas-bench-201 / nats-bench** — NAS benchmark APIs
- **matplotlib** — Visualization

See `environment.yml` for the full list.
