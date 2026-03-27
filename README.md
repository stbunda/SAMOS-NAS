# SAMOS-NAS

**Surrogate-Assisted Multi-Objective Search** — a framework for benchmarking surrogate-assisted evolutionary multi-objective optimization on both Neural Architecture Search (NAS) benchmarks and continuous MOO test suites.

SAMOS-NAS combines evolutionary multi-objective optimization with surrogate models (Random Forest, XGBoost, Gaussian Processes, and more) to efficiently solve expensive optimization problems. By predicting objective values with cheap surrogate evaluations and only selectively querying the true (expensive) objective, SAMOS dramatically reduces the computational cost of optimization.

The framework supports three benchmark families:

| Benchmark | Entry Point | Search Space |
|-----------|------------|--------------|
| **NASBench-101** | `main_nasbench101_baseline.py` | 26-dim integer (5 ops + 21 edges) |
| **NASBench-201** | `main_nasbench201_baseline.py` | 6-dim categorical (15,625 architectures) |
| **Continuous MOO** | `main_pymoo_benchmark.py` | WFG, ZDT, DTLZ test suites |

## Objectives

### NASBench-101

| Objective | Definition |
|-----------|------------|
| **Validation Error** | `1.0 - validation_accuracy @ epoch 12` (minimized) |
| **Model Size** | `(n_params - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS)` (minimized) |

### NASBench-201

| Objective | Definition |
|-----------|------------|
| **Validation Error** | `1.0 - validation_accuracy @ epoch 200` (minimized) |
| **FLOPs** | Normalized FLOPs (dataset-specific min/max scaling, minimized) |

Supported datasets: `cifar10-valid`, `cifar100`, `ImageNet16-120`

#### HW-NAS-Bench Extension

NASBench-201 is extended with [HW-NAS-Bench](https://github.com/RICE-EIC/HW-NAS-Bench) hardware metrics, replacing or supplementing FLOPs with device-specific latency and energy measurements. These are looked up directly from the database at zero training cost and can be used as real objectives in the inner surrogate loop.

| Hardware Objective | Key | Device |
|--------------------|-----|--------|
| `edgegpu_latency` | `edge-lat` | Edge GPU inference latency |
| `edgegpu_energy` | `edge-ene` | Edge GPU energy consumption |
| `raspi4_latency` | `raspi-lat` | Raspberry Pi 4 inference latency |
| `pixel3_latency` | `pix3-lat` | Google Pixel 3 inference latency |
| `eyeriss_latency` | `eyrs-lat` | Eyeriss accelerator latency |
| `eyeriss_energy` | `eyrs-ene` | Eyeriss accelerator energy |
| `eyeriss_arithmetic_intensity` | `eyrs-ai` | Eyeriss arithmetic intensity |
| `fpga_latency` | `fpga-lat` | FPGA inference latency |
| `fpga_energy` | `fpga-ene` | FPGA energy consumption |

Hardware objectives can be selected via the `--real_obj` argument:

```bash
python main_nasbench201_baseline.py \
    --samos_xgb \
    --dataset cifar10-valid \
    --predict_obj val_err \
    --real_obj edgegpu_latency
```

### Continuous MOO (WFG, ZDT, DTLZ)

All objectives defined by the respective pymoo problem. Configurable number of objectives via `--n_obj`.

## Implemented Algorithms

### Baselines

| Method | Description |
|--------|-------------|
| `random` | One-shot uniform random sampling |
| `random_ga` | Generational random search (pymoo `GeneticAlgorithm`) |
| `nsga2` | NSGA-II with default operators |
| `nsga2-uniform` | NSGA-II with 2-point crossover + uniform mutation |
| `nsga2-xo-single` | NSGA-II with 2-point crossover + single-point mutation |
| `nsga2-no-xo-single` | NSGA-II with no crossover + single-point mutation |

### Surrogate-Assisted

| Method | Description |
|--------|-------------|
| `samos-rfr` | SAMOS with Random Forest surrogate |
| `samos-xgb` | SAMOS with XGBoost surrogate |
| `samos-xgb-xo-uniform-{50,200,1000}` | SAMOS-XGBoost variants (inner pop size) |
| `samos-xgb-xo-single-{50,200,1000}` | SAMOS-XGBoost + single-point mutation variants |
| `samos-xgb-no-xo-single-{50,200,1000}` | SAMOS-XGBoost (no crossover) variants |
| `gpsaf-default` / `gpsaf-rfr` / `gpsaf-xgb` | GPSAF (pysamoo) with various surrogates |
| `ssa-nsga2-default` / `ssa-nsga2-rfr` / `ssa-nsga2-xgb` | SSA-NSGA-II (pysamoo) with various surrogates |
| `parego` | ParEGO — Tchebycheff scalarization + Bayesian optimization |
| `cobra` | IOC-SAMO-COBRA — Constraint-dependent optimization with RBF surrogates |
| `mosmac` | SMAC3 `MultiObjectiveFacade` |

## How SAMOS Works

Each outer generation of SAMOS performs the following loop:

1. **Fit surrogates** — Train one model per objective on the evaluated archive
2. **Warm-start inner NSGA-II** — Seed the inner population with top archive members + fresh random samples
3. **Run inner NSGA-II** — Evolve candidate solutions on the surrogate problem for `n_gen_inner` generations
4. **Deduplicate** — Remove candidates already present in the evaluated archive (by canonical hash for NAS, vector comparison for continuous)
5. **Subset selection** — Pick `n_infill` diverse candidates from the inner Pareto front
6. **Evaluate** — Query the true objective function (the only expensive step)
7. **Update archive** — Merge new evaluations into the population

## Search Spaces

### NASBench-101

The NASBench-101 cell is encoded as a 26-dimensional integer vector:

- **5 operation genes** — each in `{0, 1, 2}` mapping to `conv3x3-bn-relu`, `conv1x1-bn-relu`, `maxpool3x3`
- **21 edge genes** — binary values forming the upper-triangular adjacency matrix of a 7-node DAG

Architectures are validated for connectivity (input → output path must exist) and sparsity (≤ 9 edges).

### NASBench-201

The NASBench-201 cell is encoded as a 6-dimensional categorical vector:

- **6 edge genes** — each in `{0, 1, 2, 3, 4}` selecting an operation in a 4-node cell DAG
- All 15,625 combinations are valid (no topology constraints)

### Continuous MOO

Standard pymoo problem definitions with configurable dimensionality:
- **WFG 1–9** — `n_var = 2(n_obj - 1) + 10` by default
- **ZDT 1–6** — 2-objective, 30 variables
- **DTLZ 1–7** — Scalable objectives and variables

## Project Structure

```
SAMOS-NAS/
├── main_nasbench101_baseline.py       # NASBench-101 experiments
├── main_nasbench201_baseline.py       # NASBench-201 experiments
├── main_pymoo_benchmark.py            # Continuous MOO benchmark experiments
├── analyze_nasbench101_baseline.py    # NASBench-101 post-processing
├── analyze_nasbench201_baseline.py    # NASBench-201 post-processing
├── analyze_pymoo_benchmark.py         # MOO benchmark post-processing
├── environment.yml                    # Conda environment (Python 3.11)
│
├── strategy/                          # Core optimization framework
│   ├── algorithm/
│   │   ├── algorithms.py              # RandomGA, NSGA-II wrappers
│   │   ├── gpsaf.py                   # GPSAF wrapper (pysamoo)
│   │   ├── ssansga2.py                # SSA-NSGA-II wrapper (pysamoo)
│   │   ├── parego.py                  # ParEGO (Tchebycheff + BO)
│   │   ├── cobra.py                   # IOC-SAMO-COBRA wrapper
│   │   ├── ioc_samo_cobra/            # COBRA algorithm internals (RBF surrogates)
│   │   └── bo/                        # Bayesian optimization components
│   │       ├── base.py                # BO infrastructure
│   │       ├── surrogate.py           # GP surrogate for BO
│   │       ├── acquisition.py         # Expected Improvement
│   │       └── solver.py              # LBFGS-B and EA solvers
│   ├── surrogate/
│   │   ├── samos.py                   # Full SAMOS with logging
│   │   ├── samos_minimal.py           # Lightweight SAMOS (pymoo drop-in)
│   │   ├── subset_selection.py        # Diversity-based infill selection
│   │   └── models/                    # Surrogate models
│   │       ├── rf.py                  # Random Forest (n_estimators=20)
│   │       ├── xgboost.py             # XGBoost (n_estimators=100)
│   │       ├── kriging.py             # Kriging / GPR (sklearn)
│   │       ├── gp.py                  # Gaussian Process
│   │       ├── gpr_enhanced.py        # Enhanced GPR
│   │       ├── knn.py                 # K-Nearest Neighbors
│   │       ├── carts.py               # CART ensemble (n_tree=1000)
│   │       ├── mlp.py                 # Multi-Layer Perceptron (PyTorch)
│   │       ├── rnn.py                 # RNN / LSTM / GRU (PyTorch)
│   │       └── lightning_model.py     # PyTorch Lightning training wrapper
│   ├── operations/
│   │   ├── crossover.py               # TwoPointCrossover, NoCrossover (101 & 201)
│   │   └── mutation.py                # UniformMutation, SinglePointMutation (101 & 201)
│   ├── genetics/
│   │   ├── nasbench_genetic_base.py   # NASBENCH101 / NASBENCH201 genome wrappers
│   │   ├── duplicate.py               # Canonical architecture deduplication
│   │   ├── program_config.py          # BenchConfig / DartsConfig
│   │   └── nasbench101_lib/           # NASBench-101 ModelSpec & graph utilities
│   ├── sampler.py                     # ValidRandomSampling101, ValidRandomSampling201
│   └── callbacks.py                   # NASArchiveCallback, PymooBenchmarkCallback
│
├── problem/                           # Problem definitions & data
│   ├── nasbench101/
│   │   ├── baseline_problem.py        # Real evaluation (pymoo Problem)
│   │   ├── surrogate_problem.py       # Surrogate-based inner problem
│   │   ├── precompute_lut.py          # Canonical architecture lookup table
│   │   └── utils.py                   # Constants, vector ↔ arch_str conversion
│   ├── nasbench201/
│   │   ├── baseline_problem.py        # Real evaluation (pymoo Problem)
│   │   ├── surrogate_problem.py       # Surrogate-based inner problem
│   │   └── utils.py                   # Constants, dataset-specific normalization
│   ├── pymoo/
│   │   ├── surrogate_problem.py       # Surrogate problem for continuous MOO
│   │   └── benchmark_utils.py         # Problem instantiation, Pareto fronts, ref points
│   └── data/                          # Pre-computed databases (not versioned)
│
├── analysis/                          # Visualization & reporting
│   ├── plotter.py                     # HV / IGD+ trajectory plots
│   └── latex_table_generator.py       # Publication-ready LaTeX tables
│
├── utils/                             # Shared utilities
│   ├── logger.py                      # Experiment logging & archiving
│   ├── logger2.py                     # Alternative logger
│   ├── sweep_logger.py                # Sweep experiment logger
│   └── utils.py                       # Data augmentation helpers
│
├── results/                           # Experiment outputs (per benchmark / method / seed)
│
├── NASBENCH101_BASELINE.sbatch        # SLURM: NASBench-101 experiments
├── ANALYZE_NASBENCH101_BASELINE.sbatch# SLURM: NASBench-101 analysis
├── PRECOMPUTE_LUT.sbatch              # SLURM: lookup table preprocessing
├── WFG_BENCHMARK.sbatch               # SLURM: WFG benchmark experiments
├── WFG_EXTENDED_BENCHMARK.sbatch      # SLURM: extended WFG experiments
├── submit_wfg_benchmark.sh            # SLURM: sequential WFG job submission
└── submit_wfg_extended_benchmark.sh   # SLURM: sequential extended WFG submission
```

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- For NAS benchmarks: database files placed in `problem/data/` (e.g., `data_nasbench101.pkl`)

### Installation

```bash
conda env create -f environment.yml
conda activate SAMOS_311
```

### Build the Lookup Table (NASBench-101 only, first time)

The lookup table maps canonical architecture forms to hash strings for fast deduplication:

```bash
python -m problem.nasbench101.precompute_lut
```

This produces `problem/data/nasbench101_lut.pkl`.

### Run Experiments

**NASBench-101:**

```bash
python main_nasbench101_baseline.py \
    --samos_xgb \
    --seeds 0 \
    --pop_size 20 \
    --n_gen 50 \
    --n_doe 20 \
    --n_infill 20 \
    --n_gen_inner 20 \
    --inner_pop_size 200
```

**NASBench-201:**

```bash
python main_nasbench201_baseline.py \
    --samos_xgb \
    --seeds 0 \
    --pop_size 20 \
    --n_gen 50 \
    --dataset cifar10-valid
```

**Continuous MOO (e.g., WFG):**

```bash
python main_pymoo_benchmark.py \
    --problem wfg1 wfg2 wfg3 \
    --methods nsga2 samos-xgb parego \
    --seeds 0 1 2 \
    --pop_size 20 \
    --n_gen 60
```

### Analyze Results

```bash
python analyze_nasbench101_baseline.py --pop_size 20 --n_gen 50
python analyze_nasbench201_baseline.py
python analyze_pymoo_benchmark.py
```

These generate hypervolume / IGD+ trajectory plots and LaTeX summary tables under `results/`.

## Running on a SLURM Cluster

```bash
# 1. Build lookup table (NASBench-101)
sbatch PRECOMPUTE_LUT.sbatch

# 2. Launch NASBench-101 experiments (13 methods × 30 seeds)
sbatch NASBENCH101_BASELINE.sbatch

# 3. Launch WFG benchmark experiments
sbatch WFG_BENCHMARK.sbatch
# or for the extended set:
bash submit_wfg_extended_benchmark.sh

# 4. Generate plots & tables after experiments finish
sbatch ANALYZE_NASBENCH101_BASELINE.sbatch
```

## CLI Reference

### Common Arguments (all entry points)

| Argument | Default | Description |
|----------|---------|-------------|
| `--seeds` | `0–9` | List of random seeds |
| `--pop_size` | `20` | Population size |
| `--n_gen` | `50` | Number of outer generations |
| `--n_doe` | `pop_size` | Initial design-of-experiments size (SAMOS) |
| `--n_infill` | `pop_size` | Real evaluations per SAMOS generation |
| `--n_gen_inner` | `20` | Inner NSGA-II generations (SAMOS) |
| `--inner_pop_size` | `pop_size × 10` | Inner NSGA-II population size (SAMOS) |
| `--warm_start_ratio` | `0.75` | Archive fraction to warm-start inner NSGA-II |
| `--experiment_name` | auto | Results subdirectory name |
| `--overwrite` | `False` | Re-run even if result exists |

### NASBench-101 Specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--elim_dupes` | `arch_str` | Duplicate elimination: `arch_str` (canonical hash) or `pymoo_default` |
| `--predict_obj` | `val_err_12` | Objectives approximated by surrogates |
| `--real_obj` | `n_params` | Objectives evaluated directly in inner loop |

### NASBench-201 Specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `cifar10-valid` | Dataset: `cifar10-valid`, `cifar100`, or `ImageNet16-120` |

### Continuous MOO Specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--problem` | `wfg1` | Problem name(s): `wfg1`–`wfg9`, `zdt1`–`zdt6`, `dtlz1`–`dtlz7` |
| `--methods` | `random nsga2 samos-xgb` | List of algorithm names to run |
| `--n_obj` | `2` | Number of objectives |
| `--n_var` | auto | Number of decision variables (overrides problem default) |
| `--proxy_obj_indices` | all | Indices of objectives to approximate with surrogates |
| `--no_plot` | `False` | Skip plot generation |

## Available Surrogate Models

| Model | Module | Type |
|-------|--------|------|
| Random Forest | `strategy/surrogate/models/rf.py` | Ensemble (20 trees) |
| XGBoost | `strategy/surrogate/models/xgboost.py` | Gradient Boosting (100 estimators) |
| Kriging / GPR | `strategy/surrogate/models/kriging.py` | Probabilistic (sklearn) |
| Gaussian Process | `strategy/surrogate/models/gp.py` | Probabilistic |
| Enhanced GPR | `strategy/surrogate/models/gpr_enhanced.py` | Probabilistic (advanced) |
| MLP | `strategy/surrogate/models/mlp.py` | Neural Network (PyTorch) |
| RNN / LSTM / GRU | `strategy/surrogate/models/rnn.py` | Recurrent NN (PyTorch) |
| KNN | `strategy/surrogate/models/knn.py` | Instance-based |
| CART | `strategy/surrogate/models/carts.py` | Tree ensemble (1000 trees) |

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
