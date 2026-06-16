# GA Comparison Under Varying Objectives (TEVC Journal Extension)

**Experiment 1 of the SAMOS journal extension:** Benchmark 8 multi-objective genetic algorithms across the **WFG** continuous suite and the **EvoXBench** NAS suites (CIFAR-10 / ImageNet-1K), grouped into 2-, 3-, and 4-objective sub-experiments.

This branch (`SAMOS-NAS-TEVC`) contains new experiments for the journal extension. The original PPSN-26 surrogate-assisted (SAMOS) code remains in [`depreciated/`](depreciated/) — see [`depreciated/README.md`](depreciated/README.md) for that documentation.

## Algorithms

| Key | Label | pymoo class |
|-----|-------|-------------|
| `random`    | Random        | `RandomGA` (`strategy/algorithm/algorithms.py`) |
| `nsga2`     | NSGA-II       | `NSGA2` |
| `nsga3`     | NSGA-III      | `NSGA3` (ref_dirs) |
| `moead`     | MOEA/D        | `MOEAD` (pop_size = len(ref_dirs)) |
| `sms-emoa`  | SMS-EMOA      | `SMSEMOA` |
| `rvea`      | RVEA          | `RVEA` (ref_dirs) |
| `age-moea`  | AGE-MOEA      | `AGEMOEA` |
| `age-moea2` | AGE-MOEA-II   | `AGEMOEA2` |

For the reference-direction methods (`nsga3`, `moead`, `rvea`) Das-Dennis directions are generated with the partition count chosen so `len(ref_dirs)` is nearest to `n_var` (and `≥ pop_size_floor`); that count becomes the effective population size.

## Sub-experiments

All configured in [`config/experiment_obj_ga.yaml`](config/experiment_obj_ga.yaml). 20 seeds, `n_gen = 100`, `pop_size = max(n_var, 12)`.

| Sub-exp | n_obj | WFG (n_var) | EvoXBench C10MOP pids | EvoXBench IN1KMOP pids |
|---------|-------|-------------|-----------------------|------------------------|
| 1.1 | 2 | wfg1–9 (12) | 1, 8         | 1, 4, 7 |
| 1.2 | 3 | wfg1–9 (14) | 2, 3, 9      | 3, 6 |
| 1.3 | 4 | wfg1–9 (16) | 4, 5, 6, 7   | 9 |

## Installation

```bash
conda env create -f environment.yml
conda activate SAMOS_311
```

EvoXBench data files go in `problem/data/evoxbench/` (see the [EvoXBench docs](https://github.com/EMI-Group/evoxbench)).

## Running

### Single run

```bash
# WFG
python experiment_obj_ga.py --benchmark wfg --problem wfg1 --n_obj 2 \
    --method nsga2 --seed 0 --config config/experiment_obj_ga.yaml

# EvoXBench
python experiment_obj_ga.py --benchmark c10mop --pid 1 --n_obj 2 \
    --method nsga2 --seed 0 --config config/experiment_obj_ga.yaml
```

A run is **skipped if its pickle already exists**. Pass `--overwrite` to force a redo.

### SLURM cluster

Each array task runs one seed across every method × problem/pid in the sub-experiment. The job stages code + existing results onto the node-local scratch disk (`/local/...`), runs, copies results back, and cleans up scratch.

```bash
# WFG (one sub-experiment at a time)
sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=2 OBJ_GA_WFG.sbatch
sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=3 OBJ_GA_WFG.sbatch
sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=4 OBJ_GA_WFG.sbatch

# EvoXBench
sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=2 OBJ_GA_EVOXBENCH.sbatch
sbatch --export=ALL,BENCHMARK=in1kmop,N_OBJ=2 OBJ_GA_EVOXBENCH.sbatch
# ... and N_OBJ=3, N_OBJ=4
```

Selective single-seed rerun (force overwrite):

```bash
sbatch --array=5 --export=ALL,BENCHMARK=wfg,N_OBJ=2,OVERWRITE=1 OBJ_GA_WFG.sbatch
```

### Checking coverage

`seed_coverage.py` reports how many seeds are present per experiment/algorithm:

```bash
python seed_coverage.py                      # table to console
python seed_coverage.py --format csv
python seed_coverage.py --format json --output coverage.json
```

## Analysis

```bash
python analyse_obj_ga.py --n_obj all --benchmark all \
    --config config/experiment_obj_ga.yaml --output_dir results/obj_ga/analysis
```

This:
1. Builds a shared **Pareto approximation** per EvoXBench problem (pooling all methods + seeds) and saves it to `pareto_approx.pkl`.
2. Recomputes EvoXBench HV / IGD+ trajectories against that approximation (WFG indicators are computed live during the run).
3. Emits **LaTeX tables** (mean$_{\text{std}}$, best bolded, Holm-corrected Wilcoxon vs. NSGA-II) at the `metric_checkpoints` generations.
4. Plots **convergence** curves (HV / IGD+ vs. generation, mean ± std) organized into subdirectories:
   - `plots/hv_convergence/` — HV curves per problem
   - `plots/igd_convergence/` — IGD+ curves per problem
   - `plots/hv_convergence/*_combined.png` — Combined HV plot with subplots for all n_obj (2, 3, 4)
   - `plots/igd_convergence/*_combined.png` — Combined IGD+ plot with subplots for all n_obj
5. Plots **50% attainment surfaces** (via `moocore`) for 2-objective problems only, in `plots/attainment/`.

Pass `--force` to rebuild cached Pareto approximations.

## Results layout

```
results/obj_ga/
  wfg/{n_obj}_obj/{problem}/{method}/ga_obj/seed_{seed}.pkl
  evoxbench/{suite}/pid{pid}/{method}/ga_obj/seed_{seed}.pkl
  evoxbench/{suite}/pid{pid}/ga_obj/pareto_approx.pkl
  analysis/
    tables/*.tex                          # LaTeX tables per (benchmark, n_obj, metric)
    plots/
      hv_convergence/                     # HV curves
        wfg_*.png
        evox_*.png
        *_combined.png                    # Combined 2/3/4-obj subplots
      igd_convergence/                    # IGD+ curves
        wfg_*.png
        evox_*.png
        *_combined.png
      attainment/                         # 50% EAF plots (2-obj only)
        wfg_*.png
        evox_*.png
```

Each per-seed pickle holds, with one entry per generation:

```python
{
  'var_archive':      [ndarray, ...],   # ND archive, decision space
  'obj_archive':      [ndarray, ...],   # ND archive, objective space
  'test_obj_archive': [ndarray, ...],   # true objectives
  'indicators':       [{'hv':..., 'igd_plus':...}, ...],  # WFG live; EvoXBench filled post-hoc
  'config':           dict,
  'time':             float,
}
```

## Project structure

```
SAMOS-NAS-TEVC/
├── experiment_obj_ga.py            # per-run driver (one method × problem × seed)
├── analyse_obj_ga.py               # Pareto approx, indicators, tables, plots
├── seed_coverage.py                # seed-completion reporting tool
├── config/experiment_obj_ga.yaml   # algorithms, seeds, budget, sub-experiments
├── OBJ_GA_WFG.sbatch               # SLURM array: WFG suite
├── OBJ_GA_EVOXBENCH.sbatch         # SLURM array: EvoXBench suites
├── environment.yml                 # conda environment (Python 3.11)
│
├── strategy/                       # algorithms, operators, samplers, callbacks
│   ├── algorithm/algorithms.py     # RandomGA, NSGA-II wrappers
│   ├── operations/                 # crossover, mutation
│   ├── genetics/                   # encoding & deduplication
│   └── sampler.py
│
├── problem/                        # benchmark definitions
│   ├── pymoo/benchmark_utils.py    # build_problem, get_pareto_front, default_ref_point
│   ├── evoxbench/                  # EvoXBench NAS wrappers
│   └── data/                       # benchmark data (not versioned)
│
├── analysis/                       # plotting & reporting library
│   ├── plotter.py                  # convergence + attainment plots
│   ├── convergence.py              # Pareto approximation + indicator recomputation
│   └── latex_table_generator.py    # LaTeX tables
│
└── depreciated/                    # retired PPSN-26 SAMOS codebase
```

## Key dependencies

- **[pymoo](https://pymoo.org/) 0.6.1.x** — multi-objective optimization framework
- **[moocore](https://github.com/multi-objective/moocore)** — empirical attainment functions / HV
- **[EvoXBench](https://github.com/EMI-Group/evoxbench)** — NAS benchmark suites
- **scipy** — Wilcoxon significance testing
- **matplotlib** — visualization

See `environment.yml` for the full list.
