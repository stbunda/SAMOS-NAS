# Experiment 2 — GA Performance Under Fixed Predictors (`obj_ga_pred`)

**How well does each multi-objective GA exploit a *fixed* surrogate predictor?**
This experiment simulates the inner search of a surrogate-assisted optimiser: a
predictor is trained once on a random design-of-experiments (DOE), then each GA
optimises against that frozen predictor and is scored on the **true** objectives of
the front it finds.

It reuses the same 8 algorithms, WFG/EvoXBench problems, objective groups (2/3/4),
seeds, and HV/IGD+ indicators as [Experiment 1](../obj_ga/). The new axis is the
**predictor** ∈ `{xgboost, rf, rbf}`.

## Protocol (per run = one benchmark × problem/pid × predictor × method × seed)

1. **DOE** — sample `n_doe = 100` random configurations and evaluate their **true**
   objectives (analytic for WFG; `true_eval=True` for EvoXBench).
2. **Fit** one surrogate per objective on the DOE, for the chosen predictor. The
   surrogate is **trained once and frozen** for the whole run.
3. **Warm-start** the GA from the best DOE members (rank + crowding) and run it for
   `n_gen = 100` generations against the **frozen surrogate problem** (all objectives
   predicted). `random` is the exception — it re-samples fresh candidates each
   generation, so it is *not* warm-started.
4. **Score** — the GA keeps a non-dominated archive in **predicted** space; each
   generation that archive's **true** objectives are recorded. HV/IGD+ are computed
   on the true values (live for WFG; post-hoc against a pooled Pareto approximation
   for EvoXBench).

## Predictors

Configured in [`config/experiment_obj_ga_pred.yaml`](config/experiment_obj_ga_pred.yaml).
`builder` is a class key in [`strategy/surrogate/models`](../../strategy/surrogate/models/__init__.py).

| Key | Label | Builder | Params |
|-----|-------|---------|--------|
| `xgboost` | XGBoost       | `XGBoost`   | `n_estimators=100` |
| `rf`      | Random Forest | `RFR`       | `n_estimators=100` |
| `rbf`     | RBF           | `RBF_Cubic` | `smoothing=1e-3` |

One surrogate is fitted **per objective** (seeded with `seed + obj_index`).

## Running

```bash
# Single run (WFG)
python experiments/obj_ga_pred/experiment_obj_ga_pred.py \
    --benchmark wfg --problem wfg1 --n_obj 2 --predictor rbf --method nsga2 --seed 0

# Single run (EvoXBench)
python experiments/obj_ga_pred/experiment_obj_ga_pred.py \
    --benchmark c10mop --pid 1 --n_obj 2 --predictor rf --method nsga2 --seed 0
```

A run is **skipped if its pickle already exists**; pass `--overwrite` to force a redo.

### SLURM cluster

One array task == one seed; each task loops `PREDICTOR × METHOD × {problem|pid}`.

```bash
sbatch --export=ALL,BENCHMARK=wfg,N_OBJ=2     experiments/obj_ga_pred/OBJ_GA_PRED_WFG.sbatch
sbatch --export=ALL,BENCHMARK=c10mop,N_OBJ=2  experiments/obj_ga_pred/OBJ_GA_PRED_EVOXBENCH.sbatch
# Restrict to one predictor: add ,PREDICTORS=xgboost
```

## Analysis

```bash
python experiments/obj_ga_pred/analyse_obj_ga_pred.py \
    --n_obj all --benchmark all --predictor all \
    --output_dir results/obj_ga_pred/analysis
```

This builds one EvoXBench Pareto approximation per `(suite, pid)` **pooled across all
predictors + methods + seeds** (a common scoring reference), recomputes EvoXBench
HV/IGD+ trajectories against it, and emits per-predictor LaTeX tables and convergence
/ attainment plots. Outputs are namespaced by predictor:

```
results/obj_ga_pred/analysis/
  tables/{predictor}/{wfg|evox}_{n_obj}obj_{hv|igd_plus}_table.tex
  plots/{predictor}/hv_convergence/   igd_convergence/   attainment/
```

Pass `--force` to rebuild cached Pareto approximations.

## Results layout

```
results/obj_ga_pred/
  wfg/{n_obj}_obj/{problem}/{predictor}/{method}/ga_obj_pred/seed_{seed}.pkl
  evoxbench/{suite}/pid{pid}/{predictor}/{method}/ga_obj_pred/seed_{seed}.pkl
  evoxbench/{suite}/pid{pid}/ga_obj_pred/pareto_approx.pkl   # pooled across predictors
```

Each per-seed pickle (one entry per generation):

```python
{
  'var_archive':      [ndarray, ...],   # ND archive, decision space
  'obj_archive':      [ndarray, ...],   # ND archive, PREDICTED objectives (guides search)
  'test_obj_archive': [ndarray, ...],   # TRUE objectives of the archive (scoring basis)
  'indicators':       [{'hv':..., 'igd_plus':...}, ...],  # WFG live; EvoXBench post-hoc
  'config':           dict,             # incl. predictor, n_doe, predictor_builder, train_rmse
  'time':             float,
}
```

## Key building blocks reused

- Surrogate models + registry: [`strategy/surrogate/models`](../../strategy/surrogate/models/__init__.py)
- Inner surrogate problems: [`SurrogateProblemMOO`](../../problem/pymoo/surrogate_problem.py) (WFG),
  [`SurrogateProblemEvox`](../../problem/evoxbench/surrogate_problem.py) (EvoXBench)
- Warm-start pattern: [`SAMOSMinimal`](../../strategy/surrogate/samos_minimal.py)
- Shared analysis library: [`analysis/`](../../analysis/) (`convergence`, `plotter`, `latex_table_generator`)
