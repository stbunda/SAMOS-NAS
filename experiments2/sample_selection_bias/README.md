# Sample-selection bias in SAMOS

**Question.** Under a hard model-size budget, an architecture that exceeds it
cannot be run, so its error and every hardware-dependent metric are
unobservable. SAMOS discards those points, and its objective surrogates
therefore train on the feasible population only. How much does that cost, and
does the cost depend on how the size budget enters the formulation?

Benchmark: **NB201** (`c10mop`, pid 7, 15 625 architectures, exhaustively
enumerable so every reference number here is exact). Two objectives
throughout, CDP as the constraint handler, hard gate always on.

## Design

The gate is **always `#Params <= tau_P`**, in every cell. That is the physical
premise: exceeding the size budget makes the model unrunnable. What varies:

| axis | levels |
|---|---|
| `tightness` | four `#Params` budgets: T1 4.7%, T2 14.2%, T3 25.3%, T4 51.1% of the space evaluable |
| `role` | where `#Params` sits in the formulation the optimizer sees |
| `hw` | which hardware metric plays the other budget, each at its own ~10% point |
| `arm` | `feasible` (what SAMOS does today) vs `oracle` (counterfactual) |

Roles, with `Ha` the hardware metric and `Hb` the other device's:

| role | objectives | declared constraints |
|---|---|---|
| `obj` | Err, **#Params** | Ha |
| `constr` | Err, Ha | **#Params** |
| `multi` | Err, Hb | **#Params**, Ha |

The **arms** differ in one thing only. Both gate the archive identically; the
`oracle` arm additionally trains the *objective* surrogates on the gated-out
points' true objectives (`F_oracle`, the outer problem's unmasked F). That is
physically impossible — the whole premise is that those values cannot be
measured — so it is the upper bound the realistic arm is scored against, never
a proposed method. The feasible-to-oracle gap **is** the bias impact.

Two details that keep the `feasible` arm honest:

- The **constraint** surrogate trains on archive ∪ rejection log, but only for
  columns still measurable when gated. `#Params` is a static structural count,
  known even for a model that cannot run; a hardware metric of an unrunnable
  model is not. Without that split the `feasible` arm would quietly receive
  oracle information through its latency constraint model
  (`SAMOS2.constr_observable_when_gated`).
- Thresholds are **midpoints between attained values**, not percentiles. NB201's
  `#Params` has 59 distinct values with plateaus of up to 2 421 architectures,
  so percentile thresholds collapse (its 5th and 10th percentile are the same
  number) and land exactly on a tie.

## Grid size

10 runnable `(role, hw)` pairs × 4 tightness × 2 arms = **80 cells**, 20 seeds
= 1 600 runs at `pop_size=20`, `n_evals=1200`.

Not 12 pairs: **arithmetic intensity is antagonistic to a small-model budget**
(`spearman(#Params, AI) = 0.78`). Among architectures under the T1–T3 size
budgets the largest attainable AI is 0.633, below any floor worth calling a
budget, so wherever AI is a *constraint* next to the size gate — the `obj` and
`multi` roles — the joint feasible set is **exactly empty** and HV is
identically 0. Those cells are excluded (`config.EXCLUDED`). AI is kept in the
`constr` role, where it is the *objective*: that cell is the most informative
in the grid, being the only place the gate removes precisely the good end of an
objective. Being maximize-better it is minimized as `1 - AI`, consistently in
the problem, the probe set and the indicator.

## Reading the results

`bias_gap = 1 - HV_feasible / HV_oracle` is the headline. Use it, not raw HV,
to compare **across roles**: the three roles search three different objective
spaces (Err×#Params, Err×Ha, Err×Hb), so their HVs are not on one scale, while
the arm-to-arm ratio within a cell is scale-free. `hv_frac` (HV over the exact
constrained-front HV, recomputed from the full enumeration under `true_eval`)
gives an absolute reading within a role.

`mae_gated_*` is the mechanism: each generation the live surrogates are scored
against a fixed 2 000-architecture probe set, split by which side of the gate
each architecture is on. The `gated` side is the region the realistic arm never
observes, and the feasible-vs-oracle gap there is the bias itself — the HV gap
is its consequence.

## Running

```bash
# regenerate/verify the thresholds in config.py
python experiments2/sample_selection_bias/derive_taus.py --check

# wiring self-check (includes a regression check on the shared-code edits)
python experiments2/sample_selection_bias/test_bias_wiring.py

# one cell, locally
python experiments2/sample_selection_bias/run_bias.py \
    --tightness T2 --role constr --hw edgegpu_latency --seeds 0 1 2

# the campaign (80 array tasks x 20 seeds)
sbatch experiments2/sample_selection_bias/BIAS_EVOXBENCH.sbatch

# tables + figures
python experiments2/sample_selection_bias/analyse_bias.py
```

Smoke tests go to a dedicated folder, never `results2/`:
`--results_root smoke_tests/sample_selection_bias`.

## Files

| file | role |
|---|---|
| `config.py` | the grid: thresholds, roles, hardware metrics, exclusions, per-cell spec |
| `derive_taus.py` | regenerates every threshold from the NB201 enumeration; `--check` verifies `config.py` |
| `_bias.py` | probe set + the probe-aware callback (surrogate accuracy per side of the gate) |
| `run_bias.py` | campaign runner, one `(tightness, role, hw, arm)` cell per invocation |
| `analyse_bias.py` | `runs.csv`, `cells.csv`, and the three figures |
| `test_bias_wiring.py` | assert-based self-check of the three shared-code seams |
| `BIAS_EVOXBENCH.sbatch` | 80-task array job, one seed per rsync-back |

## Shared code touched

All three seams default to the previous behaviour, and
`test_bias_wiring.check_legacy_rejection_log` is the regression proof that the
`scenario_run` campaign's rejection log is byte-identical to before.

- `problem/evoxbench/constrained_problem.py` — `gate_index`/`gate_threshold`
  (gate on one metric that need not be a declared constraint, publishing
  `G_gate` and the unmasked `F_oracle`), and `flip_obj_pos`.
- `problem/evoxbench/callbacks.py` — `flip_obj_pos`, so the indicator is
  measured in the space the run actually searched.
- `strategy/surrogate/samos2.py` — `gate_g_fn` now takes precedence over `G`;
  the rejection log records declared-constraint violations separately from the
  gate violation; `constr_observable_when_gated`; `oracle_train_key`.
