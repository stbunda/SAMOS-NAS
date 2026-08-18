# Sample-selection bias in SAMOS

**Question.** Under a hard model-size budget, an architecture that exceeds it
cannot be run, so its error and every hardware-dependent metric are
unobservable. SAMOS discards those points, and its objective surrogates
therefore train on the feasible population only. How much does that cost, and
does the cost depend on how the size budget enters the formulation?

Two benchmarks, same grid, selected with `--space`:

| space | kind | size | reference front | why |
|---|---|---|---|---|
| **NB201** (`c10mop`/7) | tabular | 15 625, enumerable | **exact** | every number is exact |
| **MobileNetV3** (`in1kmop`/9) | surrogate (MLP predictor) | 21 vars, ~1e20 | sampled (full 1M stage-1 cache) | NB201 saturates; this does not |

Two objectives throughout, CDP as the constraint handler, hard gate always on.

MobileNetV3 is the only surrogate space in the suite carrying a real hardware
metric. FLOPs is never used as a budget on either space: like `#Params` it is a
static structural count, observable even for a model that cannot be run, so it
cannot carry the evaluability premise. It appears only as an objective.

## Design

The gate is **always `#Params <= tau_P`**, in every cell. That is the physical
premise: exceeding the size budget makes the model unrunnable. What varies:

| axis | levels |
|---|---|
| `space` | NB201 (exact) or MobileNetV3 (surrogate, unsaturated) |
| `tightness` | four `#Params` budgets. NB201: 4.7 / 14.2 / 25.3 / 51.1% evaluable (its `#Params` plateaus force the offsets). MobileNetV3: 5 / 15 / 25 / 50% |
| `role` | where `#Params` sits in the formulation the optimizer sees |
| `hw` | which hardware metric plays the other budget, each at its own ~10% point. NB201: 4 metrics; MobileNetV3: `Latency` only |
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

**NB201**: 10 runnable `(role, hw)` pairs × 4 tightness × 2 arms = **80 cells**,
20 seeds = 1 600 runs at `pop_size=20`, `n_evals=1200`.

**MobileNetV3**: 3 roles × 1 hw × 4 tightness × 2 arms = **24 cells**, 20 seeds
= 480 runs, same budget. No exclusions — joint (size AND latency) feasibility
runs 1.0 / 2.6 / 4.0 / 6.8% across T1–T4, tight but never empty.

NB201 is not 12 pairs: **arithmetic intensity is antagonistic to a small-model budget**
(`spearman(#Params, AI) = 0.78`). Among architectures under the T1–T3 size
budgets the largest attainable AI is 0.633, below any floor worth calling a
budget, so wherever AI is a *constraint* next to the size gate — the `obj` and
`multi` roles — the joint feasible set is **exactly empty** and HV is
identically 0. Those cells are excluded (`config.SPACES['NB201']['excluded']`).
AI is kept in the
`constr` role, where it is the *objective*: that cell is the most informative
in the grid, being the only place the gate removes precisely the good end of an
objective. Being maximize-better it is minimized as `1 - AI`, consistently in
the problem, the probe set and the indicator.

## How MobileNetV3 differs, and why it is the better test

- **It does not saturate.** ~1e20 architectures against a 1 200-evaluation
  budget, where NB201 is solved to 95% of its exact optimum in 40-120
  evaluations. This is the direct answer to NB201's main caveat.
- **The geometry is close to the mirror image.** `spearman(#Params, Latency) =
  0.16` here, against 0.49-0.78 on NB201: the size budget and the hardware
  budget are nearly INDEPENDENT, so the constraint surrogate cannot infer one
  from the other and the joint feasible region is genuinely a product.
- **Richer fronts.** 63-95 points per objective pair, against NB201's 5-25, so
  HV moves smoothly instead of in lumps.
- **The reference is approximate, but converged.** Non-enumerable, so the
  constrained front is estimated — from the **whole 1M stage-1 cache**, not a
  subsample, re-evaluated under `true_eval` (~295 s once, then cached to npz).
  That is not a detail: a 200k subsample under-estimates `hv_max` by 0.9–3.9
  points, and does so *unevenly* — worst in the tight cells with few feasible
  architectures (T1/obj +3.9 pts vs T4/constr +0.9), which would have
  systematically inflated `hv_frac` exactly where the interesting cells are.
  At 1M it has converged: going 700k → 1M moves `hv_max` by only 0.03–0.23%.
  `hv_frac` can still sit slightly above 1 in principle; it does not in
  practice.
- **Flag when reporting the `multi` role**: `spearman(FLOPs, Latency) = 0.99`
  on this space, and FLOPs is the only metric left to serve as the `multi`
  role's second objective, so there that objective is close to a relabelling of
  the constrained metric. Unavoidable -- MobileNetV3 exposes only four metrics.
- `Err.` normalizes negative here (in1kmop normalizes against its own Pareto
  utopian/nadir). Harmless: the `(1.05, 1.05)` reference point still dominates
  every column (max 1.11 across metrics), and `hv_frac` is a ratio.

## Reading the results

`bias_gap = 1 - HV_feasible / HV_oracle` is the headline. Use it, not raw HV,
to compare **across roles**: the three roles search three different objective
spaces (Err×#Params, Err×Ha, Err×Hb), so their HVs are not on one scale, while
the arm-to-arm ratio within a cell is scale-free. `hv_frac` (HV over the
constrained-front HV, recomputed under `true_eval` — from the full enumeration
on NB201, from the full 1M stage-1 cache on MobileNetV3) gives an absolute
reading within a role.

`bias_gap` is immune to reference quality: `hv_max` is the same constant in
both arms of a cell, so it cancels out of the ratio entirely. Only the absolute
`hv_frac` and `evals_to_target` (which is defined as 95% *of the optimum*)
depend on how good the reference front is — which is why it is worth taking
the full cache rather than a subsample.

`mae_gated_*` is the mechanism: each generation the live surrogates are scored
against a fixed 2 000-architecture probe set, split by which side of the gate
each architecture is on. The `gated` side is the region the realistic arm never
observes, and the feasible-vs-oracle gap there is the bias itself — the HV gap
is its consequence.

## Results — NB201 (1 600 runs: 80 cells × 20 seeds, pop 20, 1 200 evaluations)

**The bias is severe in the surrogate and almost entirely absorbed by the
optimizer.** It converts into wasted evaluations rather than into worse search.

### 1. The surrogate is badly biased, and it scales with tightness

MAE of the error-objective surrogate on the fixed probe set, split by side of
the size budget (mean over all roles and hardware metrics):

| tightness | MAE above budget, `feasible` | MAE above budget, `oracle` | ratio | MAE below budget, `feasible` |
|---|---|---|---|---|
| T1 (4.7% evaluable) | 0.238 | 0.033 | **7.3x** | 0.088 |
| T2 (14.2%) | 0.074 | 0.032 | 2.3x | 0.088 |
| T3 (25.3%) | 0.056 | 0.033 | 1.7x | 0.067 |
| T4 (51.1%) | 0.035 | 0.024 | 1.5x | 0.060 |

The mechanism is exactly as hypothesised: the tighter the budget, the less of
the space the surrogate observes, and the worse it extrapolates above it.

### 2. Pooling the whole population is NOT a free fix

At T1 the oracle arm's surrogate is **worse inside the feasible region** than
the biased one -- MAE 0.113 vs 0.088 -- while being 7x better outside it. With
only 4.7% of draws evaluable, a full-population training set is dominated by
architectures that cannot be built, and a fixed-capacity model spends itself on
the region that does not matter. The effect reverses by T2. Any remedy should
therefore weight or condition on the feasible region, not simply pool; "train
on everything" trades in-region accuracy for out-region accuracy exactly where
the bias is worst.

### 3. The cost is paid in wasted evaluations

Share of real evaluations that were evaluable, paired by seed (Wilcoxon):

| tightness | `feasible` | `oracle` | difference | p |
|---|---|---|---|---|
| T1 | 0.185 | 0.213 | +2.8 pp | 1e-10 |
| T2 | 0.316 | 0.335 | +1.9 pp | 3e-16 |
| T3 | 0.461 | 0.492 | +3.0 pp | 2e-25 |
| T4 | 0.677 | 0.682 | +0.4 pp | 0.005 |

Small but unambiguous and consistent: the biased surrogate does send more
proposals into the unrunnable region.

### 4. Search quality is barely affected

- **Success rate** (any feasible solution found) is unchanged: 100% in `constr`
  at every tightness; in `obj`/`multi` the oracle is at most +3.4 pp ahead
  (T1 `obj` 0.883 vs 0.917) -- about two seeds out of sixty.
- **HV** converges to the same place. The bias gap is visible only early and
  only modestly. Pooled over the `constr` role (320 seed-pairs, 100% success):

  | budget | 40 | 80 | 160 | 320+ |
  |---|---|---|---|---|
  | bias gap | +3.8% | +1.4% | +0.55% | ~0 |
  | p | 1e-5 | 0.0015 | 1e-5 | n.s. |

  Consistent in sign across all four hardware metrics. By 320 evaluations it is
  gone.
- **Evaluations to 95% of the exact optimum**: median extra cost 0 in most
  cells, +20 (one generation) in `obj` at T1.

**Why absorbed.** Under CDP plus a hard gate the algorithm never needs accurate
objective predictions above the budget -- it only ranks *within* the feasible
region, where the surrogate is trained on-distribution. Bad extrapolation
misdirects proposals, the gate catches them, and the `#Params` constraint
surrogate (which does train on the rejection log, since size is observable even
for an unrunnable model) learns the boundary. The bias becomes waste, not
misdirection.

### 5. The role of #Params barely matters

The surrogate-side bias is essentially the same in all three roles. The HV
effect is cleanest in `constr` only because that role has 100% success and four
hardware metrics, i.e. more statistical power -- not because it is more
affected. `obj` and `multi` carry a joint (size AND hardware) constraint that
some seeds never satisfy, so their HV is bimodal (~0.997 or exactly 0) and
their gaps flip sign at random.

### Caveats

- **The benchmark is too easy at this budget.** 1 200 evaluations enumerates
  7.7% of a 15 625-architecture space, and cells reach 95% of the exact optimum
  in 40-120 evaluations. 81% of runs finish above 99% of optimum. The negative
  HV result therefore says the bias does not hurt *when search is easy relative
  to the budget*, not that it never hurts. **This is what the MobileNetV3 arm
  of the experiment exists to settle** — same grid, a space 1e16 times larger,
  where 1 200 evaluations cannot solve anything.
- **Budget-40 and budget-80 points at T1 are unstable.** The gated DOE consumes
  a median of 60 evaluations (max 160) before the first infill at T1, and both
  arms draw identically until the first surrogate fit, so those columns are
  ratios over mostly-zero means. Read T1 from budget 160 upward.
- Wilcoxon p-values are per cell, not corrected for the 72 comparisons in
  `budget.csv`. The `constr` early-budget effect survives on consistency of
  sign across metrics and budgets, not on any single p-value.

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
| `config.py` | `SPACES`: per-space thresholds, hardware metrics, exclusions, plus the per-cell spec |
| `derive_taus.py` | regenerates every threshold from the stage-1 cache of either space; `--check` verifies `config.py` |
| `_bias.py` | probe set + the probe-aware callback (surrogate accuracy per side of the gate) |
| `run_bias.py` | campaign runner, one `(space, tightness, role, hw, arm)` cell per invocation |
| `analyse_bias.py` | `runs.csv`, `cells.csv`, `budget.csv` and the four figures |
| `test_bias_wiring.py` | assert-based self-check of the three shared-code seams, run on both spaces |
| `BIAS_EVOXBENCH.sbatch` | array job for either space (`SPACE=`), one seed per rsync-back |

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
