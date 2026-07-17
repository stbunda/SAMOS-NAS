# Constraint thresholds — constraint2 campaign (~Q25)

One threshold per (instance, metric), targeting **~25% of the search space
feasible**. Computed from a fixed-seed (0) uniform random sample of 10,000
architectures per instance (`X[:, i] = np.random.randint(lb[i], ub[i] + 1)`,
matching `strategy/sampler.py::EvoxBenchSampler`), evaluated with
`benchmark.evaluate(X, true_eval=False)` and `benchmark.normalize()` applied
only when `not benchmark.normalized_objectives` — i.e. thresholds live in the
same space `ConstrainedEvoXBenchProblem` operates in. Feasible ⇔ metric ≤ T.

c10mop/4 and in1kmop/9 values are the Q25 rows carried over from the earlier
campaign's quartile tables (same convention, same fixed-seed samples — see
`experiments/constraint_deprecated/THRESHOLDS.md` for their full provenance
and generator scripts). c10mop/5, citysegmop/5 and citysegmop/10 were
computed 2026-07-14 with the script below.

| Instance       | Metric        | Threshold (T)        | Feasible fraction |
|----------------|---------------|----------------------|-------------------|
| c10mop/4       | #Params       | 0.3078135998873715   | 0.25              |
| c10mop/4       | Latency       | 0.36311633657280196  | 0.25              |
| c10mop/5       | #Params       | 0.11198208286674133  | 0.254             |
| c10mop/5       | EdgeGPU Lat.  | 0.5728975060318674   | 0.250             |
| in1kmop/9      | #Params       | 0.46727868434691233  | 0.25              |
| in1kmop/9      | Latency       | 0.34752772487724964  | 0.25              |
| citysegmop/5   | #Params       | 0.9997754995135822   | **0.1762**        |
| citysegmop/5   | H1 Lat.       | 0.5579798323679069   | 0.250             |
| citysegmop/10  | H2 Lat.       | 0.9999087009935012   | **0.2015**        |

## MoSegNAS spike (why two fractions are not 0.25)

~33.5% of uniformly random MoSegNAS genotypes map to one and the same
fallback architecture, whose normalized metrics all equal exactly 1.0. Its
per-column masses in the 10k sample:

- `#Params`: 17.6% of samples below 1.0, 33.5% exactly at 1.0, 48.9% above —
  achievable feasible fractions jump from 0.176 to 0.511; no ~25% threshold
  exists. T is the largest achieved value below the spike.
- `H2 Lat.` (citysegmop/10): 20.1% / 33.5% / 46.4% — same situation, T at
  the spike edge, 0.2015 feasible.
- `H1 Lat.` (citysegmop/5): the spike lies above the lower quartile, so the
  plain Q25 works (0.250 feasible).

Consequence for the campaign design: citysegmop/10 is **expensive-only**
(s2/s4). For the cheap scenarios it has no latency column in play and is the
identical experiment to citysegmop/5 (same search space; the raw
Err/FLOPs/#Params columns of the shared 10k sample are equal between the two
pids — verified; only per-instance normalization bounds differ, an affine
rescale that preserves dominance and the feasible set).

Every run's pkl records its exact design fraction in
`meta['design_feasible_fraction']`.

## Multi-constraint campaign (s5–s12): new metrics + joint fractions

Added 2026-07-17 for the constraint-type-combination scenarios (CC =
#Params+FLOPs, CE = #Params+latency, EE = latency+energy, ALL = every
constrained metric at once). **Per-metric thresholds stay at Q25 with the
exact same convention as above** (fixed user decision 2026-07-16): the
existing values are reused unchanged, the new FLOPs/energy (and
citysegmop/10 structural) values below were computed 2026-07-17 with the
same fixed-seed-0 10k sample, spike rule included. The JOINT feasible
fraction of a combination is a **measured property**, never a design
target.

### Instance × combo availability

EE/ALL need ≥ 2 expensive (predictor-/lookup-backed) metric columns —
latency AND energy. From `benchmark_meta.py` obj_names:

| Instance      | expensive columns          | CC | CE | EE | ALL |
|---------------|----------------------------|----|----|----|-----|
| c10mop/4      | Latency                    | ✓  | ✓  | –  | –   |
| c10mop/5      | EdgeGPU Lat., EdgeGPU En.  | ✓  | ✓  | ✓  | ✓   |
| in1kmop/9     | Latency                    | ✓  | ✓  | –  | –   |
| citysegmop/5  | H1 Lat., H1 En.            | ✓  | ✓  | ✓  | ✓   |
| citysegmop/10 | H2 Lat., H2 En.            | –  | –  | ✓  | ✓   |

citysegmop/10 stays excluded from CC/CE for the same duplication reason as
s1/s3 (without a device column in play it is citysegmop/5); c10mop/4 and
in1kmop/9 expose no energy column and cannot run EE/ALL.

### New per-metric Q25 thresholds

| Instance      | Metric      | Threshold (T)        | Feasible fraction |
|---------------|-------------|----------------------|-------------------|
| c10mop/4      | FLOPs       | 0.1814415868239766   | 0.25              |
| c10mop/5      | FLOPs       | 0.10810810810810814  | 0.2544            |
| c10mop/5      | EdgeGPU En. | 0.544715316639718    | 0.25              |
| in1kmop/9     | FLOPs       | 0.37192471140932504  | 0.25              |
| citysegmop/5  | FLOPs       | 0.9988584474885844   | **0.1646**        |
| citysegmop/5  | H1 En.      | 0.4869375522655389   | 0.25              |
| citysegmop/10 | #Params     | 0.9997754995135822   | **0.1762**        |
| citysegmop/10 | FLOPs       | 0.9988584474885844   | **0.1646**        |
| citysegmop/10 | H2 En.      | 0.9999647471796245   | **0.2292**        |

Bold fractions: the MoSegNAS fallback-architecture spike (33.5 % of the
sample at normalized 1.0 on every column) again swallows the lower
quartile — T sits at the largest achieved value below the spike, exactly
as for the earlier #Params / H2 Lat. cases. citysegmop/10's #Params/FLOPs
thresholds coincide numerically with citysegmop/5's (identical raw
columns, near-identical normalization bounds on these metrics).

Re-validation of the EXISTING thresholds against a fresh regeneration of
the sample reproduced every recorded fraction to ±0.002; the residual on
the citysegmop latency/energy columns is the intentional simulated
measurement noise (±2 %/±5 %) those predictors carry per evaluate() call.

### Measured JOINT feasible fractions (recorded per run in
`meta['design_feasible_fraction_joint']`)

| combo | c10mop/4 | c10mop/5 | in1kmop/9 | citysegmop/5 | citysegmop/10 |
|-------|----------|----------|-----------|--------------|----------------|
| CC    | 0.1644   | 0.2544   | 0.0825    | 0.1424       | –              |
| CE    | 0.1034   | 0.0990   | 0.0837    | 0.1167       | –              |
| EE    | –        | 0.2429   | –         | 0.2216       | 0.1994         |
| ALL   | –        | 0.0977   | –         | 0.0966       | 0.1121         |

Notes:

- **c10mop/5 CC is degenerate**: the #Params and FLOPs Q25-feasible sets
  coincide exactly (joint = both marginals = 0.2544) — on NB201 the two
  structural metrics rank architectures identically at this quantile, so
  CC there behaves as a single binding constraint. Kept (the comparison
  against CE/EE on the same instance is still informative); flag in
  analysis when reading the CC column.
- MoSeg spike interaction, as predicted: the fallback architecture is
  infeasible on every axis, so the cs joint fractions sit at or below the
  tightest single fraction (e.g. ALL @ citysegmop/5: 0.0966 vs tightest
  marginal 0.1646).
- EE joint fractions are high (0.20–0.24): latency and energy are strongly
  correlated on every device, so stacking them barely tightens the region;
  CE (structural × device metric, weakly correlated) tightens to ~0.10.

## Generator script (new instances)

Run from the repo root with the SAMOS311 environment:

```python
import numpy as np
from problem.evoxbench.utils import get_benchmark
from problem.evoxbench.benchmark_meta import metric_index

N, SEED = 10_000, 0
TARGETS = [('c10mop', 5, ['#Params', 'EdgeGPU Lat.']),
           ('citysegmop', 5, ['#Params', 'H1 Lat.']),
           ('citysegmop', 10, ['H2 Lat.'])]

for suite, pid, metrics in TARGETS:
    b = get_benchmark(suite, pid)
    lb, ub = b.search_space.lb, b.search_space.ub
    np.random.seed(SEED)
    X = np.column_stack([np.random.randint(lo, hi + 1, size=N)
                         for lo, hi in zip(lb, ub)])
    F = b.evaluate(X, true_eval=False)
    if not b.normalized_objectives:
        F = b.normalize(F)
    F = F[np.isfinite(F).all(axis=1)]
    for m in metrics:
        col = F[:, metric_index(suite, pid, m)]
        q25 = float(np.percentile(col, 25))
        if np.mean(col <= q25) > 0.30:          # quantile fell inside a spike
            q25 = float(np.max(col[col < q25]))  # -> largest value below it
        print(f'{suite}/pid{pid} {m}: T={q25!r} '
              f'feasible={np.mean(col <= q25):.4f}')
```
