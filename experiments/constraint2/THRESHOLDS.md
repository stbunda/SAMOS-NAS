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
