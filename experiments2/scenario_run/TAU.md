# TAU — scenario-run campaign operating point

Single 10% feasibility budget across all eight scenarios (S1–S8), with tau (the constraint threshold) shared by hard and soft modes. Hard mode gates evaluability; soft mode archives infeasible points.

## Tau definition and normalization

Tau is the percentile threshold in benchmark-evaluate space, exactly as `ConstrainedEvoXBenchProblem` operates on it. When `benchmark.normalized_objectives` is `False`, parquet samples are scaled via `(F - utopian) / (nadir - utopian)` before tau is computed or applied. Three scenarios are non-normalized:

- S2: ResNet-50D in1kmop/3 (objective set: Err., FLOPs)
- S3: MobileNetV3 in1kmop/9 (objective set: Err., FLOPs)
- S6, S7: MoSegNAS citysegmop/15 (objective set: Err., #Params; constrained on latency)

## Verification against results2/constraint_analysis/full/

**S1 NATS c10mop/4**: tau=0.10096685215830803 (direct match vs hv_curve.csv)

**S2 ResNet-50D in1kmop/3**: tau=0.19399442988274224 (affine-converted from non-normalized raw); matches hv_curve.csv to 1e-9 after `(raw-u)/(n-u)` rescaling.

**S3 MobileNetV3 in1kmop/9**: tau=0.39204589380971877 (affine-converted); matches hv_curve.csv to 1e-9 after normalization.

**S4 NB201 c10mop/7**: tau=0.02799552120268345 (achieves 13.97% feasible, not 10%). Reason: NB201 #Params has only 59 distinct values across 15,625 architectures; the 10th percentile sits on a value shared by 1454 architectures. The percentile cutoff evaluates to exactly 0.139712 (the joint feasible set is 4.67% strictly below + 9.30% exactly at the tied value). The benchmark analysis reports this identical 0.139712, confirming S4's hard/soft published numbers already reflect the tie.

**S5 NB201 c10mop/7**: tau=0.32976839542388914 (direct match)

**S6 MoSegNAS citysegmop/15**: tau=0.36661331274859266 (affine-converted); matches hv_curve.csv to 1e-9 after normalization. Ref point = 2.626431190600913 on #Params axis (see Ref point section below).

**S7 MoSegNAS citysegmop/15**: tau=0.7076894651685864 (affine-converted); matches hv_curve.csv to 1e-9 after normalization. Same ref point as S6.

**S8 NB201 c10mop/7**: tau=0.8965967893600464 (achieves 10.15% feasible). Eyeriss AI is a maximize metric, so this is a **floor**: tau is the *90th* percentile and feasibility is `metric >= tau`. Same tie-block effect as S4 — Eyeriss AI has only 84 distinct values across 15,625 architectures — giving 10.15% rather than exactly 10.00%.

## Reference point per scenario

HV ref point = `max(1.05, p95)` per normalized objective column within each scenario's objective set.

| Scenario | Objectives  | Ref point |
|----------|-------------|-----------|
| S1       | Err., #Params | (1.05, 1.05) |
| S2       | Err., FLOPs | (1.05, 1.05) |
| S3       | Err., FLOPs | (1.05, 1.05) |
| S4       | Err., EdgeGPU Lat. | (1.05, 1.05) |
| S5       | Err., #Params | (1.05, 1.05) |
| S6       | Err., #Params | (1.05, 2.626431190600913) |
| S7       | Err., #Params | (1.05, 2.626431190600913) |
| S8       | Err., Eyeriss Lat. | (1.05, 1.05) |

**Why S6/S7 use 2.626 instead of 1.05**: `normalize()` is `(F - utopian) / (nadir - utopian)`, and EvoXBench's utopian/nadir are the benchmark's *Pareto front* extremes, not the range of the search space. For MoSegNAS #Params they are 1.325e+05 / 4.532e+05, both well inside the reachable range, so normalized #Params has median 1.40 and reaches 4.58 — about 70% of sampled architectures normalize above 1.05. A ref point of 1.05 would credit all of them with zero hypervolume, leaving HV signal in only the bottom ~27% of the distribution. The measured p95 is 2.626.

## Penalty (h2-static_penalty weight)

Each scenario also carries a `penalty` value, used as h2-static_penalty's
fixed weight (and as h3-adaptive's starting weight w0, see
`experiments2/scenario_run/run_scenario.py`). Rule:

```
CV        = max(0, sense * (metric - tau) / tau)   over the stage-1 sample
CV_median = 50th percentile of CV over the WHOLE sample (not just infeasible rows)
span      = max over the scenario's two objectives of (ref_point[j] - min(objective_j))
penalty   = span / CV_median
```

| Scenario | CV_median | span  | penalty |
|----------|-----------|-------|---------|
| S1       | 2.4445    | 1.050 | 0.42953 |
| S2       | 0.8564    | 1.412 | 1.64875 |
| S3       | 0.4103    | 1.546 | 3.76896 |
| S4       | 8.6800    | 1.050 | 0.12097 |
| S5       | 0.8993    | 1.050 | 1.16757 |
| S6       | 0.6978    | 2.575 | 3.69063 |
| S7       | 0.6995    | 2.575 | 3.68159 |
| S8       | 0.4405    | 1.050 | 2.38340 |

**Why it exists**: `G = (m - tau) / tau` is a *relative* violation, so a tight
tau inflates CV -- the same absolute distance past the threshold produces a
much larger G when tau is small. At this campaign's 10% feasibility operating
point, >=50% of the stage-1 sample IS infeasible (feasibility is the rare
region here, not the common one), so `CV_median` is well-defined and
positive; dividing it out of `span` converts "one median-violating
architecture" into a penalty that costs exactly one objective range,
regardless of how tight or loose tau happens to be on that scenario's raw
metric scale.

Without this correction, a flat `penalty=1.0` would vary the EFFECTIVE
pressure ~30x across the suite: from 0.2x of the objective span on S6/S7 (tau
loose relative to CV) up to 6.0x on S4 (tau=0.028 sits on the #Params tie
plateau -- see the S4 note above -- so CV is inflated by the small
denominator). At the low end that collapses h2-static_penalty into
effectively unconstrained search; at the high end it collapses it into
h4-cdp's hard feasibility-first behaviour, defeating the point of having a
distinct penalty handler.

Only h2 is affected DIRECTLY by this scaling:
  - h5's eps0 is the mean DOE CV, so it self-scales to whatever tau produces.
  - h6's Pf (stochastic ranking) is scale-free -- it only compares CV
    ORDER, never magnitude.
  - h1/h4/b0-as-obj never consume CV magnitude at all (rejection,
    native CDP, and constraint-as-objective respectively).
  - h3-adaptive corrects its weight multiplicatively every generation
    regardless of scale, so only its w0 STARTING point is touched --
    penalty is used as h3's w0 for the same reason, so it starts already
    in the right neighbourhood instead of taking several generations of
    *= / /= 1.2 to correct a ~30x initial mis-scale.

## Guard: tau > 0

Every scenario's tau is strictly positive:

```
S1: 0.101
S2: 0.194
S3: 0.392
S4: 0.028
S5: 0.330
S6: 0.367
S7: 0.708
S8: 0.897
```

This matters because `G = sense * (metric - tau) / tau` divides by tau: a tau <= 0 would invert the inequality and silently turn the constraint inside out. Only `Err.` normalizes negative on any of these spaces, and `Err.` is never a constrained metric.

## Regeneration

Run `derive_tau.py --check` with the SAMOS_311 Python environment to verify:

```bash
"$USERPROFILE/AppData/Local/anaconda3/envs/SAMOS_311/python.exe" experiments2/scenario_run/derive_tau.py --check
```

This reads from `results2/constraint_analysis/full/<space>/`, applies the same normalization as `ConstrainedEvoXBenchProblem`, filters MoSegNAS x0≠0, and re-derives all tau, ref_point, and penalty values. The `--check` flag compares against `scenarios.py` and exits nonzero on any mismatch (tolerance 1e-9).
