"""Probe set + probe-aware callback for the sample-selection-bias experiment.

The headline number of the experiment is HV, but HV alone cannot say WHY a
gated run loses: the mechanism is that the objective surrogates never see the
region above the size budget, so they extrapolate into it and the inner GA
proposes points there on bad predictions. ``BiasCallback`` measures exactly
that, by scoring the live surrogates each outer generation against a fixed
random probe set of architectures, split by which side of the evaluability
gate the architecture is on:

  ``mae_evaluable`` / ``rho_evaluable``  -- the region the surrogate is trained
      on (feasible arm) or trained on regardless (oracle arm).
  ``mae_gated`` / ``rho_gated``          -- the region the feasible arm never
      observes. The feasible-vs-oracle gap HERE is the bias itself; the gap in
      HV is its consequence.

One probe set is built per (benchmark, cell, probe seed) and reused across
generations, so the per-generation cost is 2 surrogate predicts and nothing
else -- no benchmark calls after construction.
"""

import numpy as np
from scipy.stats import spearmanr

from problem.evoxbench.callbacks import FeasibilityAwareEvoxBenchCallback
from problem.evoxbench.constrained_problem import _violation


def evaluate_metrics(benchmark, X_int):
    """Benchmark metric matrix under the campaign's baseline convention
    (normalize only when the benchmark does not already), non-finite guarded
    to 1.0 -- the same space ConstrainedEvoXBenchProblem's F and G live in."""
    F = benchmark.evaluate(np.round(X_int).astype(int), true_eval=False)
    if not benchmark.normalized_objectives:
        F = benchmark.normalize(F)
    return np.where(np.isfinite(F), F, 1.0)


def build_probe(benchmark, spec, xl, xu, n=2000, seed=12345):
    """Fixed random probe set: (X, F_obj, gated_mask).

    ``F_obj`` is in the run's own objective space -- the cell's obj_indices,
    with the maximize-better column flipped exactly as the outer problem
    flips it -- so surrogate predictions are directly comparable to it.
    """
    rng = np.random.RandomState(seed)
    X = np.column_stack([rng.randint(lo, hi + 1, size=n)
                         for lo, hi in zip(np.asarray(xl, int), np.asarray(xu, int))])
    X = np.unique(X, axis=0).astype(float)
    F = evaluate_metrics(benchmark, X)
    F_obj = F[:, spec['obj_indices']]
    for pos in spec['flip_obj_pos']:
        F_obj[:, pos] = 1.0 - F_obj[:, pos]
    gated = _violation(F[:, spec['gate_index']], spec['gate_tau'], 1) > 0
    return X, F_obj, gated


def _score(pred, truth):
    """(MAE, Spearman rho) of one surrogate on one probe subset."""
    if len(truth) < 3:
        return float('nan'), float('nan')
    mae = float(np.mean(np.abs(pred - truth)))
    rho = spearmanr(pred, truth).statistic
    return mae, float(rho) if np.isfinite(rho) else float('nan')


class BiasCallback(FeasibilityAwareEvoxBenchCallback):
    """FeasibilityAwareEvoxBenchCallback plus per-generation surrogate-quality
    diagnostics on a fixed probe set.

    IGD+ is additionally disabled (``_pareto_front = None``): the benchmark's
    own Pareto front is over its native, unconstrained objective set, which is
    not this run's target. The exact CONSTRAINED front is recomputed per cell
    at analysis time from the NB201 enumeration instead, and HV against the
    fixed reference point stays the live indicator.
    """

    def __init__(self, *args, probe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pareto_front = None
        self._igd_ind = None
        self.probe = probe
        self.data['probe'] = []

    def notify(self, algorithm):
        super().notify(algorithm)
        self.data['probe'].append(self._probe_stats(algorithm))

    def _probe_stats(self, algorithm):
        """MAE / Spearman per objective surrogate, on each side of the gate.
        Empty until the first fit (the DOE generation has no surrogate yet)."""
        if self.probe is None:
            return {}
        X, F_true, gated = self.probe
        stats = {'n_gated_probe': int(gated.sum()), 'n_probe': int(len(X))}
        for i, (surrogate, pos) in enumerate(
                zip(algorithm.surrogates, algorithm.predict_obj_indices)):
            try:
                pred = np.asarray(surrogate.predict(X.astype(float))).squeeze()
            except Exception:      # not fitted yet -- DOE generation only
                return stats
            for side, mask in (('evaluable', ~gated), ('gated', gated)):
                mae, rho = _score(pred[mask], F_true[mask, pos])
                stats[f'mae_{side}_o{i}'] = mae
                stats[f'rho_{side}_o{i}'] = rho
        return stats
