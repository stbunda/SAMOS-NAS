"""strategy/operations/repair.py --- NAS repair operators for constrained EvoXBench.

Repair maps an infeasible architecture back into the feasible region.  It is only
meaningful for **a-priori** constraints (the violation is computable cheaply from
the encoding, without a noisy simulation) and for **NUAK** operator-support
constraints — exactly the QRAK classes where repair is expected to dominate.  The
technique factory must NOT attach this operator to simulated-constraint scenarios
(it would have to probe the noisy true value, which a-priori repair cannot do);
those are reported ``N/A`` instead.

Two repair modes, applied in order for the ``mixed`` scenario:

* **NUAK (binary operator support):** deterministically replace any unsupported
  categorical op value with a random supported one — exact and cheap, guarantees
  feasibility for that constraint.
* **A-priori budget (quantifiable, noiseless):** bounded greedy local search on
  the cheap raw structural metric — repeatedly change a random variable, keep the
  change if it lowers the metric, until feasible or ``max_tries`` is exhausted.
  This uses the benchmark's own (cheap, tabular) metric lookup, so it is faithful
  to the real metric rather than a hand-coded cost model and is search-space
  agnostic.
"""

from __future__ import annotations

import numpy as np
from pymoo.core.repair import Repair


class EvoXBenchRepair(Repair):
    """Repair operator for :class:`ConstrainedEvoXBenchProblem` constraints.

    Parameters
    ----------
    benchmark :
        The evoxbench benchmark (provides ``search_space`` + cheap metric lookup).
    specs :
        The list of :class:`ConstraintSpec` applied to the problem.  Only a-priori
        (``simulated=False``) budget specs and NUAK (``binary=True``) specs are
        repaired; any simulated spec raises (the factory should never pass one).
    seed :
        RNG seed for the stochastic local search / op replacement.
    max_tries :
        Maximum greedy local-search steps per infeasible individual (budget specs).
    """

    def __init__(self, benchmark, specs, seed: int = 0, max_tries: int = 30):
        super().__init__()
        simulated = [s for s in specs if (not s.binary) and s.simulated]
        if simulated:
            raise ValueError(
                'EvoXBenchRepair only supports a-priori / NUAK constraints; '
                f'got simulated budget spec(s): {[s.metric for s in simulated]}. '
                'The technique factory should report N/A instead of attaching repair.')
        self.benchmark = benchmark
        self.specs = specs
        self.nuak_specs = [s for s in specs if s.binary]
        self.budget_specs = [s for s in specs if not s.binary]
        self.rng = np.random.default_rng(seed)
        self.max_tries = max_tries

        ss = benchmark.search_space
        self.lb = np.asarray(ss.lb, dtype=int)
        self.ub = np.asarray(ss.ub, dtype=int)

    # ----- cheap raw metric lookup ------------------------------------------------------------------------------------------------------------------------

    def _raw_metric(self, X_int: np.ndarray, metric_idx: int) -> np.ndarray:
        archs = self.benchmark.search_space.decode(X_int)
        raw = np.asarray(
            self.benchmark.to_matrix(
                self.benchmark.evaluator.evaluate(archs, true_eval=True)),
            dtype=float)
        return raw[:, metric_idx]

    # ----- NUAK: exact op replacement -----------------------------------------------------------------------------------------------------------------

    def _repair_nuak(self, X_int: np.ndarray) -> np.ndarray:
        for spec in self.nuak_specs:
            unsupported = set(spec.unsupported_values)
            if not unsupported:
                continue
            cols = (list(spec.op_var_indices) if spec.op_var_indices is not None
                    else list(range(X_int.shape[1])))
            supported = {c: [v for v in range(self.lb[c], self.ub[c] + 1)
                             if v not in unsupported] for c in cols}
            for c in cols:
                bad = np.isin(X_int[:, c], list(unsupported))
                if bad.any():
                    choices = supported[c]
                    X_int[bad, c] = self.rng.choice(choices, size=int(bad.sum()))
        return X_int

    # ----- a-priori budget: greedy local search ---------------------------------------------------------------------------------------------

    def _repair_budget(self, X_int: np.ndarray) -> np.ndarray:
        for spec in self.budget_specs:
            tau = spec.threshold + spec.feasible_tol()
            for i in range(len(X_int)):
                x = X_int[i].copy()
                cur = float(self._raw_metric(x[None, :], spec.metric_idx)[0])
                tries = 0
                while cur > tau and tries < self.max_tries:
                    tries += 1
                    j = int(self.rng.integers(0, len(x)))
                    new_val = int(self.rng.integers(self.lb[j], self.ub[j] + 1))
                    if new_val == x[j]:
                        continue
                    cand = x.copy()
                    cand[j] = new_val
                    cand_metric = float(self._raw_metric(cand[None, :], spec.metric_idx)[0])
                    if cand_metric < cur:        # greedy: keep improving moves
                        x, cur = cand, cand_metric
                X_int[i] = x
        return X_int

    # ----- Repair API ---------------------------------------------------------------------------------------------------------------------------------------------------------

    def _do(self, problem, X, **kwargs):
        X_int = np.round(X).astype(int)
        X_int = np.clip(X_int, self.lb, self.ub)
        if self.nuak_specs:
            X_int = self._repair_nuak(X_int)
        if self.budget_specs:
            X_int = self._repair_budget(X_int)
        # Return in the original dtype/shape pymoo expects.
        return X_int.astype(X.dtype) if X.dtype.kind == 'f' else X_int
