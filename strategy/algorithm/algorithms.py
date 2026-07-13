import warnings

import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population


class MyNSGA2(NSGA2):
    """
    Thin wrapper around pymoo's NSGA2 that captures
    constructor intent for reproducibility.
    """

    def __init__(
        self,
        pop_size,
        sampling,
        crossover,
        mutation,
        eliminate_duplicates=True,
        seed=None,
        **kwargs,
    ):
        # ---- store intent (NOT objects) ----
        self._config = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": {
                "pop_size": pop_size,
                "eliminate_duplicates": eliminate_duplicates,
                "seed": seed,
            },
            "sampling": self._operator_config(sampling),
            "crossover": self._operator_config(crossover),
            "mutation": self._operator_config(mutation),
        }

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates,
            seed=seed,
            **kwargs,
        )

    def _operator_config(self, op):
        if op is None:
            return None

        if hasattr(op, "to_config"):
            return op.to_config()

        return {
            "class": op.__class__.__name__,
            "module": op.__class__.__module__,
            "params": self._extract_constructor_params(op),
        }

    def _extract_constructor_params(self, op):
        """
        Explicit, defensive extraction.
        Only include JSON-safe fields.
        """
        params = {}
        for k, v in vars(op).items():
            if isinstance(v, (int, float, str, bool, type(None))):
                params[k] = v
        return params

    def to_config(self):
        return self._config


class RandomGA(GeneticAlgorithm):
    """Pure random search: sample, evaluate, accumulate -- no selection.

    ``hard_gate``: when True, evaluated
    individuals with violation G > 0 are counted (``n_hf_evaluated`` /
    ``n_hf_feasible``) and DISCARDED from ``self.pop`` -- their fitness is
    unobservable under gated-hard semantics, and per the campaign's random
    definition there is NO replacement sampling: an infeasible draw just
    consumes budget. Rejected X are kept (X-only) in ``_rejected_pop``
    purely so duplicate elimination never re-proposes a known-failed
    architecture. The counters are maintained even when the gate is off so
    the feasibility-aware callback can always prefer them."""

    def __init__(
        self,
        *,
        pop_size,
        sampling,
        eliminate_duplicates=None,
        n_max_iterations=100,
        seed=None,
        hard_gate=False,
        **kwargs
    ):
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            crossover=None,
            mutation=None,
            survival=None,
            eliminate_duplicates=eliminate_duplicates,
            seed=seed,
            **kwargs
        )

        self.sampling = sampling
        self.n_max_iterations = n_max_iterations
        self.history = []
        self.hard_gate = bool(hard_gate)
        self.n_hf_evaluated = 0
        self.n_hf_feasible  = 0
        self._rejected_pop  = Population.empty()   # X-only, dedup reference

        self._config = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": {
                "pop_size": pop_size,
                "eliminate_duplicates": eliminate_duplicates,
                "seed": seed,
            },
            "sampling": self._operator_config(sampling),
        }

    def _initialize(self):
        super()._initialize()
        # self.pop is the initial random population
        self.history.append(self.pop.copy())

    def _gate_keep(self, pop):
        """Count evaluated individuals; when hard_gate is on, return only the
        feasible subset (G <= 0) and remember rejected X for dedup. No
        replacement sampling ever happens here -- infeasible draws just
        consume budget (random has no strategy)."""
        if pop is None or len(pop) == 0:
            return pop
        G = pop.get('G')
        if G is not None and np.asarray(G).size > 0:
            viol = np.asarray(G, dtype=float).reshape(len(pop), -1)[:, 0]
        else:
            viol = np.zeros(len(pop))
        self.n_hf_evaluated += len(pop)
        self.n_hf_feasible  += int(np.sum(viol <= 0))
        if not self.hard_gate:
            return pop
        feas_mask = viol <= 0
        rejected  = pop[~feas_mask]
        if len(rejected) > 0:
            self._rejected_pop = Population.merge(
                self._rejected_pop, Population.new('X', rejected.get('X')))
        return pop[feas_mask]

    def _initialize_advance(self, infills=None, **kwargs):
        # Gate the DOE too: pymoo core has already set self.pop = infills
        # (ungated) before calling this hook, so re-set it to the kept
        # subset. An all-infeasible DOE leaves pop legitimately empty.
        kept = self._gate_keep(infills)
        self.pop = kept if kept is not None else infills
        super()._initialize_advance(infills=kept, **kwargs)

    def _set_optimum(self):
        # Gated edge case: an empty (all-rejected) population makes pymoo's
        # filter_optimum return None, which the verbose output machinery
        # cannot handle. Keep the previous optimum -- display/result state
        # only, the population gate is unaffected.
        if self.hard_gate and (self.pop is None or len(self.pop) == 0):
            if self.opt is None:
                self.opt = Population.empty()
            return
        super()._set_optimum()

    def _infill(self):
        """
        Generate purely random candidates.
        Duplicate elimination is GLOBAL w.r.t. the entire current population
        (plus, under the hard gate, everything already rejected -- a
        known-failed architecture is never re-proposed).
        """
        infill = Population.empty()
        n_iter = 0

        while len(infill) < self.pop_size:
            if n_iter >= self.n_max_iterations:
                break

            n_missing = self.pop_size - len(infill)
            cand = self.sampling.do(self.problem, n_missing)

            if self.eliminate_duplicates is not None:
                # eliminate duplicates against *both* current pop and infill
                reference = Population.merge(self.pop, infill)
                if len(self._rejected_pop) > 0:
                    reference = Population.merge(reference, self._rejected_pop)
                cand = self.eliminate_duplicates.do(cand, reference)

            infill = Population.merge(infill, cand)
            n_iter += 1

        if len(infill) < self.pop_size:
            warnings.warn(
                f"Infill size {len(infill)} < requested {self.pop_size} "
                f"after {self.n_max_iterations} sampling iterations.",
                RuntimeWarning,
            )

        return infill

    def _advance(self, infills=None, **kwargs):
        """
        Evaluate infills and ACCUMULATE them into the population
        (feasible subset only under the hard gate -- see _gate_keep).
        """
        self.evaluator.eval(self.problem, infills)

        kept = self._gate_keep(infills)

        # grow population monotonically
        self.pop = Population.merge(self.pop, kept)

        self.history.append(self.pop.copy())

    def to_config(self):
        return self._config

    def _operator_config(self, op):
        if op is None:
            return None

        if hasattr(op, "to_config"):
            return op.to_config()

        return {
            "class": op.__class__.__name__,
            "module": op.__class__.__module__,
            "params": self._extract_constructor_params(op),
        }

    def _extract_constructor_params(self, op):
        """
        Explicit, defensive extraction.
        Only include JSON-safe fields.
        """
        params = {}
        for k, v in vars(op).items():
            if isinstance(v, (int, float, str, bool, type(None))):
                params[k] = v
        return params


