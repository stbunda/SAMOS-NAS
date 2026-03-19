import warnings

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

    def __init__(
        self,
        *,
        pop_size,
        sampling,
        eliminate_duplicates=None,
        n_max_iterations=100,
        seed=None,
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

    def _infill(self):
        """
        Generate purely random candidates.
        Duplicate elimination is GLOBAL w.r.t. the entire current population.
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
        Evaluate infills and ACCUMULATE them into the population.
        """
        self.evaluator.eval(self.problem, infills)

        # grow population monotonically
        self.pop = Population.merge(self.pop, infills)

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


