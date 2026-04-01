import copy

import numpy as np
from pymoo.core.mutation import Mutation


class MutateProgram(Mutation):
    def __init__(
            self,
            p_point=None,
            p_prob=None,
            p_poly=None,
            single=False,
            **kwargs
    ):
        """ Initialize the mutation operator.
            Args:
                eta (float): Distribution index controlling mutation intensity.
                p_point (float, optional): In point mutation the user decides the percentage of the total number of
                genes of a parent genotype to be mutated to create an offspring.
                p_prob (float, optional): In probabilistic mutation every gene is considered for mutation according to
                a user-defined probability. p_poly (float, optional): Perform the base polynomial mutation operation on the given population.
                prob (float, optional): Probability of mutation for each variable. Defaults to None, which sets it to 1/number of variables.
                """
        super().__init__(**kwargs)

        strategies = {
            "point": p_point is not None,
            "probabilistic": p_prob is not None,
            "polynomial": p_poly is not None,
            "single": single,
        }

        n_active = sum(strategies.values())

        if n_active != 1:
            raise ValueError(
                "Exactly one mutation strategy must be active. "
                f"Received: {', '.join(k for k, v in strategies.items() if v)}"
            )

        self.strategy = next(k for k, v in strategies.items() if v)

        self.p_point = p_point
        self.p_prob = p_prob
        self.p_poly = p_poly
        self.single = single

    def to_config(self):
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": {
                "p_point": self.p_point,
                "p_prob": self.p_prob,
                "p_poly": self.p_poly,
                "single": self.single,
            },
        }

    def _do(self, problem, X, **kwargs):
        n = X.shape[0]
        X_mut = np.empty_like(X, dtype=object)
        for i in range(n):
            # X[i, 0] is the full CGP program
            prog = copy.deepcopy(X[i, 0])
            # print('o', X[i, 0].tuple_program)

            if self.strategy == "point":
                prog.point_mutation(self.p_point)

            elif self.strategy == "probabilistic":
                prog.probabilistic_mutation(self.p_prob)

            elif self.strategy == "single":
                prog.single_active_mutation()

            X_mut[i, 0] = prog

            # self.debug_mutation(X[i, 0], X_mut[i, 0])
            # print('m', X_mut[i, 0].tuple_program)
            assert X[i, 0].tuple_program is not X_mut[i, 0].tuple_program

        return X_mut

    @staticmethod
    def debug_mutation(parent, child):
        p_active = parent.active_net_list()
        c_active = child.active_net_list()

        if p_active == c_active:
            print("⚠️ ACTIVE GRAPH UNCHANGED")

        if parent.tuple_program == child.tuple_program:
            print("⚠️ TUPLE REPRESENTATION UNCHANGED")

        if parent.tuple_genome == child.tuple_genome:
            print("⚠️ GENOTYPE UNCHANGED")

    # def _do(self, problem, X, **kwargs):
    #     X_temp = copy.deepcopy(X)
    #     for c, parent in enumerate(X_temp[0]):
    #         parent.point_mutation(p_point_replace=self.p_point)
    #     # if isinstance(X[0][0], GeneticProgram):
    #     #     X_temp[0][0] = X_temp[0][0].point_mutation(p_point_replace=self.p_point)
    #     # elif isinstance(X[0][0], CartesianGeneticProgram):
    #     #     # print(f'Old genes: {X_temp}')
    #     #     X_temp[0][0] = X_temp[0][0].point_mutation(p_point_replace=self.p_point)
    #     #     # print(f'New genes: {X_temp}')
    #     #     # for i in range(X[0, 0].normal):
    #     #     #     X_temp = self.mutation_base(X_temp, problem, i)
    #
    #     return X_temp


# ─── NASBench-101 mutation operators ─────────────────────────────────────────

from pymoo.core.variable import Real, get
from pymoo.operators.mutation.pm import mut_pm


class UniformMutation101(Mutation):
    """
    Polynomial mutation with eta=1.0 and prob=1/n_var.
    Retries up to N_VAR times for validity; falls back to parent.
    Mirrors PDNS CustomPolynomialMutation.
    """

    def __init__(self, prob=None, eta: float = 1.0, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, 100.0))

    def to_config(self):
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": {"prob": self.prob, "eta": float(get(self.eta, size=1)[0])},
        }

    def _do(self, problem, X, params=None, **kwargs):
        from problem.nasbench101.utils import (
            N_VAR as _NB101_N_VAR,
            N_OPS as _NB101_N_OPS,
            _is_valid_vec as _nb101_is_valid_vec,
        )
        X = X.astype(float)
        Xp = np.copy(X)
        eta      = get(self.eta, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))

        for i in range(len(X)):
            for _ in range(_NB101_N_VAR):          # up to N_VAR retries
                candidate = mut_pm(
                    X[i].reshape(1, -1),
                    problem.xl, problem.xu,
                    np.array([eta[i]]),
                    np.array([prob_var[i]]),
                    at_least_once=False,
                )
                if _nb101_is_valid_vec(np.round(candidate[0]).astype(int)):
                    Xp[i] = candidate[0]
                    break
            # if all retries failed, Xp[i] remains == X[i] (parent)
        return Xp


class SinglePointMutation101(Mutation):
    """
    Single-gene mutation: exactly one gene is changed per individual.
    - Op gene  (0-4):  replaced by a uniformly chosen *different* op (0-2).
    - Edge gene (5-25): bit-flipped.
    Retries up to N_VAR times for validity; falls back to parent.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        from problem.nasbench101.utils import (
            N_VAR as _NB101_N_VAR,
            N_OPS as _NB101_N_OPS,
            _is_valid_vec as _nb101_is_valid_vec,
        )
        X = X.astype(float)
        Xp = np.copy(X)

        for i in range(len(X)):
            for _ in range(_NB101_N_VAR):          # up to N_VAR retries
                candidate = X[i].copy()
                gene = np.random.randint(0, _NB101_N_VAR)
                if gene < _NB101_N_OPS:            # op gene
                    current = int(round(candidate[gene]))
                    choices = [v for v in range(3) if v != current]
                    candidate[gene] = float(np.random.choice(choices))
                else:                       # edge gene — flip
                    candidate[gene] = 1.0 - candidate[gene]
                if _nb101_is_valid_vec(np.round(candidate).astype(int)):
                    Xp[i] = candidate
                    break
            # if all retries failed, Xp[i] remains == X[i] (parent)
        return Xp


# ─── NASBench-201 mutation operators ─────────────────────────────────────────

from problem.nasbench201.utils import (
    N_VAR as _NB201_N_VAR,
    N_OPS_PER_GENE as _NB201_N_OPS,
    XL as _NB201_XL,
    XU as _NB201_XU,
)


class UniformMutation201(Mutation):
    """
    Uniform mutation for NASBench-201: each gene is independently replaced
    with probability ``prob`` (default 1/N_VAR = 1/6) by a *different*
    uniformly-chosen operation.

    All resulting architectures are valid (no validity check needed).
    """

    def __init__(self, prob: float = None, **kwargs):
        super().__init__(prob=prob, **kwargs)

    def _do(self, problem, X, **kwargs):
        X        = X.astype(float)
        Xp       = np.copy(X)
        prob_var = self.get_prob_var(problem, size=len(X))

        for i in range(len(X)):
            for gene in range(_NB201_N_VAR):
                if np.random.rand() < prob_var[i]:
                    current = int(round(X[i, gene]))
                    choices = [v for v in range(_NB201_N_OPS) if v != current]
                    Xp[i, gene] = float(np.random.choice(choices))

        return Xp


class SinglePointMutation201(Mutation):
    """
    Single-gene mutation for NASBench-201: exactly one gene is changed per
    individual to a uniformly-chosen *different* operation.

    All resulting architectures are valid (no validity check needed).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        X  = X.astype(float)
        Xp = np.copy(X)

        for i in range(len(X)):
            gene    = np.random.randint(0, _NB201_N_VAR)
            current = int(round(X[i, gene]))
            choices = [v for v in range(_NB201_N_OPS) if v != current]
            Xp[i, gene] = float(np.random.choice(choices))

        return Xp
