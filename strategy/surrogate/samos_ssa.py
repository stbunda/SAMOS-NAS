"""
SAMOS-SSA — SAMOS loop with pysamoo's cross-validated surrogate management.

Combines SAMOS's infill strategy (warm-start inner NSGA-II, subset selection,
deduplicated random padding) with SSA-NSGA-II's surrogate infrastructure
(pysamoo Surrogate / Target with automatic model selection and validation).

Drop-in pymoo Algorithm:

    from strategy.surrogate.samos_ssa import SAMOSSA

    algorithm = SAMOSSA(
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        n_doe=20, n_infill=10, n_gen_inner=30, ga_pop_size=100,
    )
    res = minimize(problem, algorithm, termination=('n_gen', 50), verbose=False)

Per-generation behaviour:
  1. Validate + fit pysamoo Surrogate on the full evaluated archive.
  2. Warm-start inner NSGA-II: 75 % best archive (rank + crowding) + 25 % fresh random.
  3. Run inner NSGA-II on ProblemFromTargets for n_gen_inner generations.
  4. Deduplicate candidates against the evaluated archive.
  5. Subset-selection to pick n_infill diverse candidates.
  6. Pad with fresh random samples when candidates are scarce.
  7. Return X-only population — the real problem evaluates F after this returns.
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pysamoo.core.surrogate import Surrogate
from pysamoo.core.target import Target

from strategy.surrogate.subset_selection import subset_selection


class SAMOSSA(Algorithm):
    """SAMOS with pysamoo's cross-validated surrogate model management.

    Parameters
    ----------
    sampling : pymoo Sampling
        Generates valid candidates for the DOE and random fill-in.
    surrogate : pysamoo Surrogate or None
        Pre-built Surrogate object.  If *None* (default), a default Surrogate
        is created during ``_setup()`` using pysamoo's standard model zoo
        (one Target per objective).
    sklearn_models : list of sklearn estimators or None
        If provided, each estimator is wrapped in a lightweight Target that
        bypasses ezmodel cross-validation (same behaviour as SklearnSSANSGA2).
        Ignored when *surrogate* is not None.
    crossover, mutation : pymoo operators
        Used by the inner NSGA-II.
    n_doe : int
        Number of candidates in the initial design-of-experiments (gen 0).
    n_infill : int
        Real evaluations added per outer generation.
    n_gen_inner : int
        Inner NSGA-II generations per outer generation.
    ga_pop_size : int or None
        Population size of the inner NSGA-II (default: n_infill * 10).
    warm_start_ratio : float
        Fraction of inner population seeded from the best archive members.
    use_subset_selection : bool
        Diversify the infill batch via subset_selection (default: True).
    """

    def __init__(self,
                 sampling,
                 surrogate=None,
                 sklearn_models=None,
                 crossover=None,
                 mutation=None,
                 n_doe=20,
                 n_infill=8,
                 n_gen_inner=30,
                 ga_pop_size=None,
                 warm_start_ratio=0.75,
                 use_subset_selection=True,
                 eliminate_duplicates=True,
                 dedup_key_fn=None,
                 **kwargs):
        super().__init__(eliminate_duplicates=False, **kwargs)
        self.sampling             = sampling
        self._surrogate_arg       = surrogate
        self._sklearn_models      = sklearn_models
        self.crossover            = crossover
        self.mutation             = mutation
        self.n_doe                = n_doe
        self.n_infill             = n_infill
        self.n_gen_inner          = n_gen_inner
        self.ga_pop_size          = ga_pop_size if ga_pop_size is not None else n_infill * 10
        self.warm_start_ratio     = warm_start_ratio
        self.use_subset_selection = use_subset_selection
        self.eliminate_duplicates = eliminate_duplicates
        self._dedup_key = dedup_key_fn if dedup_key_fn is not None \
            else lambda x: tuple(np.round(x, decimals=8).tolist())

        self.surrogate     = None   # set in _setup
        self._archive      = Population()
        self._archive_keys: set = set()
        self._init         = Initialization(sampling)
        self._prev_nd_F    = None

    # ── setup ──────────────────────────────────────────────────────────────────

    def _setup(self, problem, **kwargs):
        if self._surrogate_arg is not None:
            self.surrogate = self._surrogate_arg
        elif self._sklearn_models is not None:
            self.surrogate = _sklearn_surrogate(problem, self._sklearn_models)
        else:
            self.surrogate = _default_surrogate(problem)

    # ── DOE ────────────────────────────────────────────────────────────────────

    def _initialize_infill(self):
        return self._init.do(self.problem, self.n_doe, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills
        # Initial validation so the surrogate picks its best model
        self.surrogate.validate(infills)

    # ── surrogate-assisted infill ──────────────────────────────────────────────

    @staticmethod
    def _fmt_float(val, width):
        if val is None:
            return '-'.rjust(width)
        if val >= 10 or val * 1e5 < 1:
            text = f"%.{width - 7}E" % val
        else:
            text = f"%.{width - 3}f" % val
        return text.rjust(width)

    def _infill(self):
        F_arc = self._archive.get('F')

        # 1. Validate + fit pysamoo surrogate on the full archive
        self.surrogate.validate(self._archive)
        self.surrogate.fit(self._archive)

        # 2. Warm-start: 75 % best archive + 25 % fresh random
        topx    = max(1, int(self.ga_pop_size * self.warm_start_ratio))
        top_pop = RankAndCrowding().do(problem=self.problem, pop=self._archive, n_survive=topx)
        n_rand  = self.ga_pop_size - len(top_pop)
        if n_rand > 0:
            rand_pop = self._init.do(self.problem, n_rand, algorithm=self)
            inner_X  = np.vstack([top_pop.get('X'), rand_pop.get('X')])
        else:
            inner_X  = top_pop.get('X')
        inner_init = Population.new('X', inner_X)

        # 3. Inner NSGA-II on ProblemFromTargets
        surr_problem = self.surrogate.problem()
        inner_alg = NSGA2(
            pop_size=self.ga_pop_size,
            sampling=inner_init,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=self.eliminate_duplicates,
        )
        res = minimize(
            surr_problem, inner_alg,
            termination=('n_gen', self.n_gen_inner),
            verbose=False,
        )
        n_surr_eval = res.algorithm.evaluator.n_eval

        # ── display ───────────────────────────────────────────────────────────
        _W = {'gen': 6, 'eval': 8, 'nds': 6, 'surr': 11, 'eps': 13, 'ind': 13}
        if self.n_gen == 2:
            _header = (
                f" {'n_gen':>{_W['gen']}} | {'n_eval':>{_W['eval']}}"
                f" | {'n_nds':>{_W['nds']}} | {'n_surr_eval':>{_W['surr']}}"
                f" | {'eps':>{_W['eps']}} | {'indicator':>{_W['ind']}} "
            )
            _border = '=' * len(_header)
            print(_border)
            print(_header)
            print(_border)

        nd_idx = NonDominatedSorting().do(F_arc, only_non_dominated_front=True)
        nd_F   = F_arc[nd_idx]
        n_nds  = len(nd_idx)

        if self._prev_nd_F is not None and len(self._prev_nd_F) > 0:
            ideal_c = nd_F.min(axis=0)
            ideal_p = self._prev_nd_F.min(axis=0)
            nadir_c = nd_F.max(axis=0)
            nadir_p = self._prev_nd_F.max(axis=0)
            denom   = np.maximum(nadir_c - ideal_c, 1e-30)
            d_ideal = float(np.max(np.abs(ideal_c - ideal_p) / denom))
            d_nadir = float(np.max(np.abs(nadir_c - nadir_p) / denom))
            tol = 1e-6
            if d_ideal > tol:
                eps, ind = d_ideal, 'ideal'
            elif d_nadir > tol:
                eps, ind = d_nadir, 'nadir'
            else:
                eps, ind = max(d_ideal, d_nadir), 'f'
        else:
            eps, ind = None, None
        self._prev_nd_F = nd_F

        print(
            f" {self.n_gen:>{_W['gen']}} | {len(self._archive):>{_W['eval']}}"
            f" | {n_nds:>{_W['nds']}} | {n_surr_eval:>{_W['surr']}}"
            f" | {self._fmt_float(eps, _W['eps'])} | {str(ind or '-'):>{_W['ind']}} "
        )

        # 4. Deduplicate candidates against the evaluated archive
        cand_pop = res.pop if res.pop is not None else Population.empty()
        if len(cand_pop) > 0:
            not_dup = np.array([
                self._dedup_key(cx) not in self._archive_keys
                for cx in cand_pop.get('X')
            ], dtype=bool)
            cand_pop = cand_pop[not_dup]

        # 5. Select n_infill candidates; pad with random if scarce
        infill_pop = self._select_infill(cand_pop, F_arc)

        return Population.new('X', infill_pop.get('X'))

    def _advance(self, infills=None, **kwargs):
        # Validate surrogate on training (archive) + test (new infills)
        self.surrogate.validate(self._archive, infills)
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    # ── helpers ────────────────────────────────────────────────────────────────

    def _select_infill(self, cand_pop, F_arc):
        if len(cand_pop) == 0:
            return self._sample_dedup(self.n_infill)

        F_cand = cand_pop.get('F')
        front  = NonDominatedSorting().do(F_arc, only_non_dominated_front=True)

        if self.use_subset_selection and len(cand_pop) > self.n_infill:
            indices = subset_selection(F_cand, F_arc[front], self.n_infill)
            found   = cand_pop if indices is None else cand_pop[indices]
        else:
            found = cand_pop

        n_found = min(len(found), self.n_infill)
        found   = found[:n_found]

        if n_found < self.n_infill:
            extra = self._sample_dedup(self.n_infill - n_found, extra_ref=found)
            return Population.merge(extra, found) if len(extra) > 0 else found
        return found

    def _add_to_archive_keys(self, infills):
        if infills is not None and len(infills) > 0:
            for x in infills.get('X'):
                k = self._dedup_key(x)
                if k is not None:
                    self._archive_keys.add(k)

    def _sample_dedup(self, n, extra_ref=None, max_tries_factor=20):
        arc_keys = self._archive_keys.copy()
        if extra_ref is not None and len(extra_ref) > 0:
            arc_keys |= {self._dedup_key(x) for x in extra_ref.get('X')}

        collected = []
        for _ in range(max_tries_factor * max(n, 1)):
            if len(collected) >= n:
                break
            x   = self._init.do(self.problem, 1, algorithm=self).get('X')[0]
            key = self._dedup_key(x)
            if key not in arc_keys:
                collected.append(x)
                arc_keys.add(key)

        return Population.new('X', np.array(collected)) if collected else Population.empty()


# ── surrogate construction helpers ─────────────────────────────────────────────

def _default_surrogate(problem):
    """Build a pysamoo Surrogate with default model zoo (one Target per objective)."""
    from pysamoo.core.defaults import DEFAULT_OBJ_MODELS
    targets = [
        Target(("F", i), models=DEFAULT_OBJ_MODELS())
        for i in range(problem.n_obj)
    ]
    return Surrogate(problem, targets)


def _sklearn_surrogate(problem, sklearn_models):
    """Build a pysamoo Surrogate from sklearn estimators (bypasses cross-validation)."""
    from copy import deepcopy

    class _SklearnTarget:
        def __init__(self, label, model):
            self.label = label
            self._tmpl = deepcopy(model)
            self.best  = 'sklearn'
            self.obj   = None

        def validate(self, trn=None, tst=None, find_best=True, **kw):
            pass

        def fit(self, sols):
            key, idx = self.label
            X = sols.get('X')
            y = sols.get(key)[:, idx]
            m = deepcopy(self._tmpl)
            m.fit(X, y)
            self.obj = m

        def predict(self, X, out):
            v = self.obj.predict(X).reshape(-1, 1)
            key, idx = self.label
            out.get(key)[:, [idx]] = v

        def performance(self, indicator, model=None, func=None):
            return 0.0

    targets = [_SklearnTarget(('F', i), m) for i, m in enumerate(sklearn_models)]
    return Surrogate(problem, targets)
