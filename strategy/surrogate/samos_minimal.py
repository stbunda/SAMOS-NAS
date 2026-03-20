"""
SAMOSMinimal — surrogate-assisted multi-objective search over float/int vectors.

Drop-in pymoo Algorithm (same interface as NSGA2):

    from strategy.surrogate.samos_minimal import SAMOSMinimal

    algorithm = SAMOSMinimal(
        sampling=...,
        surrogates=...,
        surrogate_problem_factory=lambda surrs: MySurrogateProblem(surrs, ...),
        crossover=...,
        mutation=...,
        n_doe=20, n_infill=20, n_gen_inner=20, ga_pop_size=200,
    )
    results = minimize(real_problem, algorithm, termination=('n_gen', n_gen), ...)

Per-generation behaviour (matches strategy/surrogate/samos.py):
  1. Fit one surrogate per predicted objective on the full evaluated archive.
  2. Warm-start inner NSGA-II: 75 % best archive (rank + crowding) + 25 % fresh random.
  3. Run inner NSGA-II on the surrogate problem for n_gen_inner generations.
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

from strategy.surrogate.subset_selection import subset_selection


class SAMOSMinimal(Algorithm):
    """
    Parameters
    ----------
    sampling : pymoo Sampling
        Generates valid candidates for the DOE and random fill-in.
    surrogates : list
        Surrogate models, one per predicted objective.
        Each must implement ``fit(X, y)`` and ``predict(X)``.
    surrogate_problem_factory : callable
        ``factory(surrogates) -> pymoo Problem``
        Called each infill step; the returned problem is used as the inner
        NSGA-II objective.  Must share n_var / xl / xu with the real problem.
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
    use_subset_selection : bool
        Diversify the infill batch via subset_selection (default: True).
    """

    def __init__(self,
                 sampling,
                 surrogates,
                 surrogate_problem_factory,
                 crossover=None,
                 mutation=None,
                 n_doe=20,
                 n_infill=8,
                 n_gen_inner=20,
                 ga_pop_size=None,
                 warm_start_ratio=0.75,
                 use_subset_selection=True,
                 eliminate_duplicates=True,
                 dedup_key_fn=None,
                 **kwargs):
        super().__init__(eliminate_duplicates=False, **kwargs)
        self.sampling                 = sampling
        self.surrogates               = surrogates
        self.surrogate_problem_factory = surrogate_problem_factory
        self.crossover                = crossover
        self.mutation                 = mutation
        self.n_doe                    = n_doe
        self.n_infill                 = n_infill
        self.n_gen_inner              = n_gen_inner
        self.ga_pop_size              = ga_pop_size if ga_pop_size is not None else n_infill * 10
        self.warm_start_ratio         = warm_start_ratio
        self.use_subset_selection     = use_subset_selection
        self.eliminate_duplicates     = eliminate_duplicates
        # dedup_key_fn(x: np.ndarray) -> hashable: maps a decision vector to a
        # key used for archive deduplication. Defaults to a tuple of rounded ints
        # (raw-vector comparison). Override to deduplicate by canonical phenotype
        # (e.g. arch_str for NASBench-101).
        self._dedup_key = dedup_key_fn if dedup_key_fn is not None \
            else lambda x: tuple(np.round(x).astype(int).tolist())

        self._archive      = Population()
        self._archive_keys: set = set()   # incrementally maintained dedup keys
        self._init         = Initialization(sampling)
        self._prev_nd_F    = None          # for eps / indicator convergence display

    def _setup(self, problem, **kwargs):
        pass

    # ── initialisation (DOE) ──────────────────────────────────────────────────

    def _initialize_infill(self):
        """Sample the initial DOE; evaluated by the real problem right after."""
        return self._init.do(self.problem, self.n_doe, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    # ── surrogate-assisted infill ─────────────────────────────────────────────

    # ── display helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _fmt_float(val, width):
        """Format a float the same way pymoo's Column.text() does."""
        if val is None:
            return '-'.rjust(width)
        if val >= 10 or val * 1e5 < 1:
            text = f"%.{width - 7}E" % val
        else:
            text = f"%.{width - 3}f" % val
        return text.rjust(width)

    def _infill(self):
        X_arc = self._archive.get('X')   # (N, n_var) float
        F_arc = self._archive.get('F')   # (N, n_obj) float

        # 1. Fit one surrogate per predicted objective
        for s, surrogate in enumerate(self.surrogates):
            surrogate.fit(X_arc, F_arc[:, s])

        # 2. Warm-start: 75 % best archive (rank + crowding) + 25 % fresh random
        #    Strip F so the inner NSGA-II re-evaluates all on the surrogate problem.
        topx     = max(1, int(self.ga_pop_size * self.warm_start_ratio))
        top_pop  = RankAndCrowding().do(problem=self.problem, pop=self._archive, n_survive=topx)
        n_rand   = self.ga_pop_size - len(top_pop)
        if n_rand > 0:
            rand_pop = self._init.do(self.problem, n_rand, algorithm=self)
            inner_X  = np.vstack([top_pop.get('X'), rand_pop.get('X')])
        else:
            inner_X  = top_pop.get('X')
        inner_init = Population.new('X', inner_X)   # X-only → surrogate re-evaluates

        # 3. Inner NSGA-II on the surrogate problem
        surr_problem = self.surrogate_problem_factory(self.surrogates)
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

        # ── NSGA-II-style per-generation display ─────────────────────────────
        _W = {'gen': 6, 'eval': 8, 'nds': 6, 'surr': 11, 'eps': 13, 'ind': 13}
        if self.n_gen == 2:  # First call to _infill (after DOE)
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

        # eps / indicator: max change in normalised ideal or nadir (pymoo style)
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

        # Return X-only — real problem fills F after this returns
        return Population.new('X', infill_pop.get('X'))

    def _advance(self, infills=None, **kwargs):
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    # ── helpers ───────────────────────────────────────────────────────────────

    def _select_infill(self, cand_pop, F_arc):
        """Pick n_infill candidates; pad with dedup random samples if needed."""
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
        """Incrementally update _archive_keys when new individuals are added."""
        if infills is not None and len(infills) > 0:
            for x in infills.get('X'):
                k = self._dedup_key(x)
                if k is not None:
                    self._archive_keys.add(k)

    def _sample_dedup(self, n, extra_ref=None, max_tries_factor=20):
        """Sample n candidates not already in the archive (nor in extra_ref)."""
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
