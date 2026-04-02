"""
SAMOS2 — Surrogate-Assisted Multi-Objective Search, second generation.

Drop-in pymoo Algorithm with the same interface as SAMOSMinimal:

    from strategy.surrogate.samos2 import SAMOS2

    algorithm = SAMOS2(
        sampling=...,
        surrogates=[XGBoost(100, seed=0), XGBoost(100, seed=1)],
        surrogate_problem_factory=lambda surrs: MySurrogateProblem(surrs, ...),
        crossover=...,
        mutation=...,
        n_doe=20, n_infill=20,
    )
    results = minimize(real_problem, algorithm, termination=('n_gen', n_gen), ...)

Configuration derived from the SAMOS2 ablation study (WFG 2-obj, B1200_P20):

  Surrogate      : XGBoost(100)          — best mean rank across problems
  warm_start     : 0.25                  — 75 % fresh random keeps diversity
  n_gen_inner    : 5                     — fewer gens consistently better
  pop_schedule   : 500 → 50             — large early exploration, tight late refinement
  selection      : kmeans               — k-means in F-space, best rank across problems
  use_trace      : True                 — replace-parent semantics keep archive fresh
  doe_strategy   : lhs                  — Latin Hypercube for better initial coverage

  Removed (ablation showed no benefit):
    alpha=3  — surrogate dominance-tournament rounds: (0,0) ≈ (3,10) in all metrics
    beta=10  — extra inner gens from filtered pool:    same conclusion

Per-generation flow:
  1. _initialize_infill()   — LHS-based DOE
  2. _infill()
       a. Fit surrogates on archive
       b. Warm-start inner NSGA-II: 25 % best archive + 75 % fresh random
       c. Run 5 inner gens (exp-scheduled pop 500→50)
       d. Deduplicate candidates against archive
       e. k-means cluster in F-space, pick one representative per cluster
       f. Pad with random samples if pool is scarce
  3. _advance()             — if use_trace, replace each infill's closest archive
                              parent; merge infills into archive, update dedup keys
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.util.normalization import normalize
from pymoo.util.roulette import RouletteWheelSelection
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from sklearn.cluster import KMeans
from scipy.stats import qmc as _qmc


class SAMOS2(Algorithm):
    """Surrogate-assisted MOO search — second-generation algorithm.

    Parameters
    ----------
    sampling : pymoo Sampling
        Generates valid decision vectors for the DOE and random fill-in.
    surrogates : list[surrogate]
        One fitted surrogate per predicted objective.
        Each must implement ``fit(X, y)`` and ``predict(X) -> np.ndarray``.
        For use_trace to be most effective, surrogates should generalise well.
    surrogate_problem_factory : callable or None
        ``factory(surrogates) -> pymoo Problem`` used as the inner NSGA-II
        objective.  If *None*, defaults to ``SurrogateProblemMOO``.
    crossover, mutation : pymoo operators or None
        Operators for the inner NSGA-II.
    n_doe : int
        Initial DOE size (generation 0, evaluated on the real problem).
    n_infill : int
        Real evaluations added per outer generation.
    n_gen_inner : int
        Inner NSGA-II generations per outer generation (default 5).
    ga_pop_size : int
        Base inner population size when pop scheduling is disabled.
        Ignored when pop_start / pop_end are used (exp schedule active).
    pop_start, pop_end : int
        Start / end values for exponential pop scheduling.
        Default 500 → 50 (large early exploration, tight late refinement).
    warm_start_ratio : float
        Fraction of the inner pop seeded from the best archive members;
        the remainder is fresh random (default 0.25, i.e. 75 % random).
    use_trace : bool
        If True (default), each infill candidate is linked to its closest
        archive member in X-space.  In _advance() the traced parent is
        removed before adding the infill, so each infill effectively
        replaces its nearest neighbour — keeps archive diversity high.
    eliminate_duplicates : bool or pymoo DuplicateElimination
        Passed to the inner NSGA-II (default False).
    dedup_key_fn : callable or None
        Maps a decision vector to a hashable key for archive deduplication.
        Defaults to ``tuple(round(x, 8))``.
    """

    def __init__(
        self,
        sampling,
        surrogates,
        surrogate_problem_factory=None,
        crossover=None,
        mutation=None,
        n_doe=20,
        n_infill=20,
        n_gen_inner=5,
        ga_pop_size=100,
        pop_start=500,
        pop_end=50,
        warm_start_ratio=0.25,
        use_trace=True,
        eliminate_duplicates=False,
        dedup_key_fn=None,
        **kwargs,
    ):
        super().__init__(eliminate_duplicates=False, **kwargs)
        self.sampling                   = sampling
        self.surrogates                 = surrogates
        self._surrogate_problem_factory = surrogate_problem_factory
        self.crossover                  = crossover
        self.mutation                   = mutation
        self.n_doe                      = n_doe
        self.n_infill                   = n_infill
        self.n_gen_inner                = n_gen_inner
        self.ga_pop_size                = ga_pop_size
        self.pop_start                  = pop_start
        self.pop_end                    = pop_end
        self.warm_start_ratio           = warm_start_ratio
        self.use_trace                  = use_trace
        self.eliminate_duplicates       = eliminate_duplicates
        self._dedup_key = dedup_key_fn if dedup_key_fn is not None \
            else lambda x: tuple(np.round(x, decimals=8).tolist())

        self._archive      = Population()
        self._archive_keys: set = set()
        self._init         = Initialization(sampling)
        self._n_total_gen  = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _setup(self, problem, **kwargs):
        try:
            self._n_total_gen = self.termination.n_max_gen
        except AttributeError:
            self._n_total_gen = None
        self._doe_seed = getattr(self, 'seed', 0) or 0

    def _initialize_infill(self):
        """LHS-based DOE for better initial coverage."""
        xl  = self.problem.xl.astype(float)
        xu  = self.problem.xu.astype(float)
        pts = _qmc.LatinHypercube(self.problem.n_var, seed=self._doe_seed).random(self.n_doe)
        X   = _qmc.scale(pts, xl, xu)
        return Population.new('X', X)

    def _initialize_advance(self, infills=None, **kwargs):
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    # ── main infill ────────────────────────────────────────────────────────

    def _infill(self):
        X_arc = self._archive.get('X')
        F_arc = self._archive.get('F')

        # 1. Fit surrogates
        for s, surrogate in enumerate(self.surrogates):
            surrogate.fit(X_arc, F_arc[:, s])

        # 2. Warm-start inner NSGA-II (warm_start_ratio=0.25 → 75 % fresh random)
        eff_pop = self._effective_pop_size()
        n_warm  = max(1, int(eff_pop * self.warm_start_ratio))
        top_pop = RankAndCrowding().do(
            problem=self.problem, pop=self._archive, n_survive=n_warm
        )
        n_rand = eff_pop - len(top_pop)
        if n_rand > 0:
            rand_pop = self._init.do(self.problem, n_rand, algorithm=self)
            inner_X  = np.vstack([top_pop.get('X'), rand_pop.get('X')])
        else:
            inner_X = top_pop.get('X')
        inner_init = Population.new('X', inner_X)

        # 3. Run inner NSGA-II on the surrogate problem
        surr_problem = self._build_surr_problem()
        inner_alg = NSGA2(
            pop_size=eff_pop,
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
        cand_pop = res.pop if res.pop is not None else Population.empty()

        # 4. Deduplicate candidates against the archive
        cand_pop = self._dedup(cand_pop)

        # 5. k-means select n_infill candidates; pad with random if scarce
        infill_pop = self._select_infill(cand_pop, F_arc)

        # 6. Trace assignment: link each infill to closest archive parent
        if self.use_trace and len(infill_pop) > 0:
            infill_X = infill_pop.get('X')
            dists    = np.sum((infill_X[:, None, :] - X_arc[None, :, :]) ** 2, axis=2)
            parents  = dists.argmin(axis=1)
            infill_pop.set('trace_parent', parents)

        return Population.new('X', infill_pop.get('X'))

    def _advance(self, infills=None, **kwargs):
        if self.use_trace and infills is not None and len(infills) > 0:
            trace_parents = infills.get('trace_parent')
            if trace_parents is not None:
                valid_parents = [p for p in trace_parents if p is not None]
                unique_parents = np.unique(valid_parents) if valid_parents else np.array([], dtype=int)
                n_arc = len(self._archive)
                keep_mask = np.ones(n_arc, dtype=bool)
                for pidx in unique_parents:
                    if 0 <= pidx < n_arc:
                        keep_mask[pidx] = False
                self._archive = self._archive[keep_mask]
                self._archive_keys = {
                    self._dedup_key(x) for x in self._archive.get('X')
                }
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    # ── scheduling ────────────────────────────────────────────────────────

    def _effective_pop_size(self) -> int:
        """Exponential-decay schedule: pop_start → pop_end over n_total_gen."""
        if self._n_total_gen is None or self._n_total_gen <= 1:
            return self.ga_pop_size
        t    = max(0, self.n_gen - 1)
        T    = self._n_total_gen - 1
        lo   = self.pop_end
        hi   = self.pop_start
        size = hi * (lo / max(hi, 1)) ** (t / T)
        return max(2, int(round(size)))

    # ── infill selection ──────────────────────────────────────────────────

    def _select_infill(self, cand_pop: Population, F_arc: np.ndarray) -> Population:
        """k-means select n_infill diverse candidates; pad with random if needed."""
        if len(cand_pop) == 0:
            return self._sample_dedup(self.n_infill)

        if len(cand_pop) <= self.n_infill:
            # Too few candidates — take all, pad with random
            n_random = self.n_infill - len(cand_pop)
            if n_random > 0:
                extra = self._sample_dedup(n_random, extra_ref=cand_pop)
                return Population.merge(extra, cand_pop) if len(extra) > 0 else cand_pop
            return cand_pop

        # k-means clustering in normalised F-space; pick one rep per cluster
        F       = cand_pop.get('F')
        ideal   = F.min(axis=0)
        nadir   = F.max(axis=0) + 1e-16
        F_norm  = normalize(F, ideal, nadir)
        k       = min(self.n_infill, len(cand_pop))
        labels  = KMeans(n_clusters=k, random_state=0, n_init=10).fit(F_norm).labels_
        groups  = [[] for _ in range(k)]
        for idx, lbl in enumerate(labels):
            groups[lbl].append(idx)
        selected = []
        for group in groups:
            if not group:
                continue
            crowd = calc_crowding_distance(F[group])
            sel   = RouletteWheelSelection(crowd, larger_is_better=False)
            selected.append(group[sel.next()])
        return cand_pop[selected]

    # ── surrogate problem ─────────────────────────────────────────────────

    def _build_surr_problem(self):
        """Construct the inner surrogate problem for this generation."""
        if self._surrogate_problem_factory is not None:
            return self._surrogate_problem_factory(self.surrogates)
        from problem.pymoo.surrogate_problem import SurrogateProblemMOO
        return SurrogateProblemMOO(
            self.surrogates,
            self.problem.n_var,
            self.problem.xl.copy(),
            self.problem.xu.copy(),
            real_problem=self.problem,
        )

    # ── deduplication helpers ─────────────────────────────────────────────

    def _dedup(self, pop: Population) -> Population:
        """Remove individuals already in the archive."""
        if len(pop) == 0:
            return pop
        mask = np.array(
            [self._dedup_key(x) not in self._archive_keys for x in pop.get('X')],
            dtype=bool,
        )
        return pop[mask]

    def _add_to_archive_keys(self, infills):
        if infills is not None and len(infills) > 0:
            for x in infills.get('X'):
                k = self._dedup_key(x)
                if k is not None:
                    self._archive_keys.add(k)

    def _sample_dedup(self, n: int, extra_ref=None, max_tries_factor: int = 20) -> Population:
        """Sample n candidates not already in the archive."""
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
