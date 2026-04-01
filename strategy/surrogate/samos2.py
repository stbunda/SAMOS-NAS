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

Configuration derived from the SAMOS ablation study (WFG 2-obj, B1200_P20):

  Surrogate      : XGBoost(100)          — best mean rank across problems
  warm_start     : 0.25                  — 75 % fresh random keeps diversity
  n_gen_inner    : 5                     — fewer gens consistently better
  pop_schedule   : exp_dec (500 → 50)    — large exploration budget early, tight late
  alpha          : 3                     — 3 dominance-tournament rounds to refine pool
  beta           : 10                    — 10 additional gens seeded from filtered pool
  rho            : 0.5                   — sample half of beta candidates
  selection      : subset                — best across problems
  doe_strategy   : lhs                   — Latin Hypercube for better initial coverage

Per-generation flow:
  1. _initialize_infill()   — LHS-based DOE
  2. _infill()
       a. Fit surrogates on archive
       b. Warm-start inner NSGA-II: 25 % best archive + 75 % fresh random
       c. Run 5 inner gens (exp_dec scheduled pop)
       d. Deduplicate candidates against archive
       e. 3 rounds of surrogate dominance tournament  (alpha phase)
       f. 10 more inner gens seeded from filtered pool (beta phase), dedup again
       g. Subset-select n_infill diverse candidates
       h. Pad with random samples if pool is scarce
  3. _advance()             — merge infills into archive, update dedup keys
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import qmc as _qmc

from strategy.surrogate.subset_selection import subset_selection


class SAMOS2(Algorithm):
    """Surrogate-assisted MOO search — second-generation algorithm.

    Parameters
    ----------
    sampling : pymoo Sampling
        Generates valid decision vectors for the DOE and random fill-in.
    surrogates : list[surrogate]
        One fitted surrogate per predicted objective.
        Each must implement ``fit(X, y)`` and ``predict(X) -> np.ndarray``.
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
        Ignored when pop_start / pop_end are used (exp_dec schedule active).
    pop_start, pop_end : int
        Start / end values for exponential-decay pop scheduling.
        Default 500 → 50 (large early exploration, tight late refinement).
    alpha : int
        Surrogate dominance-tournament rounds applied to the candidate pool
        after deduplication and before the beta phase (default 3).
    beta : int
        Additional inner NSGA-II generations seeded from the alpha-filtered
        pool, extending candidate diversity (default 10).
    rho : float
        Fraction of beta candidates merged into the pool (default 0.5).
    warm_start_ratio : float
        Fraction of the inner pop seeded from the best archive members;
        the remainder is fresh random (default 0.25, i.e. 75 % random).
    use_subset_selection : bool
        Diversify the infill batch via subset_selection (default True).
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
        alpha=3,
        beta=10,
        rho=0.5,
        warm_start_ratio=0.25,
        use_subset_selection=True,
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
        self.alpha                      = alpha
        self.beta                       = beta
        self.rho                        = rho
        self.warm_start_ratio           = warm_start_ratio
        self.use_subset_selection       = use_subset_selection
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

        # 5. Alpha phase — surrogate dominance-tournament rounds
        cand_pop = self._alpha_tournament(cand_pop)

        # 6. Beta phase — additional inner gens seeded from filtered pool
        cand_pop = self._beta_extend(cand_pop, eff_pop, surr_problem)
        cand_pop = self._dedup(cand_pop)

        # 7. Select n_infill candidates; pad with random if scarce
        infill_pop = self._select_infill(cand_pop, F_arc)

        return Population.new('X', infill_pop.get('X'))

    def _advance(self, infills=None, **kwargs):
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    # ── alpha / beta phases ────────────────────────────────────────────────

    def _alpha_tournament(self, cand_pop: Population) -> Population:
        """``self.alpha`` rounds of pairwise surrogate dominance tournament.

        Each round: every candidate competes against a random opponent; the
        dominated one is replaced.  Pool is pruned to unique survivors.
        """
        if self.alpha <= 0 or len(cand_pop) < 2:
            return cand_pop
        X      = cand_pop.get('X')
        F_pred = np.column_stack([s.predict(X).ravel() for s in self.surrogates])
        n      = len(X)
        idx    = np.arange(n)
        rng    = np.random.default_rng()
        for _ in range(self.alpha):
            opponents = rng.integers(0, n, size=n)
            for i in range(n):
                j  = opponents[i]
                fi = F_pred[idx[i]]
                fj = F_pred[idx[j]]
                if np.all(fj <= fi) and np.any(fj < fi):
                    idx[i] = idx[j]
        return cand_pop[np.unique(idx)]

    def _beta_extend(
        self,
        cand_pop: Population,
        eff_pop: int,
        surr_problem,
    ) -> Population:
        """``self.beta`` additional inner gens seeded from alpha-filtered pool.

        Merges a ``rho``-fraction of the resulting candidates back into the pool.
        """
        if self.beta <= 0:
            return cand_pop
        seed_X = cand_pop.get('X') if len(cand_pop) > 0 else self._archive.get('X')
        n_pad  = max(0, eff_pop - len(seed_X))
        if n_pad > 0:
            pad    = self._init.do(self.problem, n_pad, algorithm=self).get('X')
            seed_X = np.vstack([seed_X, pad])
        inner_init = Population.new('X', seed_X[:eff_pop])
        inner_alg  = NSGA2(
            pop_size=eff_pop,
            sampling=inner_init,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=self.eliminate_duplicates,
        )
        res      = minimize(surr_problem, inner_alg,
                            termination=('n_gen', self.beta), verbose=False)
        beta_pop = res.pop if res.pop is not None else Population.empty()
        if len(beta_pop) == 0:
            return cand_pop
        n_take  = max(1, int(round(self.rho * len(beta_pop))))
        chosen  = beta_pop[np.random.choice(len(beta_pop), size=n_take, replace=False)]
        return Population.merge(cand_pop, chosen) if len(cand_pop) > 0 else chosen

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
        """Subset-select n_infill diverse candidates; pad with random if needed."""
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
