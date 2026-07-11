"""
SAMOS2 --- surrogate-assisted multi-objective search over float/int vectors,
with three pluggable, independently-toggleable NAS-specific contributions on
top of the SAMOSMinimal backbone.

Drop-in pymoo Algorithm (same interface as SAMOSMinimal/NSGA2):

    from strategy.surrogate.samos2 import SAMOS2

    algorithm = SAMOS2(
        sampling=..., surrogates=..., surrogate_problem_factory=...,
        crossover=..., mutation=..., predict_obj_indices=predict_idx,
        n_doe=20, n_infill=20, n_gen_inner=20, ga_pop_size=200,
    )
    results = minimize(real_problem, algorithm, termination=('n_gen', n_gen), ...)

Per-generation behaviour matches SAMOSMinimal (see strategy/surrogate/samos_minimal.py)
with three optional seams, each defaulting to the SAMOSMinimal behaviour:

  C1 encoding_spec    : one-hot categorical / passthrough-ordinal surrogate
                        features (strategy/surrogate/encoding.py). Wraps
                        ``surrogates`` transparently; the surrogate_problem_
                        factory still calls plain fit/predict/predict_std.
  C2 canonicalizer    : canonical-phenotype archive dedup + inner-GA
                        duplicate elimination, instead of raw-vector dedup
                        (strategy/surrogate/canonical.py). Optionally also
                        collapses the surrogate training set to unique
                        phenotypes (``collapse_training_set=True``).
  C3 infill_selector  : candidate-selection policy for the n_infill real
                        evaluations (strategy/surrogate/infill.py). Defaults
                        to SAMOSMinimal's diversity-only policy; swap in
                        ``AcquisitionSelector`` for uncertainty-aware infill.

With all three left at their defaults (None / False), SAMOS2 reproduces
SAMOSMinimal's archive evolution exactly given the same seed and operators,
*except* for one deliberate correctness fix: surrogates are fit against
``F_train[:, predict_obj_indices[i]]`` (matching how SurrogateProblemEvox
reads predictions back), not positionally against ``F_train[:, i]``. The
positional version silently mistrains surrogates whenever the predicted
objectives are not a contiguous prefix of the objective vector (e.g.
c10mop MOP4/5/6/7, in1kmop MOP9) -- see SAMOSMinimal for the unpatched
behaviour used by the existing gens_to_dominate.py experiment.
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from strategy.surrogate.canonical import PhenotypeDuplicateElimination
from strategy.surrogate.encoding import EncodingAwareSurrogate
from strategy.surrogate.infill import DiversitySelector


class SAMOS2(Algorithm):
    """
    Parameters
    ----------
    sampling : pymoo Sampling
        Generates valid candidates for the DOE and random fill-in.
    surrogates : list
        Surrogate models, one per predicted objective (order must match
        ``predict_obj_indices``). Each must implement ``fit(X, y)`` /
        ``predict(X)`` (and ``predict_std(X)`` if an AcquisitionSelector
        with kind='lcb'/'hvi' is used).
    surrogate_problem_factory : callable
        ``factory(surrogates) -> pymoo Problem``, called each infill step.
        When ``constr_surrogate`` is configured (see below) it is instead
        called as ``factory(surrogates, fitted_constr_surrogate) -> Problem``
        so the inner problem can read predicted constraint values.
    predict_obj_indices : list[int]
        Column (in the outer problem's F) that each entry of ``surrogates``
        predicts, in the same order as ``surrogates``. Required whenever
        len(surrogates) > 1 and the predicted objectives are not exactly
        ``range(len(surrogates))`` -- pass the same list you give to
        SurrogateProblemEvox's ``predict_obj_indices``.
    crossover, mutation : pymoo operators
        Used by the inner NSGA-II.
    n_doe, n_infill, n_gen_inner, ga_pop_size : see SAMOSMinimal.
    use_subset_selection : bool
        Diversify the infill batch via subset_selection (default True).
        Ignored if ``infill_selector`` is given explicitly.
    encoding_spec : strategy.surrogate.encoding.EncodingSpec or None
        C1 -- wraps ``surrogates`` to one-hot categorical / passthrough
        ordinal columns before fit/predict. None -> raw integer features
        (SAMOSMinimal behaviour).
    canonicalizer : strategy.surrogate.canonical.Canonicalizer or None
        C2 -- canonical-phenotype dedup key (archive + inner-GA duplicate
        elimination), instead of the raw-vector default. None -> raw-vector
        dedup (SAMOSMinimal behaviour).
    collapse_training_set : bool
        C2 (requires canonicalizer) -- collapse the surrogate training data
        to one row per unique canonical phenotype before fitting. Default
        False.
    infill_selector : object or None
        C3 -- object implementing ``select(cand_pop, F_arc, n_infill,
        surrogates=None) -> Population`` (see strategy/surrogate/infill.py).
        None -> DiversitySelector(use_subset_selection) (SAMOSMinimal
        behaviour).
    constr_surrogate : object or None
        Constraint-handling seam for the S1-S4 campaign. A model implementing
        ``fit(X, y)`` / ``predict(X)``. When given, it is fitted each outer
        generation on the archive's ``(X, G)`` right where the objective
        surrogates are fitted, then passed as the *second* argument to
        ``surrogate_problem_factory`` so the inner problem predicts G.
        None (default) leaves every existing caller untouched -- the factory
        is still called with a single argument, and cheap/exact constraints
        (constr computed inside the inner problem via the benchmark) need no
        surrogate here.

        Note: the outer problem defines ``out['G']``, so archive individuals
        carry G/CV/feasible and the ``RankAndCrowding`` archive selection plus
        the inner NSGA-II both become feasibility-first (Deb's CDP) for free
        -- see problem/evoxbench/constrained_problem.py for the verification.
    """

    def __init__(self,
                 sampling,
                 surrogates,
                 surrogate_problem_factory,
                 predict_obj_indices=None,
                 crossover=None,
                 mutation=None,
                 n_doe=20,
                 n_infill=8,
                 n_gen_inner=20,
                 ga_pop_size=None,
                 use_subset_selection=True,
                 eliminate_duplicates=True,
                 dedup_key_fn=None,
                 inner_algorithm=None,
                 encoding_spec=None,
                 canonicalizer=None,
                 collapse_training_set=False,
                 infill_selector=None,
                 constr_surrogate=None,
                 **kwargs):
        super().__init__(eliminate_duplicates=False, **kwargs)
        self.sampling                  = sampling
        self.surrogate_problem_factory = surrogate_problem_factory
        self.crossover                 = crossover
        self.mutation                  = mutation
        self.n_doe                     = n_doe
        self.n_infill                  = n_infill
        self.n_gen_inner               = n_gen_inner
        self.ga_pop_size               = ga_pop_size if ga_pop_size is not None else n_infill * 10
        self.canonicalizer             = canonicalizer
        self.collapse_training_set     = collapse_training_set
        self.constr_surrogate          = constr_surrogate

        if predict_obj_indices is None:
            if len(surrogates) > 1:
                raise ValueError(
                    'predict_obj_indices is required when len(surrogates) > 1 '
                    '(ambiguous which F column each surrogate predicts).')
            predict_obj_indices = [0]
        self.predict_obj_indices = list(predict_obj_indices)

        # C1: encoding-aware surrogate features. Transparent to
        # surrogate_problem_factory -- it always calls .fit/.predict on
        # self.surrogates, whether or not they are encoding-wrapped.
        self.surrogates = (
            [EncodingAwareSurrogate(s, encoding_spec) for s in surrogates]
            if encoding_spec is not None else surrogates
        )

        self.inner_algorithm = inner_algorithm if inner_algorithm is not None else NSGA2

        # C2: canonical-phenotype dedup key, overriding the raw-vector
        # default, unless the caller passed an explicit dedup_key_fn.
        if dedup_key_fn is not None:
            self._dedup_key = dedup_key_fn
        elif canonicalizer is not None:
            self._dedup_key = canonicalizer.key
        else:
            self._dedup_key = lambda x: tuple(np.round(x).astype(int).tolist())

        # C2: swap the inner GA's duplicate elimination to phenotype-aware
        # too, unless the caller passed a custom eliminate_duplicates.
        if canonicalizer is not None and eliminate_duplicates is True:
            self.eliminate_duplicates = PhenotypeDuplicateElimination(canonicalizer)
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # C3: infill-candidate selection policy.
        self.infill_selector = (
            infill_selector if infill_selector is not None
            else DiversitySelector(use_subset_selection=use_subset_selection)
        )

        self._archive      = Population()
        self._archive_keys: set = set()   # incrementally maintained dedup keys
        self._init         = Initialization(sampling)
        self._prev_nd_F    = None          # for eps / indicator convergence display

    def _setup(self, problem, **kwargs):
        pass

    # ── initialisation (DOE) ──────────────────────────────────────────────────

    def _initialize_infill(self):
        return self._init.do(self.problem, self.n_doe, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    # ── display helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _fmt_float(val, width):
        if val is None:
            return '-'.rjust(width)
        if val >= 10 or val * 1e5 < 1:
            text = f"%.{width - 7}E" % val
        else:
            text = f"%.{width - 3}f" % val
        return text.rjust(width)

    # ── C2 helper: training-set collapse ──────────────────────────────────────

    def _training_set(self, X_arc, F_arc):
        """Collapse to one row per unique canonical phenotype if requested."""
        if not (self.collapse_training_set and self.canonicalizer is not None):
            return X_arc, F_arc
        keys = self.canonicalizer.keys(X_arc)
        first_idx = {}
        for i, k in enumerate(keys):
            first_idx.setdefault(k, i)
        idx = np.array(sorted(first_idx.values()))
        return X_arc[idx], F_arc[idx]

    # ── surrogate-assisted infill ─────────────────────────────────────────────

    def _infill(self):
        X_arc = self._archive.get('X')   # (N, n_var) float
        F_arc = self._archive.get('F')   # (N, n_obj) float

        # 1. Fit one surrogate per predicted objective (matching column order
        #    with predict_obj_indices, not positionally).
        X_train, F_train = self._training_set(X_arc, F_arc)
        for surrogate, orig_idx in zip(self.surrogates, self.predict_obj_indices):
            surrogate.fit(X_train, F_train[:, orig_idx])

        # Constraint surrogate (campaign S1-S4): fit on the archive's (X, G)
        # here, alongside the objective surrogates. ponytail: trains on the
        # full archive X_arc/G_arc, not the (C2) collapsed objective training
        # set -- constraint scenarios don't use collapse_training_set.
        if self.constr_surrogate is not None:
            self.constr_surrogate.fit(X_arc, self._archive.get('G')[:, 0])

        top_pop  = RankAndCrowding().do(problem=self.problem, pop=self._archive, n_survive=self.ga_pop_size)
        n_rand   = self.ga_pop_size - len(top_pop)
        if n_rand > 0:
            rand_pop = self._init.do(self.problem, n_rand, algorithm=self)
            inner_X  = np.vstack([top_pop.get('X'), rand_pop.get('X')])
        else:
            inner_X  = top_pop.get('X')
        inner_init = Population.new('X', inner_X)   # X-only → surrogate re-evaluates

        # 2. Inner MOO GA (NSGA-II by default) on the surrogate problem.
        #    Pass the fitted constraint surrogate as a 2nd factory arg only
        #    when configured, so existing single-arg factories are unaffected.
        if self.constr_surrogate is not None:
            surr_problem = self.surrogate_problem_factory(self.surrogates, self.constr_surrogate)
        else:
            surr_problem = self.surrogate_problem_factory(self.surrogates)
        inner_alg = self.inner_algorithm(
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

        # 3. Deduplicate candidates against the evaluated archive
        cand_pop = res.pop if res.pop is not None else Population.empty()
        if len(cand_pop) > 0:
            not_dup = np.array([
                self._dedup_key(cx) not in self._archive_keys
                for cx in cand_pop.get('X')
            ], dtype=bool)
            cand_pop = cand_pop[not_dup]

        # 4. Select n_infill candidates; pad with random if scarce.
        #    ponytail: the infill selector ranks candidates on F only (no G);
        #    infeasible candidates may enter the infill batch, but the
        #    feasibility-first RankAndCrowding survival on the constrained
        #    archive (CDP) filters them out at the next generation.
        infill_pop = self._select_infill(cand_pop, F_arc)

        # Return X-only — real problem fills F after this returns
        return Population.new('X', infill_pop.get('X'))

    def _advance(self, infills=None, **kwargs):
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    # ── helpers ───────────────────────────────────────────────────────────────

    def _select_infill(self, cand_pop, F_arc):
        """Pick n_infill candidates via self.infill_selector; pad with
        dedup random samples if the selector returns fewer."""
        found = self.infill_selector.select(
            cand_pop, F_arc, self.n_infill, surrogates=self.surrogates,
            n_gen=self.n_gen)

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
