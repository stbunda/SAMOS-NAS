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
    constr_surrogate : object, sequence, or None
        Constraint-handling seam. A model implementing ``fit(X, y)`` /
        ``predict(X)`` (single constraint), or a sequence with one slot per
        constraint column -- a model to predict that G column, or None for
        an exact column computed inside the inner problem (so a mixed
        exact/predicted violation vector is expressible). Every non-None
        slot is fitted each outer generation on the archive's ``(X, G[:, j])``
        right where the objective surrogates are fitted, then the whole
        object (model or sequence) is passed as the *second* argument to
        ``surrogate_problem_factory``. None (default) leaves every existing
        caller untouched -- the factory is still called with a single
        argument, and all-exact constraints need no surrogate here.

        Note: the outer problem defines ``out['G']``, so archive individuals
        carry G/CV/feasible and the ``RankAndCrowding`` archive selection plus
        the inner NSGA-II both become feasibility-first (Deb's CDP) for free
        -- see problem/evoxbench/constrained_problem.py for the verification.
    hard_gate : bool
        Hard-evaluability gate. When True, every
        high-fidelity-evaluated individual whose violation is positive is
        counted (``n_hf_evaluated`` / ``n_hf_feasible``; infeasible = any
        constraint violated), logged to the rejection log (``_rejected_X`` /
        ``_rejected_G`` -- X and violation only, never F: under gated-hard
        semantics the objective values of an infeasible run are
        unobservable; one violation per row for a single constraint, the
        full (n, n_constr) violation rows otherwise), and DISCARDED -- it never enters
        ``_archive``, so objective surrogates train on feasible points only.
        The constraint surrogate instead trains on archive UNION rejection
        log (the infeasibility signal must come from somewhere). Rejected X
        keys still enter ``_archive_keys``: a known-failed architecture is
        never re-proposed (you know which ones failed). The DOE additionally
        redraws full batches (each counted + gated like any evaluation) until
        at least ``gate_min_doe_feasible`` feasible points exist, up to
        ``_GATE_MAX_DOE_BATCHES`` batches -- a gated run must not start with
        an unusable archive, but the policy stays minimal so the evaluation
        budget is not silently inflated (RuntimeError if even that fails).
        Default False = ungated behaviour, bit-exact. The counters are
        maintained (cheaply) even when False, so the feasibility-aware
        callback can always prefer them over archive-derived counts.
    gate_g_fn : callable or None
        Violation source for the gate/counters when the outer problem defines
        no ``G`` (the b0-as-obj / b0-nsga2 rows, where the constrained
        metrics are ordinary objective columns): ``gate_g_fn(pop) -> (n,) or
        (n, n_constr) array``, feasible <=> every entry of a row <= 0,
        evaluated on the already-evaluated population. Ignored whenever the
        population carries real G.
    gate_min_doe_feasible : int
        Gated-DOE viability floor (see ``hard_gate``). Default 2.
    """

    _GATE_MAX_DOE_BATCHES = 10   # extra full-n_doe redraws before giving up

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
                 hard_gate=False,
                 gate_g_fn=None,
                 gate_min_doe_feasible=2,
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
        self.hard_gate                 = bool(hard_gate)
        self.gate_g_fn                 = gate_g_fn
        self.gate_min_doe_feasible     = int(gate_min_doe_feasible)

        # High-fidelity evaluation counters + rejection log. Counters
        # run gated or not (the callback prefers them over archive-derived
        # counts); the rejection log only fills when hard_gate is True.
        self.n_hf_evaluated = 0
        self.n_hf_feasible  = 0
        self._rejected_X    = None   # (n_rej, n_var) float or None
        self._rejected_G    = None   # (n_rej,) violations, (n_rej, n_constr) for multi, or None

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

    # ── hard-evaluability gate ───────────────────────────────────────────────

    def _violations_of(self, pop):
        """Per-individual violation matrix (n, n_constr) for the
        gate/counters: the population's own G columns when the outer problem
        defines them, else ``gate_g_fn`` (b0 rows; may return (n,) or
        (n, k)), else zeros (unconstrained legacy callers). Feasible <=>
        every column <= 0."""
        G = pop.get('G')
        if G is not None and np.asarray(G).size > 0:
            return np.asarray(G, dtype=float).reshape(len(pop), -1)
        if self.gate_g_fn is not None:
            return np.asarray(self.gate_g_fn(pop), dtype=float).reshape(len(pop), -1)
        return np.zeros((len(pop), 1))

    def _gate_merge(self, infills):
        """Count every evaluated infill, then merge into the archive -- all
        of them ungated, only the feasible subset when hard_gate is on
        (infeasible X/violation go to the rejection log; their keys still
        enter _archive_keys so a known-failed arch is never re-proposed).
        Returns the merged (kept) subset."""
        if infills is None or len(infills) == 0:
            return infills
        viol     = self._violations_of(infills)     # (n, n_constr)
        viol_max = viol.max(axis=1)
        self.n_hf_evaluated += len(infills)
        self.n_hf_feasible  += int(np.sum(viol_max <= 0))

        kept = infills
        if self.hard_gate:
            feas_mask = viol_max <= 0
            kept      = infills[feas_mask]
            rejected  = infills[~feas_mask]
            if len(rejected) > 0:
                rx = rejected.get('X').astype(float)
                # Single constraint keeps the legacy 1-D log; multiple
                # constraints log the full per-constraint violation rows.
                rg = viol[~feas_mask]
                if rg.shape[1] == 1:
                    rg = rg[:, 0]
                self._rejected_X = rx if self._rejected_X is None else np.vstack([self._rejected_X, rx])
                self._rejected_G = rg if self._rejected_G is None else np.concatenate([self._rejected_G, rg], axis=0)

        self._archive = Population.merge(self._archive, kept)
        self._add_to_archive_keys(infills)   # kept AND rejected: never re-propose
        return kept

    def _set_optimum(self):
        # Gated edge case: a fully-rejected infill batch leaves self.pop
        # empty and pymoo's filter_optimum(empty) returns None, which the
        # verbose output machinery cannot handle. Keep the previous optimum
        # (display/result state only -- the archive gate is unaffected).
        if self.hard_gate and (self.pop is None or len(self.pop) == 0):
            if self.opt is None:
                self.opt = Population.empty()
            return
        super()._set_optimum()

    # ── initialisation (DOE) ──────────────────────────────────────────────────

    def _initialize_infill(self):
        return self._init.do(self.problem, self.n_doe, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        kept = self._gate_merge(infills)
        # Gated-DOE viability floor: redraw full batches -- every draw
        # evaluated, counted and gated exactly like the first -- until the
        # archive holds at least gate_min_doe_feasible feasible points. The
        # floor is deliberately minimal (default 2) so the policy almost
        # never triggers at the campaign's feasibility levels and the
        # evaluation budget is not silently inflated; the redraws that do
        # happen are visible in n_hf_evaluated (waste).
        if self.hard_gate:
            n_extra = 0
            while len(self._archive) < self.gate_min_doe_feasible:
                if n_extra >= self._GATE_MAX_DOE_BATCHES:
                    raise RuntimeError(
                        f'hard gate: DOE produced only {len(self._archive)} feasible '
                        f'point(s) after {n_extra} extra batch(es) of {self.n_doe} '
                        f'(need >= {self.gate_min_doe_feasible}). The feasible region '
                        f'is too small for random initialization at this threshold.')
                extra = self._init.do(self.problem, self.n_doe, algorithm=self)
                self.evaluator.eval(self.problem, extra)
                extra_kept = self._gate_merge(extra)
                kept = Population.merge(kept, extra_kept) if kept is not None else extra_kept
                n_extra += 1
        self.pop = kept if kept is not None else infills

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

        # Tripwire: the archive must never hold a non-finite fitness.
        # Ungated runs guard non-finite benchmark output to 1.0 upstream;
        # gated runs mask infeasible F to inf but the gate drops those rows
        # before the merge -- an inf/NaN here means the gate leaked.
        assert np.isfinite(F_arc).all(), \
            'non-finite F in SAMOS2 archive: hard-gate leak or unguarded fitness'

        # 1. Fit one surrogate per predicted objective (matching column order
        #    with predict_obj_indices, not positionally).
        X_train, F_train = self._training_set(X_arc, F_arc)
        for surrogate, orig_idx in zip(self.surrogates, self.predict_obj_indices):
            surrogate.fit(X_train, F_train[:, orig_idx])

        # Constraint surrogate(s): fit on the archive's (X, G) here, alongside
        # the objective surrogates -- one model per predicted constraint
        # column when a per-slot sequence is configured (None slots are exact,
        # computed inside the inner problem). ponytail: trains on the full
        # archive X_arc/G_arc, not the (C2) collapsed objective training set
        # -- constraint scenarios don't use collapse_training_set.
        # Hard gate: a gated archive is all-feasible (G <= 0 only), so
        # the boundary signal lives in the rejection log -- train on archive
        # UNION rejected (X, violation). Realistic: which architectures
        # failed IS observable, their objective values are not.
        if self.constr_surrogate is not None:
            G_arc = self._archive.get('G').reshape(len(self._archive), -1)
            rej_G = None
            if self.hard_gate and self._rejected_X is not None:
                rej_G = np.asarray(self._rejected_G, dtype=float
                                   ).reshape(len(self._rejected_X), -1)
            surr_list = (self.constr_surrogate
                         if isinstance(self.constr_surrogate, (list, tuple))
                         else [self.constr_surrogate])
            for j, surrogate in enumerate(surr_list):
                if surrogate is None:
                    continue
                X_c, g_c = X_arc, G_arc[:, j]
                if rej_G is not None:
                    X_c = np.vstack([X_c, self._rejected_X])
                    g_c = np.concatenate([g_c, rej_G[:, j]])
                surrogate.fit(X_c, g_c)

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
            f" {self.n_gen:>{_W['gen']}} | {self.n_hf_evaluated:>{_W['eval']}}"
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
        kept = self._gate_merge(infills)
        self.pop = kept if kept is not None else infills

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
