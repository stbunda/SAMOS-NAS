"""experiments2/scenario_run/_ssansga2.py -- SSA-NSGA-II for the scenario_run
campaign.

One class, ``ScenarioSSANSGA2``, wrapping pysamoo's SSANSGA2 (the
surrogate-assisted NSGA-II of Blank & Deb) with the three things this campaign
needs and the stock class does not have:

1. XGBoost surrogates instead of pysamoo's ezmodel RBF/Kriging ensemble, and
   -- for the 'h4-cdp' slot -- ONE MORE surrogate on the constraint column, so
   the inner NSGA-II ranks predicted candidates by constraint domination. That
   constraint surrogate is the entire difference between this method's two
   slots: 'b1-unconstrained' passes ``constr_model=None`` and gets an outer
   problem with ``n_ieq_constr == 0`` (_problems.UnconstrainedGatedProblem), so
   neither the surrogate nor the inner survival ever sees the constraint.
   Objective surrogates and every hyper-parameter are otherwise identical --
   the two slots differ in exactly one thing, which is what makes the pair
   interpretable.

2. This campaign's INTEGER operators on the inner NSGA-II. pysamoo's ``_infill``
   hardcodes a bare ``NSGA2(pop_size, sampling)``, i.e. pymoo's real-valued
   SBX/PM defaults, which would search a continuous relaxation of the NAS
   space and round only at evaluation time -- a different variation operator
   from every other method in the campaign. ``_infill`` is therefore
   re-implemented below (same selection logic, verbatim: fit, inner NSGA-II on
   the surrogate problem, dedup against the archive, k-means + crowding
   roulette down to ``n_infills``) with the operators threaded through. The
   inner seed is threaded too, where pysamoo hardcodes ``seed=1``.

3. HardGateMixin bookkeeping, with the SAME gate semantics SAMOS2 uses rather
   than the counters-only shape of _HardGateNSGA2/_HardGateCTAEA. Those two can
   leave infeasible rows in the population because the outer problem already
   masks their F to inf and NSGA-II selection simply never picks an inf row.
   Here an inf row would be TRAINED ON: the archive is the surrogate's training
   set, and an inf label poisons the model for the whole run. So under the hard
   gate infeasible infills are dropped from ``_archive`` (X-logged for the
   rejection log) and the surrogate trains on feasible points only, exactly as
   SAMOS2 documents.

   That gating creates SAMOS2's DOE problem too: at this campaign's ~10%
   feasibility a 20-point DOE is all-infeasible about 12% of the time, leaving
   an empty archive for ``surrogate.fit``. ``_initialize_advance`` therefore
   carries SAMOS2's gated-DOE viability floor -- redraw whole DOE batches,
   every draw evaluated, counted and gated like the first, until the archive
   holds at least ``gate_min_doe_feasible`` points. Those redraws are charged
   to the run's evaluation budget and visible as waste in ``n_hf_evaluated``,
   which is exactly why run_scenario.py terminates on n_evals rather than
   n_gen.

Termination, budget accounting and the callback contract are otherwise the
stock pysamoo ones: ``self.pop`` stays the DOE (pysamoo algorithms carry state
in ``_archive``, not ``pop``), and the campaign callback already prefers
``_archive`` when it is non-empty.
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.normalization import normalize
from pymoo.util.roulette import RouletteWheelSelection
from pysamoo.core.surrogate import Surrogate
from sklearn.cluster import KMeans

from _problems import HIDDEN_G_KEY
from strategy.algorithm.algorithms import HardGateMixin
from strategy.algorithm.gpsaf import _SklearnTarget
from strategy.algorithm.ssansga2 import SSANSGA2

# Cap on gated-DOE redraws before giving up, mirroring SAMOS2's own limit: at
# that point the feasible region is too small for random initialization and a
# silent infinite redraw loop would be the only alternative.
_GATE_MAX_DOE_BATCHES = 20


def _crowding(pop):
    """Crowding distances, with pymoo's un-set values read as 0 (maximally
    crowded, i.e. last preference).

    ``Survival.do`` only ranks the FEASIBLE split through RankAndCrowding; when
    that split is short of n_survive it tops up straight from the
    CV-sorted infeasible ones, which therefore never get a 'crowding' attribute
    at all. On the h4-cdp slot the inner surrogate problem does declare a
    constraint, so predicted-infeasible candidates reach the k-means selection
    below with crowding None and pysamoo's own ``.get('crowding').argsort()``
    raises. Upstream never hit this because it was only ever run
    unconstrained."""
    return np.array([0.0 if c is None else float(c) for c in pop.get('crowding')],
                    dtype=float)


class ScenarioSSANSGA2(HardGateMixin, SSANSGA2):
    """See the module docstring. ``constr_model`` None => the 'b1-unconstrained'
    slot (no constraint surrogate; pair with an outer problem declaring
    ``n_ieq_constr == 0``), a model => the 'h4-cdp' slot."""

    def __init__(self, obj_models, constr_model=None, hard_gate=False,
                 crossover=None, mutation=None, eliminate_duplicates=None,
                 inner_seed=1, gate_min_doe_feasible=2, **kwargs):
        kwargs.setdefault('output', MultiObjectiveOutput())
        super().__init__(**kwargs)
        self._init_hard_gate(hard_gate)
        self._obj_models   = list(obj_models)
        self._constr_model = constr_model
        self._crossover    = crossover
        self._mutation     = mutation
        self._elim         = eliminate_duplicates
        self._inner_seed   = int(inner_seed)
        self.gate_min_doe_feasible = int(gate_min_doe_feasible)

    # ── surrogate wiring ──────────────────────────────────────────────────

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        targets = [_SklearnTarget(('F', i), m) for i, m in enumerate(self._obj_models)]
        if self._constr_model is not None:
            if problem.n_ieq_constr != 1:
                raise ValueError(
                    f'constr_model given but the outer problem declares '
                    f'n_ieq_constr={problem.n_ieq_constr} -- the constraint surrogate '
                    f'fills exactly one G column (the scenario_run campaign is '
                    f'single-constraint throughout).')
            targets.append(_SklearnTarget(('G', 0), self._constr_model))
        elif problem.n_ieq_constr != 0:
            raise ValueError(
                f'no constr_model given, but the outer problem declares '
                f'n_ieq_constr={problem.n_ieq_constr} -- pysamoo requires one target '
                f'per declared G column, so the b1-unconstrained slot must be paired '
                f'with UnconstrainedGatedProblem.')
        self.surrogate = Surrogate(problem, targets)

    # ── hard gate ─────────────────────────────────────────────────────────

    def _gate_violation(self, pop):
        """b1-unconstrained's outer problem declares n_ieq_constr=0 and
        publishes its violation under HIDDEN_G_KEY, so the inherited 'G' lookup
        would find no constraint and count every draw feasible (same override,
        same reason, as algorithms._HardGateNSGA2._gate_violation).

        Read from there rather than copying the column onto 'G' after
        evaluation: archive members are reused verbatim as the inner NSGA-II's
        starting population, and a 1-column G on them next to the 0-column G of
        surrogate-evaluated offspring makes pymoo's ``Population.get('G')``
        ragged, which crashes the inner run's ``result()``."""
        if getattr(self.problem, 'hidden_constraints', False):
            hidden = np.asarray(pop.get(HIDDEN_G_KEY), dtype=float)
            return hidden.reshape(len(pop), -1).max(axis=1)
        return super()._gate_violation(pop)

    def _initialize_advance(self, infills=None, **kwargs):
        kept = self._gate_keep(infills)
        if self.hard_gate:
            n_extra = 0
            while len(kept) < self.gate_min_doe_feasible:
                if n_extra >= _GATE_MAX_DOE_BATCHES:
                    raise RuntimeError(
                        f'hard gate: DOE produced only {len(kept)} feasible point(s) '
                        f'after {n_extra} extra batch(es) of {self.n_initial_doe} '
                        f'(need >= {self.gate_min_doe_feasible}). The feasible region '
                        f'is too small for random initialization at this threshold.')
                extra = self.initialization.do(self.problem, self.n_initial_doe,
                                                algorithm=self)
                self.evaluator.eval(self.problem, extra, algorithm=self)
                kept = Population.merge(kept, self._gate_keep(extra))
                n_extra += 1
        super()._initialize_advance(infills=kept, **kwargs)

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills=self._gate_keep(infills), **kwargs)

    # ── infill ────────────────────────────────────────────────────────────

    def _infill(self):
        """pysamoo SSANSGA2._infill with this campaign's integer operators and
        seed threaded into the inner NSGA-II; selection logic unchanged."""
        self.surrogate.fit(self._archive)
        problem = self.surrogate.problem()

        if self.surr_sampling == 'current':
            sampling = self._archive
        elif self.surr_sampling == 'random':
            sampling = self.initialization.sampling
        else:
            raise Exception('Unknown surrogate sampling strategy.')

        algorithm = NSGA2(pop_size=self.surr_pop_size, sampling=sampling,
                          crossover=self._crossover, mutation=self._mutation,
                          eliminate_duplicates=self._elim,
                          output=MultiObjectiveOutput())
        res = minimize(problem, algorithm, ('n_gen', self.surr_n_gen),
                       seed=self._inner_seed, verbose=False)

        cand = DefaultDuplicateElimination(epsilon=self.surr_eps_elim).do(res.pop, self._archive)

        if len(cand) == 0:
            # Every inner candidate already sits in the archive. Returning an
            # empty infill set would advance a generation without consuming any
            # budget, and n_evals termination would then never be reached --
            # the run hangs. Fall back to fresh random draws, which also
            # re-injects diversity, exactly the situation that emptied cand.
            return self.initialization.do(self.problem, self.n_infills, algorithm=self)

        if len(cand) <= self.n_infills:
            return Population.new(X=cand.get('X'))

        # Scale the k-means space to the inner front, falling back to the
        # candidate set itself when there is no front to scale to: pymoo's
        # Result sets opt=None whenever the final population holds NOTHING
        # predicted-feasible (return_least_infeasible defaults to False). On
        # the h4-cdp slot that happens for real -- the constraint surrogate can
        # call every inner candidate infeasible -- and upstream pysamoo, only
        # ever run unconstrained, dereferences res.opt unguarded.
        front = res.opt if res.opt is not None and len(res.opt) > 0 else cand
        ideal = front.get('F').min(axis=0)
        nadir = front.get('F').max(axis=0) + 1e-16
        vals  = normalize(cand.get('F'), ideal, nadir)

        kmeans = KMeans(n_clusters=self.n_infills, random_state=self._inner_seed).fit(vals)
        groups = [[] for _ in range(self.n_infills)]
        for k, i in enumerate(kmeans.labels_):
            groups[i].append(k)

        S = []
        for group in groups:
            if len(group) > 0:
                fitness = _crowding(cand[group]).argsort()
                selection = RouletteWheelSelection(fitness, larger_is_better=False)
                S.append(group[selection.next()])

        return Population.new(X=cand[S].get('X'))

    def _set_optimum(self):
        # Gated edge case: an all-rejected DOE cannot happen (the viability
        # floor above), but an all-rejected infill batch leaves the archive
        # unchanged and pysamoo's non-dominated sort would still run on it.
        # Keep the previous optimum when there is nothing to sort.
        if len(self._archive) == 0:
            if self.opt is None:
                self.opt = Population.empty()
            return
        super()._set_optimum()
