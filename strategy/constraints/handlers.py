"""strategy/constraints/handlers.py -- round-2 constraint handlers (H1/H3/H5/H6).

Each handler is a small object that plugs into an existing seam -- no edits
to SAMOS2 (strategy/surrogate/samos2.py) or the constrained problem pair
(problem/evoxbench/constrained_problem.py) are required.

Feasibility convention throughout: an inequality constraint value ``g`` is
feasible iff ``g <= 0`` (matching ConstrainedEvoXBenchProblem's
``G = (metric - T) / T``). The aggregated constraint violation of an
individual is ``CV = sum(max(0, G))`` (pymoo's ``calc_cv`` for inequality
constraints); ``CV <= 0`` <=> feasible.

--------------------------------------------------------------------------
Wiring recipes (a runner engineer needs only this section + the signatures)
--------------------------------------------------------------------------

H1 RejectionSampling
  * random shape (RandomGA): replace the sampler.

        g_fn = make_benchmark_g_fn(benchmark, constr_index, threshold)
        sampler = RejectionSampling(EvoxBenchSampler(xl, xu), g_fn)
        RandomGA(pop_size=pop, sampling=sampler, eliminate_duplicates=elim)

    Every DOE / infill batch RandomGA draws is now rejection-resampled to
    feasibility (bounded retries, then padded with the least-violating).

  * samos shape (SAMOS2 inner GA): pass an infill selector (the C3 seam).

        SAMOS2(..., infill_selector=RejectionInfillSelector())

    The inner NSGA-II candidate pool already carries the inner problem's G
    (predicted for `samos`, exact for `samos-cheap` S1/S3); the selector
    drops infeasible candidates before the diversity subset step and pads
    the infill batch with the least-violating survivors.

  IMPORTANT (H3 and H5 factory hook): call ``minimize(..., copy_algorithm=False)``
  for these two. The factory below holds a live handle (``algo_ref``) to the
  running SAMOS2 instance to read its archive; pymoo's default
  ``copy_algorithm=True`` deep-copies the algorithm and runs the COPY, severing
  that handle (the state objects ``pen`` / ``eps`` still persist -- functions
  survive deepcopy referencing the originals -- but ``algo_ref[0]._archive``
  would then be the unused original's empty archive). ``copy_algorithm=False``
  makes the passed instance the running one; the FeasibilityAwareEvoxBenchCallback
  and ``results.algorithm._archive`` keep working (they already read off the
  running instance).

H3 AdaptivePenaltyProblem  (SAMOS2 inner GA; soft scenarios' adaptive variant)
    Hold ONE state object for the whole run; adapt() it once per outer
    generation on the archive feasibility ratio; wrap() the freshly-built
    inner problem inside the factory:

        pen = AdaptivePenaltyProblem(w0=1.0, target=0.5, c=1.2)
        algo_ref = []                      # late-bound handle to the SAMOS2 instance
        def factory(surrs, fitted_constr=None):
            arc = algo_ref[0]._archive     # updated at the top of _infill
            if len(arc) > 0:
                pen.adapt(feasible_fraction(arc))
            inner = ConstrainedSurrogateProblemEvox(surrs, ..., constr_surrogate=fitted_constr)
            return pen.wrap(inner)         # penalized inner problem, current weight
        algo = SAMOS2(..., surrogate_problem_factory=factory)
        algo_ref.append(algo)
        minimize(problem, algo, ..., copy_algorithm=False)

    The factory is called exactly once per outer generation (inside
    SAMOS2._infill, after the archive is updated), so adapt() runs once per
    generation with no SAMOS2 edit. Only the inner problem is wrapped, so the
    outer archive / saved pkls always keep unpenalized F and raw G.

H5 EpsilonRelaxation  (SAMOS2 inner GA; hard scenarios, primary use)
    Same per-generation factory hook as H3, but wrapping the inner problem so
    its reported G is relaxed by epsilon(t); native CDP then ranks on the
    relaxed feasibility for free:

        eps = EpsilonRelaxation(n_gen_total=n_gen)     # T = outer generations
        algo_ref = []
        def factory(surrs, fitted_constr=None):
            eps.maybe_init_eps0(algo_ref[0]._archive)  # eps0 = mean CV of the DOE
            inner = ConstrainedSurrogateProblemEvox(surrs, ..., constr_surrogate=fitted_constr)
            wrapped = eps.wrap(inner)                   # snapshots epsilon(t)
            eps.advance()                              # t -> t+1 for next generation
            return wrapped
        algo = SAMOS2(..., surrogate_problem_factory=factory)
        algo_ref.append(algo)
        minimize(problem, algo, ..., copy_algorithm=False)

    Only the inner problem's G is shifted; the outer archive keeps raw G.

H6 DominanceStochasticRanking  (SAMOS2 inner GA survival, or outer ranking)
    Use as the inner NSGA-II survival via a functools.partial inner_algorithm
    (SAMOS2 constructs the inner algorithm as
    ``inner_algorithm(pop_size=..., sampling=..., crossover=..., mutation=...,
    eliminate_duplicates=...)`` -- partial supplies ``survival``):

        from functools import partial
        from pymoo.algorithms.moo.nsga2 import NSGA2
        dsr = DominanceStochasticRanking(Pf=0.45, seed=seed)
        SAMOS2(..., inner_algorithm=partial(NSGA2, survival=dsr))

    It is an ordinary pymoo Survival, so ``NSGA2(survival=dsr)`` also ranks
    the outer constrained problem directly.
"""

from functools import partial

import numpy as np
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.core.individual import calc_cv
from pymoo.core.meta import Meta
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.util.dominator import Dominator
from pymoo.util.misc import from_dict

from strategy.surrogate.infill import DiversitySelector


# ══════════════════════════════════════════════════════════════════════════
# shared helpers
# ══════════════════════════════════════════════════════════════════════════

def _cv_of(pop):
    """Per-individual aggregated inequality violation CV = sum(max(0, G)),
    shape (len(pop),). Prefers the population's raw G (present on constrained
    problems that keep out['G']); falls back to a populated CV column, then to
    zeros (unconstrained). Feasible <=> CV <= 0."""
    n = len(pop)
    G = pop.get('G')
    G = np.asarray(G) if G is not None else None
    if G is not None and G.size > 0:
        return np.maximum(0.0, G.reshape(n, -1)).sum(axis=1)
    CV = pop.get('CV')
    CV = np.asarray(CV) if CV is not None else None
    if CV is not None and CV.size > 0:
        return CV.reshape(n, -1)[:, 0]
    return np.zeros(n)


def feasible_fraction(pop_or_g):
    """Fraction of feasible individuals (CV <= 0). Accepts a pymoo Population
    or a raw G array (shape (n,) or (n, k)). Empty input -> 0.0. Used by H3 to
    feed the archive feasibility ratio into ``adapt``."""
    if hasattr(pop_or_g, 'get'):          # Population
        cv = _cv_of(pop_or_g)
    else:
        g = np.asarray(pop_or_g, dtype=float)
        g = g.reshape(len(g), -1) if g.ndim > 1 else g.reshape(-1, 1)
        cv = np.maximum(0.0, g).sum(axis=1)
    if len(cv) == 0:
        return 0.0
    return float(np.mean(cv <= 0.0))


def make_benchmark_g_fn(benchmark, constr_index, threshold, no_norm=False):
    """Build an exact ``g_fn(X) -> violations`` for H1's RandomGA shape, using
    the same G = (metric - T) / T convention (and normalize-once baseline) as
    ConstrainedEvoXBenchProblem. Returns a 1-D array (feasible <= 0)."""
    from problem.evoxbench.constrained_problem import _violation

    def g_fn(X):
        X_int = np.round(np.asarray(X)).astype(int)
        F = benchmark.evaluate(X_int, true_eval=False)
        if not no_norm and not benchmark.normalized_objectives:
            F = benchmark.normalize(F)
        F = np.where(np.isfinite(F), F, 1.0)
        return _violation(F[:, int(constr_index)], float(threshold)).reshape(-1)

    return g_fn


# ══════════════════════════════════════════════════════════════════════════
# H1 -- rejection sampling / filtering
# ══════════════════════════════════════════════════════════════════════════

class RejectionSampling(Sampling):
    """H1 (RandomGA shape) -- feasibility rejection sampling.

    Wraps a base pymoo ``Sampling`` and a ``g_fn(X) -> violations`` (feasible
    <= 0). Each ``do(problem, n)`` oversamples in batches of ``n`` from the
    base sampler, keeps only feasible candidates, and retries until ``n``
    feasible are collected. After the retry ceiling it pads the shortfall with
    the least-violating infeasible candidates seen (never loops forever).

    Drop-in for RandomGA's ``sampling=`` argument: RandomGA calls
    ``sampling.do(problem, n)`` for both its DOE and every infill batch, so all
    of RandomGA's candidates become rejection-resampled.

    Parameters
    ----------
    base_sampling : pymoo Sampling
        The unconstrained sampler to draw raw candidates from.
    g_fn : callable
        ``g_fn(X) -> array`` of per-row violations (feasible <= 0), e.g. from
        ``make_benchmark_g_fn`` (exact benchmark constraint for `random`).
    max_resample : int
        Retry ceiling = number of batch draws before padding. Default 10.
    """

    def __init__(self, base_sampling, g_fn, max_resample: int = 10):
        super().__init__()
        self.base_sampling = base_sampling
        self.g_fn = g_fn
        # ponytail: hard ceiling of `max_resample` batch draws (default 10x the
        # requested batch) -- then pad with least-violating rather than loop
        # forever on a (near-)empty feasible region.
        self.max_resample = int(max_resample)

    def _do(self, problem, n_samples, **kwargs):
        X_batches, g_batches, n_feas = [], [], 0
        for _ in range(max(1, self.max_resample)):
            Xb = self.base_sampling.do(problem, n_samples, **kwargs).get('X')
            gb = np.asarray(self.g_fn(Xb), dtype=float).reshape(-1)
            X_batches.append(Xb)
            g_batches.append(gb)
            n_feas += int(np.sum(gb <= 0.0))
            if n_feas >= n_samples:
                break

        X_all = np.vstack(X_batches)
        g_all = np.concatenate(g_batches)
        feas = np.where(g_all <= 0.0)[0]

        if len(feas) >= n_samples:
            chosen = feas[:n_samples]
        else:
            infeas = np.where(g_all > 0.0)[0]
            infeas = infeas[np.argsort(g_all[infeas])]      # least-violating first
            need = n_samples - len(feas)
            chosen = np.concatenate([feas, infeas[:need]])
        return X_all[chosen]


class RejectionInfillSelector:
    """H1 (SAMOS2 inner-GA shape) -- feasibility filter on the infill batch.

    An infill selector for SAMOS2's C3 ``infill_selector`` seam. The inner
    NSGA-II candidate pool already carries the inner problem's G (predicted or
    exact), so this reads their violation directly: feasible candidates are
    handed to a base diversity selector; if fewer than ``n_infill`` are
    feasible it returns all feasible ones padded with the least-violating
    infeasible candidates. (SAMOS2 further pads any residual shortfall with
    fresh random samples, exactly as for the default selector.)

    Parameters
    ----------
    base_selector : object or None
        Selector applied to the feasible subset (must expose the same
        ``select(cand_pop, F_arc, n_infill, surrogates=None, n_gen=None)``
        contract as DiversitySelector). Default: DiversitySelector().
    """

    def __init__(self, base_selector=None):
        self.base_selector = base_selector or DiversitySelector(use_subset_selection=True)

    def select(self, cand_pop, F_arc, n_infill, surrogates=None, n_gen=None, **kwargs):
        if len(cand_pop) == 0:
            return Population.empty()

        cv = _cv_of(cand_pop)
        feas_mask = cv <= 0.0
        feas = cand_pop[feas_mask]

        if len(feas) >= n_infill:
            return self.base_selector.select(
                feas, F_arc, n_infill, surrogates=surrogates, n_gen=n_gen)

        # Not enough feasible candidates: keep all feasible, pad with the
        # least-violating infeasible ones. ponytail: no resampling loop here --
        # the inner GA already produced a fixed pool; SAMOS2 tops up any
        # remaining shortfall with random dedup samples.
        infeas = cand_pop[~feas_mask]
        order = np.argsort(cv[~feas_mask])
        need = n_infill - len(feas)
        pad = infeas[order[:need]]
        if len(feas) == 0:
            return pad
        if len(pad) == 0:
            return feas
        return Population.merge(feas, pad)


# ══════════════════════════════════════════════════════════════════════════
# H3 -- adaptive multiplicative penalty
# ══════════════════════════════════════════════════════════════════════════

class _PenaltyProblemMO(ConstraintsAsPenalty):
    """Multi-objective static-penalty problem: F + penalty * CV, broadcast
    across every objective column.

    pymoo 0.6.1.1's ``ConstraintsAsPenalty.do`` does ``F + penalty *
    np.reshape(CV, F.shape)``, but ``calc_cv`` returns one aggregated
    violation per individual (shape (n,)), so the reshape raises for
    n_obj > 1. Here CV is broadcast with ``CV.reshape(-1, 1)`` -- the standard
    multi-objective generalization of static penalty
    (run_constraint.py::_ConstraintsAsPenaltyMO carries the same fix).
    """

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)
        F, G, H = from_dict(out, 'F', 'G', 'H')
        out['__F__'], out['__G__'], out['__H__'] = F, G, H
        CV = calc_cv(G=G, H=H)
        out['F'] = F + self.penalty * CV.reshape(-1, 1)
        out.pop('G', None)   # pop (not del): a problem with no eq/ieq output
        out.pop('H', None)   # may omit the key entirely (e.g. n_eq_constr=0).
        return out


class AdaptivePenaltyProblem:
    """H3 -- adaptive-penalty controller/state (multiplicative weight update).

    Hold ONE instance for the whole run. Base semantics are the multi-objective
    static penalty (``_PenaltyProblemMO``: F + w * CV); the weight ``w`` moves
    each outer generation toward a target feasibility ratio:

        adapt(feasible_ratio):
            w *= c   if feasible_ratio < target   (too few feasible -> penalize harder)
            w /= c   otherwise                     (enough feasible  -> relax)
        w is clipped to [w_min, w_max].

    Usage (see module wiring recipe H3): each outer generation the factory
    calls ``adapt(feasible_fraction(archive))`` then ``wrap(inner_problem)``;
    ``wrap`` builds a penalized inner problem carrying the current weight. The
    controller and the (rebuilt-every-generation) inner problem are kept
    separate precisely because SAMOS2 discards the inner problem each
    generation -- only the weight must persist.

    Parameters
    ----------
    w0 : float
        Initial penalty weight. Default 1.0.
    target : float
        Target feasible fraction. Default 0.5.
    c : float
        Multiplicative step (> 1). Default 1.2.
    w_min, w_max : float
        Weight clip bounds. Default 1e-3, 1e3.
    """

    def __init__(self, w0: float = 1.0, target: float = 0.5, c: float = 1.2,
                 w_min: float = 1e-3, w_max: float = 1e3):
        self.weight = float(w0)
        self.target = float(target)
        self.c = float(c)
        self.w_min = float(w_min)
        self.w_max = float(w_max)

    def adapt(self, feasible_ratio: float) -> float:
        """Update and return the weight for the current generation."""
        if feasible_ratio < self.target:
            self.weight *= self.c
        else:
            self.weight /= self.c
        self.weight = float(np.clip(self.weight, self.w_min, self.w_max))
        return self.weight

    def wrap(self, problem):
        """Return a penalized copy of ``problem`` with the current weight."""
        return _PenaltyProblemMO(problem, penalty=self.weight)


# ══════════════════════════════════════════════════════════════════════════
# H5 -- epsilon relaxation
#
# pymoo-eps investigation (spec ask): can pymoo's
# AdaptiveEpsilonConstraintHandling (pymoo/constraints/eps.py) wrap our inner
# NSGA-II directly?  Outcome: NO -- it wraps the wrong clock. That class
# (a) sets eps0 = mean CV in `_initialize_advance`, i.e. from the wrapped
# algorithm's OWN initial population, and (b) schedules eps via
# `self.termination.perc`, the wrapped algorithm's own termination progress.
# SAMOS2 rebuilds and re-runs a FRESH inner NSGA-II for n_gen_inner
# generations every outer generation, so wrapping the inner run would reset
# eps0 and restart the 1->0 decay inside each outer generation, decaying over
# the inner budget rather than the whole outer run. The intended schedule is
# outer-clocked (eps0 = mean CV of the initial DOE; decay to 0 at 50% of the
# TOTAL outer generations). Hence the small custom wrapper below, driven by an
# external state the runner advances once per outer generation. (Mechanically
# one could pass inner_algorithm=partial(lambda **k:
# AdaptiveEpsilonConstraintHandling(NSGA2(**k))), but the schedule semantics
# would still be wrong.)
# ══════════════════════════════════════════════════════════════════════════

class _EpsRelaxedProblem(Meta, Problem):
    """Problem wrapper that reports the epsilon-relaxed constraint
    ``G' = G - eps`` while keeping ``n_ieq_constr`` intact, so pymoo's native
    feasibility-first survival (CDP) ranks on the relaxed feasibility. Only the
    G column is shifted; F is untouched."""

    def __init__(self, problem, eps: float):
        super().__init__(problem)      # Meta deep-copies the wrapped problem
        self._eps = float(eps)

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)
        G = out.get('G', None)
        if G is not None:
            out['G'] = np.asarray(G, dtype=float) - self._eps
        return out


class EpsilonRelaxation:
    """H5 -- pymoo-style epsilon-relaxed feasibility schedule.

    ε₀ = mean CV of the initial DOE; ε(t) = ε₀ * max(0, 1 − t / (perc_eps_until * T)),
    with ``perc_eps_until = 0.5`` and ``T`` = total outer generations. So ε
    decays linearly to 0 at 50 % of the budget and stays 0 afterward -- the
    same schedule pymoo's AdaptiveEpsilonConstraintHandling uses, but driven by
    the OUTER-loop clock (see the module-level investigation note below on why
    the pymoo class cannot be dropped onto the SAMOS2 inner GA directly).

    A small mutable state the runner advances once per outer generation. Each
    generation the factory: (1) initializes ε₀ from the DOE on the first call
    (``maybe_init_eps0``), (2) ``wrap``s the inner problem (snapshotting the
    current ε(t)), (3) ``advance``s t for the next generation.

    Parameters
    ----------
    n_gen_total : int
        T -- total number of OUTER generations of the run.
    perc_eps_until : float
        Fraction of the budget over which ε decays to 0. Default 0.5.
    """

    def __init__(self, n_gen_total: int, perc_eps_until: float = 0.5):
        self.T = int(n_gen_total)
        self.perc_eps_until = float(perc_eps_until)
        self.eps0 = None
        self.t = 0

    # -- epsilon0 initialization ------------------------------------------------
    def set_eps0(self, value: float) -> float:
        self.eps0 = float(value)
        return self.eps0

    def set_eps0_from_pop(self, pop) -> float:
        """ε₀ = mean aggregated CV over ``pop`` (the initial DOE)."""
        return self.set_eps0(float(np.mean(_cv_of(pop))) if len(pop) > 0 else 0.0)

    def maybe_init_eps0(self, pop) -> float:
        """Set ε₀ from ``pop`` only on the first call (idempotent thereafter)."""
        if self.eps0 is None and len(pop) > 0:
            self.set_eps0_from_pop(pop)
        return self.eps0 if self.eps0 is not None else 0.0

    # -- schedule ---------------------------------------------------------------
    @property
    def eps(self) -> float:
        """Current ε(t). 0 until ε₀ is initialized."""
        if self.eps0 is None:
            return 0.0
        denom = max(self.perc_eps_until * self.T, 1e-12)
        alpha = max(0.0, 1.0 - self.t / denom)
        return self.eps0 * alpha

    def advance(self) -> int:
        self.t += 1
        return self.t

    def wrap(self, problem):
        """Return an ε-relaxed copy of ``problem`` snapshotting current ε(t)."""
        return _EpsRelaxedProblem(problem, self.eps)


# ══════════════════════════════════════════════════════════════════════════
# H6 -- dominance-lifted stochastic ranking
# ══════════════════════════════════════════════════════════════════════════

class DominanceStochasticRanking(Survival):
    """H6 -- Stochastic Ranking survival (Runarsson & Yao, 2000), lifted to
    Pareto dominance.

    Runarsson, T.P. & Yao, X. (2000), "Stochastic ranking for constrained
    evolutionary optimization", IEEE Trans. Evol. Comput. 4(3):284-294.
    The original SR compares a scalar objective; here adjacent pairs are
    compared by Pareto dominance (instead of a scalar objective) with
    probability ``Pf``, and by constraint violation otherwise -- with two
    feasible solutions ALWAYS compared by dominance.

    Bubble-sort SR: N sweeps (N = len(pop)) over the population, each sweep
    comparing adjacent pairs and swapping the worse forward; stops early on a
    sweep with no swaps. For an adjacent pair (a, b) in current order:

      * both feasible, OR uniform draw u < Pf  -> compare by dominance:
            swap iff b dominates a (a dominates / non-dominated / equal -> keep).
      * otherwise                              -> compare by violation:
            swap iff CV(a) > CV(b).

    After ranking, the first ``n_survive`` individuals survive. Deterministic
    under a fixed ``seed`` / passed ``RandomState`` (the same seed + same
    population + same call sequence reproduce the order exactly).

    Usable as ``NSGA2(survival=DominanceStochasticRanking(...))`` for the SAMOS2
    inner GA (via a partial inner_algorithm, see module recipe H6) and on the
    outer constrained problem.

    Parameters
    ----------
    Pf : float
        Probability of an objective (dominance) comparison when at least one of
        the pair is infeasible. Default 0.45.
    seed : int or None
        Seed for the internal RandomState (used when ``random_state`` is None).
    random_state : np.random.RandomState or None
        Explicit RNG (takes precedence over ``seed``); advanced statefully
        across ``do`` calls.
    """

    def __init__(self, Pf: float = 0.45, seed=None, random_state=None):
        # filter_infeasible=False: SR does its own feasibility handling; the
        # base Survival must NOT pre-split feasible/infeasible.
        super().__init__(filter_infeasible=False)
        self.Pf = float(Pf)
        self.random_state = random_state if random_state is not None \
            else np.random.RandomState(seed)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        n = len(pop)
        if n_survive is None:
            n_survive = n

        F = pop.get('F')
        cv = _cv_of(pop)
        rs = self.random_state
        Pf = self.Pf

        order = np.arange(n)
        for _ in range(n):                       # up to N sweeps
            swapped = False
            for i in range(n - 1):
                a, b = int(order[i]), int(order[i + 1])
                both_feasible = (cv[a] <= 0.0) and (cv[b] <= 0.0)
                if both_feasible or rs.rand() < Pf:
                    # dominance comparison: +1 a>b, -1 b>a, 0 non-dominated/equal
                    swap = Dominator.get_relation(F[a], F[b]) == -1
                else:
                    swap = cv[a] > cv[b]
                if swap:
                    order[i], order[i + 1] = order[i + 1], order[i]
                    swapped = True
            if not swapped:
                break

        survivors = pop[order[:n_survive]]
        # Set rank/crowding so this survival is a drop-in NSGA2 inner survival:
        # NSGA2's binary tournament ('comp_by_dom_and_crowding') and
        # _set_optimum read per-individual 'rank'/'crowding', which the default
        # RankAndCrowding sets but SR does not. Rank = SR order (0 = best);
        # crowding is constant (ties fall back to a random pick, as pymoo does).
        if len(survivors) > 0:
            survivors.set('rank', np.arange(len(survivors)))
            survivors.set('crowding', np.zeros(len(survivors)))
        return survivors
