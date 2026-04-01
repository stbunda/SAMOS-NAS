"""main_samos_ablation.py — Ablation study: why does SAMOS underperform SSA-NSGA-II / GPSAF?

Isolates factors that differ between SAMOS and SSA-NSGA-II / GPSAF:

  Original ablations:
    A. Surrogate model type       (gpr, xgb, rfr)
    B. Warm-start ratio           (0.0, 0.25, 0.5, 0.75, 1.0)
    C. Candidate selection method (subset, kmeans, crowding)
    D. Inner population size      (50, 100, 200, 500)
    E. Subset selection on/off    (True, False)
    F. Surrogate model extended   (gpr, gpr_mle, gpr_matern15, gpr_matern25,
                                   gpr_white, xgb, rfr, etr, knn)
    G. Inner NSGA-II generations  (5, 10, 20, 40, 80) — refs use same value
    H. Population schedule        (constant, linear_inc, linear_dec,
                                   exp_inc, exp_dec)
    I. Alpha/beta phases          [(0,0), (3,0), (0,10), (3,10), (5,20)]
    J. Noise-aware tournament     (False, True)
    K. Trace assignment           (False, True)
    L. Uncertainty predictor      (none, extra_obj, ei_filter, exploration_bonus)

Each experiment fixes all other factors to defaults and sweeps one.
SSA-NSGA-II (default + xgb) and GPSAF (default + xgb) are included as
reference baselines in every ablation plot.

Run examples:
  python main_samos_ablation.py --problem wfg3 --ablation surrogate
  python main_samos_ablation.py --problem wfg3 --ablation inner_gens
  python main_samos_ablation.py --problem wfg1 wfg3 wfg7 --ablation alpha_beta
  python main_samos_ablation.py --problem wfg3 --ablation all
"""

import argparse
import os
import pickle
import random
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
from sklearn.cluster import KMeans
from pymoo.util.roulette import RouletteWheelSelection
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from scipy.stats import spearmanr, kendalltau

from problem.pymoo.benchmark_utils import build_problem, get_pareto_front, default_ref_point
from strategy.callbacks import PymooBenchmarkCallback
from strategy.surrogate.models import RFR, XGBoost, ETR
from strategy.surrogate.models.kriging import GPR, GPR_MLE, GPR_Matern15, GPR_Matern25, GPR_White
from strategy.surrogate.models.ensemble import EnsembleSurrogate
try:
    from strategy.surrogate.models.rbf import (
        RBF_Cubic, RBF_ThinPlateSpline, RBF_Gaussian,
        RBF_Multiquadric, RBF_InverseQuadratic, RBF_InverseMultiquadric,
    )
    _RBF_AVAILABLE = True
except ImportError:
    _RBF_AVAILABLE = False
from scipy.stats import qmc as _qmc

# Reference baselines
from strategy.algorithm.gpsaf import GPSAF, SklearnGPSAF
from strategy.algorithm.ssansga2 import SSANSGA2, SklearnSSANSGA2


# ─── Reference baseline names ────────────────────────────────────────────────

REFERENCE_BASELINES = [
    'ssa-nsga2-default',
    'ssa-nsga2-xgb',
    'gpsaf-default',
    'gpsaf-xgb',
]


# ─── DOE sampling helpers ─────────────────────────────────────────────────────

def _riesz_energy_samples(n: int, n_var: int, xl: np.ndarray, xu: np.ndarray,
                           s: float = 1.0, n_iter: int = 200,
                           lr: float = 0.01, seed: int = 0) -> np.ndarray:
    """Riesz s-energy repulsion sampling normalised to [xl, xu]."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0.0, 1.0, (n, n_var))
    for _ in range(n_iter):
        diff  = pts[:, None, :] - pts[None, :, :]      # (n, n, d)
        dist2 = np.sum(diff ** 2, axis=2) + 1e-12      # (n, n)
        np.fill_diagonal(dist2, np.inf)
        coeff = s / (dist2 ** (s / 2 + 1))             # (n, n)
        force = (coeff[:, :, None] * diff).sum(axis=1) # (n, d)
        pts   = np.clip(pts + lr * force, 0.0, 1.0)
    return xl + pts * (xu - xl)


def _doe_samples(problem, n: int, strategy: str, seed: int = 0) -> np.ndarray:
    """Return (n, n_var) DOE samples in decision space using *strategy*."""
    xl    = problem.xl.astype(float)
    xu    = problem.xu.astype(float)
    n_var = problem.n_var
    if strategy == 'uniform':
        return np.random.RandomState(seed).uniform(xl, xu, (n, n_var))
    if strategy == 'lhs':
        pts = _qmc.LatinHypercube(n_var, seed=seed).random(n)
    elif strategy == 'halton':
        pts = _qmc.Halton(n_var, scramble=True, seed=seed).random(n)
    elif strategy == 'sobol':
        import math as _math
        m   = _math.ceil(_math.log2(max(n, 1)))
        pts = _qmc.Sobol(n_var, scramble=True, seed=seed).random(2 ** m)[:n]
    elif strategy == 'riesz':
        return _riesz_energy_samples(n, n_var, xl, xu, seed=seed)
    else:
        raise ValueError(f'Unknown DOE strategy: {strategy!r}')
    return _qmc.scale(pts, xl, xu)


def _make_single_surrogate(stype: str, seed: int = 0):
    """Construct a single surrogate model from a type string."""
    stype = stype.strip().lower()
    if stype == 'gpr':
        return GPR(seed=seed)
    if stype == 'gpr_mle':
        return GPR_MLE(seed=seed)
    if stype == 'gpr_matern15':
        return GPR_Matern15(seed=seed)
    if stype == 'gpr_matern25':
        return GPR_Matern25(seed=seed)
    if stype == 'gpr_white':
        return GPR_White(seed=seed)
    if stype == 'xgb':
        return XGBoost(100, seed=seed)
    if stype == 'rfr':
        return RFR(100, seed=seed)
    if stype == 'etr':
        return ETR(100, seed=seed)
    if stype == 'knn':
        from strategy.surrogate.models.knn import KNN
        return KNN(n_neighbors=5, random_state=np.random.RandomState(seed))
    if _RBF_AVAILABLE:
        _RBF_MAP = {
            'rbf_cubic':        RBF_Cubic,
            'rbf_tps':          RBF_ThinPlateSpline,
            'rbf_gaussian':     RBF_Gaussian,
            'rbf_multiquadric': RBF_Multiquadric,
            'rbf_invquad':      RBF_InverseQuadratic,
            'rbf_invmultiquad': RBF_InverseMultiquadric,
        }
        if stype in _RBF_MAP:
            return _RBF_MAP[stype]()
    raise ValueError(f'Unknown surrogate type: {stype!r}')


# ─── Diagnostics-enhanced SAMOS ──────────────────────────────────────────────

class SAMOSAblation(Algorithm):
    """SAMOS variant that logs per-generation diagnostics for ablation studies.

    New parameters vs. original SAMOSAblation
    ------------------------------------------
    pop_schedule : str
        How the inner NSGA-II population size changes each outer generation.
        'constant'    — fixed ga_pop_size (original behaviour)
        'linear_inc'  — linearly ramp from pop_start → pop_end over n_gen gens
        'linear_dec'  — linearly ramp from pop_end → pop_start
        'exp_inc'     — exponentially ramp from pop_start → pop_end
        'exp_dec'     — exponentially ramp from pop_end → pop_start
    pop_start, pop_end : int
        Range for scheduled population sizes (used when pop_schedule != 'constant').
    alpha : int
        Number of surrogate-based tournament rounds applied to candidates after
        dedup and before final subset selection (GPSAF alpha phase analogue).
    beta : int
        Additional inner NSGA-II generations run on the surrogate to extend the
        candidate pool (GPSAF beta phase analogue).
    rho : float
        Replacement probability for the beta phase (fraction of beta candidates
        merged into the candidate pool; default 0.5).
    noise_tournament : bool
        If True and the surrogate exposes predict_std(), replace the inner
        NSGA-II binary tournament with a noise-aware variant that uses the
        optimistic lower bound  F_hat - noise_k * std  for dominance comparison.
    noise_k : float
        Scale factor for the noise-aware tournament (default 1.0).
    use_trace : bool
        If True, each infill candidate is linked to its closest archive member
        (parent) in X-space.  In _advance(), the traced parent is removed before
        merging so each infill effectively replaces its origin.
    uncertainty_mode : str
        Strategy for using predicted uncertainty in infill selection.
        'none'              — standard behaviour
        'extra_obj'         — add -predict_std as an extra inner objective
        'ei_filter'         — keep only candidates above median predict_std
        'exploration_bonus' — boost subset-selection score by predict_std
        'hvi_rerank'        — re-rank infill candidates by predicted hypervolume improvement
    doe_strategy : str
        Initial DOE sampling strategy for the first generation.
        'uniform' (default), 'lhs', 'halton', 'sobol', 'riesz'
    """

    def __init__(self,
                 sampling,
                 surrogates,
                 crossover=None,
                 mutation=None,
                 n_doe=20,
                 n_infill=20,
                 n_gen_inner=20,
                 ga_pop_size=200,
                 warm_start_ratio=1.0,
                 use_subset_selection=True,
                 selection_method='subset',  # 'subset', 'kmeans', 'crowding'
                 eliminate_duplicates=False,
                 dedup_key_fn=None,
                 # ── new ──────────────────────────────────────────────── #
                 pop_schedule='constant',
                 pop_start=50,
                 pop_end=1000,
                 alpha=0,
                 beta=0,
                 rho=0.5,
                 noise_tournament=False,
                 noise_k=1.0,
                 use_trace=False,
                 uncertainty_mode='none',
                 doe_strategy='uniform',
                 **kwargs):
        super().__init__(eliminate_duplicates=False, **kwargs)
        self.sampling             = sampling
        self.surrogates           = surrogates
        self.crossover            = crossover
        self.mutation             = mutation
        self.n_doe                = n_doe
        self.n_infill             = n_infill
        self.n_gen_inner          = n_gen_inner
        self.ga_pop_size          = ga_pop_size
        self.warm_start_ratio     = warm_start_ratio
        self.use_subset_selection = use_subset_selection
        self.selection_method     = selection_method
        self.eliminate_duplicates = eliminate_duplicates
        self._dedup_key = dedup_key_fn if dedup_key_fn is not None \
            else lambda x: tuple(np.round(x, decimals=8).tolist())
        # new
        self.pop_schedule    = pop_schedule
        self.pop_start       = pop_start
        self.pop_end         = pop_end
        self.alpha           = alpha
        self.beta            = beta
        self.rho             = rho
        self.noise_tournament = noise_tournament
        self.noise_k         = noise_k
        self.use_trace       = use_trace
        self.uncertainty_mode = uncertainty_mode
        self.doe_strategy    = doe_strategy

        self._archive      = Population()
        self._archive_keys: set = set()
        self._init         = Initialization(sampling)
        self.diagnostics   = []
        self._n_total_gen  = None   # set in _setup from termination

    def _setup(self, problem, **kwargs):
        # Resolve total number of outer generations for population scheduling.
        try:
            self._n_total_gen = self.termination.n_max_gen
        except AttributeError:
            self._n_total_gen = None
        # Capture seed from the algorithm (set by pymoo minimize via seed= arg)
        self._doe_seed = getattr(self, 'seed', 0) or 0

    def _initialize_infill(self):
        if self.doe_strategy == 'uniform':
            return self._init.do(self.problem, self.n_doe, algorithm=self)
        X = _doe_samples(self.problem, self.n_doe, self.doe_strategy,
                        seed=getattr(self, '_doe_seed', 0))
        return Population.new('X', X)

    def _initialize_advance(self, infills=None, **kwargs):
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    # ── helpers ────────────────────────────────────────────────────────────

    def _effective_pop_size(self):
        """Return inner NSGA-II population size for the current generation."""
        if self.pop_schedule == 'constant' or self._n_total_gen is None:
            return self.ga_pop_size
        t  = max(0, self.n_gen - 1)
        T  = max(1, self._n_total_gen - 1)
        lo, hi = self.pop_start, self.pop_end
        if self.pop_schedule == 'linear_inc':
            size = lo + (hi - lo) * t / T
        elif self.pop_schedule == 'linear_dec':
            size = hi - (hi - lo) * t / T
        elif self.pop_schedule == 'exp_inc':
            size = lo * (hi / max(lo, 1)) ** (t / T)
        elif self.pop_schedule == 'exp_dec':
            size = hi * (lo / max(hi, 1)) ** (t / T)
        else:
            size = self.ga_pop_size
        return max(2, int(round(size)))

    def _can_predict_std(self):
        """True if ALL surrogates have been fitted and support predict_std."""
        return all(hasattr(s, 'predict_std') and
                   not isinstance(getattr(s, 'predict_std', None), type)
                   for s in self.surrogates)

    def _surrogates_std(self, X):
        """Return (n_candidates, n_objectives) array of surrogate std estimates."""
        stds = []
        for s in self.surrogates:
            try:
                stds.append(s.predict_std(X).ravel())
            except NotImplementedError:
                stds.append(np.zeros(len(X)))
        return np.column_stack(stds)

    # ── alpha phase (surrogate-based tournament) ───────────────────────────

    def _alpha_tournament(self, cand_pop):
        """Run self.alpha rounds of pairwise surrogate-dominance tournament.

        Each round replaces every individual with the better of itself and a
        random opponent (based on predicted F).  The population size is unchanged.
        """
        if self.alpha <= 0 or len(cand_pop) < 2:
            return cand_pop, 0
        X = cand_pop.get('X')
        F_pred = np.column_stack([s.predict(X).ravel() for s in self.surrogates])
        n = len(X)
        survivors = np.arange(n)
        rng = np.random.default_rng()
        for _ in range(self.alpha):
            opponents = rng.integers(0, n, size=n)
            for i in range(n):
                j = opponents[i]
                fi, fj = F_pred[survivors[i]], F_pred[survivors[j]]
                # i dominates j?
                if np.all(fi <= fj) and np.any(fi < fj):
                    pass  # keep i
                elif np.all(fj <= fi) and np.any(fj < fi):
                    survivors[i] = survivors[j]
                # else: tie → keep current
        unique_survivors = np.unique(survivors)
        return cand_pop[unique_survivors], len(cand_pop) - len(unique_survivors)

    # ── beta phase (additional surrogate generations) ──────────────────────

    def _beta_extend(self, cand_pop, effective_pop_size):
        """Run self.beta additional inner NSGA-II generations then merge candidates."""
        if self.beta <= 0:
            return cand_pop, 0
        from problem.pymoo.surrogate_problem import SurrogateProblemMOO
        surr_problem = SurrogateProblemMOO(
            self.surrogates, self.problem.n_var,
            self.problem.xl.copy(), self.problem.xu.copy(),
            real_problem=self.problem,
        )
        # seed inner NSGA-II from current candidate pool (or archive if pool is empty)
        if len(cand_pop) > 0:
            init_X = cand_pop.get('X')
        else:
            init_X = self._archive.get('X')
        n_pad = max(0, effective_pop_size - len(init_X))
        if n_pad > 0:
            pad = self._init.do(self.problem, n_pad, algorithm=self).get('X')
            init_X = np.vstack([init_X, pad])
        inner_init = Population.new('X', init_X[:effective_pop_size])
        inner_alg = NSGA2(
            pop_size=effective_pop_size,
            sampling=inner_init,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=self.eliminate_duplicates,
        )
        res = minimize(surr_problem, inner_alg,
                       termination=('n_gen', self.beta), verbose=False)
        beta_pop = res.pop if res.pop is not None else Population.empty()
        # randomly sample rho fraction of beta candidates to add
        n_take = max(1, int(round(self.rho * len(beta_pop))))
        idx = np.random.choice(len(beta_pop), size=min(n_take, len(beta_pop)), replace=False)
        beta_sample = beta_pop[idx]
        if len(cand_pop) > 0:
            merged = Population.merge(cand_pop, beta_sample)
        else:
            merged = beta_sample
        return merged, len(beta_sample)

    # ── noise-aware tournament selection ───────────────────────────────────

    def _make_noise_aware_tournament(self):
        """Return a TournamentSelection that uses F_hat - noise_k*std."""
        from pymoo.operators.selection.tournament import TournamentSelection

        surrogates = self.surrogates
        noise_k    = self.noise_k

        def _comp(pop, P, **kwargs):
            # P: (n_tournaments, 2) indices into pop
            S = np.full(P.shape[0], -1, dtype=int)
            for i, (a, b) in enumerate(P):
                X_pair = pop[[a, b]].get('X')
                F_pair = np.column_stack([s.predict(X_pair).ravel() for s in surrogates])
                try:
                    stds = np.column_stack([s.predict_std(X_pair).ravel() for s in surrogates])
                    F_opt = F_pair - noise_k * stds
                except NotImplementedError:
                    F_opt = F_pair
                fa, fb = F_opt[0], F_opt[1]
                if np.all(fa <= fb) and np.any(fa < fb):
                    S[i] = a
                elif np.all(fb <= fa) and np.any(fb < fa):
                    S[i] = b
                else:
                    S[i] = a if np.random.rand() < 0.5 else b
            return S

        return TournamentSelection(func_comp=_comp)

    def _hvi_rerank(self, cand_pop: Population, F_arc: np.ndarray) -> Population:
        """Re-rank candidates by predicted hypervolume improvement (descending).

        Uses an adaptive reference point = 1.1 × worst observed values, so no
        problem-specific prior is needed.
        """
        if len(cand_pop) == 0:
            return cand_pop
        cand_X    = cand_pop.get('X')
        surr_F    = np.column_stack([s.predict(cand_X).ravel() for s in self.surrogates])
        ref_point = F_arc.max(axis=0) * 1.1
        hv_ind    = HV(ref_point=ref_point)
        pf_idx    = NonDominatedSorting().do(F_arc, only_non_dominated_front=True)
        pf_F      = F_arc[pf_idx]
        hv_base   = float(hv_ind(pf_F))
        hvi_scores = np.zeros(len(cand_pop))
        for i, f_cand in enumerate(surr_F):
            aug_F         = np.vstack([pf_F, f_cand[None, :]])
            hvi_scores[i] = float(hv_ind(aug_F)) - hv_base
        order = np.argsort(-hvi_scores)
        return cand_pop[order]

    # ── main infill ─────────────────────────────────────────────────────────

    def _infill(self):
        X_arc = self._archive.get('X')
        F_arc = self._archive.get('F')
        diag  = {'gen': self.n_gen, 'n_archive': len(self._archive)}

        # ── 1. Fit surrogates + compute diagnostics ────────────────────────
        rmses, spearmans, kendalls = [], [], []
        for s, surrogate in enumerate(self.surrogates):
            surrogate.fit(X_arc, F_arc[:, s])
            pred = surrogate.predict(X_arc).ravel()
            true = F_arc[:, s]
            rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
            rho, _ = spearmanr(pred, true)
            tau, _ = kendalltau(pred, true)
            rmses.append(rmse)
            spearmans.append(float(rho))
            kendalls.append(float(tau))
        diag['surrogate_rmse']     = rmses
        diag['surrogate_spearman'] = spearmans
        diag['surrogate_kendall']  = kendalls

        # ── 2. Effective population size for this generation ───────────────
        effective_pop_size = self._effective_pop_size()
        diag['effective_pop_size'] = effective_pop_size

        # ── 3. Warm-start inner NSGA-II ────────────────────────────────────
        topx    = max(1, int(effective_pop_size * self.warm_start_ratio))
        top_pop = RankAndCrowding().do(problem=self.problem, pop=self._archive, n_survive=topx)
        n_rand  = effective_pop_size - len(top_pop)
        if n_rand > 0:
            rand_pop = self._init.do(self.problem, n_rand, algorithm=self)
            inner_X  = np.vstack([top_pop.get('X'), rand_pop.get('X')])
        else:
            inner_X  = top_pop.get('X')
        inner_init = Population.new('X', inner_X)
        diag['n_warm_archive'] = len(top_pop)
        diag['n_warm_random']  = max(0, n_rand)

        # ── 4. Build inner surrogate problem ──────────────────────────────
        from problem.pymoo.surrogate_problem import SurrogateProblemMOO

        if self.uncertainty_mode == 'extra_obj':
            # extra_obj: augment the inner problem with -predict_std objective
            surr_problem = _UncertaintySurrogateProblem(
                self.surrogates, self.problem.n_var,
                self.problem.xl.copy(), self.problem.xu.copy(),
                real_problem=self.problem,
            )
        else:
            surr_problem = SurrogateProblemMOO(
                self.surrogates, self.problem.n_var,
                self.problem.xl.copy(), self.problem.xu.copy(),
                real_problem=self.problem,
            )

        # ── 5. Inner NSGA-II ───────────────────────────────────────────────
        selection_op = None
        if self.noise_tournament and self._can_predict_std():
            selection_op = self._make_noise_aware_tournament()

        inner_alg_kwargs = dict(
            pop_size=effective_pop_size,
            sampling=inner_init,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=self.eliminate_duplicates,
        )
        if selection_op is not None:
            inner_alg_kwargs['selection'] = selection_op

        inner_alg = NSGA2(**inner_alg_kwargs)
        res = minimize(
            surr_problem, inner_alg,
            termination=('n_gen', self.n_gen_inner),
            verbose=False,
        )
        diag['n_surr_eval'] = res.algorithm.evaluator.n_eval

        # ── 6. Deduplicate ─────────────────────────────────────────────────
        cand_pop = res.pop if res.pop is not None else Population.empty()
        n_before_dedup = len(cand_pop)
        if len(cand_pop) > 0:
            not_dup = np.array([
                self._dedup_key(cx) not in self._archive_keys
                for cx in cand_pop.get('X')
            ], dtype=bool)
            cand_pop = cand_pop[not_dup]
        diag['n_before_dedup'] = n_before_dedup
        diag['n_after_dedup']  = len(cand_pop)

        # ── 7. Alpha phase (surrogate tournament) ─────────────────────────
        cand_pop, n_alpha_eliminated = self._alpha_tournament(cand_pop)
        diag['n_alpha_eliminated'] = n_alpha_eliminated

        # ── 8. Beta phase (additional surrogate gens) ─────────────────────
        cand_pop, n_beta_added = self._beta_extend(cand_pop, effective_pop_size)
        diag['n_beta_added'] = n_beta_added
        # Dedup again after beta
        if n_beta_added > 0 and len(cand_pop) > 0:
            not_dup2 = np.array([
                self._dedup_key(cx) not in self._archive_keys
                for cx in cand_pop.get('X')
            ], dtype=bool)
            cand_pop = cand_pop[not_dup2]

        # ── 9. Uncertainty filter / infill criterion ───────────────────────
        mean_std_selected = 0.0
        if self.uncertainty_mode in ('ei_filter', 'exploration_bonus') \
                and len(cand_pop) > 0 and self._can_predict_std():
            cand_X  = cand_pop.get('X')
            std_arr = self._surrogates_std(cand_X).mean(axis=1)   # (n_cand,)
            if self.uncertainty_mode == 'ei_filter':
                threshold = np.median(std_arr)
                keep = std_arr >= threshold
                if keep.sum() >= 1:
                    cand_pop = cand_pop[keep]
            # exploration_bonus: pre-sort candidates by std desc so subset
            # selection naturally picks high-uncertainty candidates first
            elif self.uncertainty_mode == 'exploration_bonus':
                order = np.argsort(-std_arr)
                cand_pop = cand_pop[order]
            mean_std_selected = float(std_arr[:len(cand_pop)].mean()) \
                if len(cand_pop) > 0 else 0.0
        elif self.uncertainty_mode == 'hvi_rerank' and len(cand_pop) > 0:
            cand_pop = self._hvi_rerank(cand_pop, F_arc)
        diag['mean_surrogate_std'] = mean_std_selected

        # ── 10. Select final infill batch ──────────────────────────────────
        infill_pop, n_random_fill = self._select_infill(cand_pop, F_arc)
        diag['n_selected']    = len(infill_pop) - n_random_fill
        diag['n_random_fill'] = n_random_fill

        # ── 11. Trace assignment ───────────────────────────────────────────
        if self.use_trace and len(infill_pop) > 0:
            infill_X  = infill_pop.get('X')
            arc_X     = X_arc
            # squared L2 distances; shape (n_infill, n_archive)
            dists     = np.sum((infill_X[:, None, :] - arc_X[None, :, :]) ** 2, axis=2)
            parents   = dists.argmin(axis=1)
            infill_pop.set('trace_parent', parents)
            diag['trace_parents'] = parents.tolist()
        else:
            diag['trace_parents'] = []

        # ── 12. Surrogate-vs-real correlation on selected candidates ───────
        infill_X = infill_pop.get('X')
        real_out = {}
        self.problem._evaluate(infill_X, real_out)
        real_F   = real_out['F']
        surr_F   = np.column_stack([s.predict(infill_X).ravel() for s in self.surrogates])
        infill_corrs = []
        for j in range(real_F.shape[1]):
            rho_val, _ = spearmanr(surr_F[:, j], real_F[:, j])
            infill_corrs.append(float(rho_val) if not np.isnan(rho_val) else 0.0)
        diag['infill_surrogate_vs_real_spearman'] = infill_corrs

        self.diagnostics.append(diag)
        return Population.new('X', infill_X)

    def _advance(self, infills=None, **kwargs):
        if self.use_trace and infills is not None and len(infills) > 0:
            trace_parents = infills.get('trace_parent')
            if trace_parents is not None:
                # Remove traced parents from archive (replace-parent semantics)
                # Filter out None values (individuals without a traced parent)
                valid_parents = [p for p in trace_parents if p is not None]
                unique_parents = np.unique(valid_parents) if valid_parents else np.array([], dtype=int)
                n_arc = len(self._archive)
                keep_mask = np.ones(n_arc, dtype=bool)
                for pidx in unique_parents:
                    if 0 <= pidx < n_arc:
                        keep_mask[pidx] = False
                diag_entry = self.diagnostics[-1] if self.diagnostics else {}
                diag_entry['n_replaced_by_trace'] = int((~keep_mask).sum())
                self._archive = self._archive[keep_mask]
                # Remove displaced keys
                self._archive_keys = {
                    self._dedup_key(x) for x in self._archive.get('X')
                }
        else:
            if self.diagnostics:
                self.diagnostics[-1]['n_replaced_by_trace'] = 0

        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    def _select_infill(self, cand_pop, F_arc):
        """Returns (infill_pop, n_random_fill)."""
        if len(cand_pop) == 0:
            return self._sample_dedup(self.n_infill), self.n_infill

        if self.selection_method == 'kmeans' and len(cand_pop) > self.n_infill:
            return self._select_kmeans(cand_pop), 0
        elif self.selection_method == 'crowding' and len(cand_pop) > self.n_infill:
            return self._select_crowding(cand_pop), 0
        else:
            # subset selection (SAMOS default)
            return self._select_subset(cand_pop, F_arc)

    def _select_subset(self, cand_pop, F_arc):
        """SAMOS-style: non-dominated front + crowding distance."""
        from strategy.surrogate.subset_selection import subset_selection
        F_cand = cand_pop.get('F')
        front  = NonDominatedSorting().do(F_arc, only_non_dominated_front=True)

        if self.use_subset_selection and len(cand_pop) > self.n_infill:
            indices = subset_selection(F_cand, F_arc[front], self.n_infill)
            found   = cand_pop if indices is None else cand_pop[indices]
        else:
            found = cand_pop

        n_found = min(len(found), self.n_infill)
        found   = found[:n_found]
        n_random = 0

        if n_found < self.n_infill:
            n_random = self.n_infill - n_found
            extra = self._sample_dedup(n_random, extra_ref=found)
            return (Population.merge(extra, found) if len(extra) > 0 else found), n_random
        return found, 0

    def _select_kmeans(self, cand_pop):
        """SSA-NSGA-II-style: k-means clustering in normalised F-space."""
        F = cand_pop.get('F')
        ideal = F.min(axis=0)
        nadir = F.max(axis=0) + 1e-16
        vals = normalize(F, ideal, nadir)

        n_clusters = min(self.n_infill, len(cand_pop))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(vals)
        groups = [[] for _ in range(n_clusters)]
        for k, i in enumerate(kmeans.labels_):
            groups[i].append(k)

        S = []
        for group in groups:
            if len(group) > 0:
                fitness = cand_pop[group].get("crowding")
                if fitness is None:
                    fitness = calc_crowding_distance(F[group])
                order = fitness.argsort()
                selection = RouletteWheelSelection(order, larger_is_better=False)
                I = group[selection.next()]
                S.append(I)

        return cand_pop[S]

    def _select_crowding(self, cand_pop):
        """Simple: non-dominated sort + crowding, take top n_infill."""
        F = cand_pop.get('F')
        fronts = NonDominatedSorting().do(F)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.n_infill:
                selected.extend(front)
            else:
                remaining = self.n_infill - len(selected)
                crowding = calc_crowding_distance(F[front])
                order = np.argsort(-crowding)
                selected.extend(front[order[:remaining]])
                break
        return cand_pop[selected]

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


# ─── Uncertainty-augmented inner surrogate problem ───────────────────────────

class _UncertaintySurrogateProblem(Problem):
    """Wraps SurrogateProblemMOO and appends -mean_std as an extra objective.

    This makes the inner NSGA-II explicitly trade off objective quality vs.
    exploration (high surrogate uncertainty).
    """

    def __init__(self, surrogates, n_var, xl, xu, real_problem=None):
        from problem.pymoo.surrogate_problem import SurrogateProblemMOO
        self._base = SurrogateProblemMOO(surrogates, n_var, xl, xu,
                                          real_problem=real_problem)
        self._surrogates = surrogates
        n_obj_aug = self._base.n_obj + 1  # +1 for uncertainty objective
        super().__init__(n_var=n_var, n_obj=n_obj_aug, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        base_out = {}
        self._base._evaluate(X, base_out, *args, **kwargs)
        F_base = base_out['F']  # shape (n, n_base_obj)
        # Uncertainty: mean std across objectives (negate so NSGA-II minimises it)
        stds = []
        for s in self._surrogates:
            try:
                stds.append(s.predict_std(X).ravel())
            except NotImplementedError:
                stds.append(np.zeros(len(X)))
        mean_std = np.column_stack(stds).mean(axis=1, keepdims=True)
        out['F'] = np.hstack([F_base, -mean_std])  # minimise -std = maximise std


# ─── Reference baseline runner ───────────────────────────────────────────────

def run_reference_baseline(method, seed, problem_name, pop_size, n_gen,
                            n_obj=2, n_var=None, n_gen_inner=20):
    """Run SSA-NSGA-II or GPSAF as a reference baseline."""
    np.random.seed(seed)
    random.seed(seed)

    problem   = build_problem(problem_name, n_obj, n_var)
    pf        = get_pareto_front(problem, problem.n_obj)
    ref_point = default_ref_point(problem_name, problem.n_obj)
    callback  = PymooBenchmarkCallback(pf, ref_point)
    sampling  = FloatRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation  = PM(eta=20)

    n_infill_ = pop_size
    n_doe_    = pop_size
    inner_ps  = pop_size * 10

    if method.startswith('gpsaf-') or method.startswith('ssa-nsga2-'):
        if method.startswith('gpsaf-'):
            surrogate_type = method[len('gpsaf-'):]
        else:
            surrogate_type = method[len('ssa-nsga2-'):]

        sklearn_models = None
        if surrogate_type in ('rfr', 'xgb'):
            rng = np.random.RandomState(seed)
            sklearn_models = [
                (RFR(20, seed=rng.randint(0, 2**31 - 1))
                 if surrogate_type == 'rfr' else
                 XGBoost(100, seed=rng.randint(0, 2**31 - 1)))
                for _ in range(problem.n_obj)
            ]

        if method.startswith('ssa-nsga2-'):
            if surrogate_type == 'default':
                algorithm = SSANSGA2(
                    n_infills=n_infill_, surr_pop_size=inner_ps,
                    surr_n_gen=n_gen_inner, n_initial_doe=n_doe_,
                )
            else:
                algorithm = SklearnSSANSGA2(
                    sklearn_models=sklearn_models,
                    n_infills=n_infill_, surr_pop_size=inner_ps,
                    surr_n_gen=n_gen_inner, n_initial_doe=n_doe_,
                )
        else:
            base_algo = NSGA2(pop_size=pop_size, crossover=crossover, mutation=mutation)
            if surrogate_type == 'default':
                algorithm = GPSAF(
                    base_algo, n_initial_doe=n_doe_,
                    n_max_infills=n_infill_, beta=n_gen_inner,
                )
            else:
                algorithm = SklearnGPSAF(
                    base_algo, sklearn_models=sklearn_models,
                    n_initial_doe=n_doe_, n_max_infills=n_infill_,
                    beta=n_gen_inner,
                )
    else:
        raise ValueError(f'Unknown reference method: {method}')

    results = minimize(
        problem=problem, algorithm=algorithm,
        termination=('n_gen', n_gen), seed=seed,
        callback=callback, save_history=False, verbose=False,
    )
    return results.algorithm.callback.data


# ─── Ablation configurations ─────────────────────────────────────────────────

def _make_surrogates(stype, n_obj, seed):
    """Build one surrogate per objective.

    *stype* may be:
    - a simple key, e.g. ``'xgb'``, ``'gpr'``, ``'rbf_cubic'``
    - a ``'+'``-separated ensemble spec, e.g. ``'gpr+xgb'`` → one
      :class:`EnsembleSurrogate` per objective whose members are those types
    """
    rng = np.random.RandomState(seed)
    member_types = [t.strip() for t in stype.split('+')]
    surrs = []
    for _ in range(n_obj):
        s = int(rng.randint(0, 2**31 - 1))
        if len(member_types) == 1:
            surrs.append(_make_single_surrogate(member_types[0], seed=s))
        else:
            members = [_make_single_surrogate(mt, seed=s) for mt in member_types]
            surrs.append(EnsembleSurrogate(members))
    return surrs


# Default SAMOSAblation kwargs shared across all ablations (can be overridden)
_SAMOS_DEFAULTS = dict(
    warm_start_ratio=1.0,
    selection_method='subset',
    use_subset_selection=True,
    ga_pop_size=200,
    pop_schedule='constant',
    pop_start=50,
    pop_end=1000,
    alpha=0,
    beta=0,
    rho=0.5,
    noise_tournament=False,
    use_trace=False,
    uncertainty_mode='none',
)

ABLATION_CONFIGS = {
    # ── original ablations ──────────────────────────────────────────────── #
    'surrogate': {
        'sweep_param': 'surrogate_type',
        'values': ['gpr', 'xgb', 'rfr'],
        'defaults': dict(**_SAMOS_DEFAULTS),
    },
    'warm_start': {
        'sweep_param': 'warm_start_ratio',
        'values': [0.0, 0.25, 0.5, 0.75, 1.0],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'xgb'}),
    },
    'selection': {
        'sweep_param': 'selection_method',
        'values': ['subset', 'kmeans', 'crowding'],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'xgb'}),
    },
    'inner_pop': {
        'sweep_param': 'ga_pop_size',
        'values': [50, 100, 200, 500],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'xgb'}),
    },
    'subset_sel': {
        'sweep_param': 'use_subset_selection',
        'values': [True, False],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'xgb'}),
    },
    # ── new ablations ───────────────────────────────────────────────────── #
    'surrogate_ext': {
        'sweep_param': 'surrogate_type',
        'values': ['gpr', 'gpr_mle', 'gpr_matern15', 'gpr_matern25',
                   'gpr_white', 'xgb', 'rfr', 'etr', 'knn'],
        'defaults': dict(**_SAMOS_DEFAULTS),
    },
    'inner_gens': {
        'sweep_param': 'n_gen_inner',
        'values': [5, 10, 20, 40, 80],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'xgb'}),
    },
    'pop_schedule': {
        'sweep_param': 'pop_schedule',
        'values': ['constant', 'linear_inc', 'linear_dec', 'exp_inc', 'exp_dec'],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'xgb',
                            'pop_start': 50, 'pop_end': 1000}),
    },
    'alpha_beta': {
        'sweep_param': 'alpha_beta_pair',   # special: decoded in run_ablation_single
        'values': [(0, 0), (3, 0), (0, 10), (3, 10), (5, 20)],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'xgb'}),
    },
    'noise_tournament': {
        'sweep_param': 'noise_tournament',
        'values': [False, True],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'gpr'}),
    },
    'trace': {
        'sweep_param': 'use_trace',
        'values': [False, True],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'xgb'}),
    },
    'uncertainty': {
        'sweep_param': 'uncertainty_mode',
        'values': ['none', 'extra_obj', 'ei_filter', 'exploration_bonus', 'hvi_rerank'],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'gpr'}),
    },
    # ── batch-2 ablations ───────────────────────────────────────────────── #
    'ensemble': {
        'sweep_param': 'surrogate_type',
        'values': ['gpr', 'xgb', 'rfr', 'etr', 'knn',
                   'gpr+xgb', 'gpr+rfr', 'gpr+etr', 'gpr+knn',
                   'xgb+rfr', 'knn+xgb', 'gpr+xgb+rfr'],
        'defaults': dict(**_SAMOS_DEFAULTS),
    },
    'doe_strategy': {
        'sweep_param': 'doe_strategy',
        'values': ['uniform', 'lhs', 'halton', 'sobol', 'riesz'],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'xgb'}),
    },
    'rbf_surrogate': {
        'sweep_param': 'surrogate_type',
        'values': ['rbf_cubic', 'rbf_tps', 'rbf_gaussian',
                   'rbf_multiquadric', 'rbf_invquad', 'rbf_invmultiquad'],
        'defaults': dict(**_SAMOS_DEFAULTS),
    },
    'infill_criterion': {
        'sweep_param': 'uncertainty_mode',
        'values': ['none', 'hvi_rerank', 'ei_filter', 'exploration_bonus', 'extra_obj'],
        'defaults': dict(**{**_SAMOS_DEFAULTS, 'surrogate_type': 'gpr'}),
    },
}


def run_ablation_single(problem_name, seed, ablation, sweep_value, pop_size, n_gen,
                        n_obj=2, n_var=None, n_gen_inner=20, defaults=None,
                        pop_start=50, pop_end=1000):
    np.random.seed(seed)
    random.seed(seed)

    problem   = build_problem(problem_name, n_obj, n_var)
    pf        = get_pareto_front(problem, problem.n_obj)
    ref_point = default_ref_point(problem_name, problem.n_obj)
    callback  = PymooBenchmarkCallback(pf, ref_point)
    sampling  = FloatRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation  = PM(eta=20)

    cfg = dict(defaults) if defaults else {}

    # Decode the sweep value into cfg
    sweep_param = ABLATION_CONFIGS[ablation]['sweep_param']
    if sweep_param == 'alpha_beta_pair':
        alpha_val, beta_val = sweep_value
        cfg['alpha'] = alpha_val
        cfg['beta']  = beta_val
    elif sweep_param == 'n_gen_inner':
        n_gen_inner = sweep_value   # Override inner gens for this run
    else:
        cfg[sweep_param] = sweep_value

    # Pull + remove surrogate_type so it doesn't reach SAMOSAblation kwargs
    surrogate_type = cfg.pop('surrogate_type', 'xgb')
    surrogates = _make_surrogates(surrogate_type, problem.n_obj, seed)

    # Apply pop_start/pop_end from CLI if not already in cfg
    cfg.setdefault('pop_start', pop_start)
    cfg.setdefault('pop_end', pop_end)

    # Remove keys that are not SAMOSAblation parameters
    cfg.pop('alpha_beta_pair', None)

    algorithm = SAMOSAblation(
        sampling=sampling,
        surrogates=surrogates,
        crossover=crossover,
        mutation=mutation,
        n_doe=pop_size,
        n_infill=pop_size,
        n_gen_inner=n_gen_inner,
        dedup_key_fn=lambda x: tuple(np.round(x, 4).tolist()),
        **cfg,
    )

    results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=('n_gen', n_gen),
        seed=seed,
        callback=callback,
        save_history=False,
        verbose=False,
    )

    data = results.algorithm.callback.data
    data['diagnostics'] = results.algorithm.diagnostics
    data['config'] = {
        'ablation': ablation,
        'sweep_param': sweep_param,
        'sweep_value': str(sweep_value),
        'surrogate_type': surrogate_type,
        'n_gen_inner': n_gen_inner,
        **cfg,
    }
    return data


def main(args):
    ablations = list(ABLATION_CONFIGS.keys()) if args.ablation == 'all' else [args.ablation]

    for prob in args.problem:

        # ── run reference baselines first ─────────────────────────────────
        budget_folder = f"B{args.n_gen * args.pop_size}_P{args.pop_size}"
        ref_root = os.path.join(
            'results', args.experiment_name, prob, budget_folder, 'reference'
        )

        for ref_method in REFERENCE_BASELINES:
            save_dir = os.path.join(ref_root, ref_method)
            os.makedirs(save_dir, exist_ok=True)

            for seed in args.seeds:
                out_path = os.path.join(save_dir, f'seed_{seed}.pkl')
                if os.path.exists(out_path) and not args.overwrite:
                    print(f'  [SKIP] ref:{ref_method}/seed_{seed}')
                    continue

                print(f'  [RUN] ref:{ref_method}  seed={seed}')
                try:
                    data = run_reference_baseline(
                        ref_method, seed, prob, args.pop_size, args.n_gen,
                        args.n_obj, args.n_var, args.n_gen_inner,
                    )
                    with open(out_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f'    Saved -> {out_path}')
                except Exception as e:
                    print(f'    [FAIL] {ref_method} seed={seed}: {e}')

        # ── run ablation sweeps ──────────────────────────────────────────
        for ablation in ablations:
            config      = ABLATION_CONFIGS[ablation]
            sweep_param = config['sweep_param']
            values      = config['values']
            defaults    = config['defaults']

            print(f'\n{"="*80}')
            print(f'Ablation: {ablation}  |  sweep: {sweep_param}  |  problem: {prob}')
            print(f'{"="*80}')

            results_root = os.path.join(
                'results', args.experiment_name, prob, budget_folder, f'ablation_{ablation}'
            )

            for val in values:
                method_name = f'{sweep_param}={val}'
                save_dir = os.path.join(results_root, method_name)
                os.makedirs(save_dir, exist_ok=True)

                for seed in args.seeds:
                    out_path = os.path.join(save_dir, f'seed_{seed}.pkl')
                    if os.path.exists(out_path) and not args.overwrite:
                        print(f'  [SKIP] {method_name}/seed_{seed}')
                        continue

                    print(f'  [RUN] {sweep_param}={val}  seed={seed}')
                    data = run_ablation_single(
                        problem_name=prob,
                        seed=seed,
                        ablation=ablation,
                        sweep_value=val,
                        pop_size=args.pop_size,
                        n_gen=args.n_gen,
                        n_obj=args.n_obj,
                        n_var=args.n_var,
                        n_gen_inner=args.n_gen_inner,
                        defaults=defaults,
                        pop_start=args.pop_start,
                        pop_end=args.pop_end,
                    )

                    with open(out_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f'    Saved -> {out_path}')

            # ── plot ablation results ────────────────────────────────────
            _plot_ablation(prob, ablation, values, sweep_param, results_root,
                           ref_root, args.n_gen, args.pop_size, args.n_obj, args.n_var)

    # ── cross-benchmark summary (only when multiple problems requested) ───
    if len(args.problem) > 1:
        for ablation in ablations:
            config      = ABLATION_CONFIGS[ablation]
            sweep_param = config['sweep_param']
            values      = config['values']
            summary = _aggregate_ablation(
                args.problem, ablation, values, sweep_param,
                args.experiment_name, args.n_gen, args.pop_size,
                args.n_obj, args.n_var,
            )
            _plot_cross_benchmark_summary(
                ablation, values, sweep_param, summary,
                args.experiment_name, args.n_gen, args.pop_size,
            )



def _aggregate_ablation(problems, ablation, values, sweep_param,
                        experiment_name, n_gen, pop_size, n_obj, n_var):
    """Return a dict  {problem: {str(val): final_hv_mean}}  over all seeds."""
    summary = {}
    budget_folder = f"B{n_gen * pop_size}_P{pop_size}"
    for prob in problems:
        results_root = os.path.join(
            'results', experiment_name, prob, budget_folder, f'ablation_{ablation}'
        )
        problem   = build_problem(prob, n_obj, n_var)
        ref_point = default_ref_point(prob, problem.n_obj)
        hv_ind    = HV(ref_point=ref_point)

        row = {}
        for val in values:
            method_name = f'{sweep_param}={val}'
            save_dir    = os.path.join(results_root, method_name)
            if not os.path.isdir(save_dir):
                continue
            finals = []
            for seed_file in sorted(os.listdir(save_dir)):
                if not seed_file.endswith('.pkl'):
                    continue
                with open(os.path.join(save_dir, seed_file), 'rb') as fh:
                    data = pickle.load(fh)
                hv_curve = [ind['hv'] for ind in data['indicators']]
                if hv_curve:
                    finals.append(hv_curve[-1])
            if finals:
                row[str(val)] = float(np.mean(finals))
        summary[prob] = row
    return summary


def _plot_cross_benchmark_summary(ablation, values, sweep_param, summary,
                                  experiment_name, n_gen, pop_size):
    """Heatmap + rank bar chart of final-HV across problems × sweep values."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    probs     = list(summary.keys())
    val_strs  = [str(v) for v in values]
    # Filter to values that have data in at least one problem
    val_strs  = [v for v in val_strs if any(v in summary[p] for p in probs)]
    if not val_strs or not probs:
        return

    # Build HV matrix (probs × vals); fill unknowns with NaN
    hv_mat = np.full((len(probs), len(val_strs)), np.nan)
    for pi, prob in enumerate(probs):
        for vi, val in enumerate(val_strs):
            hv_mat[pi, vi] = summary[prob].get(val, np.nan)

    # Row-wise Min–Max normalisation so each problem is on [0,1]
    row_min = np.nanmin(hv_mat, axis=1, keepdims=True)
    row_max = np.nanmax(hv_mat, axis=1, keepdims=True)
    norm_mat = (hv_mat - row_min) / np.maximum(row_max - row_min, 1e-12)

    # Rank per problem (1 = best), ignore NaN
    rank_mat = np.full_like(norm_mat, np.nan)
    for pi in range(len(probs)):
        row = norm_mat[pi]
        valid = ~np.isnan(row)
        if valid.any():
            sorted_idx = np.argsort(-row[valid])  # descending
            ranks      = np.empty(valid.sum())
            ranks[sorted_idx] = np.arange(1, valid.sum() + 1)
            rank_mat[pi, valid] = ranks

    mean_ranks = np.nanmean(rank_mat, axis=0)

    # ── Figure ──────────────────────────────────────────────────────────
    fig, (ax_heat, ax_rank) = plt.subplots(1, 2, figsize=(max(12, len(val_strs) * 1.2), 5))

    # Heatmap
    im = ax_heat.imshow(norm_mat, aspect='auto', cmap='RdYlGn',
                         vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(val_strs)))
    ax_heat.set_xticklabels(val_strs, rotation=45, ha='right', fontsize=8)
    ax_heat.set_yticks(range(len(probs)))
    ax_heat.set_yticklabels(probs, fontsize=9)
    for pi in range(len(probs)):
        for vi in range(len(val_strs)):
            v = norm_mat[pi, vi]
            if not np.isnan(v):
                ax_heat.text(vi, pi, f'{v:.2f}', ha='center', va='center', fontsize=7,
                             color='black' if 0.2 < v < 0.85 else 'white')
    plt.colorbar(im, ax=ax_heat, fraction=0.03)
    ax_heat.set_title(f'Normalised final HV\n(ablation: {ablation}, sweep: {sweep_param})',
                       fontsize=10)

    # Rank bar chart (lower = better)
    bar_x = np.arange(len(val_strs))
    ax_rank.bar(bar_x, mean_ranks, color='steelblue', alpha=0.8)
    ax_rank.set_xticks(bar_x)
    ax_rank.set_xticklabels(val_strs, rotation=45, ha='right', fontsize=8)
    ax_rank.set_ylabel('Mean rank (1 = best)')
    ax_rank.set_title(f'Cross-benchmark avg rank\n(ablation: {ablation})', fontsize=10)
    ax_rank.invert_yaxis()   # rank 1 at top

    fig.tight_layout()
    budget_folder = f"B{n_gen * pop_size}_P{pop_size}"
    out_dir = os.path.join('results', experiment_name, '_cross_benchmark',
                           budget_folder, f'ablation_{ablation}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'cross_benchmark_{ablation}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Cross-benchmark plot saved -> {out_path}')


def _plot_ablation(prob, ablation, values, sweep_param, results_root,
                   ref_root, n_gen, pop_size, n_obj, n_var):
    """Generate HV/IGD+ trajectory + surrogate-diagnostics plots for one ablation.

    Reference baselines (SSA-NSGA-II, GPSAF) are drawn as dashed lines.
    A second figure is produced for new diagnostic channels when present.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    problem    = build_problem(prob, n_obj, n_var)
    pf         = get_pareto_front(problem, problem.n_obj)
    ref_point  = default_ref_point(prob, problem.n_obj)
    hv_ind     = HV(ref_point=ref_point)
    hv_ceiling = float(hv_ind(pf))

    # ── primary 3×3 figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(21, 15))
    (ax_hv,  ax_igd,         ax_rmse,
     ax_sp,  ax_nfill,       ax_infill_corr,
     ax_ps,  ax_alpha_beta,  ax_misc) = axes.flat

    n_colours = max(len(values), 10)
    colours   = plt.cm.tab10(np.linspace(0, 1, n_colours))

    # ── Helper: load all seeds and build trajectory arrays ────────────────
    def _load_trajectories(save_dir):
        if not os.path.isdir(save_dir):
            return None
        buckets = {k: [] for k in (
            'hv', 'igd', 'rmse', 'spearman', 'nfill', 'infill_corr',
            'pop_size_sched', 'alpha_beta_diag', 'surrogate_std',
            'trace_replaced',
        )}
        for seed_file in sorted(os.listdir(save_dir)):
            if not seed_file.endswith('.pkl'):
                continue
            with open(os.path.join(save_dir, seed_file), 'rb') as f:
                data = pickle.load(f)
            buckets['hv'].append([ind['hv'] for ind in data['indicators']])
            buckets['igd'].append([ind['igd_plus'] for ind in data['indicators']])
            if 'diagnostics' in data:
                diags = data['diagnostics']
                buckets['rmse'].append(
                    [np.mean(d.get('surrogate_rmse', [0])) for d in diags])
                buckets['spearman'].append(
                    [np.mean(d.get('surrogate_spearman', [0])) for d in diags])
                buckets['nfill'].append(
                    [d.get('n_random_fill', 0) for d in diags])
                buckets['infill_corr'].append(
                    [np.mean(d.get('infill_surrogate_vs_real_spearman', [0]))
                     for d in diags])
                buckets['pop_size_sched'].append(
                    [d.get('effective_pop_size', 0) for d in diags])
                buckets['alpha_beta_diag'].append(
                    [d.get('n_alpha_eliminated', 0) + d.get('n_beta_added', 0)
                     for d in diags])
                buckets['surrogate_std'].append(
                    [d.get('mean_surrogate_std', 0.0) for d in diags])
                buckets['trace_replaced'].append(
                    [d.get('n_replaced_by_trace', 0) for d in diags])
        if not buckets['hv']:
            return None
        return buckets

    def _plot_mean_std(ax, data_list, color, label, linestyle='-', **kw):
        if not data_list:
            return
        min_len = min(len(d) for d in data_list)
        arr = np.array([d[:min_len] for d in data_list])
        x   = np.arange(1, min_len + 1) * pop_size
        mu  = arr.mean(axis=0)
        sd  = arr.std(axis=0)
        ax.plot(x, mu, color=color, label=label, linestyle=linestyle, **kw)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.10, color=color)

    # ── Plot ablation sweep variants (solid lines) ────────────────────────
    for vi, val in enumerate(values):
        method_name = f'{sweep_param}={val}'
        save_dir    = os.path.join(results_root, method_name)
        traj        = _load_trajectories(save_dir)
        if traj is None:
            continue
        c     = colours[vi % len(colours)]
        label = str(val)
        _plot_mean_std(ax_hv,         traj['hv'],            c, label)
        _plot_mean_std(ax_igd,        traj['igd'],           c, label)
        _plot_mean_std(ax_rmse,       traj['rmse'],          c, label)
        _plot_mean_std(ax_sp,         traj['spearman'],      c, label)
        _plot_mean_std(ax_nfill,      traj['nfill'],         c, label)
        _plot_mean_std(ax_infill_corr,traj['infill_corr'],   c, label)
        _plot_mean_std(ax_ps,         traj['pop_size_sched'],c, label)
        _plot_mean_std(ax_alpha_beta, traj['alpha_beta_diag'],c, label)
        _plot_mean_std(ax_misc,       traj['surrogate_std'], c, label)

    # ── Plot reference baselines (dashed lines) ──────────────────────────
    ref_colours = {
        'ssa-nsga2-default': 'black',
        'ssa-nsga2-xgb':     'dimgrey',
        'gpsaf-default':     'royalblue',
        'gpsaf-xgb':         'cornflowerblue',
    }
    for ref_method in REFERENCE_BASELINES:
        save_dir = os.path.join(ref_root, ref_method)
        traj     = _load_trajectories(save_dir)
        if traj is None:
            continue
        c = ref_colours.get(ref_method, 'grey')
        _plot_mean_std(ax_hv,  traj['hv'],  c, ref_method, linestyle='--', linewidth=1.5)
        _plot_mean_std(ax_igd, traj['igd'], c, ref_method, linestyle='--', linewidth=1.5)

    ax_hv.axhline(hv_ceiling, color='grey', linestyle=':', alpha=0.5, label='PF ceiling')

    titles = {
        ax_hv:          'HV (↑)',
        ax_igd:         'IGD+ (↓)',
        ax_rmse:        'Surrogate RMSE (train ↓)',
        ax_sp:          'Surrogate Spearman ρ (train ↑)',
        ax_nfill:       'Random fill-in count (↓)',
        ax_infill_corr: 'Infill surr-vs-real ρ (↑)',
        ax_ps:          'Effective pop size',
        ax_alpha_beta:  'α eliminated + β added per gen',
        ax_misc:        'Mean surrogate std of selected',
    }
    for ax, title in titles.items():
        ax.set_title(title)
        ax.set_xlabel('Evaluations')
        ax.legend(fontsize=7)

    fig.suptitle(
        f'{prob.upper()} — Ablation: {ablation} (sweep: {sweep_param})\n'
        f'pop={pop_size}, {n_gen} gens = {pop_size * n_gen} evals, mean ± std\n'
        f'Dashed = reference baselines (SSA-NSGA-II, GPSAF)',
        fontsize=13,
    )
    fig.tight_layout()
    out_path = os.path.join(results_root, f'ablation_{ablation}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Plot saved -> {out_path}')


if __name__ == '__main__':
    _all_ablations = list(ABLATION_CONFIGS.keys())
    parser = argparse.ArgumentParser(description='SAMOS ablation study')
    parser.add_argument('--problem', type=str, nargs='+', default=['wfg1', 'wfg3', 'wfg7'],
                        help='WFG problem(s) to run on (default: wfg1 wfg3 wfg7)')
    parser.add_argument('--n_obj',   type=int, default=2)
    parser.add_argument('--n_var',   type=int, default=None)
    parser.add_argument('--ablation', type=str, default='all',
                        choices=_all_ablations + ['all'],
                        help='Which ablation to run (default: all)')
    parser.add_argument('--seeds',      type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--pop_size',   type=int, default=20)
    parser.add_argument('--n_gen',      type=int, default=60)
    parser.add_argument('--n_gen_inner',type=int, default=20,
                        help='Inner NSGA-II generations (also passed to GPSAF/SSA-NSGA-II refs)')
    parser.add_argument('--pop_start',  type=int, default=50,
                        help='Start population size for pop_schedule ablation')
    parser.add_argument('--pop_end',    type=int, default=1000,
                        help='End population size for pop_schedule ablation')
    parser.add_argument('--experiment_name', type=str, default='samos_ablation/2_obj')
    parser.add_argument('--overwrite',  action='store_true')
    args = parser.parse_args()
    print(f'Arguments: {args}')
    main(args)
