"""strategy/surrogate/infill.py --- Infill-selection strategies (SAMOS2 C3).

SAMOSMinimal hard-codes a single infill policy: rank surrogate-optimised
candidates by non-domination + crowding-distance diversity
(:func:`subset_selection`), ignoring any uncertainty the surrogate might
carry about its own predictions. These classes make that choice pluggable so
diversity-only selection (the SAMOS baseline) can be compared against
uncertainty-aware acquisition, on otherwise identical algorithm state.

Both classes implement:

    select(cand_pop, F_arc, n_infill, surrogates=None) -> Population
        Returns AT MOST n_infill individuals; SAMOS2 pads any shortfall with
        fresh random samples, same as SAMOSMinimal.
"""

from __future__ import annotations

import numpy as np
from pymoo.core.population import Population
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from strategy.surrogate.subset_selection import subset_selection


class DiversitySelector:
    """Default SAMOS policy: non-dominated + crowding-distance diverse subset
    of the (mean) predicted candidates. Identical to SAMOSMinimal's built-in
    candidate-ranking step."""

    def __init__(self, use_subset_selection: bool = True):
        self.use_subset_selection = use_subset_selection

    def select(self, cand_pop, F_arc, n_infill, surrogates=None, **kwargs):
        if len(cand_pop) == 0:
            return Population.empty()
        F_cand = cand_pop.get('F')
        front  = NonDominatedSorting().do(F_arc, only_non_dominated_front=True)
        if self.use_subset_selection and len(cand_pop) > n_infill:
            indices = subset_selection(F_cand, F_arc[front], n_infill)
            found   = cand_pop if indices is None else cand_pop[indices]
        else:
            found = cand_pop
        return found[:min(len(found), n_infill)]


class AcquisitionSelector:
    """Uncertainty-aware infill selection via a lower-confidence-bound (LCB)
    shift of the predicted objectives, using ensemble/quantile
    ``predict_std`` (RFR, ETR, GPR, EnsembleSurrogate -- not XGBoost, which
    raises NotImplementedError for predict_std).

    kind='lcb' : rank by non-domination + crowding on the LCB-shifted
        objectives F - kappa * sigma (optimistic estimate; exploration via
        kappa). Cheap/real objectives are untouched (sigma=0 -- they are
        exact evaluations, not surrogate predictions).
    kind='hvi' : rank candidates by the hypervolume contribution their
        LCB-shifted point would add to the *true* archive's non-dominated
        front. This is a deterministic, cheap approximation to Bayesian
        Expected Hypervolume Improvement (no closed-form EI integral over
        the predictive distribution) -- the "exploration" comes only from
        the optimistic LCB shift, not from integrating over uncertainty.

    Parameters
    ----------
    predict_obj_indices : sequence[int]
        Column (in F_cand / F_arc) that each entry of ``surrogates`` predicts,
        in the same order as ``surrogates`` -- mirrors the mapping already
        passed to the surrogate_problem_factory (e.g. SurrogateProblemEvox).
    kappa : float
        Exploration weight for the LCB shift.
    ref_point : np.ndarray or None
        Hypervolume reference point (normalised objective space); required
        for kind='hvi'.
    kappa_schedule : None or 'linear'
        'linear' decays the effective kappa from *kappa* at the first infill
        step down to 0 at the final outer generation (explore early, exploit
        late). Requires ``total_gens``. None (default) keeps kappa constant.
    total_gens : int or None
        Total outer generations of the run; required when kappa_schedule is
        set (the algorithm itself does not know its termination).
    """

    def __init__(self, predict_obj_indices, kind: str = 'lcb', kappa: float = 2.0,
                 ref_point=None, use_subset_selection: bool = True,
                 kappa_schedule: str = None, total_gens: int = None):
        if kind not in ('lcb', 'hvi'):
            raise ValueError(f"kind must be 'lcb' or 'hvi', got {kind!r}")
        if kind == 'hvi' and ref_point is None:
            raise ValueError("kind='hvi' requires ref_point")
        if kappa_schedule not in (None, 'linear'):
            raise ValueError(f"kappa_schedule must be None or 'linear', got {kappa_schedule!r}")
        if kappa_schedule is not None and total_gens is None:
            raise ValueError("kappa_schedule requires total_gens")
        self.predict_obj_indices = list(predict_obj_indices)
        self.kind = kind
        self.kappa = kappa
        self.ref_point = None if ref_point is None else np.asarray(ref_point, dtype=float)
        self.use_subset_selection = use_subset_selection
        self.kappa_schedule = kappa_schedule
        self.total_gens = total_gens

    def _effective_kappa(self, n_gen):
        """kappa at outer generation *n_gen*. The first infill happens at
        outer gen 2 (gen 1 is the DOE), so 'linear' maps gen 2 -> kappa and
        gen total_gens -> 0."""
        if self.kappa_schedule is None or n_gen is None:
            return self.kappa
        t = (n_gen - 2) / max(self.total_gens - 2, 1)
        return self.kappa * float(np.clip(1.0 - t, 0.0, 1.0))

    def _lcb(self, X_cand, F_cand, surrogates, kappa):
        F_lcb = F_cand.copy()
        for s, orig_idx in zip(surrogates, self.predict_obj_indices):
            sigma = s.predict_std(X_cand)
            F_lcb[:, orig_idx] = F_cand[:, orig_idx] - kappa * sigma
        return F_lcb

    def select(self, cand_pop, F_arc, n_infill, surrogates=None, n_gen=None, **kwargs):
        if len(cand_pop) == 0:
            return Population.empty()
        if surrogates is None:
            raise ValueError('AcquisitionSelector requires surrogates=...')

        X_cand = cand_pop.get('X')
        F_cand = cand_pop.get('F')
        F_lcb  = self._lcb(X_cand, F_cand, surrogates, self._effective_kappa(n_gen))

        if self.kind == 'lcb':
            front = NonDominatedSorting().do(F_arc, only_non_dominated_front=True)
            if self.use_subset_selection and len(cand_pop) > n_infill:
                indices = subset_selection(F_lcb, F_arc[front], n_infill)
                found   = cand_pop if indices is None else cand_pop[indices]
            else:
                found = cand_pop
            return found[:min(len(found), n_infill)]

        # kind == 'hvi'
        nd_idx = NonDominatedSorting().do(F_arc, only_non_dominated_front=True)
        F_nd = F_arc[nd_idx]
        hv = HV(ref_point=self.ref_point)
        base = hv(F_nd)
        scores = np.empty(len(cand_pop))
        for i in range(len(cand_pop)):
            scores[i] = hv(np.vstack([F_nd, F_lcb[i]])) - base
        order = np.argsort(-scores)[:n_infill]
        return cand_pop[order]
