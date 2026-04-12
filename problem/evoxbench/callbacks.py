"""
Callback for evoxbench benchmark runs.

Maintains a cumulative non-dominated archive each generation, re-evaluates
it with ``true_eval=True`` (mean over all runs), and records HV / IGD+
against the benchmark's pre-computed Pareto front.
"""

import numpy as np
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class EvoxBenchCallback(Callback):
    """
    Per-generation callback for evoxbench runs.

    Parameters
    ----------
    benchmark :
        The evoxbench benchmark instance.  Used to obtain the reference
        Pareto front and to re-evaluate the archive with ``true_eval=True``.
    no_norm : bool
        When ``True``, ``test_obj_archive`` stores raw (un-normalized)
        true-eval objectives.  Per-run indicators are then provisional only
        (computed with an ad-hoc ref-point = max × 1.05); reliable HV / IGD+
        values are computed post-hoc via ``analyze_evoxbench.py --no_norm``
        using empirically derived bounds from all runs.
        When ``False`` (default), the benchmark's built-in normalization is
        applied, matching the original behaviour.
    """

    def __init__(self, benchmark, no_norm: bool = False) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.no_norm   = no_norm

        n_obj = benchmark.evaluator.n_objs

        if not no_norm:
            pareto_front_raw = benchmark.pareto_front   # (n_pts, n_obj) or None
            if pareto_front_raw is not None and len(pareto_front_raw) > 0:
                # pareto_front is stored in raw objective space; normalize it to
                # the same [utopian→0, nadir→1] scale used by test_obj in notify().
                pf_norm = benchmark.normalize(pareto_front_raw)
                pf_norm = np.where(np.isfinite(pf_norm), pf_norm, 1.0)
                self._pareto_front = pf_norm
                # After normalization nadir ≈ 1.0, so 1.05 is always a valid ref.
                ref_point = np.ones(n_obj) * 1.05
            else:
                # No reference front available — use a unit ref-point as fallback
                self._pareto_front = None
                ref_point = np.ones(n_obj) * 1.05
            self._hv_ind  = HV(ref_point=ref_point)
            self._igd_ind = IGDPlus(self._pareto_front) if self._pareto_front is not None else None
        else:
            # no_norm mode: indicators are provisional; real indicators are
            # computed post-hoc with empirical bounds by the analysis scripts.
            self._pareto_front = None
            self._hv_ind       = None   # rebuilt per-generation from live data
            self._igd_ind      = None

        self.data['var_pop']          = []
        self.data['obj_pop']          = []
        self.data['var_archive']      = []
        self.data['obj_archive']      = []
        self.data['test_obj_archive'] = []
        self.data['indicators']       = []
        self.data['time']             = None

    # ─── archive maintenance ──────────────────────────────────────────────────

    @staticmethod
    def _update_archive(
        var_arch: list,
        obj_arch: list,
        var_new: np.ndarray,
        obj_new: np.ndarray,
    ):
        """Add a candidate to the non-dominated archive (in-place on copies)."""
        var_arch = list(var_arch)
        obj_arch = list(obj_arch)

        # Check whether the new solution is dominated by any existing member
        for obj_existing in obj_arch:
            if all(obj_existing[k] <= obj_new[k] for k in range(len(obj_new))):
                # new solution is dominated — discard
                return var_arch, obj_arch

        # Remove any existing members dominated by the new solution
        dominated = [
            idx for idx, obj_existing in enumerate(obj_arch)
            if all(obj_new[k] <= obj_existing[k] for k in range(len(obj_new)))
            and not all(obj_new[k] == obj_existing[k] for k in range(len(obj_new)))
        ]
        for idx in sorted(dominated, reverse=True):
            var_arch.pop(idx)
            obj_arch.pop(idx)

        var_arch.append(var_new)
        obj_arch.append(obj_new)
        return var_arch, obj_arch

    # ─── notify ───────────────────────────────────────────────────────────────

    def notify(self, algorithm) -> None:
        # Use full archive for SAMOS; current population for others
        if hasattr(algorithm, '_archive') and len(algorithm._archive) > 0:
            src_pop = algorithm._archive
        else:
            src_pop = algorithm.pop

        var_pop = src_pop.get('X')
        obj_pop = src_pop.get('F')

        # Rebuild cumulative non-dominated archive
        var_arch = list(self.data['var_archive'][-1]) if self.data['var_archive'] else []
        obj_arch = list(self.data['obj_archive'][-1]) if self.data['obj_archive'] else []

        for var_ind, obj_ind in zip(var_pop, obj_pop):
            var_arch, obj_arch = self._update_archive(var_arch, obj_arch, var_ind, obj_ind)

        # Test re-evaluation with true_eval=True
        if var_arch:
            X_arch   = np.array([np.round(v).astype(int) for v in var_arch])
            test_obj = self.benchmark.evaluate(X_arch, true_eval=True)
            if not self.no_norm and not self.benchmark.normalized_objectives:
                test_obj = self.benchmark.normalize(test_obj)
            test_obj = np.where(np.isfinite(test_obj), test_obj, np.nan)
            # Re-filter to non-dominated on the test objectives (finite rows only)
            finite_mask = np.isfinite(test_obj).all(axis=1)
            test_obj    = test_obj[finite_mask]
            if len(test_obj) > 0:
                nd_idx      = NonDominatedSorting().do(test_obj, only_non_dominated_front=True)
                test_obj_nd = test_obj[nd_idx]
            else:
                test_obj_nd = np.empty((0, self.benchmark.evaluator.n_objs))
        else:
            test_obj_nd = np.empty((0, self.benchmark.evaluator.n_objs))

        # Indicators
        if len(test_obj_nd) > 0:
            if self.no_norm:
                # Provisional: build an ad-hoc ref-point from the current archive.
                # These values are for live monitoring only; reliable indicators
                # are computed post-hoc with empirical bounds by analyze_evoxbench.
                live_ref = np.max(test_obj_nd, axis=0) * 1.05
                live_hv_ind = HV(ref_point=live_ref)
                indicators = {'hv': float(live_hv_ind(test_obj_nd)), 'igd_plus': float('nan')}
            else:
                indicators = {
                    'hv':       float(self._hv_ind(test_obj_nd)),
                    'igd_plus': float(self._igd_ind(test_obj_nd)) if self._igd_ind is not None else float('nan'),
                }
        else:
            indicators = {'hv': 0.0, 'igd_plus': float('nan')}

        self.data['var_pop'].append(var_pop)
        self.data['obj_pop'].append(obj_pop)
        self.data['var_archive'].append(var_arch)
        self.data['obj_archive'].append(obj_arch)
        self.data['test_obj_archive'].append(test_obj_nd)
        self.data['indicators'].append(indicators)
        self.data['time'] = algorithm.problem.time
