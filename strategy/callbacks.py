"""
All pymoo Callback classes used by SAMOS-NAS experiments in one file.

  NASArchiveCallback    — NASBench-101/201 baseline runs (generic)
  PymooBenchmarkCallback — continuous MOO benchmark runs (ZDT, DTLZ, …)
"""

import numpy as np
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# ─── NAS benchmarks (generic) ─────────────────────────────────────────────────

class NASArchiveCallback(Callback):
    """
    Generic callback for NASBench-101/201 baseline runs.

    Maintains a cumulative Pareto archive each generation, re-evaluates it
    via a test oracle, and records HV/IGD+ indicators.

    Parameters
    ----------
    bench_db : dict
        Benchmark look-up table.
    pareto_ref : np.ndarray
        Reference Pareto front for IGD+.
    update_fn : callable
        ``_update_archive(var_arch, obj_arch, var_new, obj_new)`` — adds a
        candidate to the non-dominated archive.
    test_fn : callable
        ``_test_archive(var_arch, bench_db, **test_kwargs)`` — re-evaluates
        the archive on test objectives and returns ``(var_list, obj_array)``.
    **test_kwargs
        Extra keyword arguments forwarded to *test_fn* (e.g. ``test_key``,
        ``min_flops``, ``max_flops`` for NASBench-201).
    """

    REF_POINT = np.array([1.05, 1.05])

    def __init__(
        self,
        bench_db: dict,
        pareto_ref: np.ndarray,
        update_fn,
        test_fn,
        **test_kwargs,
    ) -> None:
        super().__init__()
        self._update_archive = update_fn
        self._test_archive   = test_fn
        self._test_kwargs    = test_kwargs

        self.bench_db   = bench_db
        self.pareto_ref = pareto_ref
        self._hv_ind    = HV(ref_point=self.REF_POINT)
        self._igd_ind   = IGDPlus(pareto_ref)

        self.data['var_pop']          = []
        self.data['obj_pop']          = []
        self.data['var_archive']      = []
        self.data['obj_archive']      = []
        self.data['test_var_archive'] = []
        self.data['test_obj_archive'] = []
        self.data['indicators']       = []
        self.data['time']             = None

    def notify(self, algorithm) -> None:
        var_pop = algorithm.pop.get('X')
        obj_pop = algorithm.pop.get('F')

        var_arch = self.data['var_archive'][-1].copy() if self.data['var_archive'] else []
        obj_arch = self.data['obj_archive'][-1].copy() if self.data['obj_archive'] else []

        for var_ind, obj_ind in zip(var_pop, obj_pop):
            var_arch, obj_arch = self._update_archive(var_arch, obj_arch, var_ind, obj_ind)

        test_var_arch, test_obj_arch = self._test_archive(
            var_arch, self.bench_db, **self._test_kwargs
        )

        if len(test_obj_arch) > 0:
            indicators = {
                'hv':       float(self._hv_ind(test_obj_arch)),
                'igd_plus': float(self._igd_ind(test_obj_arch)),
            }
        else:
            indicators = {'hv': 0.0, 'igd_plus': np.inf}

        self.data['var_pop'].append(var_pop)
        self.data['obj_pop'].append(obj_pop)
        self.data['var_archive'].append(var_arch)
        self.data['obj_archive'].append(obj_arch)
        self.data['test_var_archive'].append(test_var_arch)
        self.data['test_obj_archive'].append(test_obj_arch)
        self.data['indicators'].append(indicators)
        self.data['time'] = algorithm.problem.time


# ─── Continuous MOO benchmarks ────────────────────────────────────────────────

class PymooBenchmarkCallback(Callback):
    """
    Callback for continuous MOO benchmark runs (ZDT, DTLZ, …).

    Records per-generation HV and IGD+ against the problem's known Pareto front.
    For SAMOS the full evaluated archive is used; for other algorithms the
    current generation's population is used.
    """

    def __init__(self, pareto_ref: np.ndarray, ref_point: np.ndarray) -> None:
        super().__init__()
        self.pareto_ref = pareto_ref
        self.ref_point  = ref_point
        self._hv_ind    = HV(ref_point=ref_point)
        self._igd_ind   = IGDPlus(pareto_ref)

        self.data['indicators'] = []
        self.data['obj_pop']    = []
        self.data['var_pop']    = []

    def notify(self, algorithm) -> None:
        # SAMOS accumulates a monotonically growing archive; all others expose
        # the current population via algorithm.pop.
        if hasattr(algorithm, '_archive') and len(algorithm._archive) > 0:
            pop = algorithm._archive
        else:
            pop = algorithm.pop

        F = pop.get('F')
        X = pop.get('X')
        n_obj = self.ref_point.shape[0]
        self.data['obj_pop'].append(F.copy() if F is not None else np.empty((0, n_obj)))
        self.data['var_pop'].append(X.copy() if X is not None else np.empty((0, 0)))

        if F is not None and len(F) > 0:
            nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
            nd_F   = F[nd_idx]
            hv     = float(self._hv_ind(nd_F))
            igd    = float(self._igd_ind(nd_F))
        else:
            hv, igd = 0.0, float('inf')

        self.data['indicators'].append({'hv': hv, 'igd_plus': igd})
