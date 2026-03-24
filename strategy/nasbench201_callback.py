"""
PDNS-style pymoo Callback for NASBench-201 baseline runs.

Records per-generation indicators (HV, IGD+) and archive snapshots in the
same format as the NASBench-101 PDNS pipeline so that analysis scripts can
consume them directly.
"""

import numpy as np
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from problem.nasbench201_utils import _update_archive, _test_archive


class PDNSStyleCallback201(Callback):
    """
    Records per-evaluation indicators in PDNS format for NASBench-201.

    Uses test-accuracy objectives (test_err × flops_norm) for HV/IGD+
    computation, mirroring the NASBench-101 callback.
    """

    REF_POINT = np.array([1.05, 1.05])

    def __init__(
        self,
        bench_db: dict,
        pareto_ref: np.ndarray,
        test_key: str,
        test_min_flops: float,
        test_max_flops: float,
    ) -> None:
        super().__init__()
        self.bench_db       = bench_db
        self.pareto_ref     = pareto_ref
        self.test_key       = test_key
        self.test_min_flops = test_min_flops
        self.test_max_flops = test_max_flops
        self._hv_ind        = HV(ref_point=self.REF_POINT)
        self._igd_ind       = IGDPlus(pareto_ref)

        self.data['var_pop']          = []
        self.data['obj_pop']          = []
        self.data['var_archive']      = []
        self.data['obj_archive']      = []
        self.data['test_var_archive'] = []
        self.data['test_obj_archive'] = []
        self.data['indicators']       = []
        self.data['time']             = None

    def notify(self, algorithm):
        var_pop = algorithm.pop.get('X')
        obj_pop = algorithm.pop.get('F')

        var_arch = self.data['var_archive'][-1].copy() if self.data['var_archive'] else []
        obj_arch = self.data['obj_archive'][-1].copy() if self.data['obj_archive'] else []

        for var_ind, obj_ind in zip(var_pop, obj_pop):
            var_arch, obj_arch = _update_archive(var_arch, obj_arch, var_ind, obj_ind)

        test_var_arch, test_obj_arch = _test_archive(
            var_arch, self.bench_db,
            self.test_key, self.test_min_flops, self.test_max_flops,
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
