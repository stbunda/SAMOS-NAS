from typing import Dict, List, Optional

import numpy as np
from pymoo.core.problem import Problem

from utils.logger2 import EvolutionLogger


class MO_Fitness_NASBench101(Problem):
    """
    Multi-objective fitness function for NASBench-101.

    Objectives are looked up directly from the pre-built flat dictionary
    (data_nasbench101.pkl) keyed by ``program.arch_str`` (MD5 hash).
    NASBench-101 has a single dataset (CIFAR-10), so there is no ``dataset``
    dimension unlike MO_Fitness for NASBench-201.
    """

    def __init__(
        self,
        objectives: List[str],
        data: Dict,
        n_var: int = 1,
        logger: Optional[EvolutionLogger] = None,
        calc_pareto: bool = False,
        verbose: int = 1,
        **kwargs,
    ):
        super().__init__(n_var=n_var, n_obj=len(objectives), n_constr=0, requires_kwargs=True)
        self.problem_type = 'NASBENCH101'
        self.objectives   = objectives
        self.data         = data
        self.n_var        = n_var
        self.verbose      = verbose
        self.generation   = 0
        self.metadata     = None
        self.logger       = logger

        if logger is not None:
            display_objectives = []
            for obj in self.objectives:
                if 'acc' in obj:
                    display_objectives.append(obj + ' Error')
                elif 'params' in obj:
                    display_objectives.append(obj)
                else:
                    display_objectives.append(obj)
            self.logger.set_objectives(display_objectives)

            if calc_pareto:
                F_all, _ = self.extract_objectives()
                mask = self.pareto_front(F_all)
                self.logger.set_true_pareto(F_all[mask])

        self.xl = np.zeros(n_var)
        self.xu = 0.999999 * np.ones(n_var)

    def to_config(self):
        self.metadata = {
            'class':      self.__class__.__name__,
            'module':     self.__class__.__module__,
            'objectives': self.objectives,
            'n_var':      self.n_var,
        }
        return self.metadata

    def extract_objectives(self):
        X, archs = [], []
        for arch, entry in self.data.items():
            metrics, valid = [], True
            for obj in self.objectives:
                if obj not in entry:
                    valid = False
                    break
                metrics.append(transform_metric_101(obj, entry[obj]))
            if valid:
                X.append(metrics)
                archs.append(arch)
        return np.array(X), archs

    @staticmethod
    def pareto_front(X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if MO_Fitness_NASBench101._dominates(X[j], X[i]):
                    is_pareto[i] = False
                    break
        return is_pareto

    @staticmethod
    def _dominates(p, q):
        return all(pi <= qi for pi, qi in zip(p, q)) and any(
            pi < qi for pi, qi in zip(p, q)
        )

    def _evaluate(self, x, out, real_objectives=None, *args, **kwargs):
        objectives = self.objectives if real_objectives is None else real_objectives
        all_metrics, all_programs, all_times = [], [], []

        for program_instance in x:
            program = program_instance[0]
            all_programs.append(program)

            arch_hash = program.arch_str
            try:
                entry = self.data[arch_hash]
            except KeyError:
                # Invalid arch not in lookup table — fill with worst-case values
                metrics = [1.0 if 'acc' in obj else float('inf') for obj in objectives]
                all_metrics.append(metrics)
                all_times.append([0.0, 0.0])
                continue

            metrics = []
            for obj in objectives:
                metric = transform_metric_101(obj, entry[obj])
                metrics.append(metric)

            train_time = entry.get('train_time_12', 0.0)
            all_times.append([train_time, 0.0])
            all_metrics.append(metrics)

        all_metrics  = np.asarray(all_metrics, dtype=float)
        all_programs = np.asarray(all_programs, dtype=object)

        if self.verbose and self.logger is not None and real_objectives is None:
            self.logger.log_evaluation(
                gen=self.generation,
                F=all_metrics,
                X=all_programs,
                time=np.array(all_times, dtype=float),
                evals=len(x),
            )
        self.generation += 1

        if real_objectives is None:
            out['F'] = all_metrics
        else:
            return all_metrics


def transform_metric_101(objective: str, metric: float) -> float:
    """
    Transform a raw NASBench-101 metric into a minimisation-compatible value.

    - Accuracy fields (val_acc*, test_acc*) are in [0, 1].
      Converted to error: 1.0 - metric.
    - n_params is kept as raw integer count.
    """
    if 'acc' in objective:
        return 1.0 - metric
    return metric
