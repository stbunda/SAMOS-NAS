from typing import Dict, List

import numpy as np
from pymoo.core.problem import Problem
import sympy as sp
from scipy.stats import rankdata

from utils.logger2 import EvolutionLogger


class MO_Fitness(Problem):
    def __init__(self,
                 objectives: List[str],
                 data: Dict,
                 dataset: str,
                 n_var: int = 1,
                 logger: EvolutionLogger = None,
                 verbose: int = 1,
                 calc_pareto: bool = False,
                 **kwargs):
        super().__init__(n_var=n_var, n_obj=len(objectives), n_constr=0, requires_kwargs=True)
        self.problem_type = 'NASBENCH'
        self.objectives = objectives
        self.data = data
        self.dataset = dataset
        self.n_var = n_var
        self.verbose = verbose
        self.generation = 0
        self.metadata = None

        # Logger
        self.logger = logger
        if logger is not None:
            new_objectives = []
            for objective in self.objectives:
                if 'latency' in objective:
                    objective = objective + ' (ms)'
                if 'energy' in objective:
                    objective = objective + ' (mJ)'
                if 'params' in objective:
                    objective = objective + ' (Bytes)'
                if 'epoch' in objective:
                    objective = objective + ' Error %'
                new_objectives.append(objective)
            self.logger.set_objectives(new_objectives)

            if calc_pareto:
                # Extract and set true Pareto front from all available data
                F_all, archs = self.extract_objectives()
                mask = self.pareto_front(F_all)
                pareto_points = F_all[mask]
                self.logger.set_true_pareto(pareto_points)

        # Bounds for real-valued variables
        self.xl = np.zeros(n_var)
        self.xu = 0.999999 * np.ones(n_var)

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            'objectives': self.objectives,
            'n_var': self.n_var,
        }
        return self.metadata

    def extract_objectives(self):
        X = []
        archs = []

        for arch, data in self.data.items():
            if self.dataset not in data:
                continue

            metrics = []
            valid = True
            for obj in self.objectives:
                if obj not in data[self.dataset]:
                    valid = False
                    break
                metric = transform_metric(obj, data[self.dataset][obj])
                metrics.append(metric)

            if valid:
                X.append(metrics)
                archs.append(arch)

        return np.array(X), archs

    def pareto_front(self, X):
        n = X.shape[0]
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(X[j], X[i]):
                    is_pareto[i] = False
                    break
        return is_pareto

    @staticmethod
    def _dominates(p, q):
        """
        p dominates q (both minimization)
        """
        return all(pi <= qi for pi, qi in zip(p, q)) and any(
            pi < qi for pi, qi in zip(p, q)
        )

    def _evaluate(self, x, out, real_objectives=None, *args, **kwargs):
        all_metrics = []
        all_programs = []
        all_times = []
        objectives = self.objectives if real_objectives is None else real_objectives
        for m, program_instance in enumerate(x):
            program = program_instance[0]
            all_programs.append(program)
            metrics = []
            try:
                time_12 = self.data[program.arch_str][self.dataset]['time-12']
                time_200 = self.data[program.arch_str][self.dataset]['time-200']
            except KeyError:
                print('time data was not found!')
                time_12 = None
                time_200 = None

            for objective in objectives:
                metric = self.data[program.arch_str][self.dataset][objective]
                metric = transform_metric(objective, metric)
                metrics.append(metric)
            all_metrics.append(metrics)
            all_times.append([time_12, time_200])
        all_metrics = np.asarray(all_metrics, dtype=float)
        all_programs = np.asarray(all_programs, dtype=object)
        if self.verbose:
            self.logger.log_evaluation(
                gen=self.generation,
                F=all_metrics,
                X=all_programs,
                time=np.array(all_times, dtype=float),
                evals=len(x),
            )
            # print('added a new generation to the log')
        self.generation += 1
        if real_objectives is None:
            out['F'] = all_metrics
        else:
            return np.array(all_metrics, dtype=float)

def transform_metric(objective, metric):
    if 'energy' in objective:
        metric = 1000 * metric  # convert Joule to mili Joule

    if 'params' in objective:
        metric = metric * 1e6  # Is in MB

    if 'flops' in objective:
        metric = metric * 1e6  # Is in MegaFLOPS

    if 'epoch' in objective:
        metric = 100 - metric  # convert accuracy to error
    return metric