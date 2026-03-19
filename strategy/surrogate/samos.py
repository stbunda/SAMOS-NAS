import pickle
import time
from typing import List

import matplotlib
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import ZeroToOneNormalization
from scipy.stats import stats

from problem.nas_problem import MO_Fitness as MO_Fitness_NAS
from problem.symbolic_problem import MO_Fitness as MO_Fitness_Symbolic
from problem.nasbench_problem import MO_Fitness as MO_Fitness_NASBENCH
from problem.nasbench101_problem import MO_Fitness_NASBench101 as MO_Fitness_NASBENCH101
from problem.darts_problem import DartsFitness
from strategy.surrogate.models.rf import RFR
from strategy.surrogate.subset_selection import subset_selection
from utils.logger import operator_config

matplotlib.use('Agg')


class SAMOS(Algorithm):
    def __init__(self,
                 sample_space: Sampling,  # Define search space
                 problem: Problem = None,  # Define search problem
                 surrogates: List=None,
                 use_subset_selection: bool = False,
                 n_doe: int = 100,  # Nr of initial architectures to fit surrogates
                 n_infill: int = 8,  # Nr of architectures to train
                 n_gen_candidates: int = 20,  # Nr of generations of NSGA-II loop
                 n_gen_surrogate: int = 30,  # Nr of generations of the surrogates model
                 n_var: int = None,  # Nr of variables
                 predict_objectives: List[str] = None,
                 real_objectives: List[str] = None,
                 logger=None,
                 sbatch='',
                 use_archive=None,
                 continue_id=None,
                 genetic_algorithm=None,  # Genetic algorithm parameters
                 ga_problem=None,
                 pad_to: int = 0,
                 verbose=1,  # Verbose 0: silent, Verbose 1: Images, Verbose 2: Text only
                 random_state: np.random.RandomState = None,
                 **kwargs):

        super().__init__(eliminate_duplicates=False, **kwargs)
        self.sample_space = sample_space
        self.problem = problem
        self.logger = logger
        self.genetic_algorithm = genetic_algorithm
        self.use_subset_selection = use_subset_selection
        self.ga_problem = ga_problem
        self.sbatch = sbatch
        self.use_archive = use_archive
        self.continue_id = continue_id
        self.verbose = verbose
        self.random_state = random_state
        self.n_doe = n_doe
        self.n_infill = n_infill
        self.n_gen_candidates = n_gen_candidates
        self.n_gen_surrogate = n_gen_surrogate
        self.n_var = n_var
        self.predict_objectives = predict_objectives
        self.real_objectives = real_objectives
        self.it = 0
        self.pad_to = pad_to
        self.metadata = None

        self.n_surrogates = len(self.predict_objectives)
        self.surrogates = surrogates
        self.predictions = np.zeros((self.n_gen_surrogate + 1, self.n_surrogates, self.n_infill))
        self.non_duplicate_infill = np.zeros(self.n_infill, dtype=bool)

        self.initialization = Initialization(sample_space)
        self._archive = Population()  # all solutions that have been evaluated so far
        self.infills = None  # here always the most recent infill solutions are stored

        # Per-generation surrogate overhead timing (filled by _infill; read by callbacks)
        # Keys: 'fit', 'nsga', 'select', 'total'  (all in seconds)
        self.last_infill_timing: dict = {}

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "sample_space": operator_config(self.sample_space),
            "problem": operator_config(self.problem),
            "surrogates": operator_config(self.surrogates),
            "n_doe": self.n_doe,
            "n_infill": self.n_infill,
            "n_gen_candidates": self.n_gen_candidates,
            "n_gen_surrogate": self.n_gen_surrogate,
            "predict_objectives": self.predict_objectives,
            "real_objectives": self.real_objectives,
            "sbatch": self.sbatch,
            "genetic_algorithm": self.genetic_algorithm,
            "use_archive": self.use_archive,
            "continue_id": self.continue_id,
            "ga_problem": self.ga_problem,
            "verbose": self.verbose,
            "n_var": self.n_var,
        }
        return self.metadata

    def _setup(self, problem, **kwargs):
        # initialize surrogates predictor
        if self.surrogates is None:
            self.surrogates = [
                RFR(100, seed=self.random_state.randint(0, 2 ** 31 - 1))
                for _ in range(self.n_surrogates)
            ]

    def _initialize_infill(self):
        if self.continue_id is not None:
            if self.continue_id[0] is not None:
                with open(self.continue_id[0], "rb") as f:
                    self.infills = pickle.load(f)
                    return self.infills
        if self.use_archive is not None:
            with open(self.use_archive, "rb") as f:
                self.infills = pickle.load(f)
        else:
            # Get n_doe samples from sample space and evaluate
            self.infills = self.initialization.do(self.problem, self.n_doe, algorithm=self)
        return self.infills

    def _initialize_advance(self, infills=None, **kwargs):
        # Merge the new infills with the previously evaluated population
        self.infills = infills
        self._archive = Population.merge(self._archive, infills)
        self.logger.save_archive(self._archive)

    def _advance(self, infills=None, **kwargs):
        if self.verbose: print(f'################### ADVANCING GENERATION {self.it} #########################')
        self.infills = infills

        program_tuples = [_x[0].tuple_program for _x in infills.get('X')]
        program_list = np.array([
            np.pad(t, (0, max(0, self.pad_to - len(t))), constant_values=0)
            for t in program_tuples
        ])
        pred = np.zeros((self.n_surrogates, self.n_infill))
        for s, surrogate in enumerate(self.surrogates):
            pred[s] = surrogate.predict(program_list).squeeze()

        # ---- Generation summary ----
        if self.verbose:
            F_inf = infills.get('F')
            F_arc = self._archive.get('F') if len(self._archive) > 0 else F_inf
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            arc_nd = F_arc[NonDominatedSorting().do(F_arc, only_non_dominated_front=True)]
            print(f"  Gen {self.it} infills ({len(F_inf)} candidates):")
            obj0 = self.predict_objectives[0] if self.predict_objectives else 'obj0'
            obj1 = self.real_objectives[0]    if self.real_objectives    else 'obj1'
            print(f"    {'#':>3}  {obj0:>18}  {obj1:>12}  {'surrogate_pred':>14}")
            for i, row in enumerate(F_inf):
                surr_str = f"{pred[0, i]:.2f}" if self.n_surrogates > 0 else "-"
                print(f"    {i+1:>3}  {row[0]:>18.3f}  {row[1]:>12.4f}  {surr_str:>14}")
            print(f"  Archive Pareto front ({len(arc_nd)} pts): "
                  f"{obj0} [{arc_nd[:,0].min():.2f} - {arc_nd[:,0].max():.2f}]  "
                  f"{obj1} [{arc_nd[:,1].min():.4f} - {arc_nd[:,1].max():.4f}]")

        self.logger.log_candidate(gen=self.it,
                                  X=infills.get('X'),
                                  F=infills.get('F'),
                                  Fpred=pred,
                                  evals=len(infills),
                                  surrogates=self.surrogates,
                                  non_duplicates=self.non_duplicate_infill
                                  )

        self._archive = Population.merge(self._archive, infills)
        self.logger.plot_hypervolume_2d_surrogate(self.it)
        self.logger.plot_surrogate_error(self.it)
        self.logger.plot_surrogate_rank_correlation_trajectory()
        self.it += 1

    # def plot_progress(self, infills, archive):
    #     # Get all results
    #     F = self._archive.get('F')
    #
    #     # Get surrogates archive results
    #     Fa = archive.get('F')
    #
    #     # Get candidate results
    #     Fc = infills.get('F')
    #
    #     # Error predictions
    #     # archive_x = np.array([_x[0].nd_program for _x in archive.get('X')])
    #     # infills_x = np.array([_x[0].nd_program for _x in infills.get('X')])
    #
    #     # if isinstance(self.surrogates[0], RNN):
    #     #     archive_x = [x[0].nd_program3 for x in archive.get('X')]
    #     #     infills_x = [x[0].nd_program3 for x in infills.get('X')]
    #     # else:
    #     archive_x = np.array([_x[0].nd_program for _x in archive.get('X')])
    #     infills_x = np.array([_x[0].nd_program for _x in infills.get('X')])
    #
    #     # a_error_pred = np.array([surrogates.predict(archive_x).squeeze() for surrogates in self.surrogates])
    #     # c_error_pred = np.array([surrogates.predict(infill_x).squeeze() for surrogates in self.surrogates])
    #
    #
    #     # Actual error
    #     a_error = Fa[:, :self.n_surrogates]
    #     c_error = Fc[:, :self.n_surrogates]
    #
    #     a_error_pred = np.zeros_like(a_error)
    #     c_error_pred = np.zeros_like(c_error)
    #
    #     # check for accuracy predictor's performance
    #     for s, surrogates in enumerate(self.surrogates):
    #         a_error_pred[:, s] = surrogates.predict(archive_x).squeeze()
    #         c_error_pred[:, s] = surrogates.predict(infills_x).squeeze()
    #         if s == self.predict_objectives.index('mse'):
    #             a_error_pred[:, s] = np.expm1(a_error_pred[:, s])
    #             c_error_pred[:, s] = np.expm1(c_error_pred[:, s])
    #
    #         rmse, rho, tau = get_correlation(
    #             np.hstack((a_error_pred[:, s], c_error_pred[:, s])),
    #             np.hstack((a_error[:, s], c_error[:, s]))
    #         )
    #
    #         print(f"fitting {surrogates}-{s}: RMSE = {rmse:.4f}, Spearmans Rho = {rho:.4f}, Kendalls Tau = {tau:.4f}")
    #
    #     # Save population
    #     # TODO: save population
    #
    #     # # Clip error for plotting
    #     # a_error = np.clip(a_error, 1e-6, 1e6)
    #     # c_error = np.clip(c_error, 1e-6, 1e6)
    #     # a_error_pred = np.clip(a_error_pred, 1e-6, 1e6)
    #     # c_error_pred = np.clip(c_error_pred, 1e-6, 1e6)
    #     # F = np.clip(F, 1e-6, 1e6)
    #     # Fa = np.clip(Fa, 1e-6, 1e6)
    #     # Fc = np.clip(Fc, 1e-6, 1e6)
    #
    #
    #     # calculate hypervolume
    #     hv = self._calc_hv(F)
    #     pF = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    #     pFc = Fc[NonDominatedSorting().do(Fc, only_non_dominated_front=True)]
    #     pFa = Fa[NonDominatedSorting().do(Fa, only_non_dominated_front=True)]
    #
    #     # print iteration-wise statistics
    #     print("Iter {}: hv = {:.2f}".format(self.it, hv))
    #     F_line = self.pareto_line(pF)
    #     Fa_line = self.pareto_line(pFa)
    #
    #     ### ========= Plot Objective Space =========
    #     plt.figure(figsize=(18, 10))
    #     plt.scatter(Fa[:, 0], Fa[:, 1], alpha=0.3, label='Archive', color='green')
    #     plt.plot(Fa_line[0, :], Fa_line[1, :], label='NDS Archive', marker='*', color='green')
    #     plt.scatter(pFa[:, 0], pFa[:, 1], label='NDS Archive', marker='*', color='green')
    #
    #     plt.scatter(c_error_pred[:, 0], Fc[:, 1], label=f'Predictions Gen {self.it}', marker='v',
    #                 color='orange')
    #
    #     plt.scatter(Fc[:, 0], Fc[:, 1], alpha=0.7, label=f'Evaluated Gen {self.it}', color='blue')
    #     plt.plot(F_line[0, :], F_line[1, :] , label=f'NDS Evaluated Gen {self.it}', marker='*', color='blue')
    #     plt.scatter(pFc[:, 0], pFc[:, 1], label=f'NDS Evaluated Gen {self.it}', marker='*', color='blue')
    #
    #     plt.scatter(pF[:, 0], pF[:, 1], label=f'NDS total population', marker='*')
    #
    #     plt.legend(loc='upper right')
    #     plt.xlabel(f'{self.predict_objectives[0]} Error')
    #     if self.n_surrogates > 1:
    #         plt.ylabel(f'{self.predict_objectives[1]} Error')
    #     else:
    #         plt.ylabel(f'{self.real_objectives[0]} Error')
    #     # plt.xlim(-1, 100)
    #     if self.predict_objectives[0] == 'mse':
    #         plt.xscale('log')
    #     plt.title("Objective Space 0-1")
    #     plt.savefig(os.path.join(self.logger_params['plot_path'], f"objective_space_i{self.it}.png"))
    #     plt.close()
    #
    #     #### make symbolic regression plot ####
    #     if self.predict_objectives[0] in ['mse', 'mae', 'rmse']:
    #         x_data = self.problem.x_data
    #         y_data = self.problem.y_data
    #         X = self._archive.get('X')
    #         pX = X[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    #
    #         plt.figure(figsize=(18, 10))
    #         plt.scatter(x_data, y_data, label='Reference')
    #         for i, x in enumerate(pX):
    #             pe = x[0].execute(x_data)
    #
    #             def truncate(s, length=60):
    #                 st = s.__str__()
    #                 return st[:length] + '...' if len(st) > length else st
    #             plt.scatter(x_data, pe, label=f'{truncate(x[0])}: {pF[i, 0]:0.3f}')
    #         plt.xlabel('X')
    #         plt.ylabel('Y')
    #         plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #                    fancybox=True, shadow=True, ncol=3, fontsize = 8)
    #         plt.title(f"Symbolic Regression Approximation")
    #         plt.savefig(os.path.join(self.logger_params['plot_path'], f"symbolic_regression_approx_i{self.it}.png"))
    #         plt.close()
    #
    #
    #
    #     ### ========= Plot Surrogate Error =========
    #     for s, surrogates in enumerate(self.surrogates):
    #         plt.figure(figsize=(18, 10))
    #         plt.scatter(a_error[:, s], a_error_pred[:, s], label='Archive')
    #         plt.scatter(c_error[:, s], c_error_pred[:, s], label=f'Gen {self.it}')
    #         plt.plot([np.min(a_error[:, s]), np.max(a_error[:, s])], [np.min(a_error[:, s]), np.max(a_error[:, s])], 'r--', label="y = x (Ideal)")
    #         plt.xlabel(f'{self.problem.objectives[s]} Error')
    #         plt.ylabel('Predicted Error')
    #         # if self.predict_objectives[s] == 'mse':
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         plt.xlim([np.min(a_error[:, s]), np.max(a_error[:, s])])
    #         plt.ylim([np.min(a_error_pred[:, s]), np.max(a_error_pred[:, s])])
    #         plt.legend()
    #         plt.title(f"{surrogates} Error Prediction")
    #         plt.savefig(os.path.join(self.logger_params['plot_path'], f"error_prediction_s{s}_i{self.it}.png"))
    #         plt.close()
    #
    # @staticmethod
    # def pareto_line(front):
    #     pareto_sorted = front[np.argsort(front[:, 0])]
    #     # Create step-wise points
    #     x_step = []
    #     y_step = []
    #
    #     for i in range(len(pareto_sorted) - 1):
    #         x_step.extend([pareto_sorted[i, 0], pareto_sorted[i + 1, 0]])  # Extend x with horizontal movement
    #         y_step.extend([pareto_sorted[i, 1], pareto_sorted[i, 1]])  # Extend y with constant value
    #
    #     # Add last point to finish the front
    #     x_step.append(pareto_sorted[-1, 0])
    #     y_step.append(pareto_sorted[-1, 1])
    #     return np.array([x_step, y_step])
    #
    # def _calc_hv(self, F):
    #     # calculate hypervolume on the non-dominated set of F
    #     norm_F = self.normalize(F, objectives=self.problem.objectives)
    #
    #     front = NonDominatedSorting().do(norm_F, only_non_dominated_front=True)
    #     nd_F = norm_F[front, :]
    #     hv_metric = Hypervolume(ref_point=np.array([1.01, 1.01]),
    #                             norm_ref_point=False,
    #                             zero_to_one=True,
    #                             ideal=norm_F.min(axis=0),
    #                             nadir=norm_F.max(axis=0))
    #
    #     hv = hv_metric.do(nd_F)
    #     return hv
    #
    # def normalize(self, F, objectives):
    #     F_norm = np.zeros_like(F)
    #     for o, obj in enumerate(objectives):
    #         if obj == 'Classification_Error':
    #             F_norm[:, o] = F[:, o] / 100
    #         elif obj == 'MACs' or obj == 'Parameters':
    #             log = np.log10(F[:, o])
    #             F_norm[:, o] = (log - np.min(log)) / (np.max(log) - np.min(log))
    #         else:
    #             F_norm[:, o] = (F[:, o] - np.min(F[:, o])) / (np.max(F[:, o]) - np.min(F[:, o]))
    #
    #     return F_norm

    def define_surrogate_algorithm(self):
        nsga_pop_size = self.genetic_algorithm.get('population_size', 40)
        topx = int(nsga_pop_size * 0.75)

        topx_pop = RankAndCrowding().do(problem=self.problem, pop=self._archive, n_survive=topx)
        random_pop = self.sample_space(self.problem, nsga_pop_size - topx)
        infill_sample_space = Population.merge(topx_pop, random_pop)

        return NSGA2(pop_size=nsga_pop_size,
                     sampling=infill_sample_space,
                     eliminate_duplicates=self.genetic_algorithm.get('dedup', False),
                     )

    def define_surrogate_problem(self, other_objective_functions):
        pass

    def _infill(self):
        if self.continue_id is not None:
            if self.continue_id[0] is not None:
                self.it = self.continue_id[1] + 1
                if self.verbose: print(
                    f'################### CONTINUE {self.continue_id[0]} - {self.continue_id[1]} #########################')
            self.continue_id = None

        if self.verbose: print('################### NEW INFILLS #########################')
        t_infill_start = time.perf_counter()

        # Look for the next K number of candidates for low level evaluation
        # Get non-dominated architectures from archive:
        X = self._archive.get('X')
        F = self._archive.get('F')

        # Fit the surrogates model
        t_fit_start = time.perf_counter()
        self._fit_surrogate(self._archive)
        t_fit = time.perf_counter() - t_fit_start
        if self.verbose:
            print(f'  [SAMOS timing] surrogate fitting: {t_fit:.2f}s')

        front = NonDominatedSorting().do(F, only_non_dominated_front=True)

        algorithm = self.genetic_algorithm

        if self.ga_problem.problem_type == 'SYMBOLIC REGRESSION':
            surrogate_problem = SurrogateProblemSymbolic(search_space=self.sample_space,
                                                         x_data=self.ga_problem.x_data,
                                                         y_data=self.ga_problem.y_data,
                                                         predictor=self.surrogates,
                                                         real_objectives=self.real_objectives,
                                                         predict_objectives=self.predict_objectives,
                                                         pad_to=self.pad_to)
        elif self.ga_problem.problem_type == 'NAS':
            surrogate_problem = SurrogateProblemNAS(search_space=self.sample_space,
                                                    datamodule=self.ga_problem.datamodule,
                                                    predictor=self.surrogates,
                                                    real_objectives=self.real_objectives,
                                                    predict_objectives=self.predict_objectives,
                                                    pad_to=self.pad_to)
        elif self.ga_problem.problem_type == 'NASBENCH':
            surrogate_problem = SurrogateProblemNASBENCH(search_space=self.sample_space,
                                                         data=self.ga_problem.data,
                                                         dataset=self.ga_problem.dataset,
                                                         predictor=self.surrogates,
                                                         real_objectives=self.real_objectives,
                                                         predict_objectives=self.predict_objectives,
                                                         pad_to=self.pad_to)
        elif self.ga_problem.problem_type == 'NASBENCH101':
            surrogate_problem = SurrogateProblemNASBENCH101(search_space=self.sample_space,
                                                            data=self.ga_problem.data,
                                                            predictor=self.surrogates,
                                                            real_objectives=self.real_objectives,
                                                            predict_objectives=self.predict_objectives,
                                                            pad_to=self.pad_to)
        elif self.ga_problem.problem_type == 'DARTS':
            surrogate_problem = SurrogateProblemDARTS(search_space=self.sample_space,
                                                      ga_problem=self.ga_problem,
                                                      predictor=self.surrogates,
                                                      real_objectives=self.real_objectives,
                                                      predict_objectives=self.predict_objectives,
                                                      pad_to=self.pad_to)
        else:
            raise NotImplementedError(f"No surrogate problem defined for problem_type={self.ga_problem.problem_type!r}")

        t_nsga_start = time.perf_counter()
        res = minimize(surrogate_problem,
                       algorithm,
                       ('n_gen', self.n_gen_candidates),
                       seed=self.seed,
                       save_history=True,
                       verbose=True)
        t_nsga = time.perf_counter() - t_nsga_start
        if self.verbose:
            print(f'  [SAMOS timing] inner NSGA-II ({self.n_gen_candidates} gens): {t_nsga:.2f}s')

        # ---- Initialize ----
        t_select_start = time.perf_counter()
        infill = None
        self.non_duplicate_infill = None

        # ---- Eliminate duplicates ----
        not_duplicate_pop = self.genetic_algorithm.eliminate_duplicates.do(
            res.pop, self._archive
        )

        if self.verbose:
            print(f'Nr of non duplicates: {len(not_duplicate_pop)}')

        # ---- Case 1: no found candidates ----
        if len(not_duplicate_pop) == 0:
            infill = self._sample_dedup(self.n_infill, self._archive)
            self.non_duplicate_infill = np.zeros(len(infill), dtype=bool)

        # ---- Case 2: found candidates exist ----
        elif self.use_subset_selection:
            # Find a diverse subset of the non-dominated solutions
            # Pass full multi-objective fitness for proper Pareto-based selection
            indices = subset_selection(
                not_duplicate_pop.get('F'),      # Full multi-objective fitness
                self._archive.get('F')[front],   # Archive Pareto front (multi-objective)
                self.n_infill
            )

            if self.verbose:
                n_non_dom = len(indices) if indices is not None else 0
                print(f"[SUBSET SELECTION] Non-dominated candidates found: {n_non_dom}, Need: {self.n_infill}")

            # ---------- indices is None → take all found ----------
            if indices is None:
                found = not_duplicate_pop
                n_found = len(found)

                infill_add = self._sample_dedup(
                    self.n_infill - n_found,
                    Population.merge(self._archive, found)
                )

                infill = Population.merge(infill_add, found)

                self.non_duplicate_infill = np.concatenate([
                    np.zeros(len(infill_add), dtype=bool),
                    np.ones(n_found, dtype=bool),
                ])

            # ---------- fewer than required ----------
            elif len(indices) < self.n_infill:
                found = not_duplicate_pop[indices]
                n_found = len(found)

                infill_add = self._sample_dedup(
                    self.n_infill - n_found,
                    Population.merge(self._archive, found)
                )

                infill = Population.merge(infill_add, found)

                self.non_duplicate_infill = np.concatenate([
                    np.zeros(len(infill_add), dtype=bool),
                    np.ones(n_found, dtype=bool),
                ])

            # ---------- exact selection ----------
            else:
                infill = not_duplicate_pop[indices]
                self.non_duplicate_infill = np.ones(len(infill), dtype=bool)
        else:
            # Fallback: Select without subset selection (use multi-front approach)
            F = not_duplicate_pop.get("F")
            fronts = NonDominatedSorting().do(F)

            if self.verbose:
                print(f"[NO SUBSET SELECTION] Using multi-front selection, Number of fronts: {len(fronts)}")
                for i, f in enumerate(fronts):
                    print(f"  Front {i}: {len(f)} solutions")

            selected = []

            for front in fronts:
                if len(selected) + len(front) <= self.n_infill:
                    selected.extend(front)
                else:
                    # Need to split this front using crowding distance
                    remaining = self.n_infill - len(selected)
                    crowding = calc_crowding_distance(F[front])
                    order = np.argsort(-crowding)  # descending
                    selected.extend(front[order[:remaining]])
                    break

            if self.verbose:
                print(f"  Selected: {len(selected)} solutions")



            # infill = not_duplicate_pop[selected]
            # self.non_duplicate_infill = np.ones(len(infill), dtype=bool)

            n_found = len(selected)

            if n_found < self.n_infill:
                found = not_duplicate_pop[selected]
                infill_add = self._sample_dedup(
                    self.n_infill - n_found,
                    Population.merge(self._archive, found)
                )
                infill = Population.merge(infill_add, found)
                self.non_duplicate_infill = np.concatenate([
                    np.zeros(len(infill_add), dtype=bool),
                    np.ones(n_found, dtype=bool),
                ])
            else:
                infill = not_duplicate_pop[selected]
                self.non_duplicate_infill = np.ones(len(infill), dtype=bool)



        # program_tuples = [_x[0].tuple_program for _x in infill.get('X')]
        # program_list = np.array([
        #     np.pad(t, (0, max(0, self.pad_to - len(t))), constant_values=0)
        #     for t in program_tuples
        # ])

        # for s, surrogate in enumerate(self.surrogates):
        #     self.predictions[:, s] = surrogate.predict(program_list).squeeze()

        t_select = time.perf_counter() - t_select_start
        t_infill_total = time.perf_counter() - t_infill_start
        if self.verbose:
            print(f'  [SAMOS timing] candidate selection: {t_select:.2f}s')
            print(f'  [SAMOS timing] total _infill overhead: {t_infill_total:.2f}s '
                  f'(fit={t_fit:.1f}s + nsga={t_nsga:.1f}s + select={t_select:.1f}s)')

        # Expose timing for external callbacks / reporting
        self.last_infill_timing = {
            'fit':    t_fit,
            'nsga':   t_nsga,
            'select': t_select,
            'total':  t_infill_total,
        }

        # Make sure the candidates are evaluated with real fitness function
        candidates = Population().new("X", infill.get('X'))

        return candidates

    def _sample_dedup(self, n: int, reference: 'Population', max_iters: int = 20) -> 'Population':
        """Sample ``n`` candidates that are not already in *reference*.

        Wraps ``self.sample_space`` with duplicate elimination so that
        random fill-in points added to infill batches are not wasted on
        already-evaluated architectures.
        """
        sampled = Population.empty()
        for _ in range(max_iters):
            if len(sampled) >= n:
                break
            needed = n - len(sampled)
            cand = self.sample_space(self.problem, needed)
            combined_ref = Population.merge(reference, sampled)
            cand = self.genetic_algorithm.eliminate_duplicates.do(cand, combined_ref)
            sampled = Population.merge(sampled, cand)
        return sampled

    def _fit_surrogate(self, datapoints):
        X = datapoints.get('X')
        F = datapoints.get('F')

        assert len(X) > 1, "Need at least 2 training samples to fit the surrogate"

        print(f'Fitting surrogates to {len(X)} training samples')

        program_tuples = [_x[0].tuple_program for _x in X]
        program_list = np.array([
            np.pad(t, (0, max(0, self.pad_to - len(t))), constant_values=0)
            for t in program_tuples
        ])
        for s, surrogate in enumerate(self.surrogates):
            surrogate.fit(x=program_list, y=F[:, s])
            surrogate_prediction = surrogate.predict(program_list).squeeze()

            rmse = np.sqrt(((F[:, s] - surrogate_prediction) ** 2).mean())
            rho, _ = stats.spearmanr(surrogate_prediction, F[:, s])
            tau, _ = stats.kendalltau(surrogate_prediction, F[:, s])
            print(f"Training results: RMSE = {rmse:.3f}, rho = {rho:.3f}, tau = {tau:.3f}")


class SurrogateProblemSymbolic(MO_Fitness_Symbolic):
    def __init__(self,
                 search_space,
                 x_data,
                 y_data,
                 predictor,
                 predict_objectives,
                 real_objectives,
                 pad_to):
        super().__init__(objectives=predict_objectives + real_objectives, x_data=x_data, y_data=y_data, nvar_real=1,
                         n_constr=0, requires_kwargs=True, verbose=0)

        self.ss = search_space
        self.predictor = predictor
        self.predict_objectives = predict_objectives
        self.real_objectives = real_objectives
        self.pad_to = pad_to

        # Bounds for real-valued variables
        self.xl = np.zeros(1)
        self.xu = 0.999999 * np.ones(1)

    def _evaluate(self, x, out, *args, **kwargs):
        program_tuples = [_x[0].tuple_program for _x in x]
        program_list = np.array([
            np.pad(t, (0, max(0, self.pad_to - len(t))), constant_values=0)
            for t in program_tuples
        ])

        prediction = np.zeros((x.shape[0], len(self.predict_objectives)))
        for p, predictor in enumerate(self.predictor):
            prediction[:, p] = predictor.predict(program_list).squeeze()

        real_metrics = super()._evaluate(x, out, real_objectives=self.real_objectives)
        out['F'] = np.hstack((prediction, real_metrics))


class SurrogateProblemNAS(MO_Fitness_NAS):
    def __init__(self,
                 search_space,
                 datamodule,
                 predictor,
                 predict_objectives,
                 real_objectives,
                 pad_to):
        super().__init__(objectives=predict_objectives + real_objectives, datamodule=datamodule, nvar_real=1,
                         n_constr=0, requires_kwargs=True, verbose=0)

        self.ss = search_space
        self.predictor = predictor
        self.predict_objectives = predict_objectives
        self.real_objectives = real_objectives
        self.pad_to = pad_to

        # Bounds for real-valued variables
        self.xl = np.zeros(1)
        self.xu = 0.999999 * np.ones(1)

    def _evaluate(self, x, out, *args, **kwargs):
        program_tuples = [_x[0].tuple_program for _x in x]
        program_list = np.array([
            np.pad(t, (0, max(0, self.pad_to - len(t))), constant_values=0)
            for t in program_tuples
        ])

        prediction = np.zeros((x.shape[0], len(self.predict_objectives)))
        for p, predictor in enumerate(self.predictor):
            prediction[:, p] = predictor.predict(program_list).squeeze()

        real_metrics = super()._evaluate(x, out, real_objectives=self.real_objectives)
        out['F'] = np.hstack((prediction, real_metrics))

class SurrogateProblemNASBENCH(MO_Fitness_NASBENCH):
    def __init__(self,
                 search_space,
                 data,
                 dataset,
                 predictor,
                 predict_objectives,
                 real_objectives,
                 pad_to):
        super().__init__(objectives=predict_objectives + real_objectives, data=data, dataset=dataset, nvar_real=1,
                         n_constr=0, requires_kwargs=True, verbose=0)

        self.ss = search_space
        self.predictor = predictor
        self.predict_objectives = predict_objectives
        self.real_objectives = real_objectives
        self.pad_to = pad_to

        # Bounds for real-valued variables
        self.xl = np.zeros(1)
        self.xu = 0.999999 * np.ones(1)

    def _evaluate(self, x, out, *args, **kwargs):
        program_tuples = [_x[0].tuple_program for _x in x]
        program_list = np.array([
            np.pad(t, (0, max(0, self.pad_to - len(t))), constant_values=0)
            for t in program_tuples
        ])

        prediction = np.zeros((x.shape[0], len(self.predict_objectives)))
        for p, predictor in enumerate(self.predictor):
            prediction[:, p] = predictor.predict(program_list).squeeze()

        real_metrics = super()._evaluate(x, out, real_objectives=self.real_objectives)
        out['F'] = np.hstack((prediction, real_metrics))


class SurrogateProblemNASBENCH101(MO_Fitness_NASBENCH101):
    def __init__(self,
                 search_space,
                 data,
                 predictor,
                 predict_objectives,
                 real_objectives,
                 pad_to):
        super().__init__(objectives=predict_objectives + real_objectives, data=data,
                         n_var=1, verbose=0)

        self.ss = search_space
        self.predictor = predictor
        self.predict_objectives = predict_objectives
        self.real_objectives = real_objectives
        self.pad_to = pad_to

        self.xl = np.zeros(1)
        self.xu = 0.999999 * np.ones(1)

    def __deepcopy__(self, memo):
        # Avoid deep-copying the 423K-entry data dict on every pymoo _post_advance call.
        # We create a shallow copy and share the data reference.
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'data':
                setattr(result, k, v)  # share by reference
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def _evaluate(self, x, out, *args, **kwargs):
        program_tuples = [_x[0].tuple_program for _x in x]
        program_list = np.array([
            np.pad(t, (0, max(0, self.pad_to - len(t))), constant_values=0)
            for t in program_tuples
        ])

        prediction = np.zeros((x.shape[0], len(self.predict_objectives)))
        for p, predictor in enumerate(self.predictor):
            prediction[:, p] = predictor.predict(program_list).squeeze()

        real_metrics = super()._evaluate(x, out, real_objectives=self.real_objectives)
        if real_metrics is None or len(self.real_objectives) == 0:
            out['F'] = prediction
        else:
            out['F'] = np.hstack((prediction, real_metrics))


class SurrogateProblemDARTS(Problem):
    """
    Lightweight surrogate inner-loop problem for DARTS search.

    Predicts objectives via the RFR surrogate (no GPU).
    Computes 'params' cheaply by instantiating the network without training.
    All other real objectives are filled with zeros in the inner loop.
    """

    def __init__(self,
                 search_space,
                 ga_problem: DartsFitness,
                 predictor,
                 predict_objectives,
                 real_objectives,
                 pad_to):
        n_obj = len(predict_objectives) + len(real_objectives)
        super().__init__(n_var=1, n_obj=n_obj, n_constr=0, requires_kwargs=True)

        self.ss = search_space
        self.predictor = predictor
        self.predict_objectives = predict_objectives
        self.real_objectives = real_objectives
        self.pad_to = pad_to

        # Geometry needed for params counting only
        self.init_channels = ga_problem.init_channels
        self.n_classes     = ga_problem.n_classes
        self.layers        = ga_problem.layers

        self.xl = np.zeros(1)
        self.xu = 0.999999 * np.ones(1)

    def _evaluate(self, x, out, *args, **kwargs):
        from problem.darts_problem import DartsNetwork, _count_parameters

        program_tuples = [_x[0].tuple_program for _x in x]
        program_list = np.array([
            np.pad(t, (0, max(0, self.pad_to - len(t))), constant_values=0)
            for t in program_tuples
        ])

        # Surrogate predictions (cheap -- no GPU)
        prediction = np.zeros((x.shape[0], len(self.predict_objectives)))
        for p, predictor in enumerate(self.predictor):
            prediction[:, p] = predictor.predict(program_list).squeeze()

        # Real objectives computed without training
        real_cols = []
        for obj in self.real_objectives:
            if obj == 'params':
                col = []
                for _x in x:
                    genome = _x[0]
                    try:
                        model = DartsNetwork(
                            genome=genome,
                            init_channels=self.init_channels,
                            n_classes=self.n_classes,
                            layers=self.layers,
                            auxiliary=False,
                        )
                        col.append(_count_parameters(model))
                    except Exception:
                        col.append(0.0)
                real_cols.append(col)
            else:
                real_cols.append([0.0] * x.shape[0])

        real_metrics = np.column_stack(real_cols) if real_cols else np.zeros((x.shape[0], 0))
        out['F'] = np.hstack((prediction, real_metrics))


def get_correlation(prediction, target):
    import scipy.stats as stats

    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau


class MyNormalization(ZeroToOneNormalization):

    def forward(self, X):
        return super().forward(X) * 200 - 100

    def backward(self, X):
        return super().backward((X + 100) / 200)
