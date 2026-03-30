"""main_samos_ablation.py — Ablation study: why does SAMOS underperform SSA-NSGA-II / GPSAF?

Isolates five factors that differ between SAMOS and SSA-NSGA-II / GPSAF:
  A. Surrogate model type       (gpr, xgb, rfr)
  B. Warm-start ratio           (0.0, 0.25, 0.5, 0.75, 1.0)
  C. Candidate selection method (subset, kmeans, crowding)
  D. Inner population size      (50, 100, 200, 500)
  E. Subset selection on/off    (True, False)

Each experiment fixes all other factors to defaults and sweeps one.
SSA-NSGA-II (default + xgb) and GPSAF (default + xgb) are included as
reference baselines in every ablation plot.

Run examples:
  python main_samos_ablation.py --problem wfg3 --ablation surrogate
  python main_samos_ablation.py --problem wfg3 --ablation warm_start
  python main_samos_ablation.py --problem wfg3 --ablation selection
  python main_samos_ablation.py --problem wfg3 --ablation inner_pop
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
from strategy.surrogate.models import RFR, XGBoost
from strategy.surrogate.models.kriging import GPR

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


# ─── Diagnostics-enhanced SAMOS ──────────────────────────────────────────────

class SAMOSAblation(Algorithm):
    """SAMOS variant that logs per-generation diagnostics for ablation studies.

    Logs to self.diagnostics (list of dicts) per generation:
      - surrogate_rmse:     per-objective RMSE on training set
      - surrogate_spearman: per-objective Spearman rho on training set
      - n_candidates:       candidates surviving dedup
      - n_random_fill:      random padding added
      - inner_hv:           HV of inner NSGA-II result on surrogate
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

        self._archive      = Population()
        self._archive_keys: set = set()
        self._init         = Initialization(sampling)
        self.diagnostics   = []

    def _setup(self, problem, **kwargs):
        pass

    def _initialize_infill(self):
        return self._init.do(self.problem, self.n_doe, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        self._archive = Population.merge(self._archive, infills)
        self._add_to_archive_keys(infills)
        self.pop = infills

    def _infill(self):
        X_arc = self._archive.get('X')
        F_arc = self._archive.get('F')
        diag = {'gen': self.n_gen, 'n_archive': len(self._archive)}

        # 1. Fit surrogates + compute diagnostics
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
        diag['surrogate_rmse'] = rmses
        diag['surrogate_spearman'] = spearmans
        diag['surrogate_kendall'] = kendalls

        # 2. Warm-start
        topx    = max(1, int(self.ga_pop_size * self.warm_start_ratio))
        top_pop = RankAndCrowding().do(problem=self.problem, pop=self._archive, n_survive=topx)
        n_rand  = self.ga_pop_size - len(top_pop)
        if n_rand > 0:
            rand_pop = self._init.do(self.problem, n_rand, algorithm=self)
            inner_X  = np.vstack([top_pop.get('X'), rand_pop.get('X')])
        else:
            inner_X  = top_pop.get('X')
        inner_init = Population.new('X', inner_X)
        diag['n_warm_archive'] = len(top_pop)
        diag['n_warm_random'] = max(0, n_rand)

        # 3. Inner NSGA-II on surrogate
        from problem.pymoo.surrogate_problem import SurrogateProblemMOO
        surr_problem = SurrogateProblemMOO(
            self.surrogates, self.problem.n_var,
            self.problem.xl.copy(), self.problem.xu.copy(),
            real_problem=self.problem,
        )
        inner_alg = NSGA2(
            pop_size=self.ga_pop_size,
            sampling=inner_init,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=self.eliminate_duplicates,
        )
        res = minimize(
            surr_problem, inner_alg,
            termination=('n_gen', self.n_gen_inner),
            verbose=False,
        )
        diag['n_surr_eval'] = res.algorithm.evaluator.n_eval

        # 4. Deduplicate
        cand_pop = res.pop if res.pop is not None else Population.empty()
        n_before_dedup = len(cand_pop)
        if len(cand_pop) > 0:
            not_dup = np.array([
                self._dedup_key(cx) not in self._archive_keys
                for cx in cand_pop.get('X')
            ], dtype=bool)
            cand_pop = cand_pop[not_dup]
        diag['n_before_dedup'] = n_before_dedup
        diag['n_after_dedup'] = len(cand_pop)

        # 5. Select candidates
        infill_pop, n_random_fill = self._select_infill(cand_pop, F_arc)
        diag['n_selected'] = len(infill_pop) - n_random_fill
        diag['n_random_fill'] = n_random_fill

        # 6. Evaluate candidates on REAL problem to measure surrogate-to-real correlation
        infill_X = infill_pop.get('X')
        real_out = {}
        self.problem._evaluate(infill_X, real_out)
        real_F = real_out['F']
        surr_F = np.column_stack([s.predict(infill_X).ravel() for s in self.surrogates])
        infill_corrs = []
        for j in range(real_F.shape[1]):
            rho, _ = spearmanr(surr_F[:, j], real_F[:, j])
            infill_corrs.append(float(rho) if not np.isnan(rho) else 0.0)
        diag['infill_surrogate_vs_real_spearman'] = infill_corrs

        self.diagnostics.append(diag)
        return Population.new('X', infill_X)

    def _advance(self, infills=None, **kwargs):
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
    rng = np.random.RandomState(seed)
    surrs = []
    for _ in range(n_obj):
        s = rng.randint(0, 2**31 - 1)
        if stype == 'xgb':
            surrs.append(XGBoost(100, seed=s))
        elif stype == 'rfr':
            surrs.append(RFR(100, seed=s))
        elif stype == 'gpr':
            surrs.append(GPR(seed=s))
        else:
            raise ValueError(f'Unknown surrogate type: {stype}')
    return surrs


ABLATION_CONFIGS = {
    'surrogate': {
        'sweep_param': 'surrogate_type',
        'values': ['gpr', 'xgb', 'rfr'],
        'defaults': dict(warm_start_ratio=1.0, selection_method='subset',
                         use_subset_selection=True, ga_pop_size=200),
    },
    'warm_start': {
        'sweep_param': 'warm_start_ratio',
        'values': [0.0, 0.25, 0.5, 0.75, 1.0],
        'defaults': dict(surrogate_type='xgb', selection_method='subset',
                         use_subset_selection=True, ga_pop_size=200),
    },
    'selection': {
        'sweep_param': 'selection_method',
        'values': ['subset', 'kmeans', 'crowding'],
        'defaults': dict(surrogate_type='xgb', warm_start_ratio=1.0,
                         use_subset_selection=True, ga_pop_size=200),
    },
    'inner_pop': {
        'sweep_param': 'ga_pop_size',
        'values': [50, 100, 200, 500],
        'defaults': dict(surrogate_type='xgb', warm_start_ratio=1.0,
                         selection_method='subset', use_subset_selection=True),
    },
    'subset_sel': {
        'sweep_param': 'use_subset_selection',
        'values': [True, False],
        'defaults': dict(surrogate_type='xgb', warm_start_ratio=1.0,
                         selection_method='subset', ga_pop_size=200),
    },
}


def run_ablation_single(problem_name, seed, ablation, sweep_value, pop_size, n_gen,
                         n_obj=2, n_var=None, n_gen_inner=20, defaults=None):
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
    cfg[ABLATION_CONFIGS[ablation]['sweep_param']] = sweep_value

    surrogate_type = cfg.pop('surrogate_type', 'xgb')
    surrogates = _make_surrogates(surrogate_type, problem.n_obj, seed)

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
        'sweep_param': ABLATION_CONFIGS[ablation]['sweep_param'],
        'sweep_value': sweep_value,
        'surrogate_type': surrogate_type,
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
            config = ABLATION_CONFIGS[ablation]
            sweep_param = config['sweep_param']
            values = config['values']
            defaults = config['defaults']

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
                    )

                    with open(out_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f'    Saved -> {out_path}')

            # ── plot ablation results ────────────────────────────────────
            _plot_ablation(prob, ablation, values, sweep_param, results_root,
                           ref_root, args.n_gen, args.pop_size, args.n_obj, args.n_var)


def _plot_ablation(prob, ablation, values, sweep_param, results_root,
                   ref_root, n_gen, pop_size, n_obj, n_var):
    """Generate HV/IGD+ trajectory + surrogate-diagnostics plots for one ablation.

    Reference baselines (SSA-NSGA-II, GPSAF) are drawn as dashed lines.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    problem   = build_problem(prob, n_obj, n_var)
    pf        = get_pareto_front(problem, problem.n_obj)
    ref_point = default_ref_point(prob, problem.n_obj)
    hv_ind    = HV(ref_point=ref_point)
    hv_ceiling = float(hv_ind(pf))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_hv, ax_igd, ax_rmse = axes[0]
    ax_spearman, ax_nfill, ax_infill_corr = axes[1]

    n_colours = max(len(values), 10)
    colours = plt.cm.tab10(np.linspace(0, 1, n_colours))

    # ── Helper: load seeds from a directory and compute mean trajectories ──
    def _load_trajectories(save_dir):
        all_hv, all_igd = [], []
        all_rmse, all_spearman, all_nfill, all_infill_corr = [], [], [], []
        if not os.path.isdir(save_dir):
            return None
        for seed_file in sorted(os.listdir(save_dir)):
            if not seed_file.endswith('.pkl'):
                continue
            with open(os.path.join(save_dir, seed_file), 'rb') as f:
                data = pickle.load(f)
            hvs  = [ind['hv'] for ind in data['indicators']]
            igds = [ind['igd_plus'] for ind in data['indicators']]
            all_hv.append(hvs)
            all_igd.append(igds)
            if 'diagnostics' in data:
                diags = data['diagnostics']
                all_rmse.append([np.mean(d['surrogate_rmse']) for d in diags])
                all_spearman.append([np.mean(d['surrogate_spearman']) for d in diags])
                all_nfill.append([d['n_random_fill'] for d in diags])
                all_infill_corr.append([np.mean(d['infill_surrogate_vs_real_spearman']) for d in diags])
        if not all_hv:
            return None
        return dict(hv=all_hv, igd=all_igd, rmse=all_rmse,
                    spearman=all_spearman, nfill=all_nfill, infill_corr=all_infill_corr)

    def _plot_mean_std(ax, data_list, color, label, linestyle='-', **kw):
        if not data_list:
            return
        min_len = min(len(d) for d in data_list)
        arr = np.array([d[:min_len] for d in data_list])
        x = np.arange(1, min_len + 1) * pop_size
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        ax.plot(x, mu, color=color, label=label, linestyle=linestyle, **kw)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.10, color=color)

    # ── Plot ablation sweep variants (solid lines) ────────────────────────
    for vi, val in enumerate(values):
        method_name = f'{sweep_param}={val}'
        save_dir = os.path.join(results_root, method_name)
        traj = _load_trajectories(save_dir)
        if traj is None:
            continue
        c = colours[vi]
        label = str(val)
        _plot_mean_std(ax_hv, traj['hv'], c, label)
        _plot_mean_std(ax_igd, traj['igd'], c, label)
        _plot_mean_std(ax_rmse, traj['rmse'], c, label)
        _plot_mean_std(ax_spearman, traj['spearman'], c, label)
        _plot_mean_std(ax_nfill, traj['nfill'], c, label)
        _plot_mean_std(ax_infill_corr, traj['infill_corr'], c, label)

    # ── Plot reference baselines (dashed lines) ──────────────────────────
    ref_colours = {
        'ssa-nsga2-default': 'black',
        'ssa-nsga2-xgb':    'dimgrey',
        'gpsaf-default':    'royalblue',
        'gpsaf-xgb':        'cornflowerblue',
    }
    for ref_method in REFERENCE_BASELINES:
        save_dir = os.path.join(ref_root, ref_method)
        traj = _load_trajectories(save_dir)
        if traj is None:
            continue
        c = ref_colours.get(ref_method, 'grey')
        _plot_mean_std(ax_hv, traj['hv'], c, ref_method, linestyle='--', linewidth=1.5)
        _plot_mean_std(ax_igd, traj['igd'], c, ref_method, linestyle='--', linewidth=1.5)
        # Reference baselines don't have diagnostics — only HV/IGD

    ax_hv.axhline(hv_ceiling, color='grey', linestyle=':', alpha=0.5, label='PF ceiling')
    ax_hv.set_title('HV (↑)')
    ax_hv.set_xlabel('Evaluations')
    ax_hv.legend(fontsize=7)

    ax_igd.set_title('IGD+ (↓)')
    ax_igd.set_xlabel('Evaluations')
    ax_igd.legend(fontsize=7)

    ax_rmse.set_title('Surrogate RMSE (train, ↓)')
    ax_rmse.set_xlabel('Evaluations')
    ax_rmse.legend(fontsize=7)

    ax_spearman.set_title('Surrogate Spearman ρ (train, ↑)')
    ax_spearman.set_xlabel('Evaluations')
    ax_spearman.legend(fontsize=7)

    ax_nfill.set_title('Random fill-in count (↓)')
    ax_nfill.set_xlabel('Evaluations')
    ax_nfill.legend(fontsize=7)

    ax_infill_corr.set_title('Infill surr-vs-real Spearman ρ (↑)')
    ax_infill_corr.set_xlabel('Evaluations')
    ax_infill_corr.legend(fontsize=7)

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
    parser = argparse.ArgumentParser(description='SAMOS ablation study')
    parser.add_argument('--problem', type=str, nargs='+', default=['wfg3'],
                        help='WFG problem(s) to run on (default: wfg3)')
    parser.add_argument('--n_obj', type=int, default=2)
    parser.add_argument('--n_var', type=int, default=None)
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['surrogate', 'warm_start', 'selection', 'inner_pop',
                                 'subset_sel', 'all'],
                        help='Which ablation to run (default: all)')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--pop_size', type=int, default=20)
    parser.add_argument('--n_gen', type=int, default=60)
    parser.add_argument('--n_gen_inner', type=int, default=20)
    parser.add_argument('--experiment_name', type=str, default='samos_ablation/2_obj')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    print(f'Arguments: {args}')
    main(args)
