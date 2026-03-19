# logger.py
import copy
import itertools
import os
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, EngFormatter

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import stats

from utils.plotter import RunMetrics, RunPlotter

matplotlib.use("Agg")


# ============================================================
# Per-generation canonical state
# ============================================================

@dataclass
class GenerationStats:
    # Raw, logged data
    X: "np.ndarray | None"             # Population
    F: np.ndarray                      # (N, M) objectives
    time: np.ndarray
    evals: int                         # cumulative eval count
    prev: "GenerationStats | None" = None

    # Derived / cached
    pareto_gen: Optional[np.ndarray] = None
    hv_gen: Optional[float] = None
    nhv_gen: Optional[float] = None

    metric_stats: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def cum_evals(self) -> int:
        if self.prev is None:
            return self.evals
        return self.prev.cum_evals + self.evals

@dataclass
class CandidateStats:
    X: "np.ndarray | None"
    F: np.ndarray
    Fpred: np.ndarray
    evals: int
    prev: "CandidateStats | None" = None
    surrogates: list = None
    non_duplicates: np.ndarray = None

    @property
    def cum_evals(self) -> int:
        if self.prev is None:
            return self.evals
        return self.prev.cum_evals + self.evals

# ============================================================
# Evolution Logger
# ============================================================

class EvolutionLogger:
    """
    Single-ingress evolution logger.

    The ONLY method called from Problem._evaluate is `log_evaluation`.

    Everything else (Pareto fronts, hypervolume, statistics, plots)
    is derived lazily from raw logged data.
    """

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def __init__(
        self,
        name: str,
        save_dir: str = "results",
        version: Optional[str] = None,
            absolute_dir: str = None,
        autosave_every: int = 1,
    ):
        self.name = name
        self.save_dir = save_dir
        self.version = version or self._default_version()
        self.autosave_every = autosave_every
        self.absolute_dir = absolute_dir

        self.objectives: Optional[List[str]] = None
        self.generations: Dict[int, GenerationStats] = {}
        self.candidates: Dict[int, CandidateStats] = {}
        self.true_pareto = None

        # Persistent location (always exists conceptually)
        self.base_dir = self._resolve(self.save_dir, name, self.version)

        # Scratch location (optional)
        self.absolute_base_dir = (
            self._resolve(
                os.path.join(self.absolute_dir, save_dir), name, self.version)
            if self.absolute_dir is not None
            else None
        )

        self.pkl_dir = os.path.join(self.base_dir , "pkl")

        self._make_dirs()

        self.meta = {
            "name": name,
            "version": self.version,
            "created": datetime.now().isoformat(),
            "environment": self._environment_info(),
        }

    def __deepcopy__(self, memo):
        return self

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    def set_objectives(self, objectives: List[str]) -> None:
        self.objectives = objectives

    def set_true_pareto(self, pareto_front: np.ndarray) -> None:
        self.true_pareto = pareto_front

    def register_metadata(
        self,
        *,
        dataset=None,
        program_config=None,
        problem=None,
        algorithm=None,
        arguments=None,
    ):
        meta = dict(self.meta)
        meta.update(
            {
                "dataset": self._serialize(dataset),
                "program_config": self._serialize(program_config),
                "problem": self._serialize(problem),
                "algorithm": self._serialize(algorithm),
                "arguments": self._serialize(arguments),
            }
        )

        with open(os.path.join(self.base_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # --------------------------------------------------------
    # Single ingress (from Problem._evaluate)
    # --------------------------------------------------------

    def log_evaluation(
            self,
            *,
            gen: int,
            F: np.ndarray,
            time: np.ndarray,
            evals: int,
            X: Optional[np.ndarray] = None,
    ) -> None:
        prev = self.generations.get(gen - 1) if gen > 0 else None
        self.generations[gen] = GenerationStats(
            X=X,
            F=np.asarray(F, dtype=float),
            time=time,
            evals=int(evals),
            prev=prev,
        )

        if gen % self.autosave_every == 0:
            self.save()

    def log_candidate(self, *,
                      gen: int,
                      X: np.ndarray = None,
                      F: np.ndarray,
                      evals: int,
                      surrogates = None,
                      Fpred: Optional[np.ndarray] = None,
                      non_duplicates: Optional[np.ndarray] = None,
                      ) -> None:
        prev = self.candidates.get(gen - 1) if gen > 0 else None
        if Fpred is None:
            Fpred = np.zeros_like(F[:len(surrogates)])
            for s, surrogate in enumerate(surrogates):
                Fpred[:, s] = surrogate.predict(X).squeeze()
        self.candidates[gen] = CandidateStats(
            X=X,
            F=F,
            Fpred=Fpred.T,
            evals=int(evals),
            prev=prev,
            surrogates=surrogates,
            non_duplicates=non_duplicates,
        )

        # print('yolo')


    # --------------------------------------------------------
    # Pareto fronts
    # --------------------------------------------------------

    def pareto_front_gen(self, gen: int) -> np.ndarray:
        """Pareto front of generation gen only."""
        state = self.generations[gen]

        if state.pareto_gen is None:
            nds = NonDominatedSorting()
            idx = nds.do(state.F, only_non_dominated_front=True)
            state.pareto_gen = state.F[idx]

        return state.pareto_gen

    def pareto_front_archive(self, upto_gen: Optional[int] = None) -> np.ndarray:
        """Pareto front over all generations up to gen."""
        Fs = [
            g.F
            for k, g in self.generations.items()
            if upto_gen is None or k <= upto_gen
        ]

        F_all = np.vstack(Fs)
        nds = NonDominatedSorting()
        idx = nds.do(F_all, only_non_dominated_front=True)
        return F_all[idx]

    def pareto_front_archive_X(self, upto_gen: Optional[int] = None) -> np.ndarray:
        """Pareto front over all generations up to gen."""
        F_all = np.vstack([
            g.F
            for k, g in self.generations.items()
            if upto_gen is None or k <= upto_gen
        ])

        X_all = np.concatenate([
            g.X
            for k, g in self.generations.items()
            if upto_gen is None or k <= upto_gen
        ])

        nds = NonDominatedSorting()
        idx = nds.do(F_all, only_non_dominated_front=True)
        return X_all[idx]

    # --------------------------------------------------------
    # Hypervolume
    # --------------------------------------------------------

    def _hv_reference_point(self) -> np.ndarray:
        if self.true_pareto is not None:
            return np.max(self.true_pareto, axis=0) * 1.1
        return self._global_max(upto_gen=max(self.generations)) * 1.1

    def hypervolume_gen(self, gen: int, normalized: bool = False) -> float:
        state = self.generations[gen]
        Fp = self.pareto_front_gen(gen)

        if normalized and self.true_pareto is not None:
            ref = self._hv_reference_point()
            hv_true = self._true_hypervolume()
            hv_cur = float(HV(ref_point=ref)(Fp))
            return hv_cur / hv_true

        if normalized:
            f_min = np.zeros(state.F.shape[1])
            f_max = self._global_max(upto_gen=gen)
            denom = np.maximum(f_max - f_min, 1e-12)

            F_norm = (Fp - f_min) / denom
            ref = np.ones(F_norm.shape[1])

            return float(HV(ref_point=ref)(F_norm))

        ref = self._hv_reference_point()
        return float(HV(ref_point=ref)(Fp))

    def hypervolume_archive(self, gen: int, normalized: bool = False) -> float:
        Fp = self.pareto_front_archive(upto_gen=gen)

        if normalized and self.true_pareto is not None:
            ref = self._hv_reference_point()
            hv_true = self._true_hypervolume()
            hv_cur = float(HV(ref_point=ref)(Fp))
            return hv_cur / hv_true

        if normalized:
            f_min = np.zeros(Fp.shape[1])
            f_max = self._global_max(upto_gen=max(self.generations))
            denom = np.maximum(f_max - f_min, 1e-12)

            F_norm = (Fp - f_min) / denom
            ref = np.ones(F_norm.shape[1])

            return float(HV(ref_point=ref)(F_norm))

        ref = self._hv_reference_point()
        return float(HV(ref_point=ref)(Fp))

    def _true_hypervolume(self) -> Optional[float]:
        if self.true_pareto is None:
            return None
        ref = self._hv_reference_point()
        return float(HV(ref_point=ref)(self.true_pareto))

    # def normalized_hypervolume_gen(self, gen: int) -> float:
    #     """
    #     Normalized HV with:
    #     - fixed lower bound = 0
    #     - upper bound = global max up to gen
    #     """
    #     state = self.generations[gen]
    #
    #     if state.nhv_gen is None:
    #         f_min = np.zeros(state.F.shape[1])
    #         f_max = self._global_max(upto_gen=gen)
    #         denom = np.maximum(f_max - f_min, 1e-12)
    #
    #         Fp = self.pareto_front_gen(gen)
    #         F_norm = (Fp - f_min) / denom
    #
    #         ref = np.ones(F_norm.shape[1]) * 1.0
    #         state.nhv_gen = float(HV(ref_point=ref)(F_norm))
    #
    #     return state.nhv_gen
    #
    # def normalized_hypervolume_archive(self, gen: int) -> float:
    #     """Best-so-far normalized HV (archive Pareto)."""
    #     f_min = np.zeros(self.generations[gen].F.shape[1])
    #     f_max = self._global_max(upto_gen=max(self.generations))
    #     denom = np.maximum(f_max - f_min, 1e-12)
    #
    #     Fp = self.pareto_front_archive(upto_gen=gen)
    #     F_norm = (Fp - f_min) / denom
    #
    #     ref = np.ones(F_norm.shape[1])*1.0
    #     return float(HV(ref_point=ref)(F_norm))

    # --------------------------------------------------------
    # Metric statistics
    # --------------------------------------------------------

    def scalar_metric_stats(self, gen: int) -> Dict[str, Dict[str, float]]:
        state = self.generations[gen]

        if state.metric_stats is None:
            stats = {}
            for name, vals in state.metrics.items():
                v = np.asarray(vals)
                stats[name] = {
                    "mean": float(np.mean(v)),
                    "median": float(np.median(v)),
                    "std": float(np.std(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "q25": float(np.percentile(v, 25)),
                    "q75": float(np.percentile(v, 75)),
                }
            state.metric_stats = stats

        return state.metric_stats

    # --------------------------------------------------------
    # Objective history helpers
    # --------------------------------------------------------
    @property
    def F(self) -> np.ndarray:
        Fs = [g.F for _, g in sorted(self.generations.items())]

        if not Fs:
            raise ValueError("No generations logged yet.")

        return np.vstack(Fs)

    @property
    def X(self) -> np.ndarray:
        Xs = [g.X for _, g in sorted(self.generations.items())]

        if not Xs:
            raise ValueError("No generations logged yet.")

        return np.vstack(Xs)

    def F_upto(self, gen: int) -> np.ndarray:
        Fs = [
            g.F
            for k, g in sorted(self.generations.items())
            if k <= gen
        ]

        if not Fs:
            raise ValueError(
                f"No generations found with index <= {gen}. "
                f"Available generations: {sorted(self.generations)}"
            )

        return np.vstack(Fs)

    def X_upto(self, gen: int) -> np.ndarray:
        Xs = [
            g.X
            for k, g in sorted(self.generations.items())
            if k <= gen
        ]

        if not Xs:
            raise ValueError(
                f"No generations found with index <= {gen}. "
                f"Available generations: {sorted(self.generations)}"
            )

        return np.vstack(Xs)

    @staticmethod
    def _improves_front_2d(F_pred_2d: np.ndarray, F_front: np.ndarray) -> np.ndarray:
        """
        Return boolean mask: True if predicted point is NOT dominated
        by the current Pareto front (2D minimization).
        """
        improves = np.ones(len(F_pred_2d), dtype=bool)

        for i, p in enumerate(F_pred_2d):
            # dominated if any front point is <= in both and < in at least one
            dominated = np.any(
                np.all(F_front <= p, axis=1) &
                np.any(F_front < p, axis=1)
            )
            improves[i] = not dominated

        return improves

    def pareto_size_gen(self, gen: int) -> int:
        return len(self.pareto_front_gen(gen))

    def pareto_front_global(self, upto_gen: Optional[int] = None) -> np.ndarray:
        Fs = [
            g.F
            for k, g in self.generations.items()
            if upto_gen is None or k <= upto_gen
        ]

        if not Fs:
            raise ValueError(
                "No objective data available to compute global Pareto front."
            )

        F_all = np.vstack(Fs)

        nds = NonDominatedSorting()
        idx = nds.do(F_all, only_non_dominated_front=True)

        return F_all[idx]

    @staticmethod
    def pareto_envelope_2d(F):
        order = np.argsort(F[:, 0])
        F = F[order]

        front = []
        best_y = np.inf
        for x, y in F:
            if y <= best_y:
                front.append((x, y))
                best_y = y

        return np.array(front)

    @staticmethod
    def points_on_pareto_front_2d(F_points, F_front):
        """
        Return boolean mask indicating which points in F_points are on the Pareto front.
        Tolerance is used for floating point comparison.
        """
        on_front = np.zeros(len(F_points), dtype=bool)

        for i, point in enumerate(F_points):
            # Check if this point is in the front (within floating point tolerance)
            is_on_front = np.any(
                np.allclose(F_front, point, atol=1e-9, rtol=1e-9)
            )
            on_front[i] = is_on_front

        return on_front

    # --------------------------------------------------------
    # Plotting (read-only)
    # --------------------------------------------------------

    def plot_hypervolume_trajectory(self, normalize=True, archive: bool = False):
        gens = sorted(self.generations)
        x = [self.generations[g].cum_evals for g in gens]

        if archive:
            y = [self.hypervolume_archive(g, normalize) for g in gens]
            label = "Archive Pareto"
            if normalize:
                fname = "normalized_hypervolume_archive.png"
            else:
                fname = "hypervolume_archive.png"
        else:
            y = [self.hypervolume_gen(g, normalize) for g in gens]
            label = "Generation Pareto"
            if normalize:
                fname = "normalized_hypervolume_gen.png"
            else:
                fname = "hypervolume_gen.png"

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, linewidth=2, label=label)
        plt.xlabel("Number of Evaluations")
        if normalize:
            plt.ylabel("Normalized Hypervolume")
            plt.title(f"{self.name} - Normalized Hypervolume")
        else:
            plt.ylabel("Hypervolume")
            plt.title(f"{self.name} Hypervolume")

        if self.true_pareto is not None:
            if normalize: # normalized HV converges to exactly 1.0 by definition
                y_true = 1.0
            else: # raw HV ceiling
                y_true = self._true_hypervolume()

            plt.axhline(
                y_true,
                linestyle="--",
                linewidth=1.8,
                color="black",
                alpha=0.85,
                label="True Pareto ceiling",
            )


        plt.grid(True)
        plt.legend()

        self.save_plot(
            filename=fname,
            save_fn=lambda p: plt.savefig(p, dpi=150, bbox_inches="tight", format="png"),
        )
        plt.close()

    def plot_hypervolume_2d(
            self,
            gen: int,
            idx_x: int = 0,
            idx_y: int = 1,
            reference_point: tuple = None,
            show_all_points: bool = True,
    ):
        """
        Plot 2D hypervolume with Pareto front and optional improvement vs gen-1.
        """

        # ---- Get fronts ----
        F_curr = self.pareto_front_archive(gen)
        if F_curr is None:
            return

        F_curr = F_curr[:, [idx_x, idx_y]]
        F_front = self.pareto_envelope_2d(F_curr)

        F_prev = self.pareto_front_archive(gen - 1)
        F_prev_front = None
        if F_prev is not None:
            F_prev_front = self.pareto_envelope_2d(
                F_prev[:, [idx_x, idx_y]]
            )

        # ---- Reference point ----
        if reference_point is None:
            ref_full = self._global_max(upto_gen=gen)
            reference_point = ref_full[[idx_x, idx_y]] * 1.1

        rx, ry = reference_point

        plt.figure(figsize=(7, 7))

        # ---- Hypervolume area ----
        prev_y = ry
        for x, y in F_front:
            plt.fill_between(
                [x, rx],
                y,
                prev_y,
                color="#cfe6f5",
                alpha=0.8,
                zorder=2,
            )
            prev_y = y

        # ---- Hypervolume improvement ----
        if F_prev_front is not None:
            prev_y = ry
            for x, y in F_front:
                y_old = np.interp(x, F_prev_front[:, 0], F_prev_front[:, 1],
                                  left=prev_y, right=F_prev_front[-1, 1])
                if y < y_old:
                    plt.fill_between(
                        [x, rx],
                        y,
                        y_old,
                        color="#cdeccf",
                        alpha=0.7,
                        zorder=3,
                    )
                prev_y = y

        # ---- Background points ----
        if show_all_points:
            F_all = self.F[:, [idx_x, idx_y]]
            plt.scatter(
                F_all[:, 0],
                F_all[:, 1],
                s=15,
                alpha=0.2,
                color="black",
                zorder=1,
            )

        # ---- Pareto fronts ----
        if self.true_pareto is not None:
            F_true_2d = self.true_pareto[:, [idx_x, idx_y]]
            F_true_line = self.pareto_envelope_2d(F_true_2d)

            plt.step(
                F_true_line[:, 0],
                F_true_line[:, 1],
                where="post",
                linestyle=":",
                linewidth=2.5,
                color="black",
                label="Pareto Front",
                zorder=5,
            )
        plt.step(
            F_front[:, 0],
            F_front[:, 1],
            where="post",
            linewidth=2.5,
            label="Pareto Approximation",
            zorder=4,
        )

        if F_prev_front is not None:
            plt.step(
                F_prev_front[:, 0],
                F_prev_front[:, 1],
                where="post",
                linestyle="--",
                alpha=0.7,
                label=f"Gen {gen - 1}",
                zorder=3,
            )

        # ---- Reference point ----
        plt.scatter(
            [rx],
            [ry],
            marker="*",
            s=300,
            color="red",
            label="Reference point",
            zorder=5,
        )

        # Highlight new samples
        # ---- New samples split by dominance ----
        F_new = self.generations[gen].F[:, [idx_x, idx_y]]
        is_nd = np.any(np.all(np.isclose(F_new[:, None, :], F_front[None, :, :], atol=1e-9), axis=2), axis=1)

        plt.scatter(
            F_new[~is_nd, 0],
            F_new[~is_nd, 1],
            s=50,
            facecolors="none",
            edgecolors="gray",
            alpha=0.6,
            linewidths=1.2,
            label="New",
            zorder=3,
        )

        plt.scatter(
            F_new[is_nd, 0],
            F_new[is_nd, 1],
            s=70,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=2.0,
            label="New (Pareto)",
            zorder=4,
        )

        # ---- Labels ----
        plt.xlabel(self.objectives[idx_x])
        plt.ylabel(self.objectives[idx_y])

        plt.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.9,
        )

        ax = plt.gca()
        if rx > 1e3:
            plt.xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.xaxis.set_major_formatter(EngFormatter())
            plt.xlabel(f'{self.objectives[idx_x]} (log scale)')
        if ry > 1e3:
            plt.yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(EngFormatter())
            plt.ylabel(f'{self.objectives[idx_y]} (log scale)')

        plt.title(f"{self.name} - Hypervolume (Gen {gen})")
        plt.grid(True)

        filename = f"hypervolume_2d_{idx_x}_{idx_y}_gen_{gen}.png"
        self.save_plot(
            filename=filename,
            save_fn=lambda p: plt.savefig(p, dpi=150, bbox_inches="tight", format="png"),
        )
        plt.close()

    def plot_objective_progress(
            self,
            normalize: bool = False,
    ):
        """
        Plot per-objective progress over generations.

        For each objective:
          - min (best)
          - median
          - interquartile range (25-75%)

        Parameters
        ----------
        normalize : bool
            If True, normalize objectives using:
            lower bound = 0
            upper bound = global max over all generations
        """

        if self.objectives is None:
            raise ValueError("Objectives not set. Call set_objectives(...) first.")

        gens = sorted(self.generations)

        # Collect global bounds once if needed
        if normalize:
            f_min = np.zeros(len(self.objectives))
            f_max = self._global_max()
            denom = np.maximum(f_max - f_min, 1e-12)

        for obj_idx, obj_name in enumerate(self.objectives):

            mins, maxs, stds, means, medians, q25, q75 = [], [], [], [], [], [],[]

            for g in gens:
                vals = self.generations[g].F[:, obj_idx]

                if normalize:
                    vals = (vals - f_min[obj_idx]) / denom[obj_idx]

                means.append(np.mean(vals))
                medians.append(np.median(vals))
                stds.append(np.std(vals))
                mins.append(np.min(vals))
                maxs.append(np.max(vals))
                q25.append(np.percentile(vals, 25))
                q75.append(np.percentile(vals, 75))

            means = np.array(means)
            medians = np.array(medians)
            stds = np.array(stds)
            mins = np.array(mins)
            maxs = np.array(maxs)
            q25 = np.array(q25)
            q75 = np.array(q75)

            plt.figure(figsize=(8, 5))

            # Mean
            plt.plot(gens, means, label="Mean", linewidth=2)

            # Median
            plt.plot(gens, medians, linestyle="--", label="Median", linewidth=1.5)

            # Best-in-generation (per metric)
            plt.plot(gens, mins, linestyle=":", label="Best (gen)", linewidth=1)

            # Interquartile range
            plt.fill_between(
                gens,
                q25,
                q75,
                alpha=0.25,
                label="IQR (25-75%)",
            )

            # Optional: min-max envelope
            plt.fill_between(
                gens,
                mins,
                maxs,
                alpha=0.1,
                label="Min-Max",
            )

            plt.xlabel("Generation")
            plt.ylabel(obj_name)

            ax = plt.gca()
            if max(maxs) > 1e3:
                plt.yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10))
                ax.yaxis.set_major_formatter(EngFormatter())
                plt.ylabel(f'{obj_name} (log scale)')

            plt.legend()
            plt.grid(True)

            title = f"{self.name} - {obj_name} Progress"
            if normalize:
                title += " (normalized)"
            plt.title(title)

            suffix = "_normalized" if normalize else ""

            filename = f"objective_{obj_name}_progress{suffix}.png"
            self.save_plot(
                filename=filename,
                save_fn=lambda p: plt.savefig(p, dpi=150, bbox_inches="tight", format="png"),
            )
            plt.close()

    def plot_regression_approximation(self, gen: int, x_data: np.ndarray, y_data: np.ndarray):
        pX = self.pareto_front_archive_X(gen)
        pF = self.pareto_front_archive(gen)

        plt.figure(figsize=(18, 10))
        plt.scatter(x_data, y_data, label='Reference')
        for i, x in enumerate(pX):
            pe = x.execute(x_data)

            def truncate(s, length=60):
                st = s.__str__()
                return st[:length] + '...' if len(st) > length else st

            plt.scatter(x_data, pe, label=f'{truncate(x)}: {pF[i, 0]:0.3f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=3, fontsize=8)
        plt.title(f"Symbolic Regression Approximation")

        filename = f"symbolic_regression_approx_i{gen}.png"
        self.save_plot(
            filename=filename,
            save_fn=lambda p: plt.savefig(p, dpi=150, bbox_inches="tight", format="png"),
        )
        plt.close()


    def plot_surrogate_error(self,
                                      gen: int,
                                      ):
        def quantile_error(y_true, y_pred, q=10):
            q_true = pd.qcut(y_true, q=q, labels=False, duplicates="drop")
            q_pred = pd.qcut(y_pred, q=q, labels=False, duplicates="drop")
            return np.mean(np.abs(q_true - q_pred))

        Fc = self.candidates[gen].F
        n_surrogates = len(self.candidates[gen].surrogates)

        Fc_pred = self.candidates[gen].Fpred

        fig = plt.figure(figsize=(7* n_surrogates, 7))
        width_ratios = [1] * n_surrogates + [0.05]
        gs = GridSpec(1, n_surrogates + 1, figure=fig, width_ratios=width_ratios)

        axes = np.empty((1, n_surrogates), dtype=object)

        for i in range(n_surrogates):
            axes[0, i] = fig.add_subplot(gs[0, i])

        cax = fig.add_subplot(gs[:, -1])  # colorbar axis
        for a, ax in enumerate(axes.flat):
            rmse = np.sqrt(((Fc[:, a] - Fc_pred[:, a]) ** 2).mean())
            rho, _ = stats.spearmanr(Fc_pred[:, a], Fc[:, a])
            tau, _ = stats.kendalltau(Fc_pred[:, a], Fc[:, a])
            qe = quantile_error(Fc[:, a], Fc_pred[:, a])

            clip = np.percentile(Fc[:, a], 99.5)
            mask = Fc[:, a] < clip

            hb = ax.hexbin(
                Fc[:, a][mask],
                Fc_pred[:, a][mask],
                gridsize=60,
                # bins="log",
                mincnt=1,
                cmap="viridis"
            )

            # Identity line (log-space!)
            lims = [
                min(Fc[:, a][mask].min(), Fc_pred[:, a][mask].min()),
                max(Fc[:, a][mask].max(), Fc_pred[:, a][mask].max())
            ]
            ax.plot(lims, lims, "r--", linewidth=1, label="y = x")

            ax.set_title(
                f"{self.candidates[gen].surrogates[a]}\n"
                f"RMSE = {rmse:.3f}, ρ = {rho:.3f}, τ = {tau:.3f}, qe = {qe:.3f}"
            )
            if self.objectives[a] == 'log1p_mse':
                ax.set_xlabel("True log(1+MSE)")
                ax.set_ylabel("Predicted log(1+MSE)")
            else:
                ax.set_xlabel(f"True {self.objectives[a]}")
                ax.set_ylabel(f"Predicted {self.objectives[a]}")

        cb = fig.colorbar(hb, cax=cax)
        cb.set_label("count")
        fig.suptitle(f"Surrogate Performance", fontsize=18)

        plt.tight_layout()

        filename = f"hexbin_surrogate_prediction_performance_gen_{gen}.png"

        self.save_plot(
            filename=filename,
            save_fn=lambda p: plt.savefig(p, dpi=150, bbox_inches="tight", format="png",),
        )

        plt.close()

    def plot_hypervolume_2d_surrogate(self,
                             gen: int,
                             ):
        obj_pairs = list(itertools.combinations(range(len(self.objectives)), 2))
        n_plots = len(obj_pairs)
        fig, axes = plt.subplots(
            1, n_plots,
            figsize=(7 * n_plots, 7),
            squeeze=False,
        )

        for ax, (idx_x, idx_y) in zip(axes.flat, obj_pairs):
            F = self.F_upto(gen)[:, [idx_x, idx_y]]
            Fa = self.F_upto(gen - 1)[:, [idx_x, idx_y]] if gen > 1 else None
            Fc = self.candidates[gen].F
            n_surrogates = len(self.candidates[gen].surrogates)

            # Combine evaluated data with candidates to get the complete Pareto front
            # This ensures the blue area includes the latest non-dominating solutions
            F_combined = np.vstack([F, Fc[:, [idx_x, idx_y]]])
            F_front = self.pareto_envelope_2d(F_combined)
            F_prev_front = self.pareto_envelope_2d(Fa) if gen > 1 else None

            Fc_pred = self.candidates[gen].Fpred
            Fc_true = Fc[:, :n_surrogates]

            # ---- Reference point ----
            ref_full = self._global_max(upto_gen=gen)
            reference_point = ref_full[[idx_x, idx_y]] * 1.1

            rx, ry = reference_point

            # ---- Hypervolume area ----
            prev_y = ry
            for x, y in F_front:
                ax.fill_between([x, rx], y, prev_y, color="#cfe6f5", alpha=0.8, zorder=2)
                prev_y = y

            # ---- Hypervolume improvement ----
            if F_prev_front is not None:
                prev_y = ry
                for x, y in F_front:
                    y_old = np.interp(x, F_prev_front[:, 0], F_prev_front[:, 1], left=prev_y, right=F_prev_front[-1, 1])
                    if y < y_old:
                        ax.fill_between(
                            [x, rx], y, y_old,
                            color="#cdeccf", alpha=0.7, zorder=3
                        )
            # ---- Background points ----
            if Fa is not None:
                ax.scatter(Fa[:, 0], Fa[:, 1], s=15, alpha=0.2, color="black", zorder=1)

            # ---- Pareto fronts ----
            if self.true_pareto is not None:
                F_true_2d = self.true_pareto[:, [idx_x, idx_y]]
                F_true_line = self.pareto_envelope_2d(F_true_2d)

                ax.step(
                    F_true_line[:, 0],
                    F_true_line[:, 1],
                    where="post",
                    linestyle=":",
                    linewidth=2.5,
                    color="black",
                    label="Pareto Front",
                    zorder=5,
                )

            ax.step(F_front[:, 0], F_front[:, 1], where="post", linewidth=2.5, label="Pareto Approximation", zorder=4)
            if F_prev_front is not None:
                ax.step(F_prev_front[:, 0], F_prev_front[:, 1], where="post", linestyle="--", alpha=0.7,
                         label=f"Gen {gen - 1}", zorder=3)


            # ---- Reference point ----
            ax.scatter([rx], [ry], marker="*", s=300, color="red", label="Reference point", zorder=5)

            # ---- New samples split by dominance ----
            F_new = self.generations[gen].F[:, [idx_x, idx_y]]
            is_nd = np.any(np.all(np.isclose(F_new[:, None, :], F_front[None, :, :], atol=1e-9), axis=2), axis=1)
            ax.scatter(F_new[~is_nd, 0], F_new[~is_nd, 1], s=50, facecolors="none", edgecolors="gray", alpha=0.6,
                        linewidths=1.2, label="New", zorder=3)

            # ---- Candidate samples split by dominance ----
            # Candidate-level predicted and true values
            F_pred_full = Fc.copy()
            F_pred_full[:, :n_surrogates] = Fc_pred

            F_pred_2d = F_pred_full[:, [idx_x, idx_y]]
            F_true_2d = Fc[:, [idx_x, idx_y]]

            # Check if ACTUAL results improve the front (not predictions)
            improves = self._improves_front_2d(F_true_2d, F_front)

            # Check which dimensions are surrogate predictions
            x_is_pred = idx_x < n_surrogates
            y_is_pred = idx_y < n_surrogates
            has_surrogate = x_is_pred or y_is_pred

            if has_surrogate:
                # Get indices of candidates with surrogate predictions
                non_dup = self.candidates[gen].non_duplicates
                if non_dup is None:
                    non_dup = np.zeros(len(improves), dtype=bool)

                improves_samos = improves & non_dup
                improves_random = improves & (~non_dup)

                # ---- Plot predicted points (green circles) ----
                ax.scatter(
                    F_pred_2d[:, 0],
                    F_pred_2d[:, 1],
                    s=50,
                    facecolors="none",
                    edgecolors="green",
                    linewidths=1.5,
                    label="Predicted",
                    zorder=4,
                    alpha=0.7,
                )

                # ---- Draw connecting lines from predicted to actual ----
                for i, (p, t) in enumerate(zip(F_pred_2d, F_true_2d)):
                    x_pred, y_pred = p
                    x_true, y_true = t

                    # Draw line based on which dimensions are predicted
                    if x_is_pred and not y_is_pred:
                        # Only x is predicted: horizontal line
                        ax.plot([x_pred, x_true], [y_pred, y_pred],
                                color="green", alpha=0.5, linewidth=1.0, zorder=3)
                    elif y_is_pred and not x_is_pred:
                        # Only y is predicted: vertical line
                        ax.plot([x_pred, x_pred], [y_pred, y_true],
                                color="green", alpha=0.5, linewidth=1.0, zorder=3)
                    else:
                        # Both predicted: diagonal line
                        ax.plot([x_pred, x_true], [y_pred, y_true],
                                color="green", alpha=0.5, linewidth=1.0, zorder=3)

                # ---- Plot actual points (colored by location) ----
                # Determine which actual results are ON the evaluated Pareto front
                on_approx_front = self.points_on_pareto_front_2d(F_true_2d, F_front)

                # Determine which are on the TRUE Pareto front (if available)
                on_true_front = np.zeros(len(F_true_2d), dtype=bool)
                if self.true_pareto is not None:
                    F_true_pareto_2d = self.true_pareto[:, [idx_x, idx_y]]
                    F_true_front_line = self.pareto_envelope_2d(F_true_pareto_2d)
                    on_true_front = self.points_on_pareto_front_2d(F_true_2d, F_true_front_line)

                # ---- Color coding ----
                # Grey circles: actual results that did NOT improve the approximation
                not_improved = ~improves
                if np.any(not_improved):
                    ax.scatter(
                        F_true_2d[not_improved, 0],
                        F_true_2d[not_improved, 1],
                        s=50,
                        facecolors="none",
                        edgecolors="gray",
                        linewidths=1.5,
                        alpha=0.5,
                        label="Did not improve",
                        zorder=3,
                    )

                # Orange circles: improved approximation but NOT on true Pareto
                improved_not_true = improves & ~on_true_front
                if np.any(improved_not_true):
                    ax.scatter(
                        F_true_2d[improved_not_true, 0],
                        F_true_2d[improved_not_true, 1],
                        s=50,
                        facecolors="none",
                        edgecolors="orange",
                        linewidths=2.0,
                        label="Improved Approximation",
                        zorder=4,
                    )

                # Red circles: ON the true Pareto front
                if np.any(on_true_front):
                    ax.scatter(
                        F_true_2d[on_true_front, 0],
                        F_true_2d[on_true_front, 1],
                        s=50,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=2.0,
                        label="On True Pareto",
                        zorder=5,
                    )

            # ---- Labels ----
            ax.set_xlabel(self.objectives[idx_x])
            ax.set_ylabel(self.objectives[idx_y])

            # ---- Set axis limits ----
            # Compute limits based on reference point and front
            x_min, x_max = 0, rx * 1.1
            y_min, y_max = 0, ry * 1.1

            if len(F_front) > 0:
                x_min = min(x_min, F_front[:, 0].min() * 0.9)
                x_max = max(x_max, F_front[:, 0].max() * 1.1)
                y_min = min(y_min, F_front[:, 1].min() * 0.9)
                y_max = max(y_max, F_front[:, 1].max() * 1.1)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            if 'epoch' in self.objectives[idx_x]:
                xlim = self.true_pareto[:, 0].min() * 2 if self.true_pareto is not None else 50
                ax.set_xlim(0, xlim)

            # ---- Axis scaling & formatting ----
            if rx > 1e3:
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(base=10))
                ax.xaxis.set_major_formatter(EngFormatter())
                ax.set_xlabel(f"{self.objectives[idx_x]} (log scale)")

            if ry > 1e3:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10))
                ax.yaxis.set_major_formatter(EngFormatter())
                ax.set_ylabel(f"{self.objectives[idx_y]} (log scale)")

            # ---- Cosmetics ----
            ax.set_title(f"{self.name} - Hypervolume (Gen {gen})")
            ax.grid(True)

        # ---- Collect and deduplicate legend entries across subplots ----
        handles, labels = [], []

        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)

        # ---- Place legend below the figure, centered ----
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),  # wrap nicely if many entries
            frameon=True,
            framealpha=0.95,
            bbox_to_anchor=(0.5, -0.05),
        )

        # Make room for the legend
        fig.subplots_adjust(bottom=0.1)

        filename = f"hypervolume_2d_surrogate_gen_{gen}.png"

        self.save_plot(
            filename=filename,
            save_fn=lambda p: plt.savefig(p, dpi=150, bbox_inches="tight", format="png",),
        )

        plt.close()

    def plot_infill_source_area(self):
        """
        Area chart of infill origin over evaluations.

        Shows how many infill points per generation were found by SAMOS
        vs randomly added, stacked cumulatively over evaluation count.
        """

        gens = sorted(self.candidates)
        if not gens:
            return

        x = []
        samos_counts = []
        random_counts = []

        for g in gens:
            c = self.candidates[g]

            if c.non_duplicates is None:
                continue

            # cumulative evaluations on x-axis
            x.append(c.cum_evals)

            n_total = int(c.evals)
            n_samos = int(np.sum(c.non_duplicates))
            n_random = n_total - n_samos

            samos_counts.append(n_samos)
            random_counts.append(n_random)

        samos_counts = np.asarray(samos_counts)
        random_counts = np.asarray(random_counts)

        plt.figure(figsize=(8, 5))

        plt.stackplot(
            x,
            samos_counts,
            random_counts,
            labels=["SAMOS infill", "Random infill"],
            alpha=0.85,
        )

        plt.xlabel("Number of Evaluations")
        plt.ylabel("Number of Infill Points")
        plt.title(f"{self.name} - Infill Source Composition")

        plt.legend(loc="upper right")
        plt.grid(True)

        self.save_plot(
            filename="infill_source_area.png",
            save_fn=lambda p: plt.savefig(p, dpi=150, bbox_inches="tight", format="png"),
        )
        plt.close()

    def plot_surrogate_rank_correlation_trajectory(self):
        """
        Plot Spearman rho and Kendall tau over cumulative evaluations.

        One line per surrogate/objective.
        """

        gens = sorted(self.candidates)
        if not gens:
            return

        # Determine number of surrogates from first generation
        n_surrogates = len(self.candidates[0].surrogates)

        # Storage
        x = []
        rhos = [[] for _ in range(n_surrogates)]
        taus = [[] for _ in range(n_surrogates)]

        for gen in gens:
            Fc = self.candidates[gen].F
            Fc_pred = self.candidates[gen].Fpred
            n_evals = self.candidates[gen].cum_evals
            x.append(n_evals)

            for i in range(n_surrogates):
                rho, _ = stats.spearmanr(Fc_pred[:, i], Fc[:, i])
                tau, _ = stats.kendalltau(Fc_pred[:, i], Fc[:, i])
                # print('=================')
                # print(f'rho: {rho}, tau: {tau}')
                # print('=================')
                rhos[i].append(rho)
                taus[i].append(tau)

        # ---- Plotting ----
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

        ax_rho, ax_tau = axes

        for i in range(n_surrogates):
            obj = self.objectives[i]
            surrogate = str(self.candidates[0].surrogates[i])

            label = f"{obj} ({surrogate})"

            ax_rho.plot(x, rhos[i], linewidth=2, label=label)
            ax_tau.plot(x, taus[i], linewidth=2, label=label)

        # ---- Cosmetics ----
        ax_rho.set_title("Spearman ρ")
        ax_tau.set_title("Kendall τ")

        for ax in axes:
            ax.set_xlabel("Number of Evaluations")
            ax.set_ylabel("Rank Correlation")
            # ax.set_ylim(-1.0, 1.0)
            ax.grid(True)

        # Shared legend below
        handles, labels = ax_rho.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=1,
            frameon=True,
            framealpha=0.95,
            bbox_to_anchor=(0.5, -0.05),
        )

        fig.suptitle(f"{self.name} - Surrogate Rank Correlation", fontsize=14)
        fig.subplots_adjust(bottom=0.1)

        self.save_plot(
            filename="surrogate_rank_correlation_trajectory.png",
            save_fn=lambda p: plt.savefig(p, dpi=150, bbox_inches="tight", format="png"),
        )
        plt.close()

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def save(self):
        run = {
            "meta": self.meta,
            "objectives": self.objectives,
            "generations": self.generations,
            "candidates": self.candidates,
        }

        with open(os.path.join(self.pkl_dir, "run.pkl"), "wb") as f:
            pickle.dump(run, f)

        n_obj = len(self.objectives)
        # x, y = RunMetrics.hv_trajectory(run, normalized=True, archive=True)
        # RunPlotter.plot_hv_trajectory(
        #     x, [y],
        #     labels=["run"],
        #     title=f"{self.name} - HV",
        #     ylabel="Normalized HV",
        # )



        if len(self.generations) > 1:
            self.plot_hypervolume_trajectory(archive=True, normalize=False)
            self.plot_hypervolume_trajectory(archive=True, normalize=True)
            if len(self.candidates) > 0:
                self.plot_infill_source_area()

            for i, j in combinations(range(n_obj), 2):
                self.plot_hypervolume_2d(
                    gen=max(self.generations),
                    idx_x=i,
                    idx_y=j
                )
        self.plot_objective_progress(normalize=False)

    def save_results(self, results):
        with open(os.path.join(self.pkl_dir, "optimization_result.pkl"), "wb") as f:
            pickle.dump(results, f)

    def save_archive(self, archive):
        with open(os.path.join(self.pkl_dir, "start_archive.pkl"), "wb") as f:
            pickle.dump(archive, f)

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def _global_max(self, upto_gen: Optional[int] = None) -> np.ndarray:
        Fs = [
            g.F
            for k, g in self.generations.items()
            if upto_gen is None or k <= upto_gen
        ]
        return np.max(np.vstack(Fs), axis=0)

    def _make_dirs(self):
        bases = [Path(self.base_dir)]

        if self.absolute_base_dir is not None:
            bases.append(Path(self.absolute_base_dir))

        for base in bases:
            (base / "plots").mkdir(parents=True, exist_ok=True)
            (base / "pkl").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve(root, name, version):
        root = os.path.abspath(os.path.normpath(root))
        return os.path.join(root, name, str(version))

    def save_plot(self, filename: str, save_fn):
        """
        save_fn: callable that takes a Path and writes the plot
                 e.g. lambda p: plt.savefig(p, ...)
        """

        targets = [Path(self.base_dir) / "plots"]

        # Always save to persistent storage

        # Also save to scratch if configured
        if self.absolute_base_dir is not None:
            targets.append(Path(self.absolute_base_dir) / "plots")

        for base in targets:
            base.mkdir(parents=True, exist_ok=True)
            path = base / filename

            tmp = path.with_suffix(path.suffix + ".tmp")
            save_fn(tmp)
            tmp.replace(path)

    @staticmethod
    def _default_version() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _environment_info():
        import sys
        import numpy
        import pymoo

        return {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "pymoo": pymoo.__version__,
        }

    def _serialize(self, obj):
        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._serialize(o) for o in obj]
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if hasattr(obj, "to_config"):
            return self._serialize(obj.to_config())
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tolist(),
            }
        raise TypeError(f"Object of type {type(obj).__name__} is not serializable")


def operator_config(op):
    # ---- 1. None ----
    if op is None:
        return None

    # ---- 2. Containers  ----
    if isinstance(op, (list, tuple)):
        return [operator_config(o) for o in op]

    # ---- 3. Explicit config always wins ----
    if hasattr(op, "to_config"):
        return op.to_config()

    # ---- 4. Stateless callables / functions ----
    if callable(op) and not hasattr(op, "__dict__"):
        return {
            "class": op.__name__,
            "module": op.__module__,
            "params": {},
        }

    # ---- 5. Stateful operators ----
    return {
        "class": op.__class__.__name__,
        "module": op.__class__.__module__,
        "params": extract_constructor_params(op),
    }

def extract_constructor_params(op):
    """
    Extract JSON-safe constructor parameters from a stateful object.
    """

    # Explicit config takes precedence
    if hasattr(op, "to_config"):
        return op.to_config()

    # No state → no params
    if not hasattr(op, "__dict__"):
        return {}

    params = {}
    for k, v in vars(op).items():
        if isinstance(v, (int, float, str, bool, type(None))):
            params[k] = v

    return params
