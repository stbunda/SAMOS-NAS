# logger.py
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import matplotlib
import numpy as np

matplotlib.use("Agg")


# ============================================================
# Per-generation canonical state
# ============================================================

@dataclass
class GenerationStats:
    # Raw, logged data
    X: "np.ndarray | None"  # Population
    F: np.ndarray            # (N, M) objectives  - only the optimisation objectives
    time: np.ndarray
    evals: int               # cumulative eval count
    prev: "GenerationStats | None" = None

    # Supplementary per-candidate data that is NOT part of the optimisation
    # objectives (e.g. accuracy at an intermediate epoch snapshot).
    # Keys are user-defined strings; values are (N,) or (N, K) np.ndarrays.
    extra: Optional[Dict[str, np.ndarray]] = None

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

        self.pkl_dir = os.path.join(self.base_dir, "pkl")

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
            extras: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        prev = self.generations.get(gen - 1) if gen > 0 else None
        self.generations[gen] = GenerationStats(
            X=X,
            F=np.asarray(F, dtype=float),
            time=time,
            evals=int(evals),
            prev=prev,
            extra=extras,
        )

        if gen % self.autosave_every == 0:
            self.save()

    def log_candidate(self, *,
                      gen: int,
                      X: np.ndarray = None,
                      F: np.ndarray,
                      evals: int,
                      surrogates=None,
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

    # --------------------------------------------------------
    # Plotting (read-only)
    # --------------------------------------------------------
    def plot_hypervolume_2d_surrogate(self,
                                      gen: int,
                                      ):
        pass

    def plot_surrogate_error(self,
                             gen: int,
                             ):
        pass

    def plot_surrogate_rank_correlation_trajectory(self):
        pass

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

    def save_results(self, results):
        pass

    def save_archive(self, archive):
        pass

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def _make_dirs(self):
        bases = [Path(self.base_dir)]

        if self.absolute_base_dir is not None:
            bases.append(Path(self.absolute_base_dir))

        for base in bases:
            (base / "pkl").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve(root, name, version):
        root = os.path.abspath(os.path.normpath(root))
        return os.path.join(root, name, str(version))

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
