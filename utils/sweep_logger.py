import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


# ============================================================
# Canonicalization + hashing
# ============================================================

def _canonicalize(obj: Any) -> Any:
    """
    Recursively canonicalize an object so that equivalent configs
    always serialize identically.
    """
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    return obj


def compute_run_id(identity_config: Dict[str, Any], length: int = 14) -> str:
    """
    Compute a stable hash for an experiment identity configuration.
    """
    canonical = _canonicalize(identity_config)
    blob = json.dumps(
        canonical,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


# ============================================================
# Run artifact paths + validation
# ============================================================

def _run_artifact_paths(run_dir: Path) -> Tuple[Path, Path]:
    """
    Canonical locations for run artifacts.
    """
    config_path = run_dir / "config.json"
    run_pkl_path = (
        run_dir
        / "CGP_SAMOS_NASBENCH201"
        / "latest"
        / "pkl"
        / "run.pkl"
    )
    return config_path, run_pkl_path


def validate_run_completion(run_dir: Path) -> Tuple[bool, Optional[str]]:
    """
    A run is considered complete iff:
      algorithm.ga.generations ==
      len(run.pkl["generations"])
    """
    try:
        config_path, run_pkl_path = _run_artifact_paths(run_dir)

        if not config_path.exists():
            return False, "missing_config.json"

        if not run_pkl_path.exists():
            return False, "missing_run.pkl"

        with open(config_path) as f:
            cfg = json.load(f)

        with open(run_pkl_path, "rb") as f:
            run = pickle.load(f)

        expected = cfg["algorithm"]["ga"]["generations"]
        observed = len(run.get("generations", {}))

        if expected != observed:
            return (
                False,
                f"generation_mismatch(expected={expected}, observed={observed})",
            )

        return True, None

    except Exception as e:
        return False, f"exception({type(e).__name__}): {e}"


# ============================================================
# Run event logging
# ============================================================

def append_event(run_dir: Path, event: str, **meta):
    """
    Append an audit event to run-local events.jsonl.
    """
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event,
        **meta,
    }
    with open(run_dir / "events.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================
# Run registry
# ============================================================

class RunRegistry:
    """
    Canonical store of *all* experiment runs.
    """

    def __init__(self, root: str = "results", run_name='runs'):
        self.root = Path(root).resolve()
        self.run_name = run_name
        self.runs_dir = self.root / run_name
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def run_exists(self, run_id: str) -> bool:
        return self.run_dir(run_id).exists()

    def register_run(
        self,
        *,
        run_id: str,
        identity_config: Dict[str, Any],
    ) -> Path:
        """
        Create a canonical run directory if it does not exist.
        """
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(
                    _canonicalize(identity_config),
                    f,
                    indent=2,
                )
            self._append_global_index(run_id)

        return run_dir

    def _append_global_index(self, run_id: str):
        index_path = self.runs_dir / "index.json"

        entry = {
            "run_id": run_id,
            "created": datetime.now().isoformat(),
            "config_path": f"{self.run_name}/{run_id}/config.json",
        }

        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
        else:
            index = []

        if any(e["run_id"] == run_id for e in index):
            return

        index.append(entry)

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)


# ============================================================
# Sweep registry (indexes only)
# ============================================================

# ============================================================
# Orchestration helper
# ============================================================

class ExperimentOrchestrator:
    """
    High-level helper that:
      - computes run identity
      - avoids duplicate completed runs
      - invalidates partial / failed runs
      - registers sweeps
    """

    def __init__(self, root: str = "results", run_name='runs'):
        self.runs = RunRegistry(root, run_name=run_name)

    def prepare_run(
        self,
        *,
        identity_config: Dict[str, Any]
    ) -> Tuple[str, Path, bool]:
        """
        Returns:
            run_id
            run_dir
            is_new (True if run must be executed)
        """
        run_id = compute_run_id(identity_config)
        run_dir = self.runs.register_run(
            run_id=run_id,
            identity_config=identity_config,
        )

        events_path = run_dir / "events.jsonl"

        # First registration
        if not events_path.exists():
            append_event(run_dir, "registered")
            return run_id, run_dir, True

        # Existing run → validate completion
        valid, reason = validate_run_completion(run_dir)

        if not valid:
            append_event(run_dir, "invalidated", reason=reason)
            append_event(run_dir, "scheduled_rerun")
            return run_id, run_dir, True

        append_event(run_dir, "skipped", reason="already_completed")
        return run_id, run_dir, False


# ============================================================
# Reconciliation CLI
# ============================================================

def reconcile_existing_runs(root: str = "results", run_name='runs') -> None:
    """
    Backfill events.jsonl for legacy runs by inspecting artifacts.
    Safe and idempotent.
    """
    runs_root = Path(root).resolve() / run_name

    if not runs_root.exists():
        raise RuntimeError(f"No runs directory found at {runs_root}")

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        events_path = run_dir / "events.jsonl"

        # Skip already-logged runs
        if events_path.exists():
            print(f'Found existing events for {run_dir}')
            continue

        append_event(run_dir, "reconciled")

        config_path, run_pkl_path = _run_artifact_paths(run_dir)

        if not config_path.exists():
            append_event(
                run_dir,
                "invalidated",
                reason="missing_config.json",
            )
            print(f'Missing config for {run_dir}')
            continue

        if not run_pkl_path.exists():
            append_event(
                run_dir,
                "invalidated",
                reason="missing_run.pkl",
            )
            print(f'Missing run.pkl for {run_dir}')
            continue

        valid, reason = validate_run_completion(run_dir)

        if valid:
            append_event(run_dir, "completed")
            print(f'Completed run for {run_dir}')
        else:
            append_event(run_dir, "invalidated", reason=reason)
            print(f'Run {run_dir} failed due to {reason}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep logger utilities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reconcile_parser = subparsers.add_parser(
        "reconcile",
        help="Backfill events.jsonl for existing runs",
    )
    reconcile_parser.add_argument(
        "--root",
        default="results",
        help="Results root directory (default: results)",
    )

    args = parser.parse_args()

    if args.command == "reconcile":
        reconcile_existing_runs(root=args.root)