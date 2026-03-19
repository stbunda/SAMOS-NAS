from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
import copy

import numpy as np


@dataclass(frozen=True)
class ProgramConfigStack:
    """
    Serializable configuration for a CGP-based program.

    This class acts as the single source of truth for:
    - Runtime instantiation (SamplingGP / CartesianGeneticStack)
    - Metadata logging (JSON)
    - Exact experiment reproduction
    """

    # ---- Core ----
    program_type: Type
    random_state: np.random.RandomState

    # ---- CGP structure ----
    row_stack: List[int]
    col_stack: List[int]
    weight_stack: Dict[str, List[Any]]

    # ---- Functions (stored by NAME, not instance) ----
    down_sample_function: List[List[str]]
    output_function: List[str]

    # ---- CGP parameters ----
    n_inputs: int
    level_back: int

    # ---- Registries (not serialized) ----
    function_map: Dict[str, Any] = field(repr=False, compare=False)
    output_function_map: Dict[str, Any] = field(repr=False, compare=False)

    # Optional
    pad_to: Optional[int] = None
    constant_range: Optional[Any] = None

    # ------------------------------------------------------------------
    # Runtime usage
    # ------------------------------------------------------------------

    def to_blueprint(self) -> Dict[str, Any]:
        """
        Returns a runtime blueprint dictionary consumable by SamplingGP.
        Functions are restored as Function instances.
        """
        return {
            "type": self.program_type,
            "random_state": self.random_state,
            "row_stack": copy.deepcopy(self.row_stack),
            "col_stack": copy.deepcopy(self.col_stack),
            "weight_stack": copy.deepcopy(self.weight_stack),
            "down_sample_function": [
                [self.output_function_map[name] for name in stage]
                for stage in self.down_sample_function
            ],
            "output_function": [
                self.output_function_map[name] for name in self.output_function
            ],
            "function_map": self.function_map,
            "n_inputs": self.n_inputs,
            "constant_range": self.constant_range,
            "level_back": self.level_back,
            "pad_to": self.pad_to,
        }

    # ------------------------------------------------------------------
    # Metadata (JSON-safe)
    # ------------------------------------------------------------------

    def to_metadata(self) -> Dict[str, Any]:
        """
        JSON-serializable representation for logging.
        """
        return {
            "program_type": self.program_type.__name__,
            "row_stack": self.row_stack,
            "col_stack": self.col_stack,
            "weight_stack": self.weight_stack,
            "down_sample_function": self.down_sample_function,
            "output_function": self.output_function,
            "n_inputs": self.n_inputs,
            "constant_range": self.constant_range,
            "level_back": self.level_back,
            "pad_to": self.pad_to,
        }

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    @classmethod
    def from_metadata(
        cls,
        metadata: Dict[str, Any],
        *,
        program_registry: Dict[str, Type],
        function_map: Dict[str, Any],
        output_function_map: Dict[str, Any],
        random_state: np.random.RandomState,
    ) -> "ProgramConfigStack":
        """
        Reconstructs a ProgramConfig from logged metadata.
        """
        program_type = program_registry[metadata["program_type"]]

        return cls(
            program_type=program_type,
            random_state=random_state,
            row_stack=metadata["row_stack"],
            col_stack=metadata["col_stack"],
            weight_stack=metadata["weight_stack"],
            down_sample_function=metadata["down_sample_function"],
            output_function=metadata["output_function"],
            n_inputs=metadata["n_inputs"],
            constant_range=metadata["constant_range"],
            level_back=metadata["level_back"],
            pad_to=metadata["pad_to"],
            function_map=function_map,
            output_function_map=output_function_map,
        )

@dataclass(frozen=True)
class ProgramConfig:
    """
    Configuration for GP and flat CGP programs.
    JSON-serializable and replayable.
    """

    program_type: Type
    function_map: Dict[str, Any]

    n_inputs: int
    constant_range: Optional[Any]
    random_state: Any

    # ---- CGP-only ----
    rows: Optional[int] = None
    columns: Optional[int] = None
    level_back: Optional[int] = None

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def to_blueprint(self) -> Dict[str, Any]:
        blueprint = {
            "type": self.program_type,
            "function_map": self.function_map,
            "n_inputs": self.n_inputs,
            "constant_range": self.constant_range,
            "random_state": self.random_state,
        }

        if self.rows is not None:
            blueprint["rows"] = self.rows
        if self.columns is not None:
            blueprint["columns"] = self.columns
        if self.level_back is not None:
            blueprint["level_back"] = self.level_back

        return blueprint

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "program_type": self.program_type.__name__,
            "n_inputs": self.n_inputs,
            "constant_range": self.constant_range,
            "rows": self.rows,
            "columns": self.columns,
            "level_back": self.level_back,
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Dict[str, Any],
        *,
        program_registry: Dict[str, Type],
        function_map: Dict[str, Any],
        random_state: Any,
    ) -> "ProgramConfig":

        return cls(
            program_type=program_registry[metadata["program_type"]],
            function_map=function_map,
            n_inputs=metadata["n_inputs"],
            constant_range=metadata["constant_range"],
            random_state=random_state,
            rows=metadata.get("rows"),
            columns=metadata.get("columns"),
            level_back=metadata.get("level_back"),
        )

@dataclass(frozen=True)
class BenchConfig:
    """
    Configuration for GP and flat CGP programs.
    JSON-serializable and replayable.
    """

    program_type: Type
    op_list: List
    random_state: np.random.RandomState

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def to_blueprint(self) -> Dict[str, Any]:
        blueprint = {
            "type": self.program_type,
            "op_list": self.op_list,
            "random_state": self.random_state,
        }

        return blueprint

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "program_type": self.program_type.__name__,
            "op_list": self.op_list,
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Dict[str, Any],
        *,
        program_registry: Dict[str, Type],
        random_state: Any,
    ) -> "BenchConfig":

        return cls(
            program_type=program_registry[metadata["program_type"]],
            op_list=metadata["op_list"],
            random_state=random_state,
        )


@dataclass(frozen=True)
class DartsConfig:
    """
    Configuration for a DARTS-style genome search.
    Drop-in replacement for BenchConfig when using DartsGenome.

    JSON-serializable and replayable via to_blueprint / to_metadata /
    from_metadata -- same contract as BenchConfig.
    """

    program_type: Type                    # DartsGenome (or subclass)
    op_list: List                         # ordered list of DARTS primitive names
    random_state: np.random.RandomState

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "type": self.program_type,
            "op_list": self.op_list,
            "random_state": self.random_state,
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "program_type": self.program_type.__name__,
            "op_list": self.op_list,
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Dict[str, Any],
        *,
        program_registry: Dict[str, Type],
        random_state: Any,
    ) -> "DartsConfig":
        return cls(
            program_type=program_registry[metadata["program_type"]],
            op_list=metadata["op_list"],
            random_state=random_state,
        )
