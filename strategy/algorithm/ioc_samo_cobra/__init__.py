"""
IOC-SAMO-COBRA bundled source.

The COBRA files use flat module imports (e.g. ``from SACOBRA import ...``),
so this package directory must be on sys.path at import time.  Adding it here
keeps the path manipulation self-contained within the project.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from cheap_SAMO_COBRA_Init import cheap_SAMO_COBRA_Init      # noqa: E402
from cheap_SAMO_COBRA_PhaseII import cheap_SAMO_COBRA_PhaseII  # noqa: E402

__all__ = ["cheap_SAMO_COBRA_Init", "cheap_SAMO_COBRA_PhaseII"]
