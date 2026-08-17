"""IOC-SAMO-COBRA -- vendored third-party source. See VENDORED.md.

Upstream (https://github.com/RoydeZomer/IOC-SAMO-COBRA) is a flat script repo:
no setup.py, no pyproject.toml, no package directory, and modules that import
each other by bare name (``from SACOBRA import plog``) on the assumption that
the run happens from inside that directory. There is nothing to pip-install,
which is why the files are copied here instead of being declared a dependency.

The bare imports have been rewritten as relative imports, so this is an
ordinary subpackage: no sys.path manipulation at import time, and none of
upstream's generically-named modules (``lhs``, ``halton``, ``hypervolume``,
``SACOBRA``, ...) are exposed as top-level names where they could shadow, or be
shadowed by, anything else.

Both public entry points keep their upstream signatures::

    cobra = cheap_SAMO_COBRA_Init(problem, feval=..., batch=..., nCores=1,
                                  initDesPoints=..., cobraSeed=..., initDesign='LHS')
    cobra = cheap_SAMO_COBRA_PhaseII(cobra)

``problem`` is duck-typed by upstream, not a base class to subclass: it must
expose ``lower``, ``upper``, ``nObj``, ``nConstraints``, ``cheapConstr``,
``cheapObj``, ``ref``, ``evaluate(x) -> [F, G]`` and
``cheap_evaluate(x) -> [F, G]``.
"""

from .cheap_SAMO_COBRA_Init import cheap_SAMO_COBRA_Init
from .cheap_SAMO_COBRA_PhaseII import cheap_SAMO_COBRA_PhaseII

__all__ = ['cheap_SAMO_COBRA_Init', 'cheap_SAMO_COBRA_PhaseII']
