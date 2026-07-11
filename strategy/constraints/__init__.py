"""strategy.constraints -- reusable constraint-handling components (round 2).

Four constraint handlers implemented as small, pluggable pieces that wire
into the existing method shapes (RandomGA / SAMOS2 inner GA / constrained
outer problem) WITHOUT editing SAMOS2, the runner, or the problem classes:

  H1  RejectionSampling          -- reject/regenerate infeasible offspring.
  H3  AdaptivePenaltyProblem     -- multiplicative penalty toward a target
                                    feasibility ratio.
  H5  EpsilonRelaxation          -- pymoo-style epsilon-relaxed feasibility.
  H6  DominanceStochasticRanking -- Runarsson & Yao SR, dominance-lifted.

See strategy/constraints/handlers.py for the public API and the per-shape
wiring recipes.
"""

from strategy.constraints.handlers import (
    AdaptivePenaltyProblem,
    DominanceStochasticRanking,
    EpsilonRelaxation,
    RejectionInfillSelector,
    RejectionSampling,
    feasible_fraction,
    make_benchmark_g_fn,
)

__all__ = [
    'RejectionSampling',
    'RejectionInfillSelector',
    'make_benchmark_g_fn',
    'AdaptivePenaltyProblem',
    'EpsilonRelaxation',
    'DominanceStochasticRanking',
    'feasible_fraction',
]
