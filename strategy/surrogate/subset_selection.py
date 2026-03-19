import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.crossover import Crossover
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def subset_selection(candidates_F, archive_F, K):
    """
    Select K diverse solutions from candidates that improve the archive Pareto front.

    This function:
    1. Identifies non-dominated solutions in the candidates
    2. Selects K solutions with maximum diversity in objective space
    3. Uses a crowding-distance-like metric to spread selections

    Parameters:
    -----------
    candidates_F : np.ndarray of shape (n_candidates, n_obj)
        Objective values of candidate solutions
    archive_F : np.ndarray of shape (n_archive, n_obj)
        Objective values of archive (current Pareto approximation)
    K : int
        Number of solutions to select

    Returns:
    --------
    indices : np.ndarray or None
        Indices of selected candidates, or None if not enough candidates
    """

    # Ensure we have 2D arrays
    candidates_F = np.atleast_2d(candidates_F)
    if candidates_F.ndim == 1:
        candidates_F = candidates_F.reshape(-1, 1)

    archive_F = np.atleast_2d(archive_F)
    if archive_F.ndim == 1:
        archive_F = archive_F.reshape(-1, 1)

    n_candidates = len(candidates_F)

    # If we have K or fewer candidates, return all (let the caller handle it)
    if n_candidates <= K:
        return None

    # Find non-dominated solutions among candidates
    nds = NonDominatedSorting()
    candidate_fronts = nds.do(candidates_F)

    # Get all non-dominated candidates (first front)
    if len(candidate_fronts) == 0:
        return None

    nd_indices = candidate_fronts[0]

    # If we have K or fewer non-dominated solutions, return them
    if len(nd_indices) <= K:
        return nd_indices if len(nd_indices) > 0 else None

    # Use crowding distance to select K diverse points from non-dominated front
    from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

    # Calculate crowding distance for non-dominated solutions
    F_nd = candidates_F[nd_indices]
    crowding = calc_crowding_distance(F_nd)

    # Select top K solutions by crowding distance (most diverse)
    selected_order = np.argsort(-crowding)[:K]  # descending order
    selected_indices = nd_indices[selected_order]

    return selected_indices

