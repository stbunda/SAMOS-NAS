# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:17:20 2018

@author: r.dewinter

VENDORED, MODIFIED: the original computed the indicator with
``pygmo.hypervolume``. pygmo is a conda-only build of the pagmo2 C++ library
and is not otherwise a dependency of this repo (nor guaranteed present in the
HPC env), while pymoo -- which IS a hard dependency, and already supplies every
other hypervolume in this project -- computes the same quantity. Same
semantics: points not dominating the reference are excluded, an empty set
scores 0.
"""

import numpy as np
from pymoo.indicators.hv import HV


def hypervolume(pointset, ref):
    """Compute the absolute hypervolume of a *pointset* according to the
    reference point *ref*.
    """
    pointset = np.asarray(pointset, dtype=float)
    ref = np.asarray(ref, dtype=float)
    pointset = pointset[np.all(pointset <= ref, axis=1)]

    if len(pointset) == 0:
        return 0
    return float(HV(ref_point=ref)(pointset))
