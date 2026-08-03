"""Colourblind-safe palette and axis helpers shared by every figure.

Qualitative: Okabe-Ito. Diverging: seaborn 'vlag'. Sequential: viridis.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito qualitative palette (colourblind-safe).
OKABE_ITO = [
    '#000000', '#E69F00', '#56B4E9', '#009E73',
    '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
]

# Four-region classifier colours for analyzer (f). Ordered background -> foreground:
# infeasible-dominated is normally the bulk of the population, so it is drawn
# first (and sits under) the smaller, more informative classes.
REGION_COLORS = {
    'infeasible-dominated':  '#999999',
    'feasible-dominated':    '#56B4E9',
    'infeasible-valuable':   '#D55E00',
    'feasible-nondominated': '#009E73',
}

# The local non-dominated front among 'infeasible-valuable' points (see
# analyzer (f)) is drawn as its own colour + marker shape rather than folded
# into REGION_COLORS, so it never gets confused with 'feasible-nondominated'
# -- distinct hue (Okabe-Ito yellow -- the one entry unused elsewhere in this
# palette, so it can't be mistaken for a shade of an existing region colour
# the way a second blue would be) and a distinct marker shape from the
# circles everything else uses.
REGION_MARKERS = {'feasible-nondominated': 'o', 'infeasible-nondominated': '^'}
INFEASIBLE_NONDOMINATED_COLOR = '#F0E442'

# 'vlag' ships with seaborn; importing it registers the colormap with
# matplotlib. Fall back to a stdlib diverging map if seaborn is absent.
try:
    import seaborn as _sns  # noqa: F401  (import registers 'vlag')
    DIVERGING = 'vlag'
except Exception:
    DIVERGING = 'RdBu_r'
SEQUENTIAL = 'viridis'

plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'font.size': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})


def apply_row_spacing(fig, nrow: int) -> None:
    """Widen vertical spacing on multi-row small-multiple grids.

    Default subplot spacing collides with x-axis labels once there is a row
    below, and with the figure suptitle once there is a row above it.
    """
    if nrow > 1:
        fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.90)


def apply_log(ax, which: str = 'x'):
    """Set a log scale on the requested axis/axes."""
    if 'x' in which:
        ax.set_xscale('log')
    if 'y' in which:
        ax.set_yscale('log')


def savefig(fig, path: str) -> None:
    """Tight, rasterization-friendly save; caller sets rasterized on artists."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
