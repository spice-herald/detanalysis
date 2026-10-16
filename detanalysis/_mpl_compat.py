"""Compatibility shims for newer matplotlib releases.

matplotlib 3.9 removed ``matplotlib.cm.get_cmap``, deprecated since 3.7.
vaex 4.19 still calls it from its plotting helpers (``vaex/utils.py`` and
``vaex/legacy.py``), so every ``viz.heatmap`` raises ``AttributeError`` on
matplotlib >= 3.9 whatever the caller does. detanalysis itself still calls it
too, from the colormap handling in its plotting methods.

Restoring the function is the least invasive fix: it reinstates the documented
behaviour of the removed API, affects only installations where it is already
missing, and needs no changes inside vaex.
"""

import matplotlib
import matplotlib.cm


def _get_cmap(name=None, lut=None):
    """Stand-in for the removed ``matplotlib.cm.get_cmap``."""
    if isinstance(name, matplotlib.colors.Colormap):
        return name
    if name is None:
        name = matplotlib.rcParams["image.cmap"]
    cmap = matplotlib.colormaps[name]
    return cmap if lut is None else cmap.resampled(lut)


def install():
    """Restore ``matplotlib.cm.get_cmap`` when the running matplotlib lacks it."""
    if not hasattr(matplotlib.cm, "get_cmap"):
        matplotlib.cm.get_cmap = _get_cmap
        return True
    return False
