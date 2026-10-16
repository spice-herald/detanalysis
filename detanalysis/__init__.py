from ._mpl_compat import install as _install_mpl_compat

# vaex 4.19 calls matplotlib.cm.get_cmap, removed in matplotlib 3.9
_install_mpl_compat()

from .analyzer import Analyzer
from .semiautocuts import Semiautocut, MasterSemiautocuts
from .scatterplotter import ScatterPlotter
from .photon_calibration import PhotonCalibration
from .func import *
