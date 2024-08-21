#first, some imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [10, 6.5]

#import things associated with detanalysis including semiautocut
import vaex as vx
from detanalysis import Analyzer, Semiautocut, MasterSemiautocuts

#for reading in data
from pytesdaq.io.hdf5 import H5Reader
import pytesdaq.io.hdf5 as h5io
h5 = h5io.H5Reader()

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector



# Inspired by https://matplotlib.org/stable/gallery/widgets/lasso_selector_demo_sgskip.html
class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        print('Verts', verts)
        # print('Path', path)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        #self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()



#location of the rq files generated with detprocess
file_path = '/sdata1/runs/run26/processed/rqgen_feature_I2_D20230405_T154904'

#location of the triggered data on which detprocess was run. Used for plotting
#example events
path_to_triggered_data = '/sdata1/runs/run26/processed/triggered_I2_D20230405_T134718'

myanalyzer = Analyzer(file_path, series=None)

# show number of events 
myanalyzer.describe()

#df is the vaex dataframe with all the RQs
df = myanalyzer.df

print(df.get_column_names())

cut_pars_baseline = {'sigma_upper':1.4}


baseline_cut = Semiautocut(df, cut_rq = 'baseline', channel_name='Melange025pcLeft',
                               cut_pars=cut_pars_baseline)

bcut_mask = baseline_cut.do_cut(lgcdiagnostics=True)

fig, axes, data = baseline_cut.plot_vs_time(lgchours=True, lgcdiagnostics=True, show=False)
ax1, ax2 = axes
xdata, ydata = data

offsets =  np.column_stack((xdata.evaluate(), ydata.evaluate()))

vcollection = mpl.collections.RegularPolyCollection(3, sizes=(1,), offsets=offsets)
selector = SelectFromCollection(ax1, vcollection)

def accept(event):
    if event.key == "enter":
        print("Selected points:")
        print(selector.xys[selector.ind])
        selector.disconnect()
        ax1.set_title("")
        fig.canvas.draw()

fig.canvas.mpl_connect("key_press_event", accept)

plt.show()

