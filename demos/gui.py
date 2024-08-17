import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
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
        print('Path', path)
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



if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of samples
    n_samples = 1000 # Define the means of the Gaussian distributions for each feature
    means = [0, 1, 2, 3]

    # Define the covariance matrix to introduce correlations between the features# The diagonal elements are the variances (since variance = standard deviation^2)# The off-diagonal elements are the covariances, controlling the correlations
    covariance_matrix = [
        [1.0, 0.8, 0.5, 0.3],  # Variances and covariances for feature 1
        [0.8, 1.0, 0.6, 0.4],  # Variances and covariances for feature 2
        [0.5, 0.6, 1.0, 0.7],  # Variances and covariances for feature 3
        [0.3, 0.4, 0.7, 1.0]   # Variances and covariances for feature 4
    ]

    # Generate the dataset using a multivariate Gaussian distribution
    data = np.random.multivariate_normal(means, covariance_matrix, size=n_samples)

    # Create a DataFrame with the generated data
    df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    hist = ax.hist2d(df['Feature1'], df['Feature2'], bins=30, cmap='plasma', norm=mpl.colors.LogNorm())
    vcollection = mpl.collections.RegularPolyCollection(3, sizes=(1,), offsets=df[['Feature1', 'Feature2']].to_numpy())
    selector = SelectFromCollection(ax, vcollection)


    fig.colorbar(hist[3], label='Counts (log scale)')

    ax.set_title('Feature1 vs Feature2')
    ax.set_xlabel('Feature1')
    ax.set_ylabel('Feature2')

    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            print(selector.xys[selector.ind])
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)

    plt.show()
