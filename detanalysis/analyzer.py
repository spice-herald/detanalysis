import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import vaex as vx
from glob import glob
import os
import pytesdaq.io as h5io
import math

_all__ = ['Analyzer']


class Analyzer:
    """
    Class to analyze features (aka RQs) stored in  
    hdf5 files using vaex format (https://vaex.io). 
    Using vaex functions, the data is analyzed in a 
    similar way than pandas dataframe.

    The analyzer works best with vaex files, however data
    saved with pandas can still be analyzed using 
    'load_from_pandas=True' argument. In that case, 
    the entire dataframe is loaded to memory so works 
    only for small samples. 

    """
    
    def __init__(self, paths, series=None,
                 load_from_pandas=False):
        """
        Initialize analyzer class

        Parameters
        ----------
        
        paths :  str or list
          Directory or list of directories and/or files

        series : str or list
          series name or list of series
       
        """

        # initialize files info
        self._file_list = None
        self._nfiles = None


        # initialize dataframe info
        self._df = None
        self._is_df_filtered = False
        self._nevents = None
        self._nevents_nofilter = None
        self._nfeatures = None
        self._feature_names = None
        self._load_from_pandas = load_from_pandas

        # add files and open
        self.add_files(paths, series=series,
                       load_from_pandas=load_from_pandas)


        
        
    @property
    def df(self):
        return self._df

    @property
    def is_df_filtered(self):
        return self._is_df_filtered
        
    @property
    def nevents(self):
        return self._nevents

    @property
    def nfiles(self):
        return self._nfiles

    @property
    def nfeatures(self):
        return self._nfeatures

    @property
    def feature_names(self):
        return self._feature_names
    


    def describe(self):
        """
        Function to display number of files, events,
        and features
        
        Parameters
        ----------
        None


        Return
        ------
        None

        """
        print('Number of files: ' + str(self._nfiles))
        print('Number of events: ' + str(self._nevents))
        print('Number of features: ' + str(self._nfeatures))
        print('Is DataFrame filtered? ' + str(self._is_df_filtered))
        

    def get_unit(self, feature_exp):
        """
        Function to get feature or expression units
        Wrapper to vaex "df.unit(...)

        Parameters
        ----------
        
        feature_exp : str
           feature name or expression such as "x" or "x+y" 

        Return
        ------

        unit : astropy.unit.Units object
           feature units
        

        """
        return self._df.unit(feature_exp)



    
    def get_values(self, feature_exp, cut=None, **kwargs):
        """
        Function get the values of a feature or expression
        Wrapper to vaex "df.evaluate"

        Parameters
        ----------

        feature_exp : str or vaex expression object
           feature name or expression such as "x" or "x+y" 
           or vaex expression object, e.g. df.x or df.x+df.y .

        cut : str or vaex expression object, optional
           selection name or expression such as "x<1"
           or vaex expression object, e.g. df.x<1, 
           default is no cut (None)

        kwargs : dict
           extra keywords to be passed to vaex "evaluate", see
           documentation 
           https://vaex.readthedocs.io/en/latest/api.html#dataframe-class
           

        Return
        ------

        values : numpy array
          values of the expression or feature name
        

        """

        return self._df.evaluate(feature_exp,
                                 selection=cut,
                                 **kwargs)
    


    
    def register_cut(self, cut, cut_name, mode='replace'):
        """
        Function to register a cut with a specific name
        Wrapper to vaex df.select()

        Parameters
        ----------

        cut : str or vaex expression object
           expression such as "x<1"
           or vaex expression object, e.g. df.x<1, 
     
        cut_name : str
            selection name

        mode : str, optional
           Possible boolean operator: replace/and/or/xor/subtract
           default: replace
        
        Return
        ------
        None
 

        """
        
        self._df.select(cut,
                        name=cut_name,
                        mode=mode)


        
    def register_cut_box(self, spaces, limits,
                         cut_name, mode='replace'):
        """
        Function to register a box with a specific name
        Wrapper to vaex df.select_box()

        Parameters
        ----------

        spaces : list
            list of feature name or expression
            e.g ['x','y'], x,y=features

        limits : list
            sequence of shape 
            e.g. [(x1,x2), (y1,y2)]

        cut_name : str
            selection name

        mode : str, optional
           Possible boolean operator: replace/and/or/xor/subtract
           default: replace
        
        Return
        ------
        None
 

        """
        
        self._df.select_box(spaces, limits,
                            name=cut_name,
                            mode=mode)


        

    def apply_global_filter(self, cut, mode='replace'):
        """
        Function to filter dataframe
        Wrapper to vaex  df.filter()

        Parameters
        ----------

        cut : str or vaex expression object
           expression such as "x<1"
           or vaex expression object, e.g. df.x<1, 
           or cut/selection name that has been 
           already registered

        mode : str, optional
           Possible boolean operator: replace/and/or/xor/subtract
           default: replace

        Return
        ------
        
        None
        """

        # filter
        if mode=='replace':
            self._df.drop_filter()
            self._df = self._df.filter(cut,
                                       mode='replace')
        else:
            self._df = self._df.filter(cut,
                                       mode=mode)

        # dataframe info
        self._is_df_filtered = True
        self._fill_df_info()


        eff_percent = (
            self._nevents/self._nevents_nofilter*100
        )
        
        print('Filter applied!')
        print('Number of events after filter: '
              + f'{self._nevents}'
              + f' ({eff_percent:.1f}%)')
        
              

        
    def drop_global_filter(self):
        """
        Function to drop global dataframe
        filter
        Wrapper to df.drop_filter()

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        
        # drop filter
        self._df = self._df.drop_filter()
   
        # dataframe  info
        self._is_df_filtered = False
        self._fill_df_info()



    def add_feature(self, name, expression):
        """
        Function to add a new feature (virtual column)
        Wrapper to vaex df.add_virtual_column()

        Parameters
        ----------
        
        name : str
           name of the new feature

        expression : str or vaex vaex expression object
            expression such as "sqrt(x**2+y**2)
            or vaex object e.g df.x+2

        Return
        ------

        None

        """
        
        self._df.add_virtual_column(name, expression)


        
    def clean(self):
        """
        Clean filter, reload data
        """
        if self._df is not None:
            self._df.drop_filter() 

        self.add_files(self._file_list,
                       load_from_pandas=self._load_from_pandas,
                       reset=True)
        
    
        
    def add_files(self, paths, series=None,
                  load_from_pandas=False,
                  reset=False):
        """
        Function to add new files
        
        Parameters
        ----------

        paths : str or list
          directory or file name or list 
          of files/directories 
        

        series : str or list of str, optional
          series name or list or series to be selected
          default: all files
      
        Return
        ------
       
        None

        """

        # extract file list
        files = (
            self._extract_file_names(paths,
                                     series=series)
        )

    
        if reset or self._file_list is None:
            self._file_list = files
        else:
            self._file_list.append(files)
            self._file_list.sort()
            # unique
            
        self._nfiles = len(self._file_list)


        # open files
        if load_from_pandas:
            self._df = None
            for afile in files:
                vaex_df = vx.from_pandas(pd.read_hdf(afile, 'detprocess_df'))
                if  self._df is None:
                    self._df = vaex_df
                else:
                    self._df = vx.concat([self._df, vaex_df])
        else:
            self._df = vx.open_many(self._file_list)
        

        # fill dataframe info
        self._is_df_filtered = False
        self._fill_df_info()

        # keep track of events without filter
        self._nevents_nofilter = self._nevents



        
        
    def hist(self, feature_x, cuts=None,
             shape=64, limits='minmax', normalize=None,
             logx=False, logy=True,
             figsize=(9,6), colors=None, colormap=None,
             title=None, labels=None,
             xlabel=None, ylabel=None,
             what='count(*)',
             ax=None, **kwargs): 
        """
        Display histrogram using vaex visualization
        interface (wrapper to matplotlib)
        
        Parameters
        ----------

        feature_x : str or vaex expression object 
            feature name or expression to be displayed
            e.g. df.x, 'x', or 'x+y'
        
        cuts : str or list, or vaex expression objects
               optional
            selection(s) to be used
            e.g [None, df.x<2] or [None, 'mycut']
            default: None

        shape : int or list. optional
             shape for the array where the statistic is calculated on, 
             if only an integer is given, it is used for all dimensions, 
             e.g shape=128, shape=[128, 256], default=64

        limits : str or list, optional
            description for the min and max values of x 
            e.g. 'minmax' (default), '99.7%', '[0, 10]'


        normalize :  boolean, optional
            normalization function, currently only True, False/None 
            is supported, Default: None for no normalization

        logx : boolean, optional
            if True, use log scale x axis,otherwise lin scale
             default=False

        logy : boolean, optional
            if True, use log scale y axis,otherwise lin scale
             default=True

        figsize : tuple, optional
            passed to matplotlib plt.figure for setting the 
            figure size
        
        colors : str or list, optional
           list of colors, list should be same length as cuts
           or a string for one or no cuts. Default: use colormap
           argument instead
            
        colormap : str, optional
            matplotlib colormap 
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
            Default: ['b','r','g', 'c','m','y'] or 'viridis' if more cuts

        title : str, optional
            figure title, default: no title

        labels : str or list, optional
            labels for legend, list should be same length as cuts
            or a string for one or no cuts
            default: no legend
        
        xlabel : str, optional 
            label X axis, default: feature name

        ylabel : str, optional 
            label Y axis, default: what argument ("count(*)")
         
        what :  str, optional
           what to plot, e.g 'mean(x)', 'sum(x)'
           default: 'count(*)'

        ax : axes.Axes object, optional
            option to pass an existing Matplotlib Axes object to plot over,
            if it already exists.


        kwargs : dict, optional
            extra argument passed to maplotlib plt.plot


        
        Return
        -------
        fig : Figure
             Matplotlib Figure object. Set to None if ax is passed as a
             parameter.

        ax : axes.Axes object
             Matplotlib Axes object
        """

        
        # cuts
        if cuts is None:
            cuts = [None]
        elif not isinstance(cuts, list):
            cuts = [cuts]


        # total number of cuts
        ncuts = len(cuts)


        # normalize
        if normalize is not None:
            if normalize:
                normalize = 'normalize'
            else:
                normalize = None
                
            
        # colors
        if colors is not None:
            
            if not isinstance(colors, list):
                colors = [colors]

            if len(colors) != ncuts:
                raise ValueError('ERROR: "colors" argument should be '
                                 + 'a list of  length '
                                 + str(ncuts) + '!')     
        else:
            colors = ['blue','red','green',
                      'cyan','magenta','yellow']
            
            if ncuts>len(colors) or colormap is not None:
                if colormap is None:
                    colormap = 'viridis'
                colors = plt.cm.get_cmap(colormap)(np.linspace(0.1, 0.9,
                                                               ncuts))
            else: 
                colors = colors[0:ncuts]
                
            
        # labels
        if labels is not None:
            
            if not isinstance(labels, list):
                labels = [labels]

            if len(labels) != ncuts:
                raise ValueError('ERROR: "labels" argument should be '
                                 + 'a list of  length ' + str(ncuts) + '!')


              
        # create histogram
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
    
        
        # modify kwargs
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 2


        # loop cuts and make plot
        for icut in range(len(cuts)):

            cut = cuts[icut]
            
            if colors is not None:
                kwargs['color'] = colors[icut]
                
            label = None
            if labels is not None:
                label = labels[icut]

            # call vaex.viz
            self._df.viz.histogram(
                feature_x, selection=cut,
                shape=shape, limits=limits,
                n=normalize,
                xlabel=xlabel,ylabel=ylabel,
                label=label,
                **kwargs
            )

            
        # add/modify parameters
        if fig is None:
            ax.tick_params(which="both", direction="in",
                           right=True, top=True)
            ax.grid(linestyle="dashed")


        if logy:
            ax.semilogy(True)

        if logx:
            ax.semilogx(True)
        
        if labels is not None:
            ax.legend(loc="best")

        if title is not None:
            ax.set_title(title)
            
            
        return fig, ax


    
            
    def heatmap(self, feature_x, feature_y, cut=None,
                what='count(*)', f='log',
                shape=256, limits='minmax',
                colormap='plasma',
                figsize=(9,6),title=None,
                xlabel=None, ylabel=None,
                ax=None, **kwargs):
        """
        Display heatmap using vaex visualization
        interface (wrapper to matplotlib)
        
        Parameters
        ----------

        feature_x : str or vaex expression object 
            X axis feature name or expression to be displayed
            e.g. df.x, 'x', or 'x+y'

        feature_y : str or vaex expression object 
            Y axis feature name or expression to be displayed
            e.g. df.x, 'x', or 'x+y' 
        
        cut : str or vaex expression objects,  optional
            selection to be used
            e.g  cut=df.x<2 or cut='mycut'
            default: None

        f :  str, option 
          transform values by: "identity" does nothing "log" or "log10" 
          will show the log of the value, default="log"

        shape : int or list. optional
             shape for the array where the statistic is calculated on, 
             if only an integer is given, it is used for all dimensions, 
             e.g shape=128, shape=[128, 256], default=64

        limits : str or list, optional
            description for the min and max values for the expressions, 
             e.g. "minmax" (default), "99.7%", [0, 10], or a list of, 
             e.g. [[0, 10], [0, 20], "minmax"]
          
        colormap : str, optional
            matplotlib colormap, default='plasma'


        figsize : tuple, optional
            passed to matplotlib plt.figure for setting the 
            figure size
        
    
        title : str, optional
            figure title, default: no title
        
        xlabel : str, optional 
            label X axis, default: feature name

        ylabel : str, optional 
            label Y axis, default: what argument ("count(*)")
         
        ax : axes.Axes object, optional
            option to pass an existing Matplotlib Axes object to plot over,
            if it already exists.

        kwargs : dict, optional
            extra argument to vaex heatmap (see 
            https://vaex.io/docs/api.html#vaex-viz )


        Return
        -------
        fig : Figure
             Matplotlib Figure object. Set to None if ax is passed as a
             parameter.

        ax : axes.Axes object
             Matplotlib Axes object

        """

        # define subplot
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # call heatmap
        self._df.viz.heatmap(feature_x, feature_y,
                             colormap=colormap,
                             shape=shape, f=f, limits=limits,
                             figsize=figsize,
                             xlabel=xlabel, ylabel=ylabel,
                             selection=cut)

        # grid
        if fig is not None:
            ax.tick_params(which="both", direction="in",
                           right=True, top=True)
            ax.grid(linestyle="dashed")

        if title is not None:
            ax.set_title(title)
        
        
        return fig, ax
                                       


    def scatter(self, feature_x, feature_y, cuts=None, 
                figsize=(9,6), ms=5, alpha=0.8,
                xlimits=None, ylimits=None,
                colors=None, colormap=None,
                title=None, labels=None,
                xlabel=None, ylabel=None,
                nb_random_samples=None,
                length_check=True,
                ax=None, **kwargs):

        """
        vizualize  data (small amounts) in 2d using a scatter 
        interface (wrapper to vaex and matplotlib).
        
        Matplolib "scatter" is slow and not appropriate for large
        amount of data (use heatmap for large samples instead). 
        There is a check that number of samples<50000. To disable check, use 
        "length_check=False"

        For large samples, you can also randomly select smaller nb
        of events using "nb_random_samples" arg (done before cuts)
       

        
        Parameters
        ----------

        feature_x : str or vaex expression object 
            X axis feature name or expression to be displayed
            e.g. df.x, 'x', or 'x+y'

        feature_y : str or vaex expression object 
            Y axis feature name or expression to be displayed
            e.g. df.x, 'x', or 'x+y' 
        

        cuts : str or list, or vaex expression objects, optional
            selection(s) to be used
            e.g [None, df.x<2] or [None, 'mycut']
            default: None

        figsize : tuple, optional
            passed to matplotlib plt.figure for setting the 
            figure size

        ms : float, optional
          The size of each marker in the scatter plot. Default is 10
    

        alpha : float, optional
          The opacity of the markers in the scatter plot, i.e. alpha.
          Default is 0.8

   
        xlimits : list, optional
            description for the X axis min and max values for the expressions, 
             e.g. [0, 10]

        ylimits : list, optional
            description for the Y axis min and max values for the expressions, 
             e.g. [0, 10]

        colors : str or list, optional
           list of colors, list should be same length as cuts
           or a string for one or no cuts. Default: use colormap
           argument instead
            
        colormap : str, optional
            matplotlib colormap 
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
            Default: ['b','r','g', 'c','m','y'] or 'viridis' if nb cuts>6
          
    
        title : str, optional
            figure title, default: no title

        labels : str or list, optional
            labels for legend, list should be same length as cuts
            or a string for one or no cuts
            default: no legend

        
        xlabel : str, optional 
            label X axis, default: feature name

        ylabel : str, optional 
            label Y axis, default: what argument ("count(*)")
         

        nb_random_samples : int, optional
          number of randomly selected events (before cut)
          default: use all events
        
        length_check : boolean, option
           if True, check that number samples <50000
           if False, no checks, default: True

        ax : axes.Axes object, optional
            option to pass an existing Matplotlib Axes object to plot over,
            if it already exists.

        kwargs : dict, optional
            extra arguments passed to matplolib plt.scatter


        Return
        -------
        fig : Figure
             Matplotlib Figure object. Set to None if ax is passed as a
             parameter.

        ax : axes.Axes object
             Matplotlib Axes object
        """

        # cuts
        if cuts is None:
            cuts = [None]
        elif not isinstance(cuts, list):
            cuts = [cuts]

        # total number of cuts
        ncuts = len(cuts)
                    
            
        # colors
        if colors is not None:
            
            if not isinstance(colors, list):
                colors = [colors]

            if len(colors) != ncuts:
                raise ValueError('ERROR: "colors" argument should be '
                                 + 'a list of  length '
                                 + str(ncuts) + '!')     
        else:
            colors = ['blue','red','green',
                      'cyan','magenta','yellow']
            
            if ncuts>len(colors) or colormap is not None:
                if colormap is None:
                    colormap = 'viridis'
                colors = plt.cm.get_cmap(colormap)(
                    np.linspace(0.1, 0.9, ncuts))
            else: 
                colors = colors[0:ncuts]

            
        # labels
        if labels is not None:
            
            if not isinstance(labels, list):
                labels = [labels]

            if len(labels) != ncuts:
                raise ValueError('ERROR: "labels" argument should '
                                 + 'a list of  length '
                                 + str(ncuts) + '!')
            


            
        # define figure
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)


        # dataframe
        df = self._df
        if nb_random_samples is not None:
            df = df.sample(n=nb_random_samples)


            
            
        # loop cuts and plot
        for icut in  range(len(cuts)):

            cut = cuts[icut]

            label = None
            if labels is not None:
                label = labels[icut]

            color = None
            if colors is not None:
               color = colors[icut] 
            
               
            # call vaex.viz
            df.viz.scatter(
                feature_x, feature_y,
                selection=cut,
                xlabel=xlabel,ylabel=ylabel,
                label=label, color=color,
                s=ms, alpha=alpha,
                length_check=length_check,
                **kwargs
            )


        # limits
        if xlimits is not None:
            ax.set_xlim(xlimits)

        if ylimits is not None:
            ax.set_ylim(ylimits)


            
        # grid
        if fig is not None:
            ax.tick_params(which="both", direction="in",
                           right=True, top=True)
            ax.grid(linestyle="dashed")

    
        # title
        if title is not None:
            ax.set_title(title)

        # legend
        if labels is not None:
            ax.legend(markerscale=2, framealpha=.9, loc='best')
            

        return fig, ax


    def interactive_selection(self, feature_x, feature_y):
        """
        wrapper to jupyter widget heatmap to define cut
        (selection is automatically stored with name 
        "default") 

        Parameters
        ----------

        feature_x : str or vaex expression object 
            X axis feature name or expression to be displayed
            e.g. df.x, 'x', or 'x+y'

        feature_y : str or vaex expression object 
            Y axis feature name or expression to be displayed
            e.g. df.x, 'x', or 'x+y' 

        Return
        ------
        None

        """

        return self._df.widget.heatmap(feature_x,
                                       feature_y)

            

        
    def plot_traces(self, channel, raw_path, cut=None,
                    nb_random_samples=None,
                    figsize=None,
                    colors=None, colormap=None,
                    length_check=True,
                    single_plot=False):
        """
        Display selected traces for a particular channel
        and selection

        Parameters
        ----------

        channel : str
          name of the channel

        raw_path : str 
          base path to raw data group directory
          Note: for old data, path needs to include group name
                for vaex data, group_name is a feature 


        cut : str or vaex expression objects,  optional
            selection to be used
            e.g  cut=df.x<2 or cut='mycut'
            default: None

        figsize : tuple, optional
            passed to matplotlib plt.figure for setting the 
            figure size. If None, figsize hard coded (depends of 
            number of traces)
 
        colors : str or list, optional
           color or list of colors. if list, it should be same length as nb traces
           Default: blue for multiple plots, use colormap for single plot
            
        colormap : str, optional
            matplotlib colormap
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
            default: plasma

        single_plot : boolean, option
            if True, display all traces in same figures
            if False, multiple plots
            default: False
        
        nb_random_samples : int, optional
          number of randomly selected events (before cut)
          default: use all events
        
        length_check : boolean, option
           
           if True, check that number traces <20 when
           multiple subplots, <100 same plot
           if False, no checks, 
           default: True

        Return
        -------
        fig : Figure
             Matplotlib Figure object. Set to None if ax is passed as a
             parameter.

        ax : axes.Axes object
             Matplotlib Axes object
        

        """


        # max number of traces
        max_traces = 20
        if single_plot:
            max_traces = 100
        
            
        # get traces
        traces, info = self.get_traces(
            channel,
            raw_path=raw_path,
            cut=cut,
            nb_random_samples=nb_random_samples,
            length_check=length_check,
            length_limit= max_traces
        )

        if traces is None:
            return None, None

        
        
        nb_events = traces.shape[0]
        nrows = math.ceil(nb_events/2)
        ncols = 2
        if  nb_events==1:
            nb_cols = 1
            
        dt = 1/info[0]['sample_rate']


            
        # colors
        if colors is not None:
            
            if not isinstance(colors, list):
                colors = [colors]
                if not single_plot:
                    colors = colors*nb_events
                
            if len(colors) != nb_events:
                raise ValueError('ERROR: "colors" argument should be '
                                 + 'a list of  length '
                                 + str(nb_events) + '!')     
        else:
            colors = ['blue','red','green',
                      'cyan','magenta','yellow']

            if not single_plot:
                colors = ['blue']*nb_events
            
            if nb_events>len(colors) or colormap is not None:
                if colormap is None:
                    colormap = 'plasma'
                colors = plt.cm.get_cmap(colormap)(
                    np.linspace(0.1, 0.9, nb_events))
            else: 
                colors = colors[0:nb_events]

            
            
        # define fig size
        fig = None
        ax = None

        if single_plot:
            if figsize is None:
                figsize=(9, 6)
            fig, ax = plt.subplots(figsize=figsize)
            
        else:
            
            if figsize is None:
                figsize = (11, 14)
                if nb_events<=2:
                    figsize = (11, 6)
                elif nb_events<=4:
                    figsize = (11, 8)
                elif nb_events<=6:
                    figsize = (11, 11)
                
    
            fig, ax = plt.subplots(nrows=nrows,
                                   ncols=ncols,
                                   figsize=figsize)
            
        bins = np.asarray(list(range(traces.shape[-1])))*dt/1000

        if not single_plot:
            ax = ax.ravel()

        for it in range(traces.shape[0]):

            ax_it = ax
            if not single_plot:
                ax_it = ax[it] 
            
            ax_it.plot(bins, traces[it,0,:]*1e6,
                        color=colors[it])
            ax_it.set_xlabel("Time [ms]")
            ax_it.set_ylabel("Amplitude [uA]")

        # add title
        plt.suptitle(channel + ' raw traces')


        return fig, ax

        
                    
    def get_traces(self, channel, raw_path,
                   cut=None,
                   nb_random_samples=None,
                   length_check=True,
                   length_limit=1000):
        """
        Get raw traces for a particular channel
        and selection


        Parameters
        ----------
        
        channel : str
          name of the channel

        raw_path : str, optional
          base path to raw data group directory

        cut : str or vaex expression objects,  optional
            selection to be used
            e.g  cut=df.x<2 or cut='mycut'
            default: None

        nb_random_samples : int, optional
          number of randomly selected events (after cut)
          default: use all events
        
        length_check : boolean, option
           if True, check that number traces <length_limit
           if False, no checks, 
           default: True

        length_limit : int 
            maximum number of traces allowed
            default: 1000


        Return
        ------

        traces :  3D numpy array
           traces [nb events, nb_channels, nb samples] in amps

        info : dict
            metadata associated to traces

   
        """
        
        df = None

        # cut
        if cut is not None:
            df = self._df.filter(cut)
        else:
            df = self._df

        if df.shape[0]==0:
            print('WARNING: No events found!')
            return None, None
            
        # random selection of events
        if (nb_random_samples is not None
            and nb_random_samples<df.shape[0]):
            df = df.sample(n=nb_random_samples)

        # number of events check
        nb_events = df.shape[0]

        if length_check and nb_events>length_limit:
            raise ValueError('ERROR: Number of traces limited to '
                             + str(length_limit) + '. Found '
                             + str(nb_events) + ' traces!'
                             + ' Use length_check=False to disable error.')
        
            
        # get series number
        series_nums = None
        event_nums = None
        group_names = None
        if 'eventnumber' in self._feature_names:
            series_nums = df.seriesnumber.values
            event_nums =  df.eventnumber.values
        else:
            series_nums = df.series_number.values
            event_nums =  df.event_number.values
            group_names = df.group_name.values

            
        # add path
        path_list = list()
        if group_names is None:
            path_list.append(raw_path)
        else:
            for group in group_names:
                group = str(group)
                path_name = raw_path
                if group not in raw_path:
                    path_name = raw_path + '/' + group
                if path_name not in path_list:
                    path_list.append(path_name)

        
        h5 = h5io.H5Reader()
        traces, info =  h5.read_many_events(
            filepath=path_list,
            detector_chans=channel,
            event_nums=event_nums,
            series_nums=series_nums,
            output_format=2,
            include_metadata=True,
            adctoamp=True)
        h5.clear()
        

        
        return traces, info



            
    

    def _fill_df_info(self):
        """
        Internal function to set some dataframe 
        parameters

        Parameters
        ----------
        None


        Return
        -------
        None


        """
        try:
            self._feature_names = self._df.get_column_names()
            self._nevents = self._df.shape[0]
            self._nfeatures = self._df.shape[1]
        except:
            self.clean()
            print('Ooops... Something went wrong. '
                  + 'Will drop filter if any and reload data')


        
    def _extract_file_names(self, paths, series=None):
        """
        Get file list from directories or 
        files

        Parameters
        ----------

        paths :  str or list
          Directory or list of directories and/or files

        series : str or list
          series name or list of series
        
        Return
        ------

        file_list : list 
          List of files (full path)
        
      

        """
        
        # convert to list if needed
        if not isinstance(paths, list):
            paths = [paths]

        # initialize
        file_list = list()
       

        # loop files 
        for a_path in paths:
            
            # case path is a directory
            if os.path.isdir(a_path):
                
                if series is not None:
                    if series == 'even' or series == 'odd':
                        file_list = glob(a_path + '/'
                                         + series + '_*.hdf5')
                    else:
                        if not isinstance(series, list):
                            series = [series]
                        for it_series in series:
                            file_name_wildcard = '*' + it_series + '_*.hdf5'
                            file_list.extend(glob(a_path + '/'
                                                  + file_name_wildcard))
                else:
                    file_list = glob(a_path + '/*.hdf5')

                    
            # case file
            elif os.path.isfile(a_path):
                                
                if a_path.find('.hdf5') != -1:
                    if series is not None:
                        if series == 'even' or series == 'odd':
                            if a_path.find(series) != -1:
                                file_list.append(a_path)
                        else:
                            if not isinstance(series, list):
                                series = [series]
                            for it_series in series:
                                if a_path.find(it_series) != -1:
                                    file_list.append(a_path)
                    else:
                        file_list.append(a_path)

            else:
                raise ValueError('File or directory "' + a_path
                                 + '" does not exist!')

            
        if not file_list:
            raise ValueError('ERROR: No data found. Check arguments!')

        # sort
        file_list.sort()
      
          
        return file_list

    
