import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import vaex as vx
from glob import glob
import os
import pytesdaq.io as h5io
import math
from pprint import pprint
import importlib
from inspect import getmembers, isfunction
import git
import qetpy as qp


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
                 analysis_repo=None,
                 load_from_pandas=False,
                 memory_cache_size='1GB'):
        """
        Initialize analyzer class

        Parameters
        ----------
        
        paths :  str or list
          Directory/file or list of directories and/or files

        series : str or list, optional
          filter file based on series name or list of series
          Default: all files in "paths"

        analysis_repo : str, optional
           path to analysis package containing 
           "cuts" and "features" directory
           Default: None (path to cuts/features can be defined
                    in load_cuts / load_derived_features directly)

        memory_cache_size : str,optional
           size of the RAM cache, default=1GB
       
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

        # intialize cut dict
        # {cut_name: {metadata dict}}
        self._cuts = None

        # Initialize derived features dict
        # {feature_name: {metadata_dict}}
        self._derived_features = None
        
        
        # add files and open
        self.add_files(paths, series=series,
                       load_from_pandas=load_from_pandas)


        # turn on memory cache
        vx.cache.memory()
        vx.settings.cache.memory_size_limit= (
            memory_cache_size
        )

        # FIXE implement multilevel cache
        # vaex.settings.cache.disk_size_limit = ...
        # vaex.cache.path = ...
        # vaex.cache.multilevel_cache()
        
        
        # analysis package directory containing "cuts", "features"
        # directories
        self._analysis_repo = None
        if analysis_repo is not None:
            self.set_analysis_repo(analysis_repo, load_func=True)
            
       
        
        
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

        if self._cuts is not None:
            print('Cuts:')
            pprint(self._cuts)
        else:
            print('No cuts have been registered!')

        if self._derived_features is not None:
            print('Derived features:')
            pprint(self._derived_features)
        else:
            print('No derived features have been added!')

              
            
        

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

        values = np.array(
            self._df.evaluate(feature_exp,
                              selection=cut,
                              **kwargs)
        )
        
        return values
    


    
    def register_cut(self, cut, name,
                     metadata=None,
                     overwrite=False,
                     mode='replace'):
        """
        Function to register a cut with a specific name
        Wrapper to vaex df.select()

        Parameters
        ----------

        cut : str or vaex expression object
           expression such as "x<1"
           or vaex expression object, e.g. df.x<1, 
     
        name : str
            cut name

        metadata : dict, optional
            optional cut metadata:
              "version" : float 
              "authors" : str
              "description": str  (short summary)
              "contact" : str (format: email (name))
              "date" : str (format: MM/DD/YYYY)

        
        overwrite : boolean, optional
           if True, overwrite cut if exist already
           Default: False

        mode : str, optional
           Possible boolean operator: replace/and/or/xor/subtract
           default: replace
        


        Return
        ------
        None
 

        """

        # check if cut already exist
        if (not overwrite
            and self._cuts is not None
            and name in self._cuts.keys()):
            print('Cut "' + name + '" already '
                  + 'registered! Use overwrite=True '
                  + ' or change name.')
            return

        # metadata
        if metadata is None:
            metadata = dict()
        
        # save cut name in dictionary
        if self._cuts is None:
            self._cuts = dict()
                
        # apply and register
        self._df.select(cut,
                        name=name,
                        mode=mode)
        
        # update dictionary
        self._cuts[name] = metadata


        
    def register_cut_box(self, features, limits,
                         name,
                         metadata=None,
                         overwrite=False,
                         mode='replace'):
        """
        Function to register a box with a specific name
        Wrapper to vaex df.select_box()

        Parameters
        ----------

        features : list
            list of feature name or expression
            e.g ['x','y'], x,y=features

        limits : list
            sequence of shape 
            e.g. [(x1,x2), (y1,y2)]

        name : str
            cut  name


        metadata : dict, optional
            optional cut metadata:
              "version" : float 
              "authors" : str
              "description": str  (short summary)
              "contact" : str (format: email (name))
              "date" : str (format: MM/DD/YYYY)
 
        overwrite : boolean, optional
           if True, overwrite cut if exist already
           Default: False


        mode : str, optional
           Possible boolean operator: replace/and/or/xor/subtract
           default: replace
        
        Return
        ------
        None
 

        """

        # check if cut already exist
        if (not overwrite
            and self._cuts is not None
            and name in self._cuts.keys()):
            print('Cut "' + name + '" already '
                  + 'registered! Use mode="replace" '
                  + ' or change name.')
            return

        # metadata
        if metadata is None:
            metadata = dict()
        
        # save cut name in dictionary
        if self._cuts is None:
            self._cuts = dict()
                

        # apply
        self._df.select_box(features, limits,
                            name=name,
                            mode=mode)

        # update dictionary
        self._cuts[name] = metadata
        
        

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



    def add_feature(self, expression, name,
                    metadata=None,
                    overwrite=False):
        """
        Function to add a new feature (virtual column)
        Wrapper to vaex df.add_virtual_column()

        Parameters
        ----------
        

        expression : str or vaex vaex expression object
            expression such as "sqrt(x**2+y**2)
            or vaex object e.g df.x+2

    
        name : str
           name of the new feature

        
        metadata : dict, optional
            optional feaure metadata:
              "version" : float 
              "authors" : str
              "description": str  (short summary)
              "contact" : str (format: email (name))
              "date" : str (format: MM/DD/YYYY)
 
        overwrite : boolean, optional
           if True, overwrite derived feature if exist already
           Default: False



        Return
        ------

        None

        """

        # check if cut already exist
        if (not overwrite
            and self._derived_features is not None
            and name in self._derived_features.keys()):
            print('Feature "' + name + '" already '
                  + 'added! Use overwrite=True '
                  + ' or change name.')
            return

        # metadata
        if metadata is None:
            metadata = dict()
        
        # save cut name in dictionary
        if self._derived_features is None:
            self._derived_features = dict()
                
        self._derived_features[name] = metadata

        # add virtual column
        self._df.add_virtual_column(name, expression)

        # update info
        self._fill_df_info()



        
    def load_cuts(self,
                  cuts_path=None,
                  overwrite=False):
        """
        Load cuts from disk
        
        Parameters
        ----------

        cuts_path : str, option
          path to the cuts directory or full file name(s)
          Default: None (use analysis git repo 
                         defined during initialization or 
                         with "set_analysis_repo")

        overwrite : boolean, optional
           if True, overwrite cut if exist already
           (case existing cut same/higher version)
           If cut script has higher version, cut is 
           automatically updated
           Default: False
            

        Return:
        ------
        None

        """

        
        # check path and get list of files
        if (cuts_path is None
            and self._analysis_repo is None):
            print('ERROR: A path to the cuts needs '
                  + 'to be provided!')
            return


        if cuts_path is None:
            cuts_path = self._analysis_repo.working_dir + '/cuts'

            # if directory not found, then walk through subdirectories
            if not os.path.isdir(cuts_path):
                for dir_tuple in os.walk(self._analysis_repo.working_dir):
                    if dir_tuple and 'cuts' in dir_tuple[1]:
                        cuts_path = dir_tuple[0]  + '/cuts'
                        break
            
            
        repo_info = self._get_repo_info()
            
        # get functions and load
        self._load_func(cuts_path,
                        is_cut=True,
                        repo_info=repo_info,
                        overwrite=overwrite)
        
        
    def load_derived_features(self,
                              features_path=None,
                              overwrite=False):
        """
        Load derived features from disk
        
        Parameters
        ----------

        features_path : str, optional
          path to the features directory or or full path file name(s)
          Default: None (use path defined during initialization)

        update_git: boolean, optional
          update to latest git version 


        overwrite : boolean, optional
           if True, overwrite feature if exist already
           (case existing feature same/higher version)
           If feature script has higher version, it is 
           automatically updated
           Default: False
            
        

        Return:
        ------
        None

        """
        # check path and get list of files
        if (features_path  is None
            and self._analysis_repo is None):
            print('ERROR: A path to features needs '
                  + 'to be provided!')
            return
        
     
        if features_path is None:
            features_path = self._analysis_repo.working_dir  + '/features'

            # if directory not found, then walk through subdirectories
            if not os.path.isdir(features_path):
                for dir_tuple in os.walk(self._analysis_repo.working_dir):
                    if dir_tuple and 'features' in dir_tuple[1]:
                        features_path = dir_tuple[0]  + '/features'
                        break

        # get repo info
        repo_info = self._get_repo_info()       

            
        # get functions and load
        self._load_func(features_path,
                        is_cut=False,
                        repo_info=repo_info,
                        overwrite=overwrite)
        


         
        
    def clean(self):
        """
        Clean filter, reload data
        """
        if self._df is not None:
            self._df.drop_filter() 

        self.add_files(self._file_list,
                       load_from_pandas=self._load_from_pandas,
                       replace=True)
        
    
        
    def add_files(self, paths, series=None,
                  load_from_pandas=False,
                  replace=False):
        """
        Function to add new files
        (Note registered cut and new feature will be 
        deleted)
        
        Parameters
        ----------

        paths : str or list
          directory or file name or list 
          of files/directories 
        

        series : str or list of str, optional
          series name or list or series to be selected
          default: all files

        load_from_pandas: boolean
          if pandas files, set to True
          default: False

        replace: boolean
          if True, replace existing files
          Default: False


      
        Return
        ------
       
        None

        """

        # extract file list
        files = (
            self._extract_file_names(paths,
                                     series=series)
        )

    
        if (replace or self._file_list is None
            or not self._file_list):
            self._file_list = files
        else:
            self._file_list.extend(files)

        # sort
        self._file_list.sort()
        
        # unique
        self._file_list = list(set(self._file_list))

        # number of files
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

        # add "index" to match pandas dataframe
        self._df['index'] = np.arange(0, len(self._df), 1)
        

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
                shape=256,
                xlimits=None, ylimits=None,
                limits=None,
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

        xlimits : list, optional
            description for the X axis min and max values for the expressions, 
             e.g. [0, 10] or 'minmax' or "99.7%" or None

        ylimits : list, optional
            description for the Y axis min and max values for the expressions, 
             e.g. [0, 10] or 'minmax' or "99.7%" or None

        limits : str or list, optional 
            description for themin and max values for the expressions
            (to be used instead of xlimits/ylimits, not in addition)
             e.g. "minmax" (default), "99.7%", [0, 10], or a list of, 
             e.g. [[0, 10], [0, 20], "minmax"]
             e.g  [None, [0,10]



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


        # limits
        if (xlimits is not None or
            ylimits is not None):

        
            if limits is not None:
                print('ERROR: "limits" parameter cannot '
                      + 'be used in the same time as '
                      + ' "xlimits/ylimits"')
                return

            limits = [xlimits, ylimits]


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
        df = self._df.copy()
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


    def set_analysis_repo(self, repo_path, load_func=True):
        """
        Set analysis github repository. It should 
        contain  "cuts" and "features" directories
 
        Parameters
        ----------
        repo_path : str
          Path the repository main directory
        
        Return
        ------
        None


        """

        # instantiate repo
        try:
            self._analysis_repo  = git.Repo(repo_path)
        except git.exc.GitError as e:
            print('\nWARNING: Problem with analysis repo "'
                  + repo_path
                  +'".\n Is it a git package?')
            
            
        
        if load_func:
            self.load_derived_features()
            self.load_cuts()



    def plot_traces(self, channels, raw_path,
                    cut=None,
                    trace_length_msec=None,
                    trace_length_samples=None,
                    pretrigger_length_msec=None,
                    pretrigger_length_samples=None,
                    nb_random_samples=None,
                    figsize=None,
                    colors=None, colormap=None,
                    nb_events_check=True,
                    single_plot=False,
                    baselinesub=True,
                    baselineinds=(5,100),
                    lpcutoff=None):
        """
        Display selected traces for a particular channel
        and selection

        Parameters
        ----------

        channels : str or list of str
          name of the channel (s)

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
           color or list of colors. if list, it should be same length as 
           nb traces for multiple traces or nb_channels for multiple channels
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
        
        nb_events_check : boolean, option
           
           if True, check that number traces <20 when
           multiple subplots, <100 same plot
           if False, no checks, 
           default: True

        baselinesub : boolean, optional
           if True: baseline subtract trace using 
           first 100 bins
        


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

            
        # convert channels into list
        if isinstance(channels, str):
            channels = [channels]


        #if (len(channels)>1 and
        #    (len(single_plot):
        #    print('WARNING: Unable to plot multiple channels for '
        #          + ' multiple events on single figure. Changing '
        #          + 'settings to display multiple plots!')
        #    single_plot = False

            
        # get traces
        traces, info = self.get_traces(
            channels,
            trace_length_msec=trace_length_msec,
            trace_length_samples=trace_length_samples,
            pretrigger_length_msec=pretrigger_length_msec,
            pretrigger_length_samples=pretrigger_length_samples,
            raw_path=raw_path,
            cut=cut,
            nb_random_samples=nb_random_samples,
            nb_events_check=nb_events_check,
            nb_events_limit=max_traces,
            baselinesub=baselinesub,
            baselineinds=baselineinds,
        )

        if traces is None:
            return None, None

        # event info
        nb_events = traces.shape[0]
        nb_channels = traces.shape[1]
        nb_bins = traces.shape[2]
        nrows = math.ceil(nb_events/2)
        ncols = 2
        if  nb_events==1:
            nb_cols = 1
            
        # sample rate
        fs = info[0]['sample_rate']
        dt = 1/fs

        # low pass filter
        if lpcutoff is not None:
            for ichan in range(nb_channels):
                traces[:,ichan,:] = qp.utils.lowpassfilter(
                    traces[:,ichan,:],
                    lpcutoff,
                    fs=fs)
            
        # check if single plot can be used
        if (single_plot 
            and nb_channels>1 and nb_events>1):
            print('WARNING: Unable to plot multiple channels for '
                  + ' multiple events on single figure. Changing '
                  + 'settings to display multiple plots!')
            single_plot = False
            
                  
        # colors
        if colors is not None:
            
            if not isinstance(colors, list):
                colors = [colors]
                colors = colors*nb_events

            if nb_channels>1:
                if len(colors)<nb_channels:
                    raise ValueError('ERROR: "colors" argument should be '
                                     + 'a list of  length '
                                     + str(nb_channels) + '!')
            elif len(colors)<nb_events:
                raise ValueError('ERROR: "colors" argument should be '
                                 + 'a list of  length '
                                 + str(nb_events) + '!')            
        else:

            # hard code a few colors
            colors = ['blue','red','green',
                      'cyan','magenta','yellow']

            # for multiple plots, just use blue
            if nb_channels==1 and not single_plot:
                colors = ['blue']*nb_events

            # nb colors
            nb_colors = nb_events
            if nb_channels>nb_colors:
                nb_colors = nb_channels
                
            # check if using color map
            if (colormap is not None
                or (len(colors)<nb_channels
                    or len(colors)<nb_events)):
                
                if colormap is None:
                    colormap = 'plasma'   
                colors = plt.cm.get_cmap(colormap)(
                    np.linspace(0.1, 0.9, nb_colors))
                

            
            
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
            
        bins = np.asarray(list(range(traces.shape[-1])))*dt*1000

        if not single_plot:
            ax = ax.ravel()

        for it in range(nb_events):
            ax_it = ax
            if not single_plot:
                ax_it = ax[it] 
            for ichan in range(nb_channels):
                chan_name = channels[ichan]
                it_color = it
                if nb_channels>1:
                    it_color = ichan
                
                ax_it.plot(bins, traces[it,ichan,:]*1e6,
                           color=colors[it_color],
                           label=chan_name)
            ax_it.legend()
            ax_it.set_xlabel("Time [ms]")
            ax_it.set_ylabel("Amplitude [uA]")

       
        return fig, ax




    
    def get_event_list(self, cut=None,
                       nb_random_samples=None,
                       nb_events_check=True,
                       nb_events_limit=1000):

        """
        Get list of events in the form  of a list 
        of dictionaries with metadata required to 
        find traces in raw data
        """


        df = None
        
        # cut
        if cut is not None:
            df = self._df.filter(cut)
        else:
            df = self._df.copy()
            
        if df.shape[0]==0:
            print('WARNING: No events found!')
            return None, None
            
        # random selection of events
        if (nb_random_samples is not None
            and nb_random_samples<df.shape[0]):
            df = df.sample(n=nb_random_samples)

        # number of events check
        nb_events = df.shape[0]

        if nb_events_check and nb_events>nb_events_limit:
            raise ValueError('ERROR: Number of events limited to '
                             + str(nb_events_limit) + '. Found '
                             + str(nb_events) + ' traces!'
                             + ' Use nb_events_check=False to disable error.')


        # get events metadata
        series_nums = None
        event_nums = None
        group_names = None
        trigger_indices = None
        
        if 'eventnumber' in self._feature_names:
            series_nums = df.seriesnumber.values
            event_nums =  df.eventnumber.values
        else:
            series_nums = df.series_number.values
            event_nums =  df.event_number.values
            if 'group_name' in  self._feature_names:
                group_names = df.group_name.values
            if 'trigger_index' in self._feature_names:
                trigger_indices = df.trigger_index.values

        event_list = list()
        for ievent in range(nb_events):
            event_dict = dict()
            if series_nums is not None:
                event_dict['series_number'] = series_nums[ievent]
            if event_nums is not None:
                event_dict['event_number'] = event_nums[ievent]
            if group_names is not None:
                event_dict['group_name'] = group_names[ievent]
            if trigger_indices is not None:
                event_dict['trigger_index'] = trigger_indices[ievent]
               
            event_list.append(event_dict)


        # display
        print('INFO: Number of events found = '
              + str(len(event_list)))
        

            
        return event_list


    
                    
    def get_traces(self, channels, raw_path,
                   trace_length_msec=None,
                   trace_length_samples=None,
                   pretrigger_length_msec=None,
                   pretrigger_length_samples=None,
                   cut=None,
                   nb_random_samples=None,
                   nb_events_check=True,
                   nb_events_limit=1000,
                   memory_limit = 2,
                   baselinesub=False,
                   baselineinds=(5,100)):
        """
        Get raw traces for a particular channel
        and selection


        Parameters
        ----------
        
        channels : str or list of str
          name of the channel(s)

        raw_path : str
          base path to data group directory

        trace_length_msec : float, optional
           trace length in milli seconds
 
        trace_length_samples : int, optional
           trace length in number of samples
        
        pretrigger_length_msec : float, optional
           pretrigger length in milli seconds
 
        pretrigger_length_samples : int, optional
           pretrigger length in number of samples
        
        cut : str or vaex expression objects,  optional
            selection to be used
            e.g  cut=df.x<2 or cut='mycut'
            default: None

        nb_random_samples : int, optional
          number of randomly selected events (after cut)
          default: use all events
        
        nb_events_check : boolean, option
           if True, check that number traces <lentgh_limit
           if False, no checks, 
           default: True

        nb_events_limit : int 
            maximum number of traces allowed
            default: 1000
            
        memory_limit : float,
            Memory limit (in Gb) passed to 
            h5.read_many_events


        Return
        ------

        traces :  3D numpy array depending 
           raw traces in amps dim=[nb events, nb_channels, nb samples]

        info : dict
            metadata associated to traces
   
        """


        # get list of events
        event_list = self.get_event_list(
            cut=cut,
            nb_random_samples=nb_random_samples,
            nb_events_check=nb_events_check,
            nb_events_limit=nb_events_limit
        )
        

        # get raw data
        h5 = h5io.H5Reader()

        traces, info =  h5.read_many_events(
            filepath=raw_path,
            detector_chans=channels,
            event_list=event_list,
            trace_length_msec=trace_length_msec,
            trace_length_samples=trace_length_samples,
            pretrigger_length_msec=pretrigger_length_msec,
            pretrigger_length_samples=pretrigger_length_samples,
            output_format=2,
            include_metadata=True,
            adctoamp=True,
            memory_limit=memory_limit,
            baselinesub=baselinesub,
            baselineinds=baselineinds
        )
        
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

    

    def _load_func(self, paths,
                   is_cut=True,
                   repo_info=None,
                   overwrite=False):
        """
        
        import cut or feature functions
        Parameters
        ----------

        paths :  str or list
          Directory or list of directories and/or python scripts

        is_cut : boolean, optional
          If True, function is a cut
          If False, function is a feature
          Default: True


        repo_info : dict, optional
          dictionary with git repo information

        overwrite : boolean, optional
           if True, overwrite cut if exist already
           (case existing cut same/higher version)
           If cut script has higher version, cut is 
           automatically updated
           Default: False
        
        
        Return
        ------
        None
      
        

        """

        # get list of python script
        file_list = list()
        if not isinstance(paths, list):
            paths = [paths]
        for filepath in paths:
            if os.path.isdir(filepath):
                file_list.extend(glob(filepath + '/*.py'))
            elif os.path.isfile(filepath):
                file_list.append(filepath)
            else:
                raise ValueError('ERROR: Unknown path or file '
                                 + filepath)

        
        # sort/unique
        file_list.sort()
        file_list = list(set(file_list))
        

            
        # loop files and load
        module_name = 'detanalysis.analyzer'
        for a_file in file_list:

            # load module
            spec = importlib.util.spec_from_file_location(module_name,
                                                          a_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # get function list
            func_list = getmembers(module, isfunction)

            # loop (multiple functions can be defined
            # in same file)
            for func_name in func_list:
                func_name = func_name[0]
           
                # get object
                func_obj = getattr(module, func_name)

                # get object metadata
                func_metadata = vars(func_obj)

                # add repo info
                if repo_info is not None:
                    func_metadata.update(repo_info)
                
                # check if func exist
                current_funcs = self._derived_features
                if  is_cut:
                    current_funcs =  self._cuts 
                
                if (not overwrite
                    and current_funcs is not None
                    and func_name in current_funcs.keys()):
                    
                    version = None
                    if 'version' in func_metadata:
                        version = func_metadata['version']

                    version_saved = None
                    if 'version' in current_funcs[func_name].keys():
                        version_saved = (
                            current_funcs[func_name]['version']
                            )

                    if (version is not None
                        and version_saved is not None
                        and float(version)<=float(version_saved)):

                        print('WARNING: Function "' + func_name
                              + '" already exist (version='
                              + str(version_saved) + ').')
                        print(' Unable to register it! '
                              +'Change version or use overwrite=True')
                        continue
                    

                # register
                vaex_expr = func_obj(self._df)
                if is_cut:
                    self.register_cut(vaex_expr,
                                      name=func_name,
                                      metadata=func_metadata,
                                      overwrite=True,
                                      mode='replace')

                else:
                    self.add_feature(vaex_expr,
                                     name=func_name,
                                     metadata=func_metadata,
                                     overwrite=True)

        
    def _get_repo_info(self):
        """
        Extract analysis repo information

        Parameters
        ----------
        None


        Return
        ------

        repo_info : dict
          dictionary with git info
        """


        # initialize
        repo_info = dict()
        repo_info['git_repo_name'] = None
        repo_info['git_repo_branch'] = None
        repo_info['git_repo_tag']  = None
        # check if repo exist
        if self._analysis_repo is None:
            print('WARNING: No git repo available. '
                  + 'Use "set_analysis_repo" function '
                  + 'to set it!')
            return repo_info
        

        repo_info['git_repo_name'] = os.path.basename(
            self._analysis_repo.working_dir
        )
        

        repo_info['git_repo_branch'] = (
            self._analysis_repo.git.branch('--show-current')
        )

        repo_info['git_repo_tag'] =  (
            self._analysis_repo.git.describe(
                '--tags', '--dirty', '--broken'
            )
        )

        return repo_info
