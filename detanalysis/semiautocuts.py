import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import vaex as vx
from glob import glob
import os
import pytesdaq.io as h5io
import math
from pprint import pprint
import importlib
from inspect import getmembers, isfunction
from copy import copy
import qetpy as qp
import git

__all__ = ['Semiautocut', 'MasterSemiautocut', 'get_trace']

h5 = h5io.H5Reader()

def get_trace(df, index, path_to_triggered_data, lgcdiagnostics):
    """
    Function to retrive a trace so it can be plotted
    
    Parameters
    ----------
    
    df : vaex dataframe
        The dataframe the trace is being retrived from
    
    index : int
        The index (in the vaex dataframe) of the trace being retrived
        
    path_to_triggered_data : str
        The path to the folder where the triggered data is stored. 
        
    lgcdiagnostics : bool, optional
        If True, prints out diagnostic statements
    """
    #df.select(df.index == index)
    
    df['index'] = np.arange(0, len(df), 1)
    df = df.sort(by='index')
    
    dump_number = int(df[df.index == index].dump_number.values[0])
    series_number = int(df[df.index == index].series_number.values[0])
    event_index = int(df[df.index == index].event_index.values[0])
    
    if df[df.index == index].trigger_type.values == 4.0:
        random_truth = False
    if df[df.index == index].trigger_type.values == 3.0:
        random_truth = True
        
    if lgcdiagnostics:
        print("dump_number: " + str(dump_number))
        print("series_number: " + str(series_number))
        print("event_index: " + str(event_index))
        print("random_truth: " + str(random_truth))
    
    if bool(random_truth):
        event_prefix = '/rand_I2_D'
    else:
        event_prefix = '/threshtrig_I2_D'
    
    file_name = path_to_triggered_data + event_prefix + str(series_number)[1:-6]  + "_T" + str(series_number)[-6:] + "_F" + str(dump_number).zfill(4) + ".hdf5"
    
    if lgcdiagnostics:
        print("File name: " + str(file_name))
    
    trace = h5.read_single_event(event_index = event_index, file_name = file_name, adctoamp = True)
    return np.asarray(trace)
    




class Semiautocut:
    """
    Class to do cuts on detanalysis vaex dataframes. Each cut (e.g.
    baseline, chi2 vs ofamp, etc.) should be a separate semiautocut
    object. This object lets you both automate the creation of cuts
    and easily tweak the cuts yourself if you want to do something
    that the automation doesn't do easily. You can also use this object
    to get all the plots that Matt likes to get from cuts.

    """
    
    def __init__(self, vaex_dataframe, cut_rq, channel_name, 
                 cut_pars, time_bins=None, ofamp_bins=None,
                 exceptions_dict={},
                 ofamp_rq=None, chi2_rq=None,
                 cut_name=None):
        """
        Initialize semiautocut class

        Parameters
        ---------
            
        vaex_dataframe :  vaex dataframe
            Dataframe of all the RQs, generated from detprocess.

        cut_rq : str
            Name of the RQ that the cuts will be done on, e.g.
            "baseline" or "lowchi2_of1x1_nodelay"

        channel_name : str
            Name of the channel, e.g. "Melange1pc1ch"
            
        cut_pars : dict
            Dictionary of the parameters for the cut. Can contain
            the following values:
                -val_upper/val_lower: either or both can be used to set
                 the cut level in the same units as the cut rq is in.
                -sigma / sigma_upper/sigma_lower: either the symmetric
                value of sigma at which the data is cut outside, or
                the upper and lower bounds for the sigma cut. Note that
                the place to cut is determined by assuming the data is
                gaussian centered around the median, and that the width of
                the distribution is set by the interquartile range (IQR)
                rather than by the true std, which can be pulled by 
                outliers.
                -percent / percent_upper/percent_lower: either the percent
                of data to pass symmetrically around the median, or the
                percentile above and below which to pass data. This is the
                true percentile of data to pass, so for percent_upper=0.6, 
                percent_lower=0.4, 20% of the data is passed
                -val_upper/val_lower: value above and/or below which to 
                pass data.
                -time_arr: array of time pairs between which to pass data.
            Example: {'sigma': 0.90}, or 
            {'pecent_upper': 0.65, 'percent_lower': 0.05} or
            {'time_arr': [[1050, 1075], [1502, 1760]]}
            
        time_bins : array or int, optional
            If not None, then the cut will be performed in each time bin
            separately. If it's an int, this is the number of bins to make,
            and the data will automatically be divided with even numbers
            of events per bin. If it's an array, these will be the start
            of the time bins.
            
        ofamp_bins : array or int, optional
            If not none, then the cut will be performed in each ofamp bin
            separately. If it's an int, this is the number of bins to make,
            and the data will automatically be divided with even spacing
            of ofamp bins. If it's an array, these will be the start values
            of the ofamp bins.
            
        exceptions_dict : dict, optional
            Elements of this dictionary are individual cut_pars dicts, the
            keys for this dictionary are the bins for which the default
            cut_pars dictionary should be overwritten with the one in
            exceptions_dict.
            For example: { {4: cut_pars_exception_1}, {19: cut_pars_exception_2}}
            
        ofamp_rq : str, optional
            If not none, uses this as the RQ name for doing ofamp binned
            cuts and plotting. If none, defaults to 'amp_of1x1_nodelay'.
        
        chi2_rq : str, optional
            If not none, uses this as the RQ name for doing chi2 based plotting.
            If none, defaults to 'lowchi2_of1x1_nodelay'.
            
        cut_name : str, optional
            If not None, uses this to name the cut (i.e. adds the RQ to the
            vaex dataframe 'cut_ofamp_silly'). If None, defaults to
            "cut_" + cut_rq + "_" + "channel_name"
            
           
        """

        self.df = vaex_dataframe
        self.cut_rq_base = cut_rq
        self.channel_name = channel_name
        self.cut_pars = cut_pars
        self.time_bins = time_bins
        self.ofamp_bins = ofamp_bins
        self.exceptions_dict = exceptions_dict
        
        self.value_lower_arr = []
        self.value_upper_arr = []
        self.time_bins_arr = None
        self.ofamp_bins_arr = None
        
        #the mask for this cut, starts out as passing nothing
        self.mask = np.zeros(len(self.df), dtype = 'bool')
        
        
        #so we know what entry we need to look at in the vaex dataframe for ofamp
        if ofamp_rq is not None:
            self.ofamp_rq = str(ofamp_rq + '_' + self.channel_name)
        else:
            self.ofamp_rq = str('amp_of1x1_nodelay_' + self.channel_name)
        #so we know what entry we need to look at in the vaex dataframe for chi2
        if chi2_rq is not None:
            self.chi2_rq = str(chi2_rq + '_' + self.channel_name)
        else:
            self.chi2_rq = str('lowchi2_of1x1_nodelay_' + self.channel_name)
        
        
        if self.cut_rq_base is ('event_time'): #this means we're doing a time cut
            self.cut_rq = self.cut_rq_base #don't include channel name in full cut rq name
        else:
            self.cut_rq = str(self.cut_rq_base + '_' + self.channel_name)
            
            if 'time_arr' in self.cut_pars:
                raise Exception('Time array is only for time based cuts')
        
        #so we know what to call the new entry of the vaex dataframe
        if cut_name is not None:
            self.cut_name = cut_name
        else:
            self.cut_name = str('cut_' + self.cut_rq)
            print("Cut name: " + str(self.cut_name))
        self.df[self.cut_name] = np.ones(len(self.df), dtype = 'bool')

                
        if (self.time_bins is not None) and (self.ofamp_bins is not None):
            raise Exception('You must do either a binned in time or in ofamp cut, not both')
            
        #make time bins if starting with number of bins
        if isinstance(self.time_bins, int):
            num_events = len(self.df)
            step_size = round(num_events/self.time_bins)
            
            time_bins_arr = np.zeros(self.time_bins)
            i = 0
            while i < len(time_bins_arr):
                percentile = float(i)/(self.time_bins) * 100.0
                time_bins_arr[i] = float(self.df.percentile_approx('event_time', percentile))
                i += 1
                
            print("Constructed time bin array: " + str(time_bins_arr))
                
            self.time_bins_arr = time_bins_arr
        elif self.time_bins is not None:
            self.time_bins_arr = self.time_bins
            
            
        #make ofamp bins if starting with number of bins
        if isinstance(self.ofamp_bins, int):
            if self.ofamp_bins < 3:
                raise Exception("Must have more than 2 ofamp bins")
            
            ofamp_bins_arr = np.zeros(self.ofamp_bins)
            ofamp_vals = self.df[self.ofamp_rq].values
            
            ofamp_bins_arr[0] = min(ofamp_vals)
            ofamp_bins_arr[1] = 0.0
            
            ofamp_bin_spacing = max(ofamp_vals)/(self.ofamp_bins - 2)
            i = 2
            while i < len(ofamp_bins_arr):
                ofamp_bins_arr[i] = ofamp_bin_spacing*(i - 1)
                i += 1
                
            self.ofamp_bins_arr = ofamp_bins_arr
        elif self.ofamp_bins is not None:
            self.ofamp_bins_arr = self.ofamp_bins
        
    def do_cut(self, lgcdiagnostics=False, include_previous_cuts=False):
        """
        Performs cuts set up during initialization.
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            If True, prints diagnostic statements
            
        include_previous_cuts : bool or array, optional
            Option to generate the automatic cut values from events that pass
            previous rounds of cuts (i.e. cutting in ofamp vs. chi2 space, 
            generating cut levels only from the distributiuon of events that
            pass baseline and slope cuts). If True, uses all RQs in the dataframe
            starting with 'cut_' and including the channel name. If an array of
            names, uses those cut RQ names.
        
        Returns
        -------
        
        mask: array of booleans that can be used
            to construct a cut RQ by specifying for
            example df['cSlope'] = slope_cut_arr
        """
            
        if lgcdiagnostics:
            print("include_previous_cuts: " + str(include_previous_cuts))
            print(" ")
            
        if 'time_arr' in self.cut_pars:
            self._do_time_cut(lgcdiagnostics=lgcdiagnostics)
        elif (self.time_bins == None) and (self.ofamp_bins == None):
            self._do_simple_cut(lgcdiagnostics=lgcdiagnostics,
                                include_previous_cuts=include_previous_cuts)
        elif self.time_bins_arr is not None:
            self._do_time_binned_cut(lgcdiagnostics=lgcdiagnostics,
                                     include_previous_cuts=include_previous_cuts)
        elif self.ofamp_bins_arr is not None:
            self._do_ofamp_binned_cut(lgcdiagnostics=lgcdiagnostics,
                                      include_previous_cuts=include_previous_cuts)
            
        #this doesn't stick around, but makes plotting easier (or maybe does?)
        self.df[self.cut_name] = self.mask 
            
        return self.mask
               
                
    def _do_time_cut(self, lgcdiagnostics=False):
        """
        Performs a cut of specific time intervals and
        sets the cut mask for the semiautocuts cut object
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            If True, prints out diagnostic statements
        """
        
        if lgcdiagnostics:
            print(" ")
            print("Doing time based cut")
        
        i = 0
        while i < len(self.cut_pars['time_arr']):
            current_time_low = self.cut_pars['time_arr'][i][0]
            current_time_high = self.cut_pars['time_arr'][i][1]
            
            if lgcdiagnostics:
                print("Cutting between event_time " + str(current_time_low) + " and " +
                      str(current_time_high))
            
            current_cuts = (self.df.event_time.values > current_time_low) & (self.df.event_time.values < current_time_high)
            
            self.mask = current_cuts & self.mask
            i += 1
            
    def _get_cut_mask(self, ofamp_lims=None, time_lims=None, cut_pars=None,
                      lgcdiagnostics=False, include_previous_cuts=False):
        """
        Gets a boolean mask of a simple cut being done in a single bin.
        Defaults to no ofamp or time limits and using the global cut
        cut_pars dictionary if no limits or override cut_pars dict is
        passed.
        
        Parameters
        ----------
        
        ofamp_lims : array
            Array of the limits in ofamp where the cut will be performed,
            formated as [lim_low, lim_high].
            
        time_lims : array
            Array of the limits in time where the cut will be performed,
            formated as [lim_low, lim_high]
            
        lgcdiagnostics : bool, optional
            If True, prints out diagnostic statements
            
        include_previous_cuts : bool or array, optional
            Option to generate the automatic cut values from events that pass
            previous rounds of cuts (i.e. cutting in ofamp vs. chi2 space, 
            generating cut levels only from the distributiuon of events that
            pass baseline and slope cuts). If True, uses all RQs in the dataframe
            starting with 'cut_' and including the channel name. If an array of
            names, uses those cut RQ names.
                
        Returns
        -------
            
        overall_mask : array
            Array that's a mask for the cut (i.e. failing cut is False,
            passing cut is True). Values outside the time of ofamp limits
            will be false, so that the mask from this function can be
            combined with the mask from other functions with a bitwise or
            and a global cut mask can be constructed.
        """
            
        if lgcdiagnostics:
            print(" ")
            print("include_previous_cuts: " + str(include_previous_cuts))
            
        if cut_pars is None:
            cut_pars = self.cut_pars
            if lgcdiagnostics:
                print("Using default cut parameters")
                
        if ofamp_lims is not None:
            lims_mask_low = self.df[self.ofamp_rq].values > ofamp_lims[0]
            lims_mask_high = self.df[self.ofamp_rq].values < ofamp_lims[1]
            
            lims_mask = lims_mask_low & lims_mask_high
            
            if lgcdiagnostics:
                print("Binning by OFAmp, " + str(ofamp_lims))
        elif time_lims is not None:
            lims_mask_low = self.df.event_time.values > time_lims[0]
            lims_mask_high = self.df.event_time.values < time_lims[1]
            
            lims_mask = lims_mask_low & lims_mask_high
            
            if lgcdiagnostics:
                print("Binning by time, " + str(time_lims))
        else:
            lims_mask = np.ones(len(self.df), dtype='bool')
        
        cut_names = []        
        if include_previous_cuts is True:
            column_names = self.df.get_column_names()
            
            i = 0
            while i < len(column_names):
                if ("cut_" in column_names[i]) and (self.channel_name in column_names[i]):
                    cut_names.append(column_names[i])
                i += 1
        elif isinstance(include_previous_cuts, list):
            cut_names = copy(include_previous_cuts)
        else:
            cut_names = []
            
        
            
        self.df['_lims_mask'] = lims_mask
        cut_names.append('_lims_mask')
        
        #reset selection 
        self.df['_trues'] = np.ones(len(self.df), dtype = 'bool')
        self.df.select('_trues', mode='replace')
        
        #select the cuts in cut_names
        i = 0
        while i < len(cut_names):
            self.df.select(cut_names[i], mode='and')
            i += 1
            
        if lgcdiagnostics:
            print("Selection list: " + str(cut_names))
            print("Cut parameters: " + str(cut_pars))
            
        #check that the cut_pars dict has logical keys
        if 'time_arr' in self.cut_pars:
            raise Exception("Can't do simple cut with a time based cut")
            
        #if there are no datapoints in the array, return an array of all false
        if lgcdiagnostics:
            print("Number of events passing selection: " + str(sum(self.df.evaluate('_trues', selection=True))))
        if sum(self.df.evaluate('_trues', selection=True)) == 0:
            return np.zeros(len(self.df), dtype = 'bool')
            
            
        
        #value based cuts
        if ('val_upper' in cut_pars):
            value_upper = cut_pars['val_upper']
            if ('val_lower' in cut_pars):
                value_lower = cut_pars['val_lower']
                
                bool_arr_lower = self.df[self.cut_rq].values > cut_pars['val_lower']
                bool_arr_upper = self.df[self.cut_rq].values < cut_pars['val_upper']
                
                cut_mask = bool_arr_lower & bool_arr_upper
            else:
                cut_mask = self.df[self.cut_rq].values < cut_pars['val_upper']
        elif ('val_lower' in self.cut_pars):
            value_lower = cut_pars['val_lower']
            cut_mask = self.df[self.cut_rq].values > cut_pars['val_lower']
                
        #percentile based cuts
        elif ('percent_upper' in cut_pars):
            value_upper = self.df.percentile_approx(self.cut_rq, cut_pars['percent_upper']*100,
                                                    selection=True)
            if ('percent_lower' in cut_pars):
                value_lower = self.df.percentile_approx(self.cut_rq, cut_pars['percent_lower']*100,     
                                                        selection=True)
                    
                bool_arr_lower = self.df[self.cut_rq].values > value_lower
                bool_arr_upper = self.df[self.cut_rq].values < value_upper
                
                cut_mask = bool_arr_lower & bool_arr_upper
            else:
                cut_mask = self.df[self.cut_rq].values < value_upper
        elif ('percent_lower' in cut_pars):
            value_lower = self.df.percentile_approx(self.cut_rq, cut_pars['percent_lower']*100, 
                                                    selection=True)
            cut_mask = self.df[self.cut_rq].values > value_lower
        elif ('percent' in cut_pars):
            percent_lower = 0.5 - 0.5 * cut_pars['percent']
            percent_upper = 0.5 + 0.5 * cut_pars['percent']
            
            value_lower = self.df.percentile_approx(self.cut_rq, percent_lower*100, selection=True)
            value_upper = self.df.percentile_approx(self.cut_rq, percent_upper*100, selection=True)
            
            bool_arr_lower = self.df[self.cut_rq].values > value_lower
            bool_arr_upper = self.df[self.cut_rq].values < value_upper
            
            cut_mask = bool_arr_lower & bool_arr_upper
           
        
        #sigma based cuts
        elif ('sigma_upper' in cut_pars):
            #for sigma calculations, if we need them
            median = self.df.percentile_approx(self.cut_rq, 50, selection=True)
            sigma = np.mean([self.df.percentile_approx(self.cut_rq, 50 - 68.27/2.0, selection=True) - median, 
                            median - self.df.percentile_approx(self.cut_rq, 50 + 68.27/2.0, selection=True)])
            sigma = np.abs(sigma)
            
            value_upper = median + sigma * self.cut_pars['sigma_upper']
                
            if ('sigma_lower' in cut_pars):
                value_lower = median - sigma * cut_pars['sigma_lower']
                
                bool_arr_lower = self.df[self.cut_rq].values > value_lower
                bool_arr_upper = self.df[self.cut_rq].values < value_upper
                
                cut_mask = bool_arr_lower & bool_arr_upper
            else:
                cut_mask = self.df[self.cut_rq].values < value_upper
        elif ('sigma_lower' in cut_pars):
            #for sigma calculations, if we need them
            median = self.df.percentile_approx(self.cut_rq, 50, selection=True)
            sigma = np.mean([self.df.percentile_approx(self.cut_rq, 50 - 68.27/2.0, selection=True) - median, 
                            median - self.df.percentile_approx(self.cut_rq, 50 + 68.27/2.0, selection=True)])
            sigma = np.abs(sigma)
            
            value_lower = median + sigma * cut_pars['sigma_lower']
            cut_mask = self.df[self.cut_rq] > value_lower
        elif('sigma' in cut_pars):
            #for sigma calculations, if we need them
            median = self.df.percentile_approx(self.cut_rq, 50, selection=True)
            sigma = np.mean([self.df.percentile_approx(self.cut_rq, 50 - 68.27/2.0, selection=True) - median, 
                            median - self.df.percentile_approx(self.cut_rq, 50 + 68.27/2.0, selection=True)])
            sigma = np.abs(sigma)
            
            value_upper = median + sigma * cut_pars['sigma']
            value_lower = median - sigma * cut_pars['sigma']
            
            
            bool_arr_lower = self.df[self.cut_rq].values > value_lower
            bool_arr_upper = self.df[self.cut_rq].values < value_upper
            
            cut_mask = bool_arr_lower & bool_arr_upper
          
        if lgcdiagnostics:
            if 'value_lower' in locals():
                print("Lower limit for cuts: " + str(value_lower))
            if 'value_upper' in locals():
                print("Upper limit for cuts: " + str(value_upper))
        
        if 'value_lower' in locals():
            self.value_lower_arr.append(value_lower)
        else:
            self.value_lower_arr.append(min(self.df[self.cut_rq].values))
        if 'value_upper' in locals():
            self.value_upper_arr.append(value_upper)
        else:
            self.value_upper_arr.append(max(self.df[self.cut_rq].values))
          
        overall_mask = cut_mask & lims_mask
        return overall_mask
            
            
    def _do_simple_cut(self, lgcdiagnostics=False, include_previous_cuts=False):
        """
        Performs a cut which isn't binned in time or ofamp.
        Sets self.mask to be the cut array.
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            Prints diagnostic statements
            
        include_previous_cuts : bool or array, optional
            Option to generate the automatic cut values from events that pass
            previous rounds of cuts (i.e. cutting in ofamp vs. chi2 space, 
            generating cut levels only from the distributiuon of events that
            pass baseline and slope cuts). If True, uses all RQs in the dataframe
            starting with 'cut_' and including the channel name. If an array of
            names, uses those cut RQ names.
            
        """
        
        self.mask = self._get_cut_mask(lgcdiagnostics=lgcdiagnostics,
                                       include_previous_cuts=include_previous_cuts)
           
    def _do_time_binned_cut(self, lgcdiagnostics=False, include_previous_cuts=False):
        """
        Performs a time binned cut, with exceptions in the
        relevent bins.
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            Prints diagnostic statements
            
        include_previous_cuts : bool or array, optional
            Option to generate the automatic cut values from events that pass
            previous rounds of cuts (i.e. cutting in ofamp vs. chi2 space, 
            generating cut levels only from the distributiuon of events that
            pass baseline and slope cuts). If True, uses all RQs in the dataframe
            starting with 'cut_' and including the channel name. If an array of
            names, uses those cut RQ names.
            
        
        """
        working_mask = np.zeros(len(self.df), dtype = 'bool')
        i = 0
        while i < len(self.time_bins_arr) - 1:
            current_exception_pars = None
            if i in self.exceptions_dict:
                current_exception_pars = self.exceptions_dict[i]
                
            if lgcdiagnostics:
                print(" ")
                print(" ")
                print("On bin " + str(i))
                    
            time_lims_arr = [self.time_bins_arr[i], self.time_bins_arr[i + 1]]
                    
            current_mask = self._get_cut_mask(time_lims=time_lims_arr, cut_pars=current_exception_pars,
                                              lgcdiagnostics=lgcdiagnostics, 
                                              include_previous_cuts=include_previous_cuts)
            working_mask = working_mask | current_mask
                
            i += 1
                
        if lgcdiagnostics:
           print(" ")
           print(" ")
           print("On bin " + str(i))
        #last bin doesn't have an upper limit
        time_lims_arr = [self.time_bins_arr[-1], max(self.df.event_time.values)]
        current_exception_pars = None
        if len(self.time_bins_arr) - 1 in self.exceptions_dict:
            current_exception_pars = self.exceptions_dict[i]
        current_mask = self._get_cut_mask(time_lims=time_lims_arr, cut_pars=current_exception_pars, 
                                          lgcdiagnostics=lgcdiagnostics,
                                          include_previous_cuts=include_previous_cuts)
        working_mask = working_mask | current_mask
        
        self.mask = working_mask
            
    def _do_ofamp_binned_cut(self, lgcdiagnostics=False, include_previous_cuts=False):
        """
        Performs a ofamp binned cut, with exceptions in the
        relevent bins.
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            Prints diagnostic statements
            
        include_previous_cuts : bool or array, optional
            Option to generate the automatic cut values from events that pass
            previous rounds of cuts (i.e. cutting in ofamp vs. chi2 space, 
            generating cut levels only from the distributiuon of events that
            pass baseline and slope cuts). If True, uses all RQs in the dataframe
            starting with 'cut_' and including the channel name. If an array of
            names, uses those cut RQ names.
            
        
        """
        working_mask = np.zeros(len(self.df), dtype = 'bool')
        i = 0
        while i < len(self.ofamp_bins_arr) - 1:
            if lgcdiagnostics:
                print(" ")
                print(" ")
                print("On bin " + str(i))
        
            current_exception_pars = None
            if i in self.exceptions_dict:
                current_exception_pars = self.exceptions_dict[i]
                    
            ofamp_lims_arr = [self.ofamp_bins_arr[i], self.ofamp_bins_arr[i + 1]]
                    
            current_mask = self._get_cut_mask(ofamp_lims=ofamp_lims_arr, cut_pars=current_exception_pars, 
                                              lgcdiagnostics=lgcdiagnostics,
                                              include_previous_cuts=include_previous_cuts)
            working_mask = working_mask | current_mask
                
            i += 1
                
        if lgcdiagnostics:
            print(" ")
            print(" ")
            print("On bin " + str(i))
                
        #last bin doesn't have an upper limit
        ofamp_lims_arr = [self.ofamp_bins_arr[-1], max(self.df[self.ofamp_rq].values)]
        current_exception_pars = None
        if len(self.ofamp_bins_arr) - 1 in self.exceptions_dict:
            current_exception_pars = self.exceptions_dict[i]
        current_mask = self._get_cut_mask(ofamp_lims=ofamp_lims_arr, cut_pars=current_exception_pars, 
                                          lgcdiagnostics=lgcdiagnostics,
                                          include_previous_cuts=include_previous_cuts)
        working_mask = working_mask | current_mask
        
        self.mask = working_mask
            
            
            
            
    #plotting functions
    def plot_vs_time(self, lgchours=False, lgcdiagnostics=False, include_previous_cuts=False):
        """
        Plots RQ vs. time, showing data that passed and failed cut
        
        Parameters
        ----------
       
        lgchours : bool, optional
            If True, plots the event_time in units of hours rather
            than seconds.
            
        lgcdiagnostics : bool, optional
            If True, prints out diagnostic statements
            
        include_previous_cuts : bool or array, optional
            Option to plot the "with cuts" data with previous rounds of
            cuts. If True, includes all cuts in the dataframe with an RQ
            including "cut_" and the channel name. If an array, includes
            all cut RQs in the array.
        """
            
        time_norm=1.0
        if lgchours:
            time_norm = 60.0 * 60.0
            
        #figures out what cuts to include in the "with cuts" plot
        cut_names = []        
        if include_previous_cuts is True:
            column_names = self.df.get_column_names()
            
            i = 0
            while i < len(column_names):
                if ("cut_" in column_names[i]) and (self.channel_name in column_names[i]):
                    cut_names.append(column_names[i])
                i += 1
        elif isinstance(include_previous_cuts, list):
            cut_names = copy(include_previous_cuts)
        else:
            cut_names = []
        cut_names.append("cut_" + str(self.cut_rq))
        if lgcdiagnostics:
            print("Cut names to include in with cuts plot: ")
            print(str(cut_names))
        
        
        #reset selection 
        self.df['_trues'] = np.ones(len(self.df), dtype = 'bool')
        self.df.select('_trues', mode='replace')
        #i = 0
        #while i < len(cut_names):
        #    self.df.select(cut_names[i], mode='and')
        #    i += 1
            
        #plot all events
        cmap = copy(mpl.cm.get_cmap('winter') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap((self.df.event_time)/time_norm, self.df[self.cut_rq], colormap = cmap,
                            f='log', colorbar_label = "log(number/bin), All Events")
                                
        #plot events passing cuts
        cmap = copy(mpl.cm.get_cmap('spring') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap((self.df.event_time)/time_norm, self.df[self.cut_rq], colormap = cmap,
                           f='log', selection=cut_names, colorbar_label = "log(number/bin), Passing Cuts")
                           
        #plot horizontal lines for cut limits                   
        i = 0
        while i < len(self.value_lower_arr):
            if self.time_bins_arr is not None:
                time_limits_arr = np.asarray(self.time_bins_arr).tolist()
                time_limits_arr.append(max(self.df.event_time.values))
                plt.hlines([float(self.value_lower_arr[i]), float(self.value_upper_arr[i])],
                            float(time_limits_arr[i])/time_norm, float(time_limits_arr[i + 1])/time_norm)
            else:
                plt.hlines([self.value_lower_arr[i], self.value_upper_arr[i]],
                            min(self.df.event_time.values)/time_norm, max(self.df.event_time.values)/time_norm)
            i += 1
                           
        plt.title("Cut: " + str(self.cut_name) + ", \n " + str(self.cut_rq) + " vs. Time")
        if lgchours:
            plt.xlabel("event_time (hours)")
        else:
            plt.xlabel("event_time (seconds)")
        plt.show()
        
        #plot zoomed in around cuts
        cmap = copy(mpl.cm.get_cmap('winter') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap((self.df.event_time)/time_norm, self.df[self.cut_rq], colormap = cmap,
                            f='log', colorbar_label = "log(number/bin), All Events")
                            
        cmap = copy(mpl.cm.get_cmap('spring') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap((self.df.event_time)/time_norm, self.df[self.cut_rq], colormap = cmap,
                            f='log', selection=cut_names,
                             colorbar_label = "log(number/bin), Passing Cuts")
                           
        plt.title("Cuts: " + str(cut_names) + ", \n " + str(self.cut_rq) + " vs. Time \n " + 
                  " Zoomed In")
        if lgchours:
            plt.xlabel("event_time (hours)")
        else:
            plt.xlabel("event_time (seconds)")
            
        #plot horizontal lines for cut limits                   
        i = 0
        while i < len(self.value_lower_arr):
            if self.time_bins_arr is not None:
                time_limits_arr = np.asarray(self.time_bins_arr).tolist()
                time_limits_arr.append(max(self.df.event_time.values))
                plt.hlines([self.value_lower_arr[i], self.value_upper_arr[i]],
                            time_limits_arr[i]/time_norm, time_limits_arr[i + 1]/time_norm)
            else:
                plt.hlines([self.value_lower_arr[i], self.value_upper_arr[i]],
                            min(self.df.event_time.values)/time_norm, max(self.df.event_time.values)/time_norm)
            i += 1
            
        center_val_y = 0.5 * (min(self.value_lower_arr) + max(self.value_upper_arr))
        delta_y = max(self.value_upper_arr) - min(self.value_lower_arr)
        plt.ylim(center_val_y - delta_y, center_val_y + delta_y)
            
        plt.show()
        
            
    def plot_vs_ofamp(self, lgcdiagnostics=False, include_previous_cuts=False):
        """
        Plots RQ vs. ofamp, showing data that passed and failed cut
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            If True, prints out diagnostic statements
            
        include_previous_cuts : bool or array, optional
            Option to plot the "with cuts" data with previous rounds of
            cuts. If True, includes all cuts in the dataframe with an RQ
            including "cut_" and the channel name. If an array, includes
            all cut RQs in the array.
        """
        
        #figures out what cuts to include in the "with cuts" plot
        cut_names = []        
        if include_previous_cuts is True:
            column_names = self.df.get_column_names()
            
            i = 0
            while i < len(column_names):
                if ("cut_" in column_names[i]) and (self.channel_name in column_names[i]):
                    cut_names.append(column_names[i])
                i += 1
        elif isinstance(include_previous_cuts, list):
            cut_names = copy(include_previous_cuts)
        else:
            cut_names = []
        cut_names.append("cut_" + str(self.cut_rq))
        if lgcdiagnostics:
            print("Cut names to include in with cuts plot: ")
            print(str(cut_names))
        
        #reset selection 
        self.df['_trues'] = np.ones(len(self.df), dtype='bool')
        self.df.select('_trues', mode='replace')
        #i = 0
        #while i < len(cut_names):
        #    self.df.select(cut_names[i], mode='and')
        #    i += 1
            
        #plot all events
        cmap = copy(mpl.cm.get_cmap('winter') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df[self.ofamp_rq], self.df[self.cut_rq], colormap = cmap,
                            f='log', colorbar_label = "log(number/bin), All Events")
                                
        #plot events passing cuts
        cmap = copy(mpl.cm.get_cmap('spring') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df[self.ofamp_rq], self.df[self.cut_rq], colormap = cmap,
                           f='log', selection=cut_names,
                            colorbar_label = "log(number/bin), Passing Cuts")
                            
        #plot horizontal lines for cut limits                   
        i = 0
        while i < len(self.value_lower_arr):
            if self.ofamp_bins_arr is not None:
                ofamp_limits_arr = np.asarray(self.ofamp_bins_arr).tolist()
                ofamp_limits_arr.append(max(self.df[self.ofamp_rq].values))
                plt.hlines([self.value_lower_arr[i], self.value_upper_arr[i]],
                            ofamp_limits_arr[i], ofamp_limits_arr[i + 1])
            i += 1
                           
        plt.title("Cut: " + str(self.cut_name) + ", \n " + str(self.cut_rq) + " vs. OFAmp")
        plt.show()
        
        
        #plot zoomed in around cuts
        cmap = copy(mpl.cm.get_cmap('winter') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df[self.ofamp_rq], self.df[self.cut_rq], colormap = cmap,
                            f='log', colorbar_label = "log(number/bin), All Events")
                            
        cmap = copy(mpl.cm.get_cmap('spring') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df[self.ofamp_rq], self.df[self.cut_rq], colormap = cmap,
                            f='log', selection=cut_names,
                            colorbar_label = "log(number/bin), Passing Cuts")
                           
        plt.title("Cuts: " + str(cut_names) + ", \n " + str(self.cut_rq) + 
                  " vs. " + str(self.ofamp_rq) + " \n Zoomed In")
            
        center_val_y = 0.5 * (min(self.value_lower_arr) + max(self.value_upper_arr))
        delta_y = max(self.value_upper_arr) - min(self.value_lower_arr)
        plt.ylim(center_val_y - delta_y, center_val_y + delta_y)
        
        #plot horizontal lines for cut limits                   
        i = 0
        while i < len(self.value_lower_arr):
            if self.ofamp_bins_arr is not None:
                ofamp_limits_arr = np.asarray(self.ofamp_bins_arr).tolist()
                ofamp_limits_arr.append(max(self.df[self.ofamp_rq].values))
                plt.hlines([self.value_lower_arr[i], self.value_upper_arr[i]],
                            ofamp_limits_arr[i], ofamp_limits_arr[i + 1])
            i += 1
            
        plt.show()
        
            
    def plot_chi2_vs_ofamp(self, lgcdiagnostics=False, include_previous_cuts=False):
        """
        Shows events passing and failing cut on ofamp vs. chi2 plot
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            If True, prints out diagnostic statements
            
        include_previous_cuts : bool or array, optional
            Option to plot the "with cuts" data with previous rounds of
            cuts. If True, includes all cuts in the dataframe with an RQ
            including "cut_" and the channel name. If an array, includes
            all cut RQs in the array.
        """
        
        #figures out what cuts to include in the "with cuts" plot
        cut_names = []        
        if include_previous_cuts is True:
            column_names = self.df.get_column_names()
            
            i = 0
            while i < len(column_names):
                if ("cut_" in column_names[i]) and (self.channel_name in column_names[i]):
                    cut_names.append(column_names[i])
                i += 1
        elif isinstance(include_previous_cuts, list):
            cut_names = copy(include_previous_cuts)
        else:
            cut_names = []
        cut_names.append("cut_" + str(self.cut_rq))
        if lgcdiagnostics:
            print("Cut names to include in with cuts plot: ")
            print(str(cut_names))
        
        #reset selection 
        self.df['_trues'] = np.ones(len(self.df), dtype='bool')
        self.df.select('_trues', mode='replace')
        #i = 0
        #while i < len(cut_names):
        #    self.df.select(cut_names[i], mode='and')
        #    i += 1
            
        #plot all events
        cmap = copy(mpl.cm.get_cmap('winter') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df[self.ofamp_rq], self.df[self.chi2_rq], colormap = cmap,
                            f='log', colorbar_label = "log(number/bin), All Events")
                                
        #plot events passing cuts
        cmap = copy(mpl.cm.get_cmap('spring') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df[self.ofamp_rq], self.df[self.chi2_rq], colormap = cmap,
                           f='log', selection=cut_names,
                            colorbar_label = "log(number/bin), Passing Cuts")
                            
        #plot horizontal lines for cut limits                   
        i = 0
        while i < len(self.value_lower_arr):
            if self.ofamp_bins_arr is not None:
                ofamp_limits_arr = np.asarray(self.ofamp_bins_arr).tolist()
                ofamp_limits_arr.append(max(self.df[self.ofamp_rq].values))
                if 'chi2' in str(self.cut_rq):
                    plt.hlines([self.value_lower_arr[i], self.value_upper_arr[i]],
                                ofamp_limits_arr[i], ofamp_limits_arr[i + 1])
            i += 1
                           
        plt.title("Cuts: " + str(cut_names) + ", \n OFAmp vs. Chi2")
        plt.show()
            
    def plot_histograms(self, lgcdiagnostics=False, include_previous_cuts=False, num_bins=100):
        """
        Plots histogram of RQ showing passing and failing events,
        either for all the data or for each bin in a binned cut.
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            If True, prints out diagnostic statements
            
        include_previous_cuts : bool or array, optional
            Option to plot the "with cuts" data with previous rounds of
            cuts. If True, includes all cuts in the dataframe with an RQ
            including "cut_" and the channel name. If an array, includes
            all cut RQs in the array.
            
        num_bins : int, optional
            Number of bins to plot in the histogram. Defaults to 100.
        """
        
        #figures out what cuts to include in the "with cuts" plot
        cut_names = []        
        if include_previous_cuts is True:
            column_names = self.df.get_column_names()
            
            i = 0
            while i < len(column_names):
                if ("cut_" in column_names[i]) and (self.channel_name in column_names[i]):
                    cut_names.append(column_names[i])
                i += 1
        elif isinstance(include_previous_cuts, list):
            cut_names = copy(include_previous_cuts)
        else:
            cut_names = []
        cut_names.append("cut_" + str(self.cut_rq))
        if lgcdiagnostics:
            print("Cut names to include in with cuts plot: ")
            print(str(cut_names))
        
        #reset selection 
        self.df['_trues'] = np.ones(len(self.df), dtype='bool')
        self.df.select('_trues', mode='replace')
        
        self.df.viz.histogram(self.cut_rq, label = "All Events", shape=num_bins)
        
        self.df.viz.histogram(self.cut_rq, label = "Passing Cut/s", selection=cut_names, shape=num_bins)
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.title("Cuts: " + str(cut_names) + ", \n " + str(self.cut_rq) + " Histogram")
        plt.show()
            
    def plot_example_events(self, num_example_events, trace_index, path_to_triggered_data,
                             time_lims=None, lp_freq=None, lgcdiagnostics=False):
        """
        Plots a certain number of example of events that pass and
        fail cuts.
        
        Parameters
        ----------
        
        num_example_events : int
            Number of example events to plot
            
        trace_index : int
            Index of the trace to plot
            
        path_to_triggered_data : str
            Path to the folder holding triggered data
                
        time_lims : array, optional
            Array of low and high limits for the time series plots
            
        lp_freq : float, optional
            Cutoff frequency for an optional trace low pass filter
            
        lgcdiagnostics : bool, optional
            If True, prints our diagnostic statements
        """
        
        if lgcdiagnostics:
            print("Cut name: " + str('cut_' + str(self.cut_rq)))
            print("Cut values: " + str(self.df[str('cut_' + str(self.cut_rq))].values))
        
        self.df.select(str('cut_' + str(self.cut_rq)))
        all_pass_indices = self.df.evaluate('index', selection = True)
        pass_indicies = np.random.choice(all_pass_indices, num_example_events)
        
        all_indices = self.df.evaluate('index')
        pass_mask = np.isin(all_indices, all_pass_indices)
        fail_mask = np.invert(pass_mask)
        all_fail_indices = all_indices[fail_mask]
        fail_indicies = np.random.choice(all_fail_indices, num_example_events)
        
        #turn off selections
        self.df.select('_trues')
        
        
        if lgcdiagnostics:
            print("Passing indices: " + str(pass_indicies))
            print("Failing indices: " + str(fail_indicies))
            
            
        traces_passing = []
        traces_failing = []
        
        i = 0
        while i < len(pass_indicies):
            traces_passing.append(get_trace(self.df, pass_indicies[i], path_to_triggered_data,
                                  lgcdiagnostics=lgcdiagnostics)[trace_index])
            traces_failing.append(get_trace(self.df, fail_indicies[i], path_to_triggered_data, 
                                  lgcdiagnostics=lgcdiagnostics)[trace_index])
            i += 1
            
        #if lgcdiagnostics:
        #    print("Passing traces: " + str(traces_passing))
        #    print("Failing traces: " + str(traces_failing))
            
        fs = int(1.25e6)
        t_arr = np.arange(0, (len(traces_passing[0]) - 0.5)/fs, 1/fs)
        i = 0
        while i < len(traces_passing):
            plt.plot(t_arr, traces_passing[i])
            i += 1
        plt.title("Events Passing Cut")
        if time_lims is not None:
            plt.xlim(time_lims[0], time_lims[1])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (amps)")
        plt.show()
        
        i = 0
        while i < len(traces_failing):
            plt.plot(t_arr, traces_failing[i])
            i += 1
        plt.title("Events Failing Cut")
        if time_lims is not None:
            plt.xlim(time_lims[0], time_lims[1])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (amps)")
        plt.show()
        
        if lp_freq is not None:
            order_ = 2
            
            i = 0
            while i < len(traces_passing):
                lp_trace_i = qp.utils.lowpassfilter(traces_passing[i], lp_freq, fs=fs, order = order_)
                plt.plot(t_arr, lp_trace_i)
                i += 1
            plt.title("Events Passing Cut, " + str(lp_freq*1e-3) + " kHz Low Pass Filted")
            if time_lims is not None:
                plt.xlim(time_lims[0], time_lims[1])
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (amps)")
            plt.show()
            
            i = 0
            while i < len(traces_passing):
                lp_trace_i = qp.utils.lowpassfilter(traces_failing[i], lp_freq, fs=fs, order = order_)
                plt.plot(t_arr, lp_trace_i)
                i += 1
            plt.title("Events Failing Cut, " + str(lp_freq*1e-3) + " kHz Low Pass Filted")
            if time_lims is not None:
                plt.xlim(time_lims[0], time_lims[1])
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (amps)")
            plt.show()
    
        
        
            
    def get_passage_fraction(self, lgcprint=False):
        """
        Calculates and returns the passage fraction for the cut.
        
        Parameters
        ----------
            
        lgcprint : bool
            If True, prints a diagostic message including the
            passage fraction and number of passing, failing and
            total events.
                
        Returns
        -------
            
        passage_fraction : float
            Fraction of events that pass the cuts (i.e. 0-1)
            
        """
            
        passage_fraction = sum(self.mask)/len(self.mask)
            
        if lgcprint:
            print("Passage fraction: " + str(passage_fraction))
            print("Number of events passing cuts: " + str(sum(self.mask)))
            print("Number of events failing cuts: " + str(len(self.mask) - sum(self.mask)))
            print("Number of total events: " + str(len(self.mask)))
                
        return passage_fraction
        

class MasterSemiautocuts:
    """
    Class for combining cuts stored in a vaex dataframe into
    one master cut called "cut_all"
    """
    
    def __init__(self, df, cuts_list, channel_name,
                 ofamp_rq=None, chi2_rq=None):
        """
        Initialize the MasterSemiautocuts class
        
        Attributes
        ----------
        
        df : vaex dataframe
            The vaex dataframe where cuts are contained
            
        cuts_list : list of strs
            A list of the column names for the cuts
            
        channel_name : str
            The name of the channel cuts are being combined for
            
        
        """
        
        self.df = df
        self.cuts_list = cuts_list
        self.channel_name = channel_name
        self.mask = np.zeros(len(df), dtype = 'bool')
        
        if ofamp_rq is not None:
            self.ofamp_rq = str(ofamp_rq + '_' + self.channel_name)
        else:
            self.ofamp_rq = str('amp_of1x1_nodelay_' + self.channel_name)
        #so we know what entry we need to look at in the vaex dataframe for chi2
        if chi2_rq is not None:
            self.chi2_rq = str(chi2_rq + '_' + self.channel_name)
        else:
            self.chi2_rq = str('lowchi2_of1x1_nodelay_' + self.channel_name)
        
    def combine_cuts(self, sat_pass_threshold=None):
        """
        Combines the cuts specified during initializtion
        into one.
        
        Parameters
        ----------
        
        sat_pass_threshold : float, optional
            If not none, this is the value above which all
            events will be passed. Used for saturated events,
            which will fail the slope and possibly chi2 cuts.
        
        """
        
        cuts_all_arr = np.ones(len(self.df), dtype = 'bool')
        
        i = 0
        while i < len(self.cuts_list):
            current_cut_arr = self.df[self.cuts_list[i]].values
            cuts_all_arr = np.logical_and(cuts_all_arr, current_cut_arr)
        
            i += 1
            
        if sat_pass_threshold is not None:
            ofamps_arr_all = self.df[self.ofamp_rq].values
            ofamp_thresh_mask = (ofamps_arr_all > sat_pass_threshold)
            
            cuts_all_arr = np.logical_or(cuts_all_arr, ofamp_thresh_mask)
            
        self.df['cut_all_' + self.channel_name] = cuts_all_arr
        self.mask = cuts_all_arr
    
    def get_passage_fraction(self, lgcprint=False):
        """
        Calculates and returns the passage fraction for the cut.
        
        Parameters
        ----------
            
        lgcprint : bool
            If True, prints a diagostic message including the
            passage fraction and number of passing, failing and
            total events.
                
        Returns
        -------
            
        passage_fraction : float
            Fraction of events that pass the cuts (i.e. 0-1)
            
        """
            
        passage_fraction = sum(self.mask)/len(self.mask)
            
        if lgcprint:
            print("Passage fraction: " + str(passage_fraction))
            print("Number of events passing cuts: " + str(sum(self.mask)))
            print("Number of events failing cuts: " + str(len(self.mask) - sum(self.mask)))
            print("Number of total events: " + str(len(self.mask)))
                
        return passage_fraction
        
    def plot_example_events(self, num_example_events, trace_index, path_to_triggered_data,
                             time_lims=None, lp_freq=None, lgcdiagnostics=False):
        """
        Plots a certain number of example of events that pass and
        fail cuts.
        
        Parameters
        ----------
        
        num_example_events : int
            Number of example events to plot
            
        trace_index : int
            Index of the trace to plot
            
        path_to_triggered_data : str
            Path to the folder holding triggered data
                
        time_lims : array, optional
            Array of low and high limits for the time series plots
            
        lp_freq : float, optional
            Cutoff frequency for an optional trace low pass filter
            
        lgcdiagnostics : bool, optional
            If True, prints our diagnostic statements
        """
        
        self.df.select(str('cut_all_' + self.channel_name))
        all_pass_indices = self.df.evaluate('index', selection = True)
        pass_indicies = np.random.choice(all_pass_indices, num_example_events)
        
        all_indices = self.df.evaluate('index')
        pass_mask = np.isin(all_indices, all_pass_indices)
        fail_mask = np.invert(pass_mask)
        all_fail_indices = all_indices[fail_mask]
        fail_indicies = np.random.choice(all_fail_indices, num_example_events)
        
        #turn off selections
        self.df.select('_trues')
        
        
        if lgcdiagnostics:
            print("Passing indices: " + str(pass_indicies))
            print("Failing indices: " + str(fail_indicies))
            
            
        traces_passing = []
        traces_failing = []
        
        i = 0
        while i < len(pass_indicies):
            traces_passing.append(get_trace(self.df, pass_indicies[i], path_to_triggered_data,
                                  lgcdiagnostics=lgcdiagnostics)[trace_index])
            traces_failing.append(get_trace(self.df, fail_indicies[i], path_to_triggered_data, 
                                  lgcdiagnostics=lgcdiagnostics)[trace_index])
            i += 1
            
        #if lgcdiagnostics:
        #    print("Passing traces: " + str(traces_passing))
        #    print("Failing traces: " + str(traces_failing))
            
        fs = int(1.25e6)
        t_arr = np.arange(0, (len(traces_passing[0]) - 0.5)/fs, 1/fs)
        i = 0
        while i < len(traces_passing):
            plt.plot(t_arr, traces_passing[i])
            i += 1
        plt.title("Events Passing All Cuts")
        if time_lims is not None:
            plt.xlim(time_lims[0], time_lims[1])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (amps)")
        plt.show()
        
        i = 0
        while i < len(traces_failing):
            plt.plot(t_arr, traces_failing[i])
            i += 1
        plt.title("Events Failing All Cuts")
        if time_lims is not None:
            plt.xlim(time_lims[0], time_lims[1])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (amps)")
        plt.show()
        
        if lp_freq is not None:
            order_ = 2
            
            i = 0
            while i < len(traces_passing):
                lp_trace_i = qp.utils.lowpassfilter(traces_passing[i], lp_freq, fs=fs, order = order_)
                plt.plot(t_arr, lp_trace_i)
                i += 1
            plt.title("Events Passing All Cuts, " + str(lp_freq*1e-3) + " kHz Low Pass Filted")
            if time_lims is not None:
                plt.xlim(time_lims[0], time_lims[1])
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (amps)")
            plt.show()
            
            i = 0
            while i < len(traces_passing):
                lp_trace_i = qp.utils.lowpassfilter(traces_failing[i], lp_freq, fs=fs, order = order_)
                plt.plot(t_arr, lp_trace_i)
                i += 1
            plt.title("Events Failing All Cuts, " + str(lp_freq*1e-3) + " kHz Low Pass Filted")
            if time_lims is not None:
                plt.xlim(time_lims[0], time_lims[1])
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (amps)")
            plt.show()
            
    def plot_chi2_vs_ofamp(self, lgcdiagnostics=False):
        """
        Shows events passing and failing cut on ofamp vs. chi2 plot
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            If True, prints out diagnostic statements
        """     
        
        
        #reset selection 
        self.df['_trues'] = np.ones(len(self.df), dtype='bool')
        self.df.select('_trues', mode='replace')
            
        #plot all events
        cmap = copy(mpl.cm.get_cmap('winter') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df[self.ofamp_rq], self.df[self.chi2_rq], colormap = cmap,
                            f='log', colorbar_label = "log(number/bin), All Events")
                                
        #plot events passing cuts
        cmap = copy(mpl.cm.get_cmap('spring') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df[self.ofamp_rq], self.df[self.chi2_rq], colormap = cmap,
                           f='log', selection='cut_all_'+self.channel_name,
                            colorbar_label = "log(number/bin), Passing Cuts")
                           
            
        plt.xlabel(self.ofamp_rq)
        plt.ylabel(self.chi2_rq)
                           
        plt.title("All Cuts, OFAmp vs. Chi2")
        plt.show()
        
    def plot_ofamp_vs_time(self, lgcdiagnostics=False):
        """
        Shows events passing and failing cut on ofamp vs. chi2 plot
        
        Parameters
        ----------
        
        lgcdiagnostics : bool, optional
            If True, prints out diagnostic statements
        """   
        
        
        #reset selection 
        self.df['_trues'] = np.ones(len(self.df), dtype='bool')
        self.df.select('_trues', mode='replace')
            
        #plot all events
        cmap = copy(mpl.cm.get_cmap('winter') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df['event_time'], self.df[self.ofamp_rq], colormap = cmap,
                            f='log', colorbar_label = "log(number/bin), All Events")
                                
        #plot events passing cuts
        cmap = copy(mpl.cm.get_cmap('spring') )
        cmap.set_bad(alpha = 0.0, color = 'Black')
        self.df.viz.heatmap(self.df['event_time'], self.df[self.ofamp_rq], colormap = cmap,
                           f='log', selection='cut_all_'+self.channel_name,
                            colorbar_label = "log(number/bin), Passing Cuts")
                            

                           
        plt.title("All Cuts, OFAmp vs. Event Time")
        plt.show()
