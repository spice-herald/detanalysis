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

__all__ = ['ScatterPlotter']

h5 = h5io.H5Reader()

class ScatterPlotter:
    """
    This class is designed to allow you to pick out events
    from a scatter plot with your mouse and then plot them easily.
    """
    
    def __init__(self, df, path_to_data, rq_1, rq_2,
                   label_1, label_2, title=None, 
                fs=int(1.25e6),
                trace_inds_to_plot=None, trace_inds_labels=None):
        """
        Initialize the ScatterPlotter class.
        
        Attributes
        ----------
        
        df : vaex dataframe
            The dataframe where the data is stored.
            
        path_to_data : str
            The path to the locaiton where the triggered
            traces are stored.
            
        rq_1 : str
            The string containing the name of the column name
            of the dataframe to plot on the x axis
            
        rq_2 : str
            The string containing the name of the column name
            of the dataframe to plot on the x axis
            
        label_1 : str
            The label for the x axis of the scatter plots
            
        label_2 : str
            The label for the x axis of the scatter plots
            
        title : str, optional
            If not None, used as a title for all scatter
            plots
            
        fs : int, optional
            Used to make the time array for the plotted events.
            Assumed to be 1.25 MHz unless specified
            
        trace_inds_to_plot : array of ints
            Array of the channels to plot when individual traces
            are plotted, for example [1, 3]. If None, all traces
            are plotted.
            
        trace_inds_labels : array of strs
            Array of the labels for the traces that are plotted, e.g.
            ['Melange1pc1ch', 'Melange25pcRight']. If None, the traces
            are not labeled
        """
        self.df = df
        self.path_to_data = path_to_data
        self.picked_inds = []
        
        self.rq_1 = rq_1
        self.rq_2 = rq_2
        self.label_1 = label_1
        self.label_2 = label_2
        self.title = title
                 
        self.fs = fs
        
        self.trace_inds_to_plot = trace_inds_to_plot
        self.trace_inds_labels = trace_inds_labels
        
    def _onpick(self, event):
        """
        Used as the matplotlib picker function
        """
        ind = np.asarray(event.ind)[0]
        self.picked_inds.append(ind)
        
    def _get_trace(self, index):
        """
        Lighter weight function to retrive a trace so it can be plotted

        Parameters
        ----------

        index : int
            The index (in the vaex dataframe) of the trace being retrived

        """
        #df.select(df.index == index)

        dump_number = int(self.df[self.df.index == index].dump_number.values[0])
        series_number = int(self.df[self.df.index == index].series_number.values[0])
        event_index = int(self.df[self.df.index == index].event_index.values[0])

        if self.df[self.df.index == index].trigger_type.values == 4.0:
            random_truth = False
        if self.df[self.df.index == index].trigger_type.values == 3.0:
            random_truth = True

        if bool(random_truth):
            event_prefix = '/rand_I2_D'
        else:
            event_prefix = '/threshtrig_I2_D'

        file_name = self.path_to_data + event_prefix + str(series_number)[1:-6]  + "_T" + str(series_number)[-6:] + "_F" + str(dump_number).zfill(4) + ".hdf5"

        trace = h5.read_single_event(event_index = event_index, file_name = file_name, adctoamp = True)
        return np.asarray(trace)
    
        
    def plot_picking_scatter(self):
        mpl.rcParams['figure.figsize'] = [8, 5.5]
        fig, ax = plt.subplots()
        ax.scatter(self.df[self.rq_1].values,
                    self.df[self.rq_2].values, 
                    s = 3, picker = True, pickradius = 5)
        ax.set_xlabel(self.label_1)
        ax.set_ylabel(self.label_2)
        
        if self.title is not None:
            ax.set_title(self.title)
        
        fig.canvas.mpl_connect('pick_event', self._onpick)
        plt.show()
        
    def plot_picked_events(self, lpcutoff=None):
        mpl.rcParams['figure.figsize'] = [6, 9]
        
        i = 0
        while i < len(self.picked_inds):
            fig, axs = plt.subplots(2)
            fig.subplots_adjust(hspace=0.5)
            
            axs[0].scatter(self.df[self.rq_1].values,
                       self.df[self.rq_2].values,
                      label = "All Data", s = 5)
            
            axs[0].scatter(self.df[self.rq_1].values[self.picked_inds[i]],
                       self.df[self.rq_2].values[self.picked_inds[i]],
                      label = "Picked Point", marker = "*")
            
            axs[0].set_xlabel(self.label_1)
            axs[0].set_ylabel(self.label_2)
            axs[0].legend()
            
            if self.title is not None:
                axs[0].set_title(self.title)

                
            retrived_traces = self._get_trace(self.picked_inds[i])
            t_arr = np.arange(1/self.fs, len(retrived_traces[0])/self.fs, 1/self.fs)
            
            if self.trace_inds_to_plot is None:
                j = 0
                while j < len(retrived_traces):
                    if self.trace_inds_labels is None:
                        label_ = "Channel " + str(j)
                    else:
                        label_ = self.trace_inds_labels[j]
                    
                    if lpcutoff is None:
                        trace_to_plot = retrived_traces[j]
                    else:
                        trace_to_plot = qp.utils.lowpassfilter(retrived_traces[j], lpcutoff,
                                                               fs=self.fs)
                        
                    axs[1].plot(t_arr, trace_to_plot, label=label_)
                    j += 1
            else:
                j = 0
                while j < len(self.trace_inds_to_plot):
                    if self.trace_inds_labels is None:
                        label_ = "Channel " + str(j)
                    else:
                        label_ = self.trace_inds_labels[j]
                      
                    if lpcutoff is None:
                        trace_to_plot = retrived_traces[self.trace_inds_to_plot[j]]
                    else:
                        trace_to_plot = qp.utils.lowpassfilter(retrived_traces[self.trace_inds_to_plot[j]],
                                                               lpcutoff,
                                                               fs=self.fs)
                        
                    axs[1].plot(t_arr, trace_to_plot, label=label_)
                    j += 1
                    
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Trace Amplitude (amps)")
            axs[1].legend()
            
            title_str = "Event " + str(self.picked_inds[i]) + ", \n" + str(self.label_1) + ": " 
            title_str += str(self.df[self.rq_1].values[self.picked_inds[i]]) + ", \n"
            title_str += str(self.label_2) + ": " 
            title_str += str(self.df[self.rq_2].values[self.picked_inds[i]]) + ", "
            if lpcutoff is not None:
                title_str += "\n " + str(lpcutoff*1e-3) + " kHz Low Pass Filtered"
            axs[1].set_title(title_str)
            
            plt.show()
            
            print(" ")
            print("-----------------")
            print(" ")
            
            i += 1
            
