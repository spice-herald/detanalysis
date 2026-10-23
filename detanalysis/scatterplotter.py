import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import vaex as vx
import os
import math
from pprint import pprint
import importlib
from inspect import getmembers, isfunction
from copy import copy
import qetpy as qp

from .analyzer import read_traces, records_from_dataframe


__all__ = ['ScatterPlotter']


class ScatterPlotter:
    """
    This class is designed to allow you to pick out events
    from a scatter plot with your mouse and then plot them easily.
    """
    
    def __init__(self, df, path_to_data, rq_1, rq_2,
                 label_1, label_2, title=None, 
                 trace_inds_to_plot=None, trace_inds_labels=None,
                 selection=None, channels=None):
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
            
        selection : vaex selection string, optional
            Vaex selection string for the displayed data.
            
        channels : str or list of str, optional
            Detector channel name(s) to read. If None, all channels are read,
            which fails for a channel that cannot be converted to amps (a TTL
            channel, for instance). "trace_inds_to_plot" selects within the
            channels that were read.
        """
        #self.df = df
        self.path_to_data = path_to_data
        self.picked_index_arr = []
        self.picked_inds = []
        
        self.rq_1 = rq_1
        self.rq_2 = rq_2
        self.label_1 = label_1
        self.label_2 = label_2
        self.title = title
                 
        self.fs = None
        
        self.trace_inds_to_plot = trace_inds_to_plot
        self.trace_inds_labels = trace_inds_labels
        self.channels = channels
        
        self.selection = selection
        
        if selection is not None:
            self.df = df[df[selection]]
        else:
            self.df = df
            
        self.index = self.df.index.values
        
    def _onpick(self, event):
        """
        Used as the matplotlib picker function
        """
        ind = np.asarray(event.ind)[0]
        self.picked_inds.append(ind)
        self.picked_index_arr.append(self.index[ind])
        
    def _get_trace(self, df_index,
                   trace_length_msec=None,
                   pretrigger_length_msec=None):
        """
        Lighter weight function to retrive a trace so it can be plotted

        Parameters
        ----------

        index : int
            The index (in the vaex dataframe) of the trace being retrived

        Returns
        -------
        trace : ndarray[channel, sample]
            Trace of the selected event, in amps.
        """

        if trace_length_msec is None:
            raise ValueError(
                'ERROR: "trace_length_msec" is required. Events are located by '
                'a trigger coordinate in the raw stream, so the window to read '
                'around it has to be given.'
            )

        # locate the event in the raw data
        df_event = self.df[self.df.index==df_index]
        record_list = records_from_dataframe(df_event)

        # get trace
        traces, info = read_traces(
            self.path_to_data,
            record_list,
            channels=self.channels,
            trace_length_msec=trace_length_msec,
            pretrigger_length_msec=pretrigger_length_msec,
        )

        # sample rate (if None)
        if self.fs is None:
            self.fs = float(info[0]['sample_rate_hz'])

        return traces[0]
    
        
    def plot_picking_scatter(self):
    
        if len(self.picked_inds) != len(self.picked_index_arr):
            self.picked_inds = []
            self.picked_index_arr = []
        
        mpl.rcParams['figure.figsize'] = [8, 5.5]
        fig, ax = plt.subplots()
        
        vals1 = self.df[self.rq_1].values
        vals2 = self.df[self.rq_2].values
        ax.scatter(vals1,
                    vals2, 
                    s = 3, picker = True, pickradius = 5,
                    )
        
        ax.set_xlabel(self.label_1)
        ax.set_ylabel(self.label_2)
        
        if self.title is not None:
            ax.set_title(self.title)
        
        fig.canvas.mpl_connect('pick_event', self._onpick)
        plt.show()
        
    def plot_picked_events(self, lpcutoff=None,
                           trace_length_msec=None,
                           pretrigger_length_msec=None):
        
        mpl.rcParams['figure.figsize'] = [6, 9]
        
        i = 0
        while i < len(self.picked_index_arr):
            fig, axs = plt.subplots(2)
            fig.subplots_adjust(hspace=0.5)
            
            
            vals1 = self.df[self.rq_1].values
            vals2 = self.df[self.rq_2].values
            
            axs[0].scatter(vals1,
                        vals2, 
                        s = 3, picker = True, pickradius = 5,
                        )
            
            axs[0].scatter(vals1[self.picked_inds[i]],
                       vals2[self.picked_inds[i]],
                      label = "Picked Point", marker = "*")
            
            axs[0].set_xlabel(self.label_1)
            axs[0].set_ylabel(self.label_2)
            axs[0].legend()
            
            if self.title is not None:
                axs[0].set_title(self.title)

            retrived_traces = self._get_trace(
                self.picked_index_arr[i],
                trace_length_msec=trace_length_msec,
                pretrigger_length_msec=pretrigger_length_msec)
            
            dt = 1/self.fs
            t_arr = np.asarray(list(range(retrived_traces.shape[-1])))*dt
                      
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
                        
                    axs[1].plot(t_arr*1e3 - pretrigger_length_msec, trace_to_plot, label=label_)
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
                        
                    axs[1].plot(t_arr*1e3 - pretrigger_length_msec, trace_to_plot, label=label_)
                    j += 1
                    
            axs[1].set_xlabel("Time Since Trigger (ms)")
            axs[1].set_ylabel("Trace Amplitude (amps)")
            axs[1].grid()
            axs[1].legend()
            
            title_str = "Event " + str(self.picked_index_arr[i]) + ", \n" + str(self.label_1) + ": " 
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
            
