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


__all__ = ['ScatterPlotter']


class ScatterPlotter:
    """
    This class is designed to allow you to pick out events
    from a scatter plot with your mouse and then plot them easily.
    """
    
    def __init__(self, df, path_to_data, rq_1, rq_2,
                 label_1, label_2, title=None, 
                 trace_inds_to_plot=None, trace_inds_labels=None,
                 selection=None):
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
        """
        self.df = df
        self.path_to_data = path_to_data
        self.picked_inds = []
        
        self.rq_1 = rq_1
        self.rq_2 = rq_2
        self.label_1 = label_1
        self.label_2 = label_2
        self.title = title
                 
        self.fs = None
        
        self.trace_inds_to_plot = trace_inds_to_plot
        self.trace_inds_labels = trace_inds_labels
        
        self.selection = selection
        
    def _onpick(self, event):
        """
        Used as the matplotlib picker function
        """
        ind = np.asarray(event.ind)[0]
        self.picked_inds.append(ind)
        
    def _get_trace(self, df_index,
                   trace_length_msec=None,
                   pretrigger_length_msec=None):
        """
        Lighter weight function to retrive a trace so it can be plotted

        Parameters
        ----------

        index : int
            The index (in the vaex dataframe) of the trace being retrived

        """
        
        # instantiate h5reader
        h5 = h5io.H5Reader()

        # get list of columns
        column_list = self.df.get_column_names()
        
        # series and dump number, and event number
        dump_number = int(self.df[self.df.index==df_index].dump_number.values[0])
        series_number = int(
            self.df[self.df.index==df_index].series_number.values[0]
        )
        event_number = int(
            self.df[self.df.index==df_index].event_number.values[0]
        )
        
        # event index
        event_index = None
        if 'event_index' in  column_list:
            event_index = int(self.df[self.df.index==df_index].event_index.values[0])
        else:
            event_index = event_number%100000
            
        # trigger index
        trigger_index = None
        if 'trigger_index' in column_list:
            trigger_index = int(
                self.df[self.df.index==df_index].trigger_index.values[0]
            )
            

        # file name
        series_name = h5io.extract_series_name(series_number)
        file_list = glob(self.path_to_data +'/*_' + series_name
                         + '_F' + str(dump_number).zfill(4)
                         + '.hdf5')
        if len(file_list) != 1:
            raise ValueError('ERROR: No raw data found')
        file_name = file_list[0]


        # sample rate (if None)
        if self.fs is None:
            info =  h5.get_metadata(file_name=file_name)
            adc_name = info['adc_list'][0]
            self.fs = float(info['groups'][adc_name]['sample_rate'])
                                

        # convert trace length to samples
        nb_samples = None
        nb_pretrigger_samples = None
        if trace_length_msec is not None:
            nb_samples = int(
                round(trace_length_msec*self.fs*1e-3)
            )
            nb_pretrigger_samples = nb_samples//2

        if pretrigger_length_msec is not None:
            nb_pretrigger_samples = int(
                round(pretrigger_length_msec*self.fs*1e-3)
            )

        if nb_samples is None:
            trigger_index = None
            
    
        # get trace
        trace  = h5.read_single_event(
            event_index=event_index,
            file_name=file_name,
            trigger_index=trigger_index,
            trace_length_samples=nb_samples,
            pretrigger_length_samples=nb_pretrigger_samples,
            adctoamp=True)
        
        return np.asarray(trace)
    
        
    def plot_picking_scatter(self):
        mpl.rcParams['figure.figsize'] = [8, 5.5]
        fig, ax = plt.subplots()
        
        vals1 = self.df[self.rq_1].values
        vals2 = self.df[self.rq_2].values
        if self.selection is not None:
            vals1 = self.df[self.selection][self.rq_1].values
            vals2 = self.df[self.selection][self.rq_2].values
            
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
        while i < len(self.picked_inds):
            fig, axs = plt.subplots(2)
            fig.subplots_adjust(hspace=0.5)
            
            vals1 = self.df[self.rq_1].values
            vals2 = self.df[self.rq_2].values
            if self.selection is not None:
                vals1 = self.df[self.selection][self.rq_1].values
                vals2 = self.df[self.selection][self.rq_2].values
            
            axs[0].scatter(vals1,
                        vals2, 
                        s = 3, picker = True, pickradius = 5,
                        )
            
            axs[0].scatter(self.df[self.rq_1].values[self.picked_inds[i]],
                       self.df[self.rq_2].values[self.picked_inds[i]],
                      label = "Picked Point", marker = "*")
            
            axs[0].set_xlabel(self.label_1)
            axs[0].set_ylabel(self.label_2)
            axs[0].legend()
            
            if self.title is not None:
                axs[0].set_title(self.title)

            retrived_traces = self._get_trace(
                self.picked_inds[i],
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
            
