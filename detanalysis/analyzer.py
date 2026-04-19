import math
import os
import uuid
import importlib
from glob import glob
from pprint import pprint
from inspect import getmembers, isfunction

import git
import numpy as np
import pandas as pd
import qetpy as qp
import pytesio as h5io
import vaex as vx
from git import GitCommandError
from matplotlib import pyplot as plt


__all__ = ["Analyzer"]


class Analyzer:
    """
    Analyze features (RQs) stored in HDF5 files using Vaex.

    Supported cut inputs
    --------------------
    - None
    - registered cut name / boolean column name (str)
    - Vaex expression
    - NumPy boolean array
    """

    def __init__(
        self,
        paths,
        series=None,
        analysis_repo=None,
        load_from_pandas=False,
        memory_cache_size="1GB",
    ):
        # file info
        self._file_list = None
        self._nfiles = None

        # dataframe info
        self._df_full = None
        self._df = None
        self._is_df_filtered = False
        self._current_filter_mask = None
        self._filter_version = 0

        self._nevents = None
        self._nevents_nofilter = None
        self._nfeatures = None
        self._feature_names = None
        self._load_from_pandas = load_from_pandas

        # metadata
        self._cuts = {}
        self._derived_features = {}

        # analysis repo
        self._analysis_repo_path = analysis_repo
        self._analysis_repo = None

        # load data
        self.add_files(paths, series=series, load_from_pandas=load_from_pandas)

        # vaex cache
        vx.cache.memory()
        vx.settings.cache.memory_size_limit = memory_cache_size

        if analysis_repo is not None:
            self.set_analysis_repo(analysis_repo, load_func=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def df(self):
        return self._df

    @property
    def df_full(self):
        return self._df_full

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

    # ------------------------------------------------------------------
    # Info / display
    # ------------------------------------------------------------------

    def list_cuts(self):
        return list(self._cuts.keys())

    def describe(self):
        print(f"Number of files: {self._nfiles}")
        print(f"Number of events in current df: {self._nevents}")
        print(f"Number of events in full df: {self._nevents_nofilter}")
        print(f"Number of features: {self._nfeatures}")
        print(f"Is DataFrame filtered? {self._is_df_filtered}")

        if self._cuts:
            print("Cuts:")
            pprint(self._cuts)
        else:
            print("No cuts have been registered!")

        if self._derived_features:
            print("Derived features:")
            pprint(self._derived_features)
        else:
            print("No derived features have been added!")

    def get_unit(self, feature_exp):
        return self._df.unit(feature_exp)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_column(self, name, df=None):
        if df is None:
            df = self._df_full
        return name in df.get_column_names()

    def _is_numpy_cut(self, cut):
        return isinstance(cut, np.ndarray)

    def _normalize_numpy_cut(self, cut, expected_length):
        arr = np.asarray(cut)
        if arr.dtype != bool:
            arr = arr.astype(bool)

        if len(arr) != expected_length:
            raise ValueError(
                f"Cut array length ({len(arr)}) does not match expected length ({expected_length})."
            )
        return arr

    def _drop_column_if_exists(self, name, df=None):
        if df is None:
            df = self._df_full
        if self._has_column(name, df=df):
            try:
                del df[name]
            except Exception:
                pass

    def _add_array_column(self, array, array_name, overwrite=True, df=None):
        if df is None:
            df = self._df_full

        array = np.asarray(array)

        if len(array) != len(df):
            raise ValueError(
                f'Unable to add column "{array_name}": array length ({len(array)}) '
                f"!= dataframe length ({len(df)})."
            )

        if self._has_column(array_name, df=df) and not overwrite:
            raise ValueError(f'Column "{array_name}" already exists.')

        df[array_name] = array

    def _resolve_cut_reference(self, cut, df=None):
        """
        Resolve:
          - registered cut name -> boolean column expression on df
          - boolean column name -> boolean column expression on df
          - otherwise -> pass through
        """
        if df is None:
            df = self._df

        if isinstance(cut, str):
            if cut in self._cuts and self._has_column(cut, df=self._df_full):
                return df[cut]
            if self._has_column(cut, df=self._df_full):
                return df[cut]
        return cut

    def _subset_df(self, df, cut=None):
        """
        Return df or a subset of df using supported cut inputs.

        Supported cut inputs:
          - None
          - NumPy boolean mask with len(df)
          - registered cut name / boolean column name
          - string Vaex expression
          - Vaex boolean expression
        """
        if cut is None:
            return df

        if self._is_numpy_cut(cut):
            arr = self._normalize_numpy_cut(cut, expected_length=len(df))
            tmp_name = f"__tmp_cut__{uuid.uuid4().hex}"
            df_tmp = df.copy()
            df_tmp[tmp_name] = arr
            return df_tmp[df_tmp[tmp_name]]

        if isinstance(cut, str):
            # registered cut name or existing boolean column
            if cut in self._cuts or self._has_column(cut, df=self._df_full):
                return df[df[cut]]

            # otherwise treat as expression string
            return df.filter(cut)

        # Vaex boolean expression
        return df[cut]

    def _full_mask_from_selection_on_df(self, selection, df):
        """
        Convert a cut/expression on `df` into a full-length mask on `_df_full`
        using the permanent `__event_index__`.
        """
        selected_df = self._subset_df(df, selection)
        selected_event_ids = np.asarray(selected_df.evaluate("__event_index__"))
        full_event_ids = np.asarray(self._df_full.evaluate("__event_index__"))
        return np.isin(full_event_ids, selected_event_ids)

    def _full_mask_from_widget_default_selection(self, df):
        """
        Convert Vaex widget 'default' selection on df into a full-length mask on _df_full.
        """
        selected_event_ids = np.asarray(df.evaluate("__event_index__", selection="default"))
        full_event_ids = np.asarray(self._df_full.evaluate("__event_index__"))
        return np.isin(full_event_ids, selected_event_ids)

    def _materialize_expr_to_full_mask(self, cut):
        """
        Convert supported cut input into a full-length boolean mask in `_df_full` row space.
        """
        if self._is_numpy_cut(cut):
            return self._normalize_numpy_cut(cut, expected_length=len(self._df_full))

        return self._full_mask_from_selection_on_df(cut, self._df_full)

    def _materialize_cut_to_column(self, cut, column_name, overwrite=True):
        mask = self._materialize_expr_to_full_mask(cut)
        self._add_array_column(mask, column_name, overwrite=overwrite, df=self._df_full)

    def _refresh_df_view(self):
        """
        Rebuild `_df` from `_df_full` and the internal global filter mask.

        `_df_full` stores only persistent columns/cuts/features.
        `_df` is rebuilt from a fresh copy when a global filter is active.
        """
        if self._current_filter_mask is None:
            self._df = self._df_full
            self._is_df_filtered = False
        else:
            self._filter_version += 1
            filter_col = f"__global_filter__{self._filter_version}"

            df_view = self._df_full.copy()
            df_view[filter_col] = self._current_filter_mask
            self._df = df_view[df_view[filter_col]]
            self._is_df_filtered = True

        self._fill_df_info()

    # ------------------------------------------------------------------
    # Core data access
    # ------------------------------------------------------------------

    def get_values(self, feature_exp, cut=None, **kwargs):
        """
        Get values of a feature or expression as a NumPy array from current view.
        """
        df_used = self._subset_df(self._df, cut=cut)
        values = np.asarray(df_used.evaluate(feature_exp, **kwargs))
        return values

    # ------------------------------------------------------------------
    # Cuts
    # ------------------------------------------------------------------

    def register_cut(
        self,
        cut,
        name,
        metadata=None,
        overwrite=False,
    ):
        """
        Register a cut as a boolean column on `_df_full`.
        """
        if not overwrite and name in self._cuts:
            print(
                f'Cut "{name}" already registered! '
                f"Use overwrite=True or choose another name."
            )
            return

        if metadata is None:
            metadata = {}

        self._materialize_cut_to_column(cut, name, overwrite=True)
        self._cuts[name] = metadata
        self._refresh_df_view()

    def register_cut_box(
        self,
        features,
        limits,
        name,
        metadata=None,
        overwrite=False,
    ):
        """
        Register a rectangular cut as a boolean column on `_df_full`.
        """
        if not overwrite and name in self._cuts:
            print(
                f'Cut "{name}" already registered! '
                f"Use overwrite=True or choose another name."
            )
            return

        if metadata is None:
            metadata = {}

        expr = (self._df_full[features[0]] >= limits[0][0]) & (
            self._df_full[features[0]] <= limits[0][1]
        )
        for feat, lim in zip(features[1:], limits[1:]):
            expr = expr & (self._df_full[feat] >= lim[0]) & (self._df_full[feat] <= lim[1])

        self._materialize_cut_to_column(expr, name, overwrite=True)
        self._cuts[name] = metadata
        self._refresh_df_view()

    def save_current_selection(self, name, metadata=None, overwrite=False):
        """
        Save current interactive Vaex default selection from `_df`
        into a boolean column on `_df_full`.
        """
        if not overwrite and name in self._cuts:
            print(
                f'Cut "{name}" already registered! '
                f"Use overwrite=True or choose another name."
            )
            return

        if metadata is None:
            metadata = {}

        mask = self._full_mask_from_widget_default_selection(self._df)
        self._add_array_column(mask, name, overwrite=True, df=self._df_full)
        self._cuts[name] = metadata
        self._refresh_df_view()

    def combine_cuts(self, cut_list, name, op="and", overwrite=False, metadata=None):
        """
        Combine cuts into a new boolean-column cut on `_df_full`.
        """
        if not cut_list:
            raise ValueError("cut_list is empty")

        if not overwrite and name in self._cuts:
            print(
                f'Cut "{name}" already registered! '
                f"Use overwrite=True or choose another name."
            )
            return

        if metadata is None:
            metadata = {}

        masks = [self._materialize_expr_to_full_mask(cut) for cut in cut_list]

        result = masks[0].copy()
        for mask in masks[1:]:
            if op == "and":
                result &= mask
            elif op == "or":
                result |= mask
            elif op == "xor":
                result ^= mask
            elif op == "subtract":
                result &= ~mask
            else:
                raise ValueError(f'Unknown op "{op}"')

        self._add_array_column(result, name, overwrite=True, df=self._df_full)
        self._cuts[name] = metadata
        self._refresh_df_view()

    # ------------------------------------------------------------------
    # Global filtering
    # ------------------------------------------------------------------

    def apply_global_filter(self, cut, mode="replace"):
        new_mask = self._materialize_expr_to_full_mask(cut)

        if mode == "replace" or self._current_filter_mask is None:
            self._current_filter_mask = new_mask.copy()
        elif mode == "and":
            self._current_filter_mask &= new_mask
        elif mode == "or":
            self._current_filter_mask |= new_mask
        elif mode == "xor":
            self._current_filter_mask ^= new_mask
        elif mode == "subtract":
            self._current_filter_mask &= ~new_mask
        else:
            raise ValueError(f'Unknown mode "{mode}"')

        expected = int(np.count_nonzero(self._current_filter_mask))

        self._refresh_df_view()

        actual = len(self._df)
        if actual != expected:
            raise RuntimeError(
                f"Global filter inconsistency: expected {expected} events, got {actual}."
            )

        eff_percent = self._nevents / self._nevents_nofilter * 100
        print("Filter applied!")
        print(f"Number of events after filter: {self._nevents} ({eff_percent:.1f}%)")

    def drop_global_filter(self):
        """
        Drop the current global filter without reloading from disk.
        """
        self._current_filter_mask = None
        self._refresh_df_view()

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------

    def add_feature(self, expression, name, metadata=None, overwrite=False):
        """
        Add a virtual column to `_df_full`.
        """
        if not overwrite and name in self._derived_features:
            print(
                f'Feature "{name}" already added! '
                f"Use overwrite=True or choose another name."
            )
            return

        if metadata is None:
            metadata = {}

        self._df_full.add_virtual_column(name, expression)
        self._derived_features[name] = metadata
        self._refresh_df_view()

    # ------------------------------------------------------------------
    # Load cut / feature functions
    # ------------------------------------------------------------------

    def load_cuts(self, cuts_path=None, overwrite=False):
        if cuts_path is None:
            if self._analysis_repo_path is None:
                print("WARNING: No path to cut scripts found! No cuts added.")
                return

            cuts_path = os.path.join(self._analysis_repo_path, "cuts")
            if not os.path.isdir(cuts_path):
                for dir_tuple in os.walk(self._analysis_repo_path):
                    if dir_tuple and "cuts" in dir_tuple[1]:
                        cuts_path = os.path.join(dir_tuple[0], "cuts")
                        break

        if not os.path.isdir(cuts_path):
            print(f"WARNING: No cut directory found in {cuts_path}!")
            return

        repo_info = self._get_repo_info()
        self._load_func(cuts_path, is_cut=True, repo_info=repo_info, overwrite=overwrite)

    def load_derived_features(self, features_path=None, overwrite=False):
        if features_path is None:
            if self._analysis_repo_path is None:
                print("WARNING: No path to feature scripts found! No features added.")
                return

            features_path = os.path.join(self._analysis_repo_path, "features")
            if not os.path.isdir(features_path):
                for dir_tuple in os.walk(self._analysis_repo_path):
                    if dir_tuple and "features" in dir_tuple[1]:
                        features_path = os.path.join(dir_tuple[0], "features")
                        break

        if not os.path.isdir(features_path):
            print(f"WARNING: No feature directory found in {features_path}!")
            return

        repo_info = self._get_repo_info()
        self._load_func(features_path, is_cut=False, repo_info=repo_info, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Data loading / reset
    # ------------------------------------------------------------------

    def clean(self):
        """
        Reload from original files and rebuild the master dataframe.

        Note:
        This removes in-memory cuts/features/columns unless you re-add them.
        """
        self.add_files(
            self._file_list,
            load_from_pandas=self._load_from_pandas,
            replace=True,
        )

    def add_files(self, paths, series=None, load_from_pandas=False, replace=False):
        files = self._extract_file_names(paths, series=series)

        if replace or self._file_list is None or not self._file_list:
            self._file_list = files
        else:
            self._file_list.extend(files)

        self._file_list = sorted(set(self._file_list))
        self._nfiles = len(self._file_list)

        if load_from_pandas:
            df_full = None
            for afile in self._file_list:
                vaex_df = vx.from_pandas(pd.read_hdf(afile, "detprocess_df"))
                if df_full is None:
                    df_full = vaex_df
                else:
                    df_full = vx.concat([df_full, vaex_df])
            self._df_full = df_full
        else:
            self._df_full = vx.open_many(self._file_list)

        # Permanent full-data event index
        self._df_full["__event_index__"] = np.arange(
            0, len(self._df_full), 1, dtype=np.int64
        )

        self._current_filter_mask = None
        self._filter_version = 0
        self._df = self._df_full
        self._is_df_filtered = False

        self._fill_df_info()
        self._nevents_nofilter = len(self._df_full)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def hist(
        self,
        feature_x,
        cuts=None,
        shape=64,
        limits="minmax",
        normalize=None,
        logx=False,
        logy=True,
        figsize=(9, 6),
        colors=None,
        colormap=None,
        title=None,
        labels=None,
        xlabel=None,
        ylabel=None,
        what="count(*)",
        ax=None,
        **kwargs,
    ):
        if cuts is None:
            cuts = [None]
        elif not isinstance(cuts, list):
            cuts = [cuts]

        ncuts = len(cuts)

        if normalize is not None:
            normalize = "normalize" if normalize else None

        if colors is not None:
            if not isinstance(colors, list):
                colors = [colors]
            if len(colors) != ncuts:
                raise ValueError(f'ERROR: "colors" should have length {ncuts}!')
        else:
            colors = ["blue", "red", "green", "cyan", "magenta", "yellow"]
            if ncuts > len(colors) or colormap is not None:
                if colormap is None:
                    colormap = "viridis"
                colors = plt.cm.get_cmap(colormap)(np.linspace(0.1, 0.9, ncuts))
            else:
                colors = colors[:ncuts]

        if labels is not None:
            if not isinstance(labels, list):
                labels = [labels]
            if len(labels) != ncuts:
                raise ValueError(f'ERROR: "labels" should have length {ncuts}!')

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        kwargs.setdefault("linewidth", 2)

        for icut, cut in enumerate(cuts):
            label = labels[icut] if labels is not None else None
            kwargs["color"] = colors[icut]
            df_used = self._subset_df(self._df, cut=cut)
            df_used.viz.histogram(
                feature_x,
                shape=shape,
                limits=limits,
                n=normalize,
                xlabel=xlabel,
                ylabel=ylabel,
                label=label,
                **kwargs,
            )

        ax.tick_params(which="both", direction="in", right=True, top=True)
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

    def heatmap(
        self,
        feature_x,
        feature_y,
        cut=None,
        what="count(*)",
        f="log",
        shape=256,
        xlimits=None,
        ylimits=None,
        limits=None,
        colormap="plasma",
        figsize=(9, 6),
        title=None,
        xlabel=None,
        ylabel=None,
        ax=None,
        **kwargs,
    ):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if xlimits is not None or ylimits is not None:
            if limits is not None:
                raise ValueError(
                    '"limits" cannot be used at the same time as "xlimits/ylimits".'
                )
            limits = [xlimits, ylimits]

        df_used = self._subset_df(self._df, cut=cut)
        df_used.viz.heatmap(
            feature_x,
            feature_y,
            colormap=colormap,
            shape=shape,
            f=f,
            limits=limits,
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            what=what,
            **kwargs,
        )

        ax.tick_params(which="both", direction="in", right=True, top=True)
        ax.grid(linestyle="dashed")

        if title is not None:
            ax.set_title(title)

        return fig, ax

    def scatter(
        self,
        feature_x,
        feature_y,
        cuts=None,
        figsize=(9, 6),
        ms=5,
        alpha=0.8,
        xlimits=None,
        ylimits=None,
        colors=None,
        colormap=None,
        title=None,
        labels=None,
        xlabel=None,
        ylabel=None,
        nb_random_samples=None,
        length_check=True,
        ax=None,
        **kwargs,
    ):
        if cuts is None:
            cuts = [None]
        elif not isinstance(cuts, list):
            cuts = [cuts]

        ncuts = len(cuts)

        if colors is not None:
            if not isinstance(colors, list):
                colors = [colors]
            if len(colors) != ncuts:
                raise ValueError(f'ERROR: "colors" should have length {ncuts}!')
        else:
            colors = ["blue", "red", "green", "cyan", "magenta", "yellow"]
            if ncuts > len(colors) or colormap is not None:
                if colormap is None:
                    colormap = "viridis"
                colors = plt.cm.get_cmap(colormap)(np.linspace(0.1, 0.9, ncuts))
            else:
                colors = colors[:ncuts]

        if labels is not None:
            if not isinstance(labels, list):
                labels = [labels]
            if len(labels) != ncuts:
                raise ValueError(f'ERROR: "labels" should have length {ncuts}!')

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        base_df = self._df
        if nb_random_samples is not None and nb_random_samples < len(base_df):
            base_df = base_df.sample(n=nb_random_samples)

        for icut, cut in enumerate(cuts):
            label = labels[icut] if labels is not None else None
            color = colors[icut]
            df_used = self._subset_df(base_df, cut=cut)
            df_used.viz.scatter(
                feature_x,
                feature_y,
                xlabel=xlabel,
                ylabel=ylabel,
                label=label,
                color=color,
                s=ms,
                alpha=alpha,
                length_check=length_check,
                **kwargs,
            )

        if xlimits is not None:
            ax.set_xlim(xlimits)
        if ylimits is not None:
            ax.set_ylim(ylimits)

        ax.tick_params(which="both", direction="in", right=True, top=True)
        ax.grid(linestyle="dashed")

        if title is not None:
            ax.set_title(title)
        if labels is not None:
            ax.legend(markerscale=2, framealpha=0.9, loc="best")

        return fig, ax

    def interactive_selection(self, feature_x, feature_y, **kwargs):
        """
        Return Vaex interactive heatmap widget on the current working dataframe.
        If a global filter is active, selection is done only on the filtered view.
        """
        return self._df.widget.heatmap(feature_x, feature_y, **kwargs)

    # ------------------------------------------------------------------
    # Analysis repo
    # ------------------------------------------------------------------

    def set_analysis_repo(self, repo_path, load_func=True):
        try:
            self._analysis_repo = git.Repo(repo_path)
            self._analysis_repo_path = self._analysis_repo.working_dir
        except git.exc.GitError:
            print(f'\nWARNING: analysis repo "{repo_path}" is not a git repository!')
            self._analysis_repo_path = repo_path

        if load_func:
            self.load_derived_features()
            self.load_cuts()

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def plot_traces(
        self,
        channels,
        raw_path,
        cut=None,
        trace_length_msec=None,
        trace_length_samples=None,
        pretrigger_length_msec=None,
        pretrigger_length_samples=None,
        nb_random_samples=None,
        figsize=None,
        colors=None,
        colormap=None,
        single_plot=False,
        baselinesub=True,
        baselineinds=(5, 100),
        lpcutoff=None,
        nb_events_limit=100,
    ):
        max_traces = nb_events_limit if single_plot else min(nb_events_limit, 20)

        if isinstance(channels, str):
            channels = [channels]

        traces, info = self.get_traces(
            channels,
            trace_length_msec=trace_length_msec,
            trace_length_samples=trace_length_samples,
            pretrigger_length_msec=pretrigger_length_msec,
            pretrigger_length_samples=pretrigger_length_samples,
            raw_path=raw_path,
            cut=cut,
            nb_random_samples=nb_random_samples,
            nb_events_limit=max_traces,
            baselinesub=baselinesub,
            baselineinds=baselineinds,
        )

        if traces is None:
            return None, None

        nb_events, nb_channels, _ = traces.shape
        nrows = math.ceil(nb_events / 2)
        ncols = 1 if nb_events == 1 else 2

        fs = info[0]["sample_rate"]
        dt = 1 / fs

        if lpcutoff is not None:
            for ichan in range(nb_channels):
                traces[:, ichan, :] = qp.utils.lowpassfilter(
                    traces[:, ichan, :],
                    lpcutoff,
                    fs=fs,
                )

        if single_plot and nb_channels > 1 and nb_events > 1:
            print(
                "WARNING: Unable to plot multiple channels for multiple events "
                "on a single figure. Switching to multiple plots."
            )
            single_plot = False

        if colors is not None:
            if not isinstance(colors, list):
                colors = [colors]
            if nb_channels > 1 and len(colors) < nb_channels:
                raise ValueError(f'ERROR: "colors" should have length at least {nb_channels}!')
            if nb_channels == 1 and len(colors) < nb_events:
                raise ValueError(f'ERROR: "colors" should have length at least {nb_events}!')
        else:
            colors = ["blue", "red", "green", "cyan", "magenta", "yellow"]
            if nb_channels == 1 and not single_plot:
                colors = ["blue"] * nb_events
            nb_colors = max(nb_events, nb_channels)
            if colormap is not None or len(colors) < nb_colors:
                if colormap is None:
                    colormap = "plasma"
                colors = plt.cm.get_cmap(colormap)(np.linspace(0.1, 0.9, nb_colors))

        if single_plot:
            if figsize is None:
                figsize = (9, 6)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            if figsize is None:
                figsize = (11, 14)
                if nb_events <= 2:
                    figsize = (11, 6)
                elif nb_events <= 4:
                    figsize = (11, 8)
                elif nb_events <= 6:
                    figsize = (11, 11)

            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        bins = np.arange(traces.shape[-1]) * dt * 1000

        if not single_plot:
            ax = np.ravel(ax)

        for it in range(nb_events):
            ax_it = ax if single_plot else ax[it]
            for ichan in range(nb_channels):
                it_color = ichan if nb_channels > 1 else it
                ax_it.plot(
                    bins,
                    traces[it, ichan, :] * 1e6,
                    color=colors[it_color],
                    label=channels[ichan],
                )
            ax_it.legend()
            ax_it.set_xlabel("Time [ms]")
            ax_it.set_ylabel("Amplitude [uA]")

        return fig, ax

    def get_event_list(
        self,
        cut=None,
        nb_random_samples=None,
        nb_events_limit=5000,
    ):
        df_used = self._subset_df(self._df, cut=cut)

        if len(df_used) == 0:
            print("WARNING: No events found!")
            return None

        # limit the number of events
        if  nb_random_samples is None:
            nb_random_samples = nb_events_limit

        
        if nb_random_samples is not None and nb_random_samples < len(df_used):
            df_used = df_used.sample(n=nb_random_samples)

        nb_events = len(df_used)

        if nb_events > nb_random_samples:
            print(f'WARNING: Number of events is limited to {nb_events_limit}. Change limit '
                  f'or use  "nb_random_samples" if needed')
            
        if "eventnumber" in self._feature_names:
            series_nums = df_used.seriesnumber.values
            event_nums = df_used.eventnumber.values
            group_names = None
            trigger_indices = None
        else:
            series_nums = df_used.series_number.values
            event_nums = df_used.event_number.values
            group_names = (
                df_used.group_name.values if "group_name" in self._feature_names else None
            )
            trigger_indices = (
                df_used.trigger_index.values if "trigger_index" in self._feature_names else None
            )

        event_list = []
        for ievent in range(nb_events):
            event_dict = {}
            if series_nums is not None:
                event_dict["series_number"] = series_nums[ievent]
            if event_nums is not None:
                event_dict["event_number"] = event_nums[ievent]
            if group_names is not None:
                event_dict["group_name"] = group_names[ievent]
            if trigger_indices is not None:
                event_dict["trigger_index"] = trigger_indices[ievent]
            event_list.append(event_dict)

        print(f"INFO: Number of events found = {len(event_list)}")
        return event_list

    def get_traces(
        self,
        channels,
        raw_path,
        trace_length_msec=None,
        trace_length_samples=None,
        pretrigger_length_msec=None,
        pretrigger_length_samples=None,
        cut=None,
        nb_random_samples=None,
        nb_events_limit=1000,
        memory_limit=2,
        baselinesub=False,
        baselineinds=(5, 100),
    ):
        event_list = self.get_event_list(
            cut=cut,
            nb_random_samples=nb_random_samples,
            nb_events_limit=nb_events_limit,
        )

        if event_list is None:
            return None, None

        h5 = h5io.H5Reader()
        traces, info = h5.read_many_events(
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
            baselineinds=baselineinds,
        )
        h5.clear()
        return traces, info

    # ------------------------------------------------------------------
    # Internal info / file handling
    # ------------------------------------------------------------------

    def _fill_df_info(self):
        try:
            self._feature_names = self._df.get_column_names()
            self._nevents = len(self._df)
            self._nfeatures = len(self._feature_names)
        except Exception:
            print("Oops... Something went wrong while refreshing dataframe info.")

    def _extract_file_names(self, paths, series=None):
        if not isinstance(paths, list):
            paths = [paths]

        file_list = []

        for a_path in paths:
            if os.path.isdir(a_path):
                if series is not None:
                    if series in ("even", "odd"):
                        file_list.extend(glob(os.path.join(a_path, f"{series}_*.hdf5")))
                    else:
                        series_list = [series] if not isinstance(series, list) else series
                        for it_series in series_list:
                            file_list.extend(
                                glob(os.path.join(a_path, f"*{it_series}_*.hdf5"))
                            )
                else:
                    file_list.extend(glob(os.path.join(a_path, "*.hdf5")))

            elif os.path.isfile(a_path):
                if ".hdf5" in a_path:
                    if series is not None:
                        if series in ("even", "odd"):
                            if series in a_path:
                                file_list.append(a_path)
                        else:
                            series_list = [series] if not isinstance(series, list) else series
                            for it_series in series_list:
                                if it_series in a_path:
                                    file_list.append(a_path)
                    else:
                        file_list.append(a_path)
            else:
                raise ValueError(f'File or directory "{a_path}" does not exist!')

        if not file_list:
            raise ValueError("ERROR: No data found. Check arguments!")

        return sorted(set(file_list))

    def _load_func(self, paths, is_cut=True, repo_info=None, overwrite=False):
        file_list = []
        if not isinstance(paths, list):
            paths = [paths]

        for filepath in paths:
            if os.path.isdir(filepath):
                file_list.extend(glob(os.path.join(filepath, "*.py")))
            elif os.path.isfile(filepath):
                file_list.append(filepath)
            else:
                raise ValueError(f"ERROR: Unknown path or file {filepath}")

        file_list = sorted(set(file_list))
        module_name = "detanalysis.analyzer"

        for a_file in file_list:
            spec = importlib.util.spec_from_file_location(module_name, a_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            func_list = getmembers(module, isfunction)

            for func_name, _ in func_list:
                func_obj = getattr(module, func_name)
                func_metadata = vars(func_obj).copy()

                if repo_info is not None:
                    func_metadata.update(repo_info)

                current_funcs = self._cuts if is_cut else self._derived_features

                if not overwrite and func_name in current_funcs:
                    version = func_metadata.get("version", None)
                    version_saved = current_funcs[func_name].get("version", None)

                    if (
                        version is not None
                        and version_saved is not None
                        and float(version) <= float(version_saved)
                    ):
                        print(
                            f'WARNING: Function "{func_name}" already exists '
                            f"(version={version_saved})."
                        )
                        print("Unable to register it! Change version or use overwrite=True")
                        continue

                vaex_expr = func_obj(self._df_full)

                if is_cut:
                    self.register_cut(
                        vaex_expr,
                        name=func_name,
                        metadata=func_metadata,
                        overwrite=True,
                    )
                else:
                    self.add_feature(
                        vaex_expr,
                        name=func_name,
                        metadata=func_metadata,
                        overwrite=True,
                    )

    def _get_repo_info(self):
        repo_info = {
            "git_repo_name": None,
            "git_repo_branch": None,
            "git_repo_tag": None,
            "git_repo_commit": None,
        }

        if self._analysis_repo is None:
            print('WARNING: No git repo available. Use "set_analysis_repo" to set it!')
            return repo_info

        repo = self._analysis_repo

        try:
            repo_info["git_repo_name"] = (
                os.path.basename(repo.working_dir) if repo.working_dir else None
            )
        except Exception:
            pass

        is_empty = False
        try:
            _ = repo.head.commit
        except ValueError:
            is_empty = True
        except Exception:
            pass

        try:
            if not is_empty:
                repo_info["git_repo_branch"] = getattr(repo.active_branch, "name", None)
        except Exception:
            repo_info["git_repo_branch"] = None

        try:
            if not is_empty:
                short = repo.git.rev_parse("--short", "HEAD")
                if repo.is_dirty():
                    short += "-dirty"
                repo_info["git_repo_commit"] = short
        except Exception:
            repo_info["git_repo_commit"] = None

        try:
            if not is_empty:
                if repo.tags:
                    desc = repo.git.describe("--tags", "--dirty", "--broken")
                else:
                    desc = repo.git.describe("--always")
                    if repo.is_dirty():
                        desc += "-dirty"
                repo_info["git_repo_tag"] = desc
        except GitCommandError:
            repo_info["git_repo_tag"] = repo_info["git_repo_commit"]
        except Exception:
            repo_info["git_repo_tag"] = repo_info["git_repo_commit"]

        return repo_info
