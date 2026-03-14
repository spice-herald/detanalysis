"""
vibration_analyzer.py
Inherits detanalysis.Analyzer with vibration-specific PSD and transfer function methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from detanalysis import Analyzer


class Vibration_Analyzer(Analyzer):
    """
    Inherits detanalysis.Analyzer, adds vibration-specific analysis methods
    for accelerometer data stored in processed HDF5 files.

    Inherits all Analyzer capabilities (vaex DataFrame access, cut management,
    histogram/scatter plotting, etc.) and adds:
      - get_psd()              : compute PSDs and ASD uncertainties per channel
      - get_transfer_function(): compute ASD transfer functions for multiple channel pairs
      - plot_psd()             : plot ASD for one or more channels with uncertainty bands
      - plot_transfer_function(): plot transfer functions for one or more channel pairs

    PSDs computed by get_psd() are cached internally so that a subsequent call to
    get_transfer_function() or plot_psd() can reuse them without re-reading the data.
    """

    # Intrinsic accelerometer noise floor (from accelerometer spec sheet).
    # Piecewise-constant: value at index k applies from FREQS_HZ[k] to FREQS_HZ[k+1].
    _NOISE_FLOOR_FREQS_HZ      = np.array([0.1, 10.0, 100.0, 1000.0])
    _NOISE_FLOOR_G_PER_SQRT_HZ = np.array([0.30, 0.10, 0.04, 0.04]) * 1e-6

    def __init__(self, paths, series=None, **kwargs):
        """
        Initialize Vibration_Analyzer.

        Parameters
        ----------
        paths : str or list
            Path(s) to processed amplitude HDF5 file(s), passed directly to
            Analyzer.__init__.
        series : str or list, optional
            Filter files by series name. Default: all files in paths.
        **kwargs
            Additional keyword arguments forwarded to Analyzer.__init__
            (e.g. analysis_repo, use_vaex_cut_handling, memory_cache_size).
        """
        super().__init__(paths, series=series, **kwargs)

        # Internal PSD cache — populated by get_psd(), consumed by get_transfer_function()
        self._psd                = None   # np.ndarray, shape (n_channels, n_freqs)
        self._variance           = None   # np.ndarray, shape (n_channels, n_freqs)
        self._freqs              = None   # np.ndarray, shape (n_freqs,)
        self._channels           = None   # list of str — channel names, index matches _psd rows
        self._transfer_functions = None   # list of dicts, populated by get_transfer_function()

    def get_psd(self, channels, return_freqs=True):
        """
        Compute PSD and ASD variance for each channel from the loaded amplitude data.

        Results are stored internally for reuse by get_transfer_function(). The
        method always recomputes when called directly (use get_transfer_function()
        to benefit from caching).

        Parameters
        ----------
        channels : list of str
            Channel names to compute PSDs for
            (e.g. ['AccelerometerGround', 'AccelerometerStage1']).
        return_freqs : bool, optional
            If True, also return the sorted frequency array. Default: True.

        Returns
        -------
        psd : np.ndarray, shape (n_channels, n_freqs)
            Power spectral density in g^2/Hz.
        variance : np.ndarray, shape (n_channels, n_freqs)
            Variance of the ASD estimate (g/√Hz)^2, via error propagation from
            amplitude uncertainties. Use np.sqrt(variance) to get 1-sigma uncertainty.
        freqs : np.ndarray, shape (n_freqs,)  [only if return_freqs=True]
            Sorted unique frequencies in Hz.
        """
        df = self.df

        freqs = sorted(df['frequency_hz'].unique())
        trace_length_msec = df['trace_length_msec'].unique()[0]

        # Frequency resolution: df = 1 / T, where T is the trace duration in seconds
        freq_resolution_hz = 1.0 / (trace_length_msec * 1e-3)

        psd = []
        variance = []

        # Loop over each requested channel and compute PSD at each frequency bin
        for channel in channels:
            chan_psd = []
            chan_var = []

            # Loop over each unique frequency bin in the dataset
            for freq in freqs:
                rows = df[df.frequency_hz == freq]
                real = rows[f'amp_real_{channel}'].values
                imag = rows[f'amp_imag_{channel}'].values
                n = len(rows)

                mean_real = real.mean()
                mean_imag = imag.mean()

                # PSD = |mean amplitude|^2 / frequency_resolution
                # Equation: PSD(f) = (Re^2 + Im^2) / df
                psd_val = (mean_real**2 + mean_imag**2) / freq_resolution_hz
                chan_psd.append(psd_val)

                # Variance of each component mean (standard error of the mean)
                var_mean_real = np.var(real, ddof=1) / n
                var_mean_imag = np.var(imag, ddof=1) / n

                # Error propagation from mean amplitudes to PSD:
                # d(PSD)/d(mean_real) = 2*mean_real / df, similarly for imag
                var_psd = (
                    (2.0 * mean_real / freq_resolution_hz)**2 * var_mean_real
                    + (2.0 * mean_imag / freq_resolution_hz)**2 * var_mean_imag
                )

                # Error propagation from PSD to ASD:
                # d(ASD)/d(PSD) = 1 / (2*sqrt(PSD))  →  var_ASD = var_PSD / (4*PSD)
                if psd_val > 0:
                    var_asd = var_psd / (4.0 * psd_val)
                else:
                    var_asd = 0.0

                chan_var.append(var_asd)

            psd.append(chan_psd)
            variance.append(chan_var)

        psd      = np.array(psd)
        variance = np.array(variance)
        freqs    = np.array(freqs)

        # Cache results for use by get_transfer_function()
        self._psd      = psd
        self._variance = variance
        self._freqs    = freqs
        self._channels = list(channels)

        if return_freqs:
            return psd, variance, freqs
        return psd, variance

    def get_transfer_function(self, channel_pairs):
        """
        Compute the ASD transfer function for one or more channel pairs.

        Uses internally cached PSDs when all required channels are already present;
        otherwise extends the PSD cache to include any missing channels (existing
        cached channels are preserved).

        Parameters
        ----------
        channel_pairs : list of [str, str]
            Each element is [channel_output, channel_input].
            E.g. [['Stage1', 'Ground'], ['Stage2', 'Ground']] computes two transfer
            functions sharing the same 'Ground' PSD computation.

        Returns
        -------
        transfer_functions : list of dict
            One dict per pair, each containing:
              'channel_output'    : str
              'channel_input'     : str
              'transfer_function' : np.ndarray, shape (n_freqs,) — ASD_out / ASD_in
              'transfer_sigma'    : np.ndarray, shape (n_freqs,) — 1-sigma uncertainty
              'freqs'             : np.ndarray, shape (n_freqs,)
        """
        # Normalize a bare pair like ['Stage1', 'Ground'] to [['Stage1', 'Ground']]
        if channel_pairs and isinstance(channel_pairs[0], str):
            channel_pairs = [channel_pairs]

        # Collect all unique channels needed across all pairs (preserve insertion order)
        all_needed = list(dict.fromkeys(ch for pair in channel_pairs for ch in pair))

        # Determine which channels are not yet in the PSD cache
        cached = self._channels if self._channels is not None else []
        missing = [ch for ch in all_needed if ch not in cached]

        # Extend the PSD cache to include missing channels without dropping existing data
        if missing:
            self.get_psd(channels=list(dict.fromkeys(cached + missing)))

        # Compute TF for each pair and collect into a list of dicts
        results = []
        for channel_output, channel_input in channel_pairs:
            idx_out = self._channels.index(channel_output)
            idx_in  = self._channels.index(channel_input)

            # Convert PSD to ASD: ASD = sqrt(PSD)
            asd_output = np.sqrt(self._psd[idx_out])
            asd_input  = np.sqrt(self._psd[idx_in])

            # Transfer function: T(f) = ASD_output(f) / ASD_input(f)
            with np.errstate(divide='ignore', invalid='ignore'):
                tf = asd_output / asd_input

            sigma_output = np.sqrt(self._variance[idx_out])
            sigma_input  = np.sqrt(self._variance[idx_in])

            # Uncertainty propagation for a ratio:
            # sigma_T / T = sqrt( (sigma_out / ASD_out)^2 + (sigma_in / ASD_in)^2 )
            with np.errstate(divide='ignore', invalid='ignore'):
                tf_sigma = tf * np.sqrt(
                    (sigma_output / asd_output) ** 2.0
                    + (sigma_input / asd_input) ** 2.0
                )

            results.append({
                'channel_output':    channel_output,
                'channel_input':     channel_input,
                'transfer_function': tf,
                'transfer_sigma':    tf_sigma,
                'freqs':             self._freqs,
            })

        # Upsert: preserve cached pairs not recomputed, then append new results
        existing = [
            d for d in (self._transfer_functions or [])
            if not any(
                d['channel_output'] == r['channel_output']
                and d['channel_input'] == r['channel_input']
                for r in results
            )
        ]
        self._transfer_functions = existing + results
        return results

    def plot_transfer_function(self, channel_pairs=None, figsize=(14, 6)):
        """
        Plot the transfer function for one or more channel pairs with 1-sigma uncertainty
        bands and a secondary dB axis.

        Uses internally cached transfer functions from a prior get_transfer_function() call
        when available. If any requested pair is not yet cached, get_transfer_function() is
        called automatically for the missing pairs before plotting.

        Parameters
        ----------
        channel_pairs : list of [str, str] or [str, str], optional
            Each element is [channel_output, channel_input]. A bare pair like
            ['Stage1', 'Ground'] is also accepted and treated as a single-pair list.
            If None, all currently cached transfer functions are plotted.
        figsize : tuple of (float, float), optional
            Figure size passed to plt.subplots. Default: (14, 6).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        # Normalize a bare pair like ['Stage1', 'Ground'] to [['Stage1', 'Ground']]
        if channel_pairs is not None and channel_pairs and isinstance(channel_pairs[0], str):
            channel_pairs = [channel_pairs]

        # If no pairs specified, plot everything currently cached
        if channel_pairs is None:
            if self._transfer_functions is None:
                raise RuntimeError(
                    "No transfer functions cached. Call get_transfer_function() first "
                    "or pass channel_pairs= explicitly."
                )
            channel_pairs = [
                [d['channel_output'], d['channel_input']]
                for d in self._transfer_functions
            ]

        # Compute any pairs not yet in the cache (existing cached pairs are preserved)
        cached_pairs = (
            {(d['channel_output'], d['channel_input']) for d in self._transfer_functions}
            if self._transfer_functions is not None
            else set()
        )
        missing_pairs = [p for p in channel_pairs if tuple(p) not in cached_pairs]
        if missing_pairs:
            self.get_transfer_function(channel_pairs=missing_pairs)

        # Build a lookup dict so we can collect pairs in the requested order
        tf_lookup = {
            (d['channel_output'], d['channel_input']): d
            for d in self._transfer_functions
        }
        pairs_to_plot = [tf_lookup[(p[0], p[1])] for p in channel_pairs]

        fig, ax = plt.subplots(figsize=figsize)
        prop_cycle_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

        all_tf_values = []
        for i, tf_dict in enumerate(pairs_to_plot):
            ch_out = tf_dict['channel_output']
            ch_in  = tf_dict['channel_input']
            freqs  = tf_dict['freqs']
            tf     = tf_dict['transfer_function']
            tf_sig = tf_dict['transfer_sigma']
            color  = prop_cycle_colors[i % len(prop_cycle_colors)]
            all_tf_values.append(tf)

            # Plot mean transfer function
            ax.loglog(
                freqs,
                tf,
                marker="o",
                markersize=0,
                linestyle="-",
                label=f"{ch_out} / {ch_in}",
                color=color,
                alpha=0.5,
            )

            # Build 1-sigma bounds; keep lower bound positive for log-scale plotting
            lower = np.maximum(tf - tf_sig, np.finfo(float).tiny)
            upper = tf + tf_sig

            # Draw uncertainty band
            ax.fill_between(
                freqs,
                lower,
                upper,
                color=color,
                alpha=0.2,
                linewidth=1,
                label="±σ/√n",
            )

        # Set y-limits across all plotted transfer functions
        all_tf_concat = np.concatenate(all_tf_values)
        ax.set_ylim(all_tf_concat.min() * 0.8, all_tf_concat.max() * 1.2)

        ax.set_xlabel("Frequency (Hz)")

        # Single-pair label names the channels; multi-pair uses a generic label
        if len(pairs_to_plot) == 1:
            ch_out = pairs_to_plot[0]['channel_output']
            ch_in  = pairs_to_plot[0]['channel_input']
            ax.set_ylabel(f"Attenuation ({ch_out}/{ch_in})")
        else:
            ax.set_ylabel("Attenuation")

        ax.grid(True, which="both", ls=":")
        ax.legend()

        # Secondary right-hand axis in dB, with ticks synced to the left axis
        secax = ax.secondary_yaxis(
            location="right",
            functions=(lambda y: 20.0 * np.log10(y), lambda db: 10.0 ** (db / 20.0)),
        )
        secax.set_ylabel("Attenuation (dB)")
        secax.set_yscale("linear")
        secax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        # Sync dB ticks to the visible left-axis ticks
        left_ticks = ax.get_yticks()
        ymin, ymax = ax.get_ylim()
        left_ticks = left_ticks[
            (left_ticks > 0) & (left_ticks >= ymin) & (left_ticks <= ymax)
        ]
        if left_ticks.size > 0:
            secax.set_yticks(20.0 * np.log10(left_ticks))
        else:
            secax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

        return fig, ax

    def plot_psd(self, channels=None, figsize=(14, 6)):
        """
        Plot the ASD for one or more channels with 1-sigma uncertainty bands and the
        intrinsic accelerometer noise floor.

        Uses internally cached PSDs from a prior get_psd() call when available.
        If any requested channel is not yet cached, get_psd() is called automatically
        for the full requested channel list before plotting.

        Parameters
        ----------
        channels : list of str, optional
            Channel names to plot, in the order they should appear.
            If None, all currently cached channels are plotted.
        figsize : tuple of (float, float), optional
            Figure size passed to plt.subplots. Default: (14, 6).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        # Resolve which channels to plot
        if channels is None:
            if self._channels is None:
                raise RuntimeError(
                    "No PSDs cached. Call get_psd() first or pass channels= explicitly."
                )
            channels = self._channels

        # Ensure PSDs are available for every requested channel; recompute if any are missing
        missing = [ch for ch in channels if self._channels is None or ch not in self._channels]
        if missing:
            self.get_psd(channels=channels)

        freqs    = self._freqs
        freq_end = np.nanmax(freqs)

        fig, ax = plt.subplots(figsize=figsize)

        # Pull colors from the default matplotlib prop cycle
        prop_cycle_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

        # Plot each channel's ASD and 1-sigma uncertainty band
        max_amp = -np.inf
        min_amp = np.inf
        for i, chan in enumerate(channels):
            chan_idx = self._channels.index(chan)
            color    = prop_cycle_colors[i % len(prop_cycle_colors)]

            amp   = np.sqrt(self._psd[chan_idx, :])
            sigma = np.sqrt(self._variance[chan_idx, :])
            max_amp = np.maximum(amp.max(), max_amp)
            min_amp = np.minimum(amp.min(), min_amp)

            # Plot mean ASD
            ax.loglog(
                freqs,
                amp,
                label=chan,
                color=color,
                alpha=0.5,
            )

            # Build 1-sigma bounds; keep lower bound positive for log-scale plotting
            lower = np.maximum(amp - sigma, np.finfo(float).tiny)
            upper = amp + sigma

            # Draw uncertainty band
            ax.fill_between(
                freqs,
                lower,
                upper,
                color=color,
                alpha=0.2,
                linewidth=1,
                label="±σ/√n",
            )

        # Plot intrinsic noise floor as piecewise-constant horizontal segments
        for k in range(len(self._NOISE_FLOOR_FREQS_HZ)):
            f_start = self._NOISE_FLOOR_FREQS_HZ[k]
            f_end   = (
                self._NOISE_FLOOR_FREQS_HZ[k + 1]
                if k < len(self._NOISE_FLOOR_FREQS_HZ) - 1
                else freq_end
            )
            ax.hlines(
                self._NOISE_FLOOR_G_PER_SQRT_HZ[k],
                f_start,
                f_end,
                colors="red",
                linestyles="dashdot",
                alpha=0.5,
                label="Intrinsic Spectral Noise" if k == 0 else None,
            )

        ax.set_xlim(np.nanmin(freqs), freq_end)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"Transducer Freq (g/$\sqrt{\mathrm{Hz}}$)")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
        ax.set_ylim([min_amp*0.8, max_amp*1.2])

        return fig, ax
