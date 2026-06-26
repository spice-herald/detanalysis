### The Vibration Analyzer inherits detanalysis.Analyzer, and adds vibration-specific PSD and transfer function calculation and visualization methods.

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
import detprocess
import pytesdaq.io as h5io
from detanalysis import Analyzer


def _count_downsampled_events(total_events, downsample_factor):
    """
    Number of events kept by the downsampling stride.

    The stride keeps the first event, then every downsample_factor-th event
    after it (1-based indices 1, 1 + downsample_factor,
    1 + 2 * downsample_factor, ...). A non-empty stream therefore always
    keeps at least the first event.

    Parameters
    ----------
    total_events : int
        Total number of events available in the raw stream.
    downsample_factor : int
        Stride; 1 keeps every event.

    Return
    ------
    n_kept : int
        Number of events the stride will process.
    """
    if total_events < 1:
        return 0
    return (total_events - 1) // downsample_factor + 1


class Vibration_Analyzer(Analyzer):
    """
    Inherits detanalysis.Analyzer, adds vibration-specific analysis methods
    for accelerometer data stored in processed HDF5 files.
    Acquire processed data from the process_transducer_sweep.py script in detprocess.

    Adds the following methods to the base Analyzer class:
        - get_psd()                       : compute PSDs and err.
                                            PSDs computed by get_psd() are cached for reuse by plot_psd().
        - get_transfer_function()         : compute transfer functions from accelerometer data
                                            Supports estimators mean_ratio, cross_correlation, and phase_locked.
                                            Transfer functions are cached for reuse by plotting methods.
        - plot_psd()                      : plot PSD for one or more channels
        - plot_transfer_function()        : plot transfer function magnitude
        - plot_transfer_function_phase()  : plot transfer function phase (complex methods only, mean_ratio not supported)

    """

    # Valid transfer function estimator name strings.
    VALID_TF_METHODS = ('mean_ratio', 'cross_correlation', 'phase_locked')

    # Default number of log-spaced points used by plotting functions when log_downsample=True. 
    # (otherwise matplotlib will sometimes produce "cell block limit exceeded" error if too much data is plotted at once)
    _LOG_DOWNSAMPLE_DEFAULT_POINTS = 20000

    # Intrinsic accelerometer noise floor.
    # Calculated from Model 393B04 Seismic Accelerometer, PCB Piezotronics, Inc. (https://www.pcb.com/products?m=393b04).
    # value at index k applies from FREQS_HZ[k] to FREQS_HZ[k+1].
    _NOISE_FLOOR_FREQS_HZ      = np.array([0.1, 10.0, 100.0, 1000.0])
    _NOISE_FLOOR_G_PER_SQRT_HZ = np.array([0.30, 0.10, 0.04, 0.04]) * 1e-6

    def __init__(self, paths, series=None, **kwargs):
        """
        Parameters
        ----------
        paths : str or list
            Path(s) to processed amplitude HDF5 file(s), passed directly to Analyzer.__init__() base class initializer.
        series : str or list, optional
            Filter files by series name. Default: all files in paths.
            Formatted like 'I2_D20260622_T221904' (without the prefix 'cont_', 'amp_', etc; and without the dump number F00001...)
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
        self._data_source        = 'processed'
        self._noise_transfer_functions = None  # dict with 'mean_ratio' and 'cross_correlation' keys

    @classmethod
    def from_raw_noise(cls, raw_path, channels, channel_pairs=None,
                       accel_gain=100.0, downsample_factor=1,
                       series=None, trace_length_samples=None,
                       trace_length_msec=None, verbose=True):
        """
        Construct a Vibration_Analyzer from raw continuous noise HDF5 data.

        Reads raw accelerometer traces, computes broadband PSDs, and derives
        transfer functions via two methods: mean-PSD-ratio and
        cross-correlation.

        Parameters
        ----------
        raw_path : str
            Path to directory containing raw continuous HDF5 files.
        channels : list of str
            Channel names to process (e.g. ['PCS1', 'PCS2']).
        channel_pairs : list of [str, str], optional
            Each element is [channel_output, channel_input] for transfer
            function computation. If None, no transfer functions are computed.
        accel_gain : float, optional
            Accelerometer gain factor. Raw ADC values are divided by this
            to convert to g. Default: 100.0.
        downsample_factor : int, optional
            Process every Nth event (1 = all events). Default: 1.
        series : str or list, optional
            Filter files by series name. Default: all files in raw_path.
        trace_length_samples : int, optional
            Desired trace length, in samples, for PSD computation. If None
            (default), the native per-event sample count stored in the raw
            HDF5 files is used (original behavior).

            If set, the samples from every (downsampled) event are
            concatenated along the time axis into a single continuous
            per-channel stream, and that stream is re-chopped into
            non-overlapping chunks of exactly ``trace_length_samples``.
            Any incomplete remainder at the end of the stream (shorter
            than one full chunk) is discarded.

            The resulting PSD has frequency resolution
            ``sample_rate / trace_length_samples``. Must be a positive
            integer >= 2. Mutually exclusive with ``trace_length_msec``.
        trace_length_msec : float, optional
            Same behavior as ``trace_length_samples``, but specified in
            milliseconds. Converted to samples via
            ``round(sample_rate * trace_length_msec / 1000)`` using the
            sample rate read from the raw HDF5 metadata. Mutually exclusive
            with ``trace_length_samples``.
        verbose : bool, optional
            Print progress information. Default: True.

        Returns
        -------
        Vibration_Analyzer
            Instance with cached PSDs and (optionally) transfer functions.
        """
        # trace_length_samples and trace_length_msec are two ways of specifying
        # the same quantity; exactly one (or neither) may be provided.
        if (trace_length_samples is not None) and (trace_length_msec is not None):
            raise ValueError(
                "trace_length_samples and trace_length_msec are mutually exclusive; "
                "pass at most one."
            )

        # Validate trace_length_samples (trace_length_msec is validated and
        # converted to samples below, once sample_rate is available).
        if trace_length_samples is not None:
            if (not isinstance(trace_length_samples, (int, np.integer))) or (trace_length_samples < 2):
                raise ValueError(
                    "trace_length_samples must be a positive integer >= 2; "
                    f"got {trace_length_samples!r}."
                )
            trace_length_samples = int(trace_length_samples)

        # Validate trace_length_msec (must be a positive finite number)
        if trace_length_msec is not None:
            if (not isinstance(trace_length_msec, (int, float, np.integer, np.floating))
                    or (not np.isfinite(trace_length_msec)) or (trace_length_msec <= 0)):
                raise ValueError(
                    "trace_length_msec must be a positive finite number; "
                    f"got {trace_length_msec!r}."
                )
            trace_length_msec = float(trace_length_msec)

        # Validate downsample_factor (process every Nth event; 1 keeps all)
        if (not isinstance(downsample_factor, (int, np.integer))) or (downsample_factor < 1):
            raise ValueError(
                "downsample_factor must be a positive integer >= 1; "
                f"got {downsample_factor!r}."
            )
        downsample_factor = int(downsample_factor)

        # Normalize channel_pairs format
        if channel_pairs is not None and channel_pairs and isinstance(channel_pairs[0], str):
            channel_pairs = [channel_pairs]

        # Create instance without calling Analyzer.__init__ (no processed DataFrame)
        instance = cls.__new__(cls)

        # Initialize Analyzer attributes to safe defaults
        instance._df = None
        instance._paths = None
        instance._series = series
        instance._cuts = {}
        instance._vaex_cut_handling = False
        instance._memory_cache_size = None
        instance._analysis_repo = None

        # Initialize Vibration_Analyzer attributes
        instance._psd = None
        instance._variance = None
        instance._freqs = None
        instance._channels = None
        instance._transfer_functions = None
        instance._data_source = 'raw_noise'
        instance._noise_transfer_functions = None

        # Load metadata via detprocess.RawData
        rawdata = detprocess.RawData(raw_path, data_type='cont', verbose=verbose)
        sample_rate = rawdata.get_sample_rate(data_type='cont')
        # Count events for the selected series only (matches the H5Reader's
        # series filtering below), so total_events and the progress bar reflect
        # what is actually read rather than the whole dataset.
        duration, total_events = rawdata.get_duration(
            series=series, data_type='cont', include_nb_events=True,
        )
        total_events = int(total_events)

        # Convert trace_length_msec to samples now that sample_rate is known.
        # n_samples = round(fs * T_msec / 1000)
        if trace_length_msec is not None:
            trace_length_samples = int(round(sample_rate * trace_length_msec / 1000.0))
            if trace_length_samples < 2:
                raise ValueError(
                    f"trace_length_msec={trace_length_msec} ms corresponds to "
                    f"{trace_length_samples} samples at fs={sample_rate} Hz; "
                    "need at least 2 samples for a meaningful FFT."
                )

        # FFT performance is best on powers of two; warn if the user picks otherwise.
        # A positive integer N is a power of two iff log2(N) is an integer.
        if trace_length_samples is not None:
            log2_len = np.log2(trace_length_samples)
            if log2_len != int(log2_len):
                warnings.warn(
                    f"trace_length_samples={trace_length_samples} is not a power of two; "
                    "FFT performance may be suboptimal."
                )

        # Fail fast if the downsampling stride keeps too few events to proceed.
        # total_events is already known, so we can raise a clear error here
        # instead of reading the entire dataset only to fail at the end.
        n_kept_events = _count_downsampled_events(total_events, downsample_factor)

        # The default path produces one trace per kept event and needs >= 2 for
        # variance estimation. The rechunk path can yield many chunks from a
        # single long event, so it only requires >= 1 kept event here (the >= 2
        # chunk requirement is enforced after the loop).
        if trace_length_samples is None:
            min_kept_required = 2
        else:
            min_kept_required = 1

        if n_kept_events < min_kept_required:
            raise ValueError(
                f"downsample_factor={downsample_factor} keeps only "
                f"{n_kept_events} of {total_events} event(s); need at least "
                f"{min_kept_required}. Reduce downsample_factor to at most "
                f"{max(total_events - 1, 1)}."
            )

        # Downsampling does not skip disk reads (every event is still read
        # before being kept or discarded), so a factor at/above the total event
        # count keeps only the first event yet still reads the whole dataset.
        if downsample_factor >= total_events:
            warnings.warn(
                f"downsample_factor={downsample_factor} is >= the total number "
                f"of events ({total_events}); only the first event is "
                "processed. Downsampling does not skip disk reads, so every "
                "event is still read from disk. Consider a smaller "
                "downsample_factor or a series filter."
            )

        if verbose:
            print(
                f"Total events available: {total_events}; processing "
                f"{n_kept_events} after downsampling (factor {downsample_factor})."
            )

        # Set up H5Reader for event streaming
        h5reader = h5io.H5Reader()
        h5reader.set_files(raw_path, series=series)

        n_channels = len(channels)

        # PSD accumulators — will be initialized on first event (need n_freqs)
        welford_mean = None   # running mean of PSD (g/sqrt(Hz)), shape (n_channels, n_freqs)
        welford_m2 = None     # running M2 for variance, shape (n_channels, n_freqs)

        # Cross-correlation accumulators (only if channel_pairs provided)
        # Each pair: sum of fft_out * conj(fft_in) and sum of |fft_in|^2
        sum_cross = None  # dict keyed by (ch_out, ch_in) → complex array
        sum_auto = None   # dict keyed by (ch_out, ch_in) → real array

        if channel_pairs is not None:
            sum_cross = {}
            sum_auto = {}

        # Counts the number of traces (one per event in the default path,
        # one per produced chunk when rechunking) folded into the PSD estimate.
        n_events_processed = 0
        event_index = 0

        # Rolling per-channel buffer used only when trace_length_samples is set
        rechunk_buffer = None
        freqs = None  # set on the first processed trace

        def _process_one_trace(trace_2d):
            """
            Fold a single (n_channels, n_samples) trace into the running PSD
            estimate and cross-correlation accumulators.

            On the first invocation, lazily initializes welford_mean / welford_m2 /
            freqs / cross-correlation accumulators based on the incoming n_samples.
            """
            nonlocal welford_mean, welford_m2, freqs, n_events_processed

            # Number of samples in this trace (may differ from native event length
            # when rechunking is enabled)
            n_samples = trace_2d.shape[-1]

            # First-trace initialization: frequency axis, PSD accumulators,
            # and (optionally) cross-correlation accumulators
            if welford_mean is None:
                n_freqs_local = n_samples // 2 + 1
                freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
                welford_mean = np.zeros((n_channels, n_freqs_local))
                welford_m2 = np.zeros((n_channels, n_freqs_local))

                if channel_pairs is not None:
                    for pair in channel_pairs:
                        key = (pair[0], pair[1])
                        sum_cross[key] = np.zeros(n_freqs_local, dtype=complex)
                        sum_auto[key] = np.zeros(n_freqs_local)

            # Bump trace counter (used by the running mean/variance below)
            n_events_processed = n_events_processed + 1

            # Compute FFT and one-sided PSD for each channel
            fft_vals_dict = {}  # cache FFTs for cross-correlation computation
            # Loop over each detector channel and update its running PSD estimate
            for ch_idx in range(n_channels):
                trace = trace_2d[ch_idx, :]

                # FFT of the trace
                fft_vals = np.fft.rfft(trace)
                fft_vals_dict[channels[ch_idx]] = fft_vals

                # Two-sided PSD: |FFT|^2 / (fs * N)
                psd_two_sided = (np.abs(fft_vals) ** 2) / (sample_rate * n_samples)

                # One-sided: double non-DC/Nyquist bins
                psd_two_sided[1:-1] = psd_two_sided[1:-1] * 2

                # PSD in g/sqrt(Hz) (square root of power spectral density)
                psd_val = np.sqrt(psd_two_sided)

                # Update the running mean and variance of the PSD (g/sqrt(Hz))
                delta = psd_val - welford_mean[ch_idx]
                welford_mean[ch_idx] = welford_mean[ch_idx] + delta / n_events_processed
                delta2 = psd_val - welford_mean[ch_idx]
                welford_m2[ch_idx] = welford_m2[ch_idx] + delta * delta2

            # Cross-correlation accumulation for each channel pair
            if channel_pairs is not None:
                # Loop over each (output, input) channel pair
                for pair in channel_pairs:
                    ch_out, ch_in = pair[0], pair[1]
                    fft_out = fft_vals_dict[ch_out]
                    fft_in = fft_vals_dict[ch_in]
                    key = (ch_out, ch_in)

                    # Accumulate cross-spectrum: fft_out * conj(fft_in)
                    sum_cross[key] = sum_cross[key] + fft_out * np.conj(fft_in)

                    # Accumulate auto-spectrum of input: |fft_in|^2
                    sum_auto[key] = sum_auto[key] + np.abs(fft_in) ** 2

        # Event loop: read raw traces and accumulate PSD statistics
        pbar = tqdm(
            total=total_events,
            desc="Processing raw noise events",
            disable=(not verbose),
        )

        # Iterate over every event in the raw HDF5 stream until exhausted
        while True:
            # Read next event from the raw HDF5 files
            event_trace, metadata = h5reader.read_next_event(
                detector_chans=channels,
                adctoamp=True,
                include_metadata=True,
            )

            # End-of-data check
            if event_trace.size == 0:
                break

            pbar.update(1)
            event_index = event_index + 1

            # Downsample: keep the first event, then every downsample_factor-th
            # event after it. This matches _count_downsampled_events and ensures
            # a non-empty stream always keeps at least the first event.
            if (event_index - 1) % downsample_factor != 0:
                continue

            # Convert ADC amplitude to g (event shape: (n_channels, n_samples))
            event_g = event_trace * (1.0 / accel_gain)

            if trace_length_samples is None:
                # Default path: treat each event as one trace (original behavior)
                _process_one_trace(event_g)
            else:
                # Rechunk path: concatenate this event onto the rolling buffer
                # and emit as many full chunks of length trace_length_samples as
                # the accumulated buffer allows. Any remainder stays in the
                # buffer for the next event.
                if rechunk_buffer is None:
                    rechunk_buffer = event_g
                else:
                    rechunk_buffer = np.concatenate(
                        (rechunk_buffer, event_g), axis=1,
                    )

                # Emit every complete chunk currently held in the buffer
                while rechunk_buffer.shape[-1] >= trace_length_samples:
                    chunk = rechunk_buffer[:, :trace_length_samples]
                    _process_one_trace(chunk)
                    rechunk_buffer = rechunk_buffer[:, trace_length_samples:]

        pbar.close()

        # Any samples left in rechunk_buffer form an incomplete trailing trace
        # and are intentionally discarded (per the rechunk semantics).
        if verbose and trace_length_samples is not None:
            leftover = 0 if rechunk_buffer is None else rechunk_buffer.shape[-1]
            print(
                f"Rechunked into {n_events_processed} trace(s) of "
                f"{trace_length_samples} samples "
                f"({leftover} trailing sample(s) discarded)."
            )

        if n_events_processed < 2:
            raise RuntimeError(
                f"Only {n_events_processed} event(s) processed — need at least 2 "
                "for variance estimation."
            )

        # Finalize PSD cache
        # Store as g^2/Hz (square of mean PSD in g/sqrt(Hz)), matching processed data convention
        instance._psd = welford_mean ** 2
        # Variance of the mean PSD estimate: M2 / (n * (n-1))
        instance._variance = welford_m2 / (n_events_processed * (n_events_processed - 1))
        instance._freqs = freqs
        instance._channels = list(channels)

        # Compute transfer functions if channel pairs were provided
        if channel_pairs is not None:
            mean_ratio_results = []
            cross_correlation_results = []

            for pair in channel_pairs:
                ch_out, ch_in = pair[0], pair[1]
                idx_out = channels.index(ch_out)
                idx_in = channels.index(ch_in)

                # --- Method 1: Mean ratio ---
                # TF = mean_amplitude_out / mean_amplitude_in
                asd_out = welford_mean[idx_out]
                asd_in = welford_mean[idx_in]

                with np.errstate(divide='ignore', invalid='ignore'):
                    tf_ratio = asd_out / asd_in

                # Uncertainty propagation for ratio of means
                # sigma of each mean PSD estimate
                sigma_out = np.sqrt(welford_m2[idx_out] / (n_events_processed * (n_events_processed - 1)))
                sigma_in = np.sqrt(welford_m2[idx_in] / (n_events_processed * (n_events_processed - 1)))

                # sigma_T / T = sqrt( (sigma_out/asd_out)^2 + (sigma_in/asd_in)^2 )
                with np.errstate(divide='ignore', invalid='ignore'):
                    tf_ratio_sigma = tf_ratio * np.sqrt(
                        (sigma_out / asd_out) ** 2.0
                        + (sigma_in / asd_in) ** 2.0
                    )

                mean_ratio_results.append({
                    'channel_output': ch_out,
                    'channel_input': ch_in,
                    'transfer_function': tf_ratio,
                    'transfer_sigma': tf_ratio_sigma,
                    'freqs': freqs,
                    'method': 'mean_ratio',
                })

                # --- Method 2: Cross-correlation ---
                # TF = |<fft_out * conj(fft_in)>| / <|fft_in|^2>
                key = (ch_out, ch_in)
                mean_cross = sum_cross[key] / n_events_processed
                mean_auto_in = sum_auto[key] / n_events_processed

                with np.errstate(divide='ignore', invalid='ignore'):
                    tf_cross = np.abs(mean_cross / mean_auto_in)

                # Propagate uncertainty on the cross-correlation TF magnitude via
                # the same ratio rule used for mean_ratio (PSD sampling variance):
                # sigma_T / T = sqrt( (sigma_out/asd_out)^2 + (sigma_in/asd_in)^2 )
                # This is valid in the high-SNR regime where cross-correlation and
                # mean-ratio TFs converge.
                with np.errstate(divide='ignore', invalid='ignore'):
                    tf_cross_sigma = tf_cross * np.sqrt(
                        (sigma_out / asd_out) ** 2.0
                        + (sigma_in / asd_in) ** 2.0
                    )

                cross_correlation_results.append({
                    'channel_output': ch_out,
                    'channel_input': ch_in,
                    'transfer_function': tf_cross,
                    'transfer_sigma': tf_cross_sigma,
                    'freqs': freqs,
                    'method': 'cross_correlation',
                })

            instance._noise_transfer_functions = {
                'mean_ratio': mean_ratio_results,
                'cross_correlation': cross_correlation_results,
            }

            # Default transfer functions point to mean_ratio results
            instance._transfer_functions = list(mean_ratio_results)

        return instance

    def get_noise_transfer_function(self, method='mean_ratio'):
        """
        Return cached noise transfer functions for the specified method.

        Only available for instances created via from_raw_noise().

        Parameters
        ----------
        method : str
            'mean_ratio' or 'cross_correlation'.

        Returns
        -------
        list of dict
            Same schema as get_transfer_function() output:
            each dict has 'channel_output', 'channel_input',
            'transfer_function', 'transfer_sigma', 'freqs'.
        """
        if self._noise_transfer_functions is None:
            raise RuntimeError(
                "No noise transfer functions available. "
                "Use from_raw_noise() with channel_pairs to compute them."
            )

        if method not in ('mean_ratio', 'cross_correlation'):
            raise ValueError(
                f"Unknown method '{method}'. Use 'mean_ratio' or 'cross_correlation'."
            )

        return self._noise_transfer_functions[method]

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
        if self._data_source == 'raw_noise':
            raise RuntimeError(
                "get_psd() is not available for raw noise instances. "
                "PSDs were already computed during from_raw_noise(); "
                "access them via self._psd, self._variance, self._freqs."
            )

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

    def _compute_tf_from_traces(self, channel_pairs, methods):
        """
        Compute transfer function estimators directly from per-trace complex amplitudes.

        Iterates over unique frequencies once and computes all requested estimators
        in a single pass, avoiding redundant data reads.

        Parameters
        ----------
        channel_pairs : list of [str, str]
            Each element is [channel_output, channel_input].
        methods : list of str
            Subset of VALID_TF_METHODS: 'mean_ratio', 'cross_correlation',
            'phase_locked'.

        Returns
        -------
        results : dict
            Keys are method names from ``methods``. Values are lists of dicts
            (one per channel pair). Each dict contains:
              'channel_output'    : str
              'channel_input'     : str
              'transfer_function' : np.ndarray — real for mean_ratio, complex
                                    for cross_correlation and phase_locked
              'transfer_sigma'    : np.ndarray — 1-sigma uncertainty (NaN where
                                    analytically intractable)
              'freqs'             : np.ndarray, shape (n_freqs,)
              'method'            : str
        """
        df = self.df

        freqs = sorted(df['frequency_hz'].unique())
        n_freqs = len(freqs)
        n_pairs = len(channel_pairs)

        # Pre-allocate output arrays for each method and each channel pair.
        # mean_ratio is real; cross_correlation and phase_locked are complex.
        arrays = {}
        for method in methods:
            if method == 'mean_ratio':
                dtype = float
            else:
                dtype = complex
            arrays[method] = {
                'tf': np.zeros((n_pairs, n_freqs), dtype=dtype),
                'sigma': np.full((n_pairs, n_freqs), np.nan),
            }

        freqs_arr = np.array(freqs)

        # Single pass over all frequencies: extract per-trace complex amplitudes
        # and compute all requested estimators at each frequency bin.
        for f_idx, freq in enumerate(freqs):
            rows = df[df.frequency_hz == freq]

            # Loop over each channel pair at this frequency
            for p_idx, (ch_out, ch_in) in enumerate(channel_pairs):
                # Build complex amplitudes from real + imaginary fit coefficients
                real_out = rows[f'amp_real_{ch_out}'].values
                imag_out = rows[f'amp_imag_{ch_out}'].values
                real_in = rows[f'amp_real_{ch_in}'].values
                imag_in = rows[f'amp_imag_{ch_in}'].values

                z_out = np.array(real_out, dtype=float) + 1j * np.array(imag_out, dtype=float)
                z_in = np.array(real_in, dtype=float) + 1j * np.array(imag_in, dtype=float)
                n_traces = len(z_out)

                # Mean complex amplitudes (used by mean_ratio and phase_locked)
                mean_z_out = np.mean(z_out)
                mean_z_in = np.mean(z_in)

                # --- Mean Ratio estimator ---
                # TF = |mean(z_out)| / |mean(z_in)|
                # Equivalent to the old PSD-based calculation (Δf cancels in the ratio).
                # This is the magnitude of the phase-locked estimator; phase is discarded.
                if 'mean_ratio' in methods:
                    abs_mean_out = np.abs(mean_z_out)
                    abs_mean_in = np.abs(mean_z_in)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        tf_mr = abs_mean_out / abs_mean_in

                    arrays['mean_ratio']['tf'][p_idx, f_idx] = tf_mr

                    # Uncertainty propagation from variance of mean real/imag components.
                    # var(mean_re) = var(re) / n, var(mean_im) = var(im) / n
                    var_mean_re_out = np.var(real_out, ddof=1) / n_traces
                    var_mean_im_out = np.var(imag_out, ddof=1) / n_traces
                    var_mean_re_in = np.var(real_in, ddof=1) / n_traces
                    var_mean_im_in = np.var(imag_in, ddof=1) / n_traces

                    # Variance of |mean(z)| via error propagation:
                    # |z| = sqrt(re^2 + im^2)
                    # d|z|/d(re) = re / |z|,  d|z|/d(im) = im / |z|
                    # var(|z|) = (re/|z|)^2 * var(re) + (im/|z|)^2 * var(im)
                    re_out = mean_z_out.real
                    im_out = mean_z_out.imag
                    re_in = mean_z_in.real
                    im_in = mean_z_in.imag

                    if abs_mean_out > 0:
                        var_abs_out = (
                            (re_out / abs_mean_out) ** 2 * var_mean_re_out
                            + (im_out / abs_mean_out) ** 2 * var_mean_im_out
                        )
                        sigma_abs_out = np.sqrt(var_abs_out)
                    else:
                        sigma_abs_out = 0.0

                    if abs_mean_in > 0:
                        var_abs_in = (
                            (re_in / abs_mean_in) ** 2 * var_mean_re_in
                            + (im_in / abs_mean_in) ** 2 * var_mean_im_in
                        )
                        sigma_abs_in = np.sqrt(var_abs_in)
                    else:
                        sigma_abs_in = 0.0

                    # Ratio uncertainty: sigma_T / T = sqrt((sigma_out/out)^2 + (sigma_in/in)^2)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        tf_sigma_mr = tf_mr * np.sqrt(
                            (sigma_abs_out / abs_mean_out) ** 2
                            + (sigma_abs_in / abs_mean_in) ** 2
                        )
                    arrays['mean_ratio']['sigma'][p_idx, f_idx] = tf_sigma_mr

                # --- Cross-Correlation estimator ---
                # TF₂ = ⟨C₁₀⟩ / ⟨P₀₀⟩  where  C₁₀,ᵢ = z_out,ᵢ · conj(z_in,ᵢ)
                #                              and  P₀₀,ᵢ = |z_in,ᵢ|²
                # Complex output — preserves phase. Only biased by input noise.
                if 'cross_correlation' in methods:
                    # Per-trace cross-spectral and auto-spectral products
                    c10 = z_out * np.conj(z_in)   # complex, shape (n_traces,)
                    p00 = np.abs(z_in) ** 2        # real, shape (n_traces,)

                    mean_c10 = np.mean(c10)
                    mean_p00 = np.mean(p00)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        tf_cc = mean_c10 / mean_p00

                    arrays['cross_correlation']['tf'][p_idx, f_idx] = tf_cc

                    # Uncertainty via ratio error propagation:
                    # sigma_|TF₂| = |TF₂| * sqrt(Var(⟨C₁₀⟩)/|⟨C₁₀⟩|² + Var(⟨P₀₀⟩)/⟨P₀₀⟩²)
                    #
                    # Var(⟨C₁₀⟩) = σ²_C₁₀ / n  where σ²_C₁₀ is the sample variance
                    # of the complex per-trace cross-spectral products.
                    # Var(⟨P₀₀⟩) = σ²_P₀₀ / n  where σ²_P₀₀ is the sample variance
                    # of the real per-trace auto-spectral values.
                    var_c10 = np.sum(np.abs(c10 - mean_c10) ** 2) / (n_traces - 1)
                    var_mean_c10 = var_c10 / n_traces

                    var_p00 = np.var(p00, ddof=1)
                    var_mean_p00 = var_p00 / n_traces

                    abs_mean_c10 = np.abs(mean_c10)
                    abs_tf_cc = np.abs(tf_cc)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        tf_sigma_cc = abs_tf_cc * np.sqrt(
                            var_mean_c10 / (abs_mean_c10 ** 2)
                            + var_mean_p00 / (mean_p00 ** 2)
                        )
                    arrays['cross_correlation']['sigma'][p_idx, f_idx] = tf_sigma_cc

                # --- Phase-Locked estimator ---
                # TF₃ = ⟨z_out'⟩ / ⟨z_in'⟩
                # Complex output — preserves phase. Unbiased (best estimator).
                # Requires active transducer excitation.
                if 'phase_locked' in methods:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        tf_pl = mean_z_out / mean_z_in

                    arrays['phase_locked']['tf'][p_idx, f_idx] = tf_pl

                    # Uncertainty via ratio error propagation:
                    # sigma_|TF₃| = |TF₃| * sqrt(Var(⟨z₁'⟩)/|⟨z₁'⟩|² + Var(⟨z₀'⟩)/|⟨z₀'⟩|²)
                    #
                    # For a complex phasor z' = A + iB:
                    #   Var(z') = σ²_A + σ²_B
                    #   Var(⟨z'⟩) = Var(z') / n = (σ²_A + σ²_B) / n
                    var_z_out = np.var(real_out, ddof=1) + np.var(imag_out, ddof=1)
                    var_mean_z_out = var_z_out / n_traces

                    var_z_in = np.var(real_in, ddof=1) + np.var(imag_in, ddof=1)
                    var_mean_z_in = var_z_in / n_traces

                    abs_mean_out = np.abs(mean_z_out)
                    abs_mean_in = np.abs(mean_z_in)
                    abs_tf_pl = np.abs(tf_pl)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        tf_sigma_pl = abs_tf_pl * np.sqrt(
                            var_mean_z_out / (abs_mean_out ** 2)
                            + var_mean_z_in / (abs_mean_in ** 2)
                        )
                    arrays['phase_locked']['sigma'][p_idx, f_idx] = tf_sigma_pl

        # Assemble result dicts from the pre-allocated arrays
        results = {}
        for method in methods:
            method_results = []
            # Loop over each channel pair and build the result dict
            for p_idx, (ch_out, ch_in) in enumerate(channel_pairs):
                method_results.append({
                    'channel_output':    ch_out,
                    'channel_input':     ch_in,
                    'transfer_function': arrays[method]['tf'][p_idx, :],
                    'transfer_sigma':    arrays[method]['sigma'][p_idx, :],
                    'freqs':             freqs_arr,
                    'method':            method,
                })
            results[method] = method_results

        return results

    def get_transfer_function(self, channel_pairs, methods=None):
        """
        Compute transfer function estimators for one or more channel pairs.

        Three estimation methods are available, each operating on per-trace
        complex amplitudes z = A + iB at each drive frequency:

          - 'mean_ratio':       |mean(z_out)| / |mean(z_in)|  (real, amplitude only)
          - 'cross_correlation': mean(z_out * conj(z_in)) / mean(|z_in|^2)  (complex)
          - 'phase_locked':     mean(z_out) / mean(z_in)  (complex)

        When ``methods`` is not provided, the mean_ratio estimator is computed
        and the return format matches the legacy API (list of dicts).

        Parameters
        ----------
        channel_pairs : list of [str, str]
            Each element is [channel_output, channel_input].
            E.g. [['Stage1', 'Ground'], ['Stage2', 'Ground']] computes transfer
            functions sharing the same 'Ground' input computation.
        methods : str or list of str, optional
            Which estimator(s) to compute. One or more of 'mean_ratio',
            'cross_correlation', 'phase_locked'. If None (default), computes
            'mean_ratio' and returns the legacy list-of-dicts format for
            backward compatibility.

        Returns
        -------
        list of dict  (when ``methods=None``)
            One dict per pair, each containing:
              'channel_output'    : str
              'channel_input'     : str
              'transfer_function' : np.ndarray, shape (n_freqs,) — real amplitude
              'transfer_sigma'    : np.ndarray, shape (n_freqs,) — 1-sigma uncertainty
              'freqs'             : np.ndarray, shape (n_freqs,)

        dict of {str: list of dict}  (when ``methods`` is provided)
            Keys are the requested method names. Values are lists of dicts
            (one per channel pair) with the same schema as above plus a
            'method' key. For 'cross_correlation' and 'phase_locked', the
            'transfer_function' array is complex-valued.
        """
        if self._data_source == 'raw_noise':
            raise RuntimeError(
                "get_transfer_function() is not available for raw noise instances. "
                "Transfer functions were already computed during from_raw_noise(); "
                "use get_noise_transfer_function() to access them."
            )

        # Normalize a bare pair like ['Stage1', 'Ground'] to [['Stage1', 'Ground']]
        if channel_pairs and isinstance(channel_pairs[0], str):
            channel_pairs = [channel_pairs]

        # Legacy backward-compatible path: methods=None → mean_ratio only, flat list output
        if methods is None:
            results = self._compute_tf_from_traces(
                channel_pairs=channel_pairs,
                methods=['mean_ratio'],
            )
            flat_results = []
            # Strip the 'method' key from each dict to match legacy schema
            for d in results['mean_ratio']:
                legacy_dict = {k: v for k, v in d.items() if k != 'method'}
                flat_results.append(legacy_dict)

            # Cache under 'mean_ratio' key internally
            self._upsert_tf_cache('mean_ratio', flat_results)
            return flat_results

        # Normalize a single method string to a list
        if isinstance(methods, str):
            methods = [methods]

        # Validate method names
        for m in methods:
            if m not in self.VALID_TF_METHODS:
                raise ValueError(
                    f"Unknown method '{m}'. "
                    f"Valid methods: {self.VALID_TF_METHODS}"
                )

        results = self._compute_tf_from_traces(
            channel_pairs=channel_pairs,
            methods=methods,
        )

        # Cache results per method
        for method_name, method_results in results.items():
            self._upsert_tf_cache(method_name, method_results)

        return results

    def _upsert_tf_cache(self, method_name, new_results):
        """
        Upsert transfer function results into the internal cache.

        Preserves cached pairs not present in ``new_results`` and replaces
        pairs that match.

        Parameters
        ----------
        method_name : str
            The estimation method key (e.g. 'mean_ratio').
        new_results : list of dict
            New TF result dicts to cache for this method.
        """
        if self._transfer_functions is None:
            self._transfer_functions = {}

        existing = self._transfer_functions.get(method_name, [])

        # Keep existing entries whose (ch_out, ch_in) pair is not in new_results
        new_pair_keys = {
            (r['channel_output'], r['channel_input']) for r in new_results
        }
        kept = [
            d for d in existing
            if (d['channel_output'], d['channel_input']) not in new_pair_keys
        ]
        self._transfer_functions[method_name] = kept + list(new_results)

    @staticmethod
    def _log_downsample_indices(freqs, n_points):
        """
        Return a sorted, deduplicated array of frequency-bin indices that are
        approximately log-spaced over the positive part of ``freqs``, or None
        if downsampling is disabled.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array (typically linearly spaced from an FFT).
        n_points : int or None
            Target number of log-spaced points. If None or if
            ``n_points >= freqs.size``, returns None (no downsampling needed).

        Returns
        -------
        idx : np.ndarray of int, or None
            Sorted indices selecting a log-spaced subset. None if no
            downsampling should be applied.
        """
        # No downsampling requested, or fewer bins than target — use full array
        if n_points is None:
            return None
        n_total = freqs.size
        if n_points >= n_total:
            return None

        # Positive frequencies only (log requires > 0). Guard against DC bin by
        # starting from the first strictly positive frequency index.
        positive_mask = freqs > 0
        positive_indices = np.where(positive_mask)[0]
        if positive_indices.size == 0:
            return None

        i_start = int(positive_indices[0])
        i_end = n_total - 1

        # Build log-spaced indices across the positive-frequency range, convert
        # to integer bin indices, deduplicate, and sort.
        log_idx = np.unique(
            np.round(
                np.logspace(np.log10(i_start + 1), np.log10(i_end + 1), n_points)
            ).astype(int) - 1
        )
        log_idx = np.clip(log_idx, 0, i_end)

        # Preserve indices outside the positive range (e.g., DC bin) at start
        if i_start > 0:
            log_idx = np.unique(np.concatenate(([0], log_idx)))

        return log_idx

    # Linestyle mapping for multi-method overlay plots
    _METHOD_LINESTYLES = {
        'mean_ratio':       ('-',  ' (RMS ratio)'),
        'cross_correlation': ('--', ' (cross-correlation)'),
        'phase_locked':     ('-.', ' (phase-locked)'),
    }

    def _resolve_tf_pairs_to_plot(self, channel_pairs, method):
        """
        Resolve which transfer function dicts to plot based on method and channel_pairs.

        Parameters
        ----------
        channel_pairs : list of [str, str] or None
            Requested pairs. If None, all available pairs are used.
        method : str or None
            For processed data: 'mean_ratio', 'cross_correlation', 'phase_locked',
            'all' (overlay all cached methods), or None (default: mean_ratio).
            For raw noise: 'mean_ratio', 'cross_correlation', 'both', or None.

        Returns
        -------
        plot_groups : list of (list_of_tf_dicts, linestyle, label_suffix)
            Each group is a set of TF dicts to plot with the given style.
        """
        groups = []

        if self._data_source == 'raw_noise' and method is not None:
            # Raw noise mode with explicit method selection
            if self._noise_transfer_functions is None:
                raise RuntimeError(
                    "No noise transfer functions available. "
                    "Use from_raw_noise() with channel_pairs to compute them."
                )

            methods_to_plot = []
            if method in ('both', 'all'):
                methods_to_plot = [('mean_ratio', '-', ' (RMS ratio)'),
                                   ('cross_correlation', '--', ' (cross-correlation)')]
            elif method in ('mean_ratio', 'cross_correlation'):
                methods_to_plot = [(method, '-', '')]
            else:
                raise ValueError(
                    f"Unknown method '{method}'. "
                    "Use 'mean_ratio', 'cross_correlation', 'both', or 'all'."
                )

            # Resolve TF dicts for each method
            for m, ls, suffix in methods_to_plot:
                tf_list = self._noise_transfer_functions[m]

                if channel_pairs is not None:
                    tf_lookup = {
                        (d['channel_output'], d['channel_input']): d
                        for d in tf_list
                    }
                    filtered = [tf_lookup[(p[0], p[1])] for p in channel_pairs]
                else:
                    filtered = list(tf_list)

                groups.append((filtered, ls, suffix))

        elif self._data_source == 'processed' and method is not None:
            # Processed data mode with explicit method selection
            if self._transfer_functions is None:
                raise RuntimeError(
                    "No transfer functions cached. Call get_transfer_function() first "
                    "or pass channel_pairs= explicitly."
                )

            # Determine which methods to overlay.
            # method can be a single string or a list of method names.
            if isinstance(method, list):
                # Explicit list of methods to overlay
                for m in method:
                    if m not in self.VALID_TF_METHODS:
                        raise ValueError(
                            f"Unknown method '{m}'. "
                            f"Valid: {self.VALID_TF_METHODS}"
                        )
                method_keys = list(method)
            elif method == 'all':
                # Overlay all cached methods
                method_keys = [
                    m for m in self.VALID_TF_METHODS
                    if m in self._transfer_functions and self._transfer_functions[m]
                ]
            elif method in self.VALID_TF_METHODS:
                method_keys = [method]
            else:
                raise ValueError(
                    f"Unknown method '{method}'. "
                    f"Valid: {self.VALID_TF_METHODS + ('all',)}"
                )

            # Build plot groups: one per method
            show_suffix = len(method_keys) > 1
            for m in method_keys:
                if m not in self._transfer_functions:
                    raise RuntimeError(
                        f"Method '{m}' not cached. "
                        f"Call get_transfer_function(methods=['{m}']) first."
                    )
                tf_list = self._transfer_functions[m]
                ls, suffix = self._METHOD_LINESTYLES[m]

                if channel_pairs is not None:
                    tf_lookup = {
                        (d['channel_output'], d['channel_input']): d
                        for d in tf_list
                    }
                    filtered = [tf_lookup[(p[0], p[1])] for p in channel_pairs]
                else:
                    filtered = list(tf_list)

                groups.append((filtered, ls, suffix if show_suffix else ''))

        else:
            # Default behavior (method=None): use mean_ratio from cache
            if self._transfer_functions is None:
                if channel_pairs is not None and self._data_source != 'raw_noise':
                    # Auto-compute mean_ratio for missing pairs
                    self.get_transfer_function(channel_pairs=channel_pairs)
                else:
                    raise RuntimeError(
                        "No transfer functions cached. Call get_transfer_function() first "
                        "or pass channel_pairs= explicitly."
                    )

            # Prefer mean_ratio from the dict cache; fall back to flat list for legacy
            if isinstance(self._transfer_functions, dict):
                tf_list = self._transfer_functions.get('mean_ratio', [])
            else:
                tf_list = self._transfer_functions

            if channel_pairs is not None:
                # Auto-compute if needed
                if self._data_source != 'raw_noise':
                    cached_pair_keys = {
                        (d['channel_output'], d['channel_input']) for d in tf_list
                    }
                    missing = [p for p in channel_pairs if tuple(p) not in cached_pair_keys]
                    if missing:
                        self.get_transfer_function(channel_pairs=missing)
                        # Re-fetch after computation
                        if isinstance(self._transfer_functions, dict):
                            tf_list = self._transfer_functions.get('mean_ratio', [])
                        else:
                            tf_list = self._transfer_functions

                tf_lookup = {
                    (d['channel_output'], d['channel_input']): d
                    for d in tf_list
                }
                pairs_list = [tf_lookup[(p[0], p[1])] for p in channel_pairs]
            else:
                pairs_list = list(tf_list)

            groups.append((pairs_list, '-', ''))

        return groups

    def plot_transfer_function(self, channel_pairs=None, figsize=(14, 6), method=None,
                               show_uncertainty=True, log_downsample=False):
        """
        Plot the transfer function magnitude for one or more channel pairs with
        1-sigma uncertainty bands and a secondary dB axis.

        Uses internally cached transfer functions from a prior get_transfer_function()
        call when available. If any requested pair is not yet cached,
        get_transfer_function() is called automatically for the missing pairs.

        For complex-valued transfer functions (cross_correlation, phase_locked),
        the magnitude is plotted. Use plot_transfer_function_phase() for phase plots.

        Parameters
        ----------
        channel_pairs : list of [str, str] or [str, str], optional
            Each element is [channel_output, channel_input]. A bare pair like
            ['Stage1', 'Ground'] is also accepted and treated as a single-pair list.
            If None, all currently cached transfer functions are plotted.
        figsize : tuple of (float, float), optional
            Figure size passed to plt.subplots. Default: (14, 6).
        method : str or list of str, optional
            Which estimation method(s) to plot:
              - 'mean_ratio', 'cross_correlation', 'phase_locked': plot one method
              - A list like ['mean_ratio', 'phase_locked']: overlay selected methods
              - 'all': overlay all cached methods with distinct linestyles
              - 'both': alias for 'all' (backward compatibility with raw noise)
              - None: default (mean_ratio for processed data)
        show_uncertainty : bool or list of bool, optional
            Whether to draw the 1-sigma uncertainty band for each method group.
            If a single bool, applies to all method groups. If a list, must have
            one entry per method group (same length as the resolved method list).
            Default: True.
        log_downsample : bool or int, optional
            If True, logarithmically downsample the plotted line and uncertainty
            band to _LOG_DOWNSAMPLE_DEFAULT_POINTS log-spaced frequency bins
            before drawing. If an int, use that many points. Useful when raw-noise
            PSDs have millions of linear bins, which can exceed matplotlib Agg's
            cell block limit during rendering. The underlying cached transfer
            function data is not modified. Default: False.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        # Resolve the number of log-spaced points to use if downsampling is on
        if log_downsample is False or log_downsample is None:
            n_downsample = None
        elif log_downsample is True:
            n_downsample = self._LOG_DOWNSAMPLE_DEFAULT_POINTS
        else:
            n_downsample = int(log_downsample)
        # Normalize a bare pair like ['Stage1', 'Ground'] to [['Stage1', 'Ground']]
        if channel_pairs is not None and channel_pairs and isinstance(channel_pairs[0], str):
            channel_pairs = [channel_pairs]

        # Resolve which TF dicts to plot, grouped by method/linestyle
        plot_groups = self._resolve_tf_pairs_to_plot(
            channel_pairs=channel_pairs, method=method,
        )

        # Normalize show_uncertainty to a list with one bool per plot group
        n_groups = len(plot_groups)
        if isinstance(show_uncertainty, bool):
            show_uncertainty_per_group = [show_uncertainty] * n_groups
        else:
            show_uncertainty_per_group = list(show_uncertainty)
            if len(show_uncertainty_per_group) != n_groups:
                raise ValueError(
                    f"show_uncertainty has {len(show_uncertainty_per_group)} entries "
                    f"but there are {n_groups} method group(s) to plot. "
                    "Pass a single bool or a list matching the number of methods."
                )

        fig, ax = plt.subplots(figsize=figsize)
        prop_cycle_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

        all_tf_values = []
        color_index = 0

        # Plot each group (one group per method when overlaying multiple methods)
        for group_idx, (pairs_to_plot, linestyle, label_suffix) in enumerate(plot_groups):
            draw_uncertainty = show_uncertainty_per_group[group_idx]

            for i, tf_dict in enumerate(pairs_to_plot):
                ch_out = tf_dict['channel_output']
                ch_in  = tf_dict['channel_input']
                freqs  = tf_dict['freqs']
                tf_raw = tf_dict['transfer_function']
                tf_sig = tf_dict['transfer_sigma']
                color  = prop_cycle_colors[(color_index + i) % len(prop_cycle_colors)]

                # For complex TFs (cross_correlation, phase_locked), plot magnitude
                tf_mag = np.abs(tf_raw) if np.iscomplexobj(tf_raw) else tf_raw
                all_tf_values.append(tf_mag)

                # Optionally select a log-spaced subset of frequency indices. This
                # keeps the plotted polygon/line size manageable for raw-noise PSDs
                # that have millions of linear FFT bins (which can otherwise exceed
                # matplotlib Agg's cell block limit when drawing fill_between).
                plot_idx = self._log_downsample_indices(freqs, n_downsample)
                plot_freqs = freqs if plot_idx is None else freqs[plot_idx]
                plot_tf_mag = tf_mag if plot_idx is None else tf_mag[plot_idx]

                # Plot transfer function magnitude
                ax.loglog(
                    plot_freqs,
                    plot_tf_mag,
                    marker="o",
                    markersize=0,
                    linestyle=linestyle,
                    label=f"{ch_out} / {ch_in}{label_suffix}",
                    color=color,
                    alpha=0.5,
                )

                # Draw uncertainty band if enabled and sigma is not all NaN
                if draw_uncertainty and not np.all(np.isnan(tf_sig)):
                    plot_tf_sig = tf_sig if plot_idx is None else tf_sig[plot_idx]

                    # Build 1-sigma bounds; keep lower bound positive for log-scale
                    lower = np.maximum(plot_tf_mag - plot_tf_sig,
                                       np.finfo(float).tiny)
                    upper = plot_tf_mag + plot_tf_sig

                    # Mask non-finite bounds — fill_between treats NaN as a polygon
                    # break, which avoids drawing to infinite y on log scale.
                    bad = ~np.isfinite(lower) | ~np.isfinite(upper)
                    lower = np.where(bad, np.nan, lower)
                    upper = np.where(bad, np.nan, upper)

                    ax.fill_between(
                        plot_freqs,
                        lower,
                        upper,
                        color=color,
                        alpha=0.2,
                        linewidth=1,
                        label=f"±σ/√n{label_suffix}",
                    )

            # For multi-method overlay, keep same color mapping across methods
            if method not in ('both', 'all') and not isinstance(method, list):
                color_index = color_index + len(pairs_to_plot)

        # Set y-limits across all plotted transfer functions
        all_tf_concat = np.concatenate(all_tf_values)
        finite_vals = all_tf_concat[np.isfinite(all_tf_concat) & (all_tf_concat > 0)]
        if finite_vals.size > 0:
            ax.set_ylim(finite_vals.min() * 0.8, finite_vals.max() * 1.2)

        ax.set_xlabel("Frequency (Hz)")

        # Collect all unique pairs across all groups for labeling
        all_pairs_flat = [
            tf_dict for group in plot_groups for tf_dict in group[0]
        ]
        unique_pair_keys = list(dict.fromkeys(
            (d['channel_output'], d['channel_input']) for d in all_pairs_flat
        ))

        # Single-pair label names the channels; multi-pair uses a generic label
        if len(unique_pair_keys) == 1:
            ch_out, ch_in = unique_pair_keys[0]
            ax.set_ylabel(f"Attenuation ({ch_out}/{ch_in})")
        else:
            ax.set_ylabel("Attenuation")

        ax.grid(True, which="both", ls=":")
        ax.tick_params(direction='in')
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

    def plot_transfer_function_phase(self, channel_pairs=None, figsize=(14, 6),
                                     method=None):
        """
        Plot the transfer function phase for one or more channel pairs.

        Only works with complex-valued transfer function methods
        ('cross_correlation' or 'phase_locked'). The 'mean_ratio' method
        discards phase information and cannot be plotted here.

        Parameters
        ----------
        channel_pairs : list of [str, str] or [str, str], optional
            Each element is [channel_output, channel_input]. A bare pair like
            ['Stage1', 'Ground'] is also accepted.
            If None, all currently cached pairs for the given method are plotted.
        figsize : tuple of (float, float), optional
            Figure size passed to plt.subplots. Default: (14, 6).
        method : str, optional
            Which estimation method to plot: 'cross_correlation', 'phase_locked',
            or 'all' (overlay both). Default: 'phase_locked' if cached, else
            'cross_correlation'.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        # Normalize a bare pair like ['Stage1', 'Ground'] to [['Stage1', 'Ground']]
        if channel_pairs is not None and channel_pairs and isinstance(channel_pairs[0], str):
            channel_pairs = [channel_pairs]

        # Determine which complex methods to plot
        complex_methods = ('cross_correlation', 'phase_locked')

        if method is None:
            # Auto-select: prefer phase_locked if cached, else cross_correlation
            if (self._transfer_functions is not None
                    and 'phase_locked' in self._transfer_functions):
                method = 'phase_locked'
            elif (self._transfer_functions is not None
                    and 'cross_correlation' in self._transfer_functions):
                method = 'cross_correlation'
            else:
                raise RuntimeError(
                    "No complex-valued transfer functions cached. "
                    "Call get_transfer_function(methods=['phase_locked']) or "
                    "get_transfer_function(methods=['cross_correlation']) first."
                )

        # Reject mean_ratio (no phase info) whether passed as string or in a list
        if method == 'mean_ratio':
            raise ValueError(
                "The 'mean_ratio' method discards phase information. "
                "Use 'cross_correlation' or 'phase_locked' for phase plots."
            )
        if isinstance(method, list) and 'mean_ratio' in method:
            raise ValueError(
                "The 'mean_ratio' method discards phase information. "
                "Use 'cross_correlation' or 'phase_locked' for phase plots."
            )

        # Build list of methods to overlay.
        # method can be a single string or a list of method names.
        if isinstance(method, list):
            for m in method:
                if m not in complex_methods:
                    raise ValueError(
                        f"Unknown or non-complex method '{m}'. "
                        f"Valid for phase plots: {complex_methods}"
                    )
            method_keys = list(method)
        elif method == 'all':
            method_keys = [
                m for m in complex_methods
                if (self._transfer_functions is not None
                    and m in self._transfer_functions
                    and self._transfer_functions[m])
            ]
            if not method_keys:
                raise RuntimeError(
                    "No complex-valued transfer functions cached. "
                    "Call get_transfer_function(methods=[...]) first."
                )
        elif method in complex_methods:
            method_keys = [method]
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Valid for phase plots: {complex_methods + ('all',)}"
            )

        fig, ax = plt.subplots(figsize=figsize)
        prop_cycle_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

        show_suffix = len(method_keys) > 1
        color_index = 0

        # Plot each method group
        for m in method_keys:
            if m not in self._transfer_functions:
                raise RuntimeError(
                    f"Method '{m}' not cached. "
                    f"Call get_transfer_function(methods=['{m}']) first."
                )

            tf_list = self._transfer_functions[m]
            ls, suffix = self._METHOD_LINESTYLES[m]

            # Filter to requested pairs if specified
            if channel_pairs is not None:
                tf_lookup = {
                    (d['channel_output'], d['channel_input']): d
                    for d in tf_list
                }
                pairs_to_plot = [tf_lookup[(p[0], p[1])] for p in channel_pairs]
            else:
                pairs_to_plot = list(tf_list)

            # Plot phase for each pair
            for i, tf_dict in enumerate(pairs_to_plot):
                ch_out = tf_dict['channel_output']
                ch_in = tf_dict['channel_input']
                freqs = tf_dict['freqs']
                tf_complex = tf_dict['transfer_function']
                color = prop_cycle_colors[(color_index + i) % len(prop_cycle_colors)]

                # Extract phase in degrees
                phase_deg = np.angle(tf_complex, deg=True)

                ax.semilogx(
                    freqs,
                    phase_deg,
                    marker="o",
                    markersize=0,
                    linestyle=ls,
                    label=f"{ch_out} / {ch_in}{suffix if show_suffix else ''}",
                    color=color,
                    alpha=0.7,
                )

            # Keep same color mapping across methods in overlay mode
            if not show_suffix:
                color_index = color_index + len(pairs_to_plot)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase (degrees)")
        ax.set_ylim(-180, 180)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.grid(True, which="both", ls=":")
        ax.legend()

        return fig, ax

    def plot_psd(
        self,
        channels=None,
        figsize=(14, 6),
        show_variance=True,
        colors=None,
        show_spectral_noise=True,
        spectral_noise_color='tab:red',
        spectral_noise_dict=None,
    ):
        """
        Plot the PSD for one or more channels with 1-sigma uncertainty bands and the
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
        show_variance : bool, optional
            If True, draw the ±1-sigma uncertainty band around each
            channel's PSD. Default: True.
        colors : dict of {str: str}, optional
            Mapping from channel name to matplotlib color. Channels not in
            the mapping fall back to the default property cycle.
        show_spectral_noise : bool, optional
            If True, draw the intrinsic accelerometer noise floor as a
            piecewise-constant horizontal line. Default: True.
        spectral_noise_color : str, optional
            Color for the intrinsic noise floor line. Default: 'tab:red'.
        spectral_noise_dict : dict, optional
            Optional dictionary specifying the noise floor to plot. If None,
            the default noise floor for the PCB Piezotronics Model 393B04 is used.
            If provided, it must contain:
              'freqs_hz' : list or np.array of float
                  Frequencies (Hz) at which the noise floor is defined.
              'noise_floor_g_per_sqrt_hz' : list or np.array of float
                  Noise floor values (g/√Hz) corresponding to the frequencies.

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

        # Ensure PSDs are available for every requested channel
        missing = [ch for ch in channels if self._channels is None or ch not in self._channels]
        if missing:
            if self._data_source == 'raw_noise':
                raise ValueError(
                    f"Channels {missing} were not included in from_raw_noise(). "
                    "Cannot compute additional PSDs from raw noise after construction."
                )
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
            # Use caller-supplied color for this channel if provided,
            # otherwise fall back to the default matplotlib prop cycle
            if (colors is not None) and (chan in colors):
                color = colors[chan]
            else:
                color = prop_cycle_colors[i % len(prop_cycle_colors)]

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

            # Optionally draw the ±1-sigma uncertainty band
            if show_variance:
                # Build 1-sigma bounds; keep lower bound positive for log-scale plotting
                lower = np.maximum(amp - sigma, np.finfo(float).tiny)
                upper = amp + sigma

                # Draw uncertainty band; the band is intentionally unlabeled
                # so it does not clutter the legend
                ax.fill_between(
                    freqs,
                    lower,
                    upper,
                    color=color,
                    alpha=0.2,
                    linewidth=1,
                )

        # Plot intrinsic noise floor as piecewise-constant horizontal segments
        if show_spectral_noise:
            # Resolve the noise floor to draw: caller-supplied dict or class default
            if spectral_noise_dict is not None:
                noise_freqs  = np.asarray(spectral_noise_dict['freqs_hz'])
                noise_values = np.asarray(spectral_noise_dict['noise_floor_g_per_sqrt_hz'])
            else:
                noise_freqs  = self._NOISE_FLOOR_FREQS_HZ
                noise_values = self._NOISE_FLOOR_G_PER_SQRT_HZ

            for k in range(len(noise_freqs)):
                f_start = noise_freqs[k]
                if k < len(noise_freqs) - 1:
                    f_end = noise_freqs[k + 1]
                else:
                    f_end = freq_end

                # Label only the first segment so the legend has a single entry
                if k == 0:
                    segment_label = "Intrinsic Spectral Noise"
                else:
                    segment_label = None

                ax.hlines(
                    noise_values[k],
                    f_start,
                    f_end,
                    colors=spectral_noise_color,
                    linestyles="dashdot",
                    alpha=0.5,
                    label=segment_label,
                )

        ax.set_xlim(np.nanmin(freqs), freq_end)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"Transducer Freq (g/$\sqrt{\mathrm{Hz}}$)")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.tick_params(direction='in')
        ax.legend()
        ax.set_ylim([min_amp*0.8, max_amp*1.2])

        return fig, ax
