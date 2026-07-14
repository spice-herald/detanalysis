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
    if total_events < 1:
        return 0
    return (total_events - 1) // downsample_factor + 1


class Vibration_Analyzer(Analyzer):
    """
    Inherits detanalysis.Analyzer, adds vibration-specific analysis methods
    for accelerometer data stored in processed HDF5 files.
    Acquire processed data from the process_transducer_sweep.py script in detprocess.

    Two data types are supported, selected by the data_type argument to the initializer:
        - 'transducer_sweep'    : processed amplitude HDF5 files (from process_transducer_sweep.py).
        - 'continuous'          : raw continuous data HDF5 files (from running the daq normally with --acquire-cont)

    Adds the following methods to the base Analyzer class:
        - calc_psd()                      : compute PSDs and err, cached for reuse by plot_psd().
        - calc_transfer_function()        : compute transfer functions from accelerometer data.
                                            Sweep data supports estimators rms-ratio, cross-correlation,
                                            and phase-locked; continuous data supports rms-ratio and
                                            cross-correlation. Results are cached for the plotting methods.
        - plot_psd()                      : plot PSD for one or more channels
        - plot_transfer_function()        : plot transfer function magnitude
        - plot_transfer_function_phase()  : plot transfer function phase (complex methods only; rms-ratio not supported)

    """

    # Valid transfer function estimator name strings.
    VALID_TF_METHODS = ('rms-ratio', 'cross-correlation', 'phase-locked')

    # Default number of log-spaced points used by plotting functions when log_downsample=True. 
    # (matplotlib will sometimes produce "cell block limit exceeded" error if too much data is plotted at once)
    _LOG_DOWNSAMPLE_DEFAULT_POINTS = 20000

    # Intrinsic accelerometer noise floor.
    # Calculated from Model 393B04 Seismic Accelerometer, PCB Piezotronics, Inc. (https://www.pcb.com/products?m=393b04).
    # value at index k applies from FREQS_HZ[k] to FREQS_HZ[k+1].
    _NOISE_FLOOR_FREQS_HZ      = np.array([0.1, 10.0, 100.0, 1000.0])
    _NOISE_FLOOR_G_PER_SQRT_HZ = np.array([0.30, 0.10, 0.04, 0.04]) * 1e-6

    # Valid data_type strings accepted by the initializer.
    VALID_DATA_TYPES = ('transducer_sweep', 'continuous')

    # Transfer function estimators available for continuous data (subset of VALID_TF_METHODS).
    CONTINUOUS_TF_METHODS = ('rms-ratio', 'cross-correlation')

    def __init__(self, paths, data_type, series=None, **kwargs):
        """
        Parameters
        ----------
        paths : str or list
            For data_type='transducer_sweep', path(s) to processed amplitude
            HDF5 file(s), passed directly to Analyzer.__init__() base class
            initializer. For data_type='continuous', the path to the directory
            containing raw continuous data HDF5 files.
        data_type : str
            Which data type to use. One of 'transducer_sweep' or 'continuous'.
        series : str or list, optional
            Filter files by series name. Default: all files in paths.
            Formatted like 'I2_D20260622_T221904' (without the prefix 'cont_', 'amp_', etc; and without the dump number F00001...)
        **kwargs
            Additional keyword arguments forwarded to Analyzer.__init__
            (e.g. analysis_repo, use_vaex_cut_handling, memory_cache_size).
            Ignored for data_type='continuous'.
        """
        if data_type not in self.VALID_DATA_TYPES:
            raise ValueError(
                f"Unknown data_type '{data_type}'. "
                f"Valid options: {self.VALID_DATA_TYPES}."
            )

        if data_type == 'transducer_sweep':
            super().__init__(paths, series=series, **kwargs)
            self._raw_path    = None
            self._data_source = 'processed'
        else:
            # Continuous data: there is no processed DataFrame to load, so
            # the base Analyzer initializer is skipped. Set the base attributes
            # to safe defaults so inherited methods do not see missing state.
            self._file_list             = None
            self._nfiles                = None
            self._df                    = None
            self._is_df_filtered        = False
            self._nevents               = None
            self._nevents_nofilter      = None
            self._nfeatures             = None
            self._feature_names         = None
            self._load_from_pandas      = False
            self._cuts                  = None
            self._derived_features      = None
            self._use_vaex_cut_handling = False
            self._analysis_repo_path    = None
            self._analysis_repo         = None

            self._raw_path    = paths
            self._series      = series
            self._data_source = 'continuous_data'

        # Shared moment cache: the statistical moments from which both the
        # PSDs and every transfer function estimator are derived. Populated by
        # _ensure_moments(), consumed by calc_psd() and calc_transfer_function().
        self._moments            = None   # dict, or None until first computed

        # Derived PSD cache, populated by calc_psd() and consumed by plot_psd()
        self._psd                = None   # np.ndarray, shape (n_channels, n_freqs)
        self._variance           = None   # np.ndarray, shape (n_channels, n_freqs)
        self._freqs              = None   # np.ndarray, shape (n_freqs,)
        self._channels           = None   # list of str; channel names, index matches _psd rows
        self._transfer_functions = None   # dict of {method: {(ch_out, ch_in): dict}}, populated by calc_transfer_function()
        self._data_type          = data_type

    def _accumulate_moments_continuous(self, channels, accel_gain=100.0,
                                       downsample_factor=1,
                                       trace_length_samples=None,
                                       trace_length_msec=None, verbose=True):
        """
        Read raw continuous data and accumulate the per-frequency moments.
        Only used for the 'continuous' data type.

        This is the moment-accumulation engine for the continuous data. It
        reads the raw accelerometer traces stored under self._raw_path,
        FFTs each trace, and accumulates the sufficient statistical moments
        required from which the transfer functions and PSDs are derived.

        For each frequency bin the following are accumulated over the trace
        ensemble (a_i is the one-sided-normalized complex FFT amplitude of
        channel i, so that ⟨|a_i|²⟩ is the one-sided PSD in g^2/Hz):
            S_ij = ⟨a_i a_j*⟩           cross-spectral density matrix (Hermitian)
            R_ij = ⟨|a_i|² |a_j|²⟩       fourth-order matrix (for uncertainties)

        The mean phasors ⟨a_i⟩ are not accumulated: continuous vibration is
        random-phase, so ⟨a_i⟩ averages to zero and the phase-locked estimator is
        not meaningful (hence unsupported for continuous data).

        Parameters
        ----------
        channels : list of str
        accel_gain : float, optional
            Accelerometer gain factor. Raw ADC values are divided by this
            to convert to g. Default: 100.0.
        downsample_factor : int, optional
            Process every Nth event (1 = all events). Default: 1.
        trace_length_samples : int, optional
            Desired trace length, in samples. If None (default), 
            the native per-event sample count stored in the raw
            HDF5 files is used.

            If set, the samples from every (downsampled) event are
            concatenated along the time axis into a single continuous
            per-channel stream, and that stream is re-chopped into
            non-overlapping chunks of exactly ``trace_length_samples``.
            Any incomplete remainder at the end of the stream (shorter
            than one full chunk) is discarded.

            Mutually exclusive with ``trace_length_msec``.
        trace_length_msec : float, optional
            Same behavior as ``trace_length_samples``, but specified in
            milliseconds. 
            Mutually exclusive with ``trace_length_samples``.
        verbose : bool, optional
            Print progress information. Default: True.

        Return
        ------
        moments : dict
            The moment cache, with keys:
              'channels' : list of str; index order for the S and R matrices
              'freqs'    : np.ndarray, shape (n_freqs,)
              'counts'   : np.ndarray, shape (n_freqs,); trace count per bin
              'S'        : np.ndarray, shape (n_channels, n_channels, n_freqs),
                           complex; the CSD matrix, whose diagonal is the PSD
              'R'        : np.ndarray, shape (n_channels, n_channels, n_freqs),
                           real; the fourth-order matrix, whose diagonal is ⟨|a_i|⁴⟩
              'm'        : None; mean phasors are not meaningful for continuous data
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

        # Load metadata via detprocess.RawData
        rawdata = detprocess.RawData(self._raw_path, data_type='cont', verbose=verbose)
        sample_rate = rawdata.get_sample_rate(data_type='cont')
        # Count events for the selected series only (matches the H5Reader's
        # series filtering below), so total_events and the progress bar reflect
        # what is actually read rather than the whole dataset.
        duration, total_events = rawdata.get_duration(
            series=self._series, data_type='cont', include_nb_events=True,
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
            if downsample_factor == 1:
                print(f"Processing {total_events} events.")
            else:
                print(
                    f"Total events available: {total_events}; processing "
                    f"{n_kept_events} after downsampling (factor {downsample_factor})."
                )

        # Set up H5Reader for event streaming
        h5reader = h5io.H5Reader()
        h5reader.set_files(self._raw_path, series=self._series)

        n_channels = len(channels)

        # Moment accumulators, initialized on the first trace (n_freqs is not
        # known until then).
        # sum_S[i, j, f] accumulates a_i(f) * conj(a_j(f)); dividing by the trace
        # count gives the CSD matrix S_ij. sum_R[i, j, f] accumulates
        # |a_i(f)|^2 * |a_j(f)|^2; dividing by the count gives the fourth-order
        # matrix R_ij used for the uncertainty propagation.
        sum_S = None         # complex, shape (n_channels, n_channels, n_freqs)
        sum_R = None         # real, shape (n_channels, n_channels, n_freqs)
        scale_sqrt = None    # per-bin one-sided normalization, shape (n_freqs,)

        # Counts the number of traces (one per event in the default path,
        # one per produced chunk when rechunking) folded into the estimate.
        n_events_processed = 0
        event_index = 0

        # Rolling per-channel buffer used only when trace_length_samples is set
        rechunk_buffer = None
        freqs = None  # set on the first processed trace

        def _process_one_trace(trace_2d):
            """
            Fold a single (n_channels, n_samples) trace into the running S and R
            moment accumulators.

            On the first invocation, lazily initializes sum_S / sum_R / freqs /
            scale_sqrt based on the incoming n_samples.
            """
            nonlocal sum_S, sum_R, scale_sqrt, freqs, n_events_processed

            # Number of samples in this trace (may differ from native event length
            # when rechunking is enabled)
            n_samples = trace_2d.shape[-1]

            # First-trace initialization: frequency axis, moment accumulators,
            # and the one-sided PSD normalization.
            if sum_S is None:
                n_freqs_local = n_samples // 2 + 1
                freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
                sum_S = np.zeros((n_channels, n_channels, n_freqs_local), dtype=complex)
                sum_R = np.zeros((n_channels, n_channels, n_freqs_local))

                # One-sided PSD normalization: |a_i|^2 = |fft_i|^2 * scale yields
                # the one-sided PSD (g^2/Hz). Interior bins are doubled to fold in
                # the negative frequencies; DC and (for even n_samples) Nyquist
                # are not. This per-bin factor is common to all channels, so it
                # cancels in every transfer function ratio.
                scale = np.full(n_freqs_local, 2.0 / (sample_rate * n_samples))
                scale[0] = 1.0 / (sample_rate * n_samples)
                if n_samples % 2 == 0:
                    scale[-1] = 1.0 / (sample_rate * n_samples)
                scale_sqrt = np.sqrt(scale)

            # Bump trace counter
            n_events_processed = n_events_processed + 1

            # FFT every channel at once and apply the one-sided normalization.
            # a has shape (n_channels, n_freqs).
            a = np.fft.rfft(trace_2d, axis=1) * scale_sqrt[None, :]
            a_conj = np.conj(a)

            # Accumulate the CSD matrix S_ij = a_i a_j* and the fourth-order
            # matrix R_ij = |a_i|^2 |a_j|^2 via outer products over the channel
            # axis at every frequency bin.
            sum_S = sum_S + a[:, None, :] * a_conj[None, :, :]
            power = (a.real ** 2 + a.imag ** 2)   # |a_i|^2, shape (n_channels, n_freqs)
            sum_R = sum_R + power[:, None, :] * power[None, :, :]

        # Event loop: read raw traces and accumulate PSD statistics
        pbar = tqdm(
            total=total_events,
            desc="Processing continuous data events",
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
                f"Only {n_events_processed} event(s) processed; need at least 2 "
                "for variance estimation."
            )

        # Finalize the moment cache. Dividing the running sums by the trace count
        # turns them into the mean moments S_ij = ⟨a_i a_j*⟩ and R_ij =
        # ⟨|a_i|² |a_j|²⟩. Every trace contributes to every frequency bin, so the
        # count is constant across the frequency axis.
        S = sum_S / n_events_processed
        R = sum_R / n_events_processed
        counts = np.full(freqs.size, n_events_processed, dtype=int)

        return {
            'channels': list(channels),
            'freqs': freqs,
            'counts': counts,
            'S': S,
            'R': R,
            'm': None,
        }

    def _moments_from_dataframe(self, channels):
        """
        Build the per-frequency moment cache from the processed sweep DataFrame.

        For each unique drive frequency, the per-trace complex amplitudes
        a_i = (amp_real_i + 1j amp_imag_i) / sqrt(Δf) are formed (the 1/sqrt(Δf)
        normalization makes ⟨|a_i|²⟩ the PSD in g^2/Hz, matching the continuous
        data treatment), and the mean phasors, CSD matrix, and fourth-order matrix are
        computed across the trace ensemble at that frequency.

        Parameters
        ----------
        channels : list of str

        Return
        ------
        moments : dict
            Same schema as _accumulate_moments_continuous(), except 'm' is the
            per-channel mean phasor array (shape (n_channels, n_freqs), complex)
            rather than None, which enables the phase-locked estimator.
        """
        df = self.df

        freqs = sorted(df['frequency_hz'].unique())
        trace_length_msec = df['trace_length_msec'].unique()[0]

        # Frequency resolution: Δf = 1 / T, where T is the trace duration in seconds
        freq_resolution_hz = 1.0 / (trace_length_msec * 1e-3)
        inv_sqrt_df = 1.0 / np.sqrt(freq_resolution_hz)

        n_channels = len(channels)
        n_freqs = len(freqs)

        S = np.zeros((n_channels, n_channels, n_freqs), dtype=complex)
        R = np.zeros((n_channels, n_channels, n_freqs))
        m = np.zeros((n_channels, n_freqs), dtype=complex)
        counts = np.zeros(n_freqs, dtype=int)

        # Loop over each unique drive frequency and reduce the per-trace
        # amplitudes into the moment matrices.
        for f_idx, freq in enumerate(freqs):
            rows = df[df.frequency_hz == freq]
            n_traces = len(rows)
            counts[f_idx] = n_traces

            # Build normalized complex amplitudes a, shape (n_channels, n_traces)
            a = np.zeros((n_channels, n_traces), dtype=complex)
            for c_idx, channel in enumerate(channels):
                real = np.array(rows[f'amp_real_{channel}'].values, dtype=float)
                imag = np.array(rows[f'amp_imag_{channel}'].values, dtype=float)
                a[c_idx, :] = (real + 1j * imag) * inv_sqrt_df

            # Mean phasor ⟨a_i⟩ per channel
            m[:, f_idx] = a.mean(axis=1)

            # CSD matrix S_ij = ⟨a_i a_j*⟩ and fourth-order matrix
            # R_ij = ⟨|a_i|² |a_j|²⟩, averaged over the trace ensemble.
            S[:, :, f_idx] = np.dot(a, np.conj(a).T) / n_traces
            power = (a.real ** 2 + a.imag ** 2)   # |a_i|^2, shape (n_channels, n_traces)
            R[:, :, f_idx] = np.dot(power, power.T) / n_traces

        return {
            'channels': list(channels),
            'freqs': np.array(freqs),
            'counts': counts,
            'S': S,
            'R': R,
            'm': m,
        }

    def _ensure_moments(self, channels, accel_gain=100.0, downsample_factor=1,
                        trace_length_samples=None, trace_length_msec=None,
                        verbose=True, force_overwrite=False):
        """
        Ensure the shared moment cache covers every requested channel.

        If the cache already exists and contains all of the requested channels,
        it is reused unchanged (this is what lets calc_transfer_function() run
        without re-reading the data after calc_psd(), and vice versa). Otherwise
        the cache is rebuilt over the union of any previously cached channels and
        the requested ones, so earlier work is not lost.

        When force_overwrite is True the existing cache is ignored and rebuilt from scratch.

        For the continuous path the extra arguments control the raw-data read;
        they are ignored when the cache is reused or for the sweep path.

        Parameters
        ----------
        channels : list of str
            Channel names that must be present in the cache.
        accel_gain, downsample_factor, trace_length_samples, trace_length_msec, verbose
            Continuous-path read parameters, forwarded to
            _accumulate_moments_continuous() when a (re)build is needed.
        force_overwrite : bool, optional
            If True, ignore any existing cache and rebuild from scratch over the
            requested channels. Default: False.

        Return
        ------
        None
        """
        channels = list(channels)

        # Reuse the existing cache when it already covers every requested channel
        # (unless a forced recomputation was requested).
        if (not force_overwrite) and self._moments is not None and all(
            ch in self._moments['channels'] for ch in channels
        ):
            return

        # Choose the channels to build. A forced rebuild starts from scratch over
        # just the requested channels; otherwise extend the union of cached and
        # requested channels so earlier work is not lost.
        if (not force_overwrite) and self._moments is not None:
            build_channels = list(dict.fromkeys(self._moments['channels'] + channels))
        else:
            build_channels = channels

        if self._data_source == 'continuous_data':
            self._moments = self._accumulate_moments_continuous(
                channels=build_channels,
                accel_gain=accel_gain,
                downsample_factor=downsample_factor,
                trace_length_samples=trace_length_samples,
                trace_length_msec=trace_length_msec,
                verbose=verbose,
            )
        else:
            self._moments = self._moments_from_dataframe(build_channels)

    @staticmethod
    def _variance_of_mean(mean_sq_magnitude, abs_mean_squared, counts):
        """
        Variance of a sample mean from its first two moments.

        For a quantity x the variance of the sample mean over N traces is
        (⟨|x|²⟩ - |⟨x⟩|²) / (N - 1). This is applied elementwise across the
        frequency axis and works for both real and complex x (the magnitudes make
        it valid for the complex cross-spectrum and mean phasors).

        Parameters
        ----------
        mean_sq_magnitude : np.ndarray
            ⟨|x|²⟩ at each frequency (real).
        abs_mean_squared : np.ndarray
            |⟨x⟩|² at each frequency (real).
        counts : np.ndarray
            Trace count N at each frequency.

        Return
        ------
        var_mean : np.ndarray
            Variance of the sample mean at each frequency. NaN where N < 2.
        """
        counts = np.asarray(counts, dtype=float)
        denom = counts - 1.0

        # Guard tiny negative values from floating-point round-off.
        pop_var = np.maximum(mean_sq_magnitude - abs_mean_squared, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            var_mean = np.where(denom > 0, pop_var / denom, np.nan)
        return var_mean

    def describe(self):
        """
        Display a summary of the loaded data.

        For the transducer sweep path this defers to Analyzer.describe(). For the
        continuous data path there is no processed DataFrame, so a message saying
        so is printed instead.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        if self._data_source == 'continuous_data':
            print("No processed dataframe to display for continuous data.\n" \
            "Only available if data_type='transducer_sweep', and processed data file is passed.")
            return
        super().describe()

    def calc_psd(self, channels, return_freqs=True, accel_gain=100.0,
                 downsample_factor=1, trace_length_samples=None,
                 trace_length_msec=None, verbose=True, force_overwrite=False):
        """
        Calculate and return the PSD.

        Technically, the PSD is derived directly from the moment cache, so most of the
        work is computing PSD and ASD variance for each channel and cache them too.

        Parameters
        ----------
        channels : list of str
        return_freqs : bool, optional
        accel_gain : float, optional
            Continuous data only. Accelerometer gain factor; raw ADC values are
            divided by this to convert to g. Ignored for sweep data. Default: 100.0.
        downsample_factor : int, optional
            Continuous data only. Process every Nth event (1 = all events).
            Ignored for sweep data. Default: 1.
        trace_length_samples : int, optional
            Continuous data only. See _accumulate_moments_continuous() for the
            rechunking behavior. Ignored for sweep data. Default: None.
        trace_length_msec : float, optional
            Continuous data only. Same as trace_length_samples but in
            milliseconds. Ignored for sweep data. Default: None.
        verbose : bool, optional
            Continuous data only. Print progress information. Default: True.
        force_overwrite : bool, optional
            If True, ignore the shared moment cache and recompute it from scratch
            (re-reading the data) before deriving the PSDs. Default: False.

        Returns
        -------
        psd : np.ndarray, shape (n_channels, n_freqs)
            Power spectral density in g^2/Hz.
        variance : np.ndarray, shape (n_channels, n_freqs)
            Variance of the ASD estimate (g/√Hz)^2, via error propagation from
            the PSD variance. Use np.sqrt(variance) to get 1-sigma uncertainty.
        freqs : np.ndarray, shape (n_freqs,)  [only if return_freqs=True]
            Sorted unique frequencies in Hz.
        """
        channels = list(channels)

        # Ensure the moment cache covers the requested channels (builds it if
        # needed; reuses it if calc_transfer_function() already did, unless
        # force_overwrite requests a fresh recomputation).
        self._ensure_moments(
            channels=channels,
            accel_gain=accel_gain,
            downsample_factor=downsample_factor,
            trace_length_samples=trace_length_samples,
            trace_length_msec=trace_length_msec,
            verbose=verbose,
            force_overwrite=force_overwrite,
        )

        moments      = self._moments
        mom_channels = moments['channels']
        freqs        = np.array(moments['freqs'])
        S            = moments['S']
        R            = moments['R']
        counts       = moments['counts']

        n_requested = len(channels)
        psd      = np.zeros((n_requested, freqs.size))
        variance = np.zeros((n_requested, freqs.size))

        # Derive PSD and ASD variance per requested channel from the diagonal of
        # the CSD matrix (PSD) and the fourth-order matrix (its variance).
        for out_idx, channel in enumerate(channels):
            i = mom_channels.index(channel)

            # PSD_i = ⟨|a_i|²⟩ = S_ii (real)
            psd_i = S[i, i, :].real
            psd[out_idx, :] = psd_i

            # Var(PSD_i) = (⟨|a_i|⁴⟩ − ⟨|a_i|²⟩²) / (N − 1) = (R_ii − S_ii²)/(N−1)
            var_psd = self._variance_of_mean(R[i, i, :].real, psd_i ** 2, counts)

            # Error propagation PSD -> ASD: d(ASD)/d(PSD) = 1/(2 sqrt(PSD))
            # so var_ASD = var_PSD / (4 PSD).
            with np.errstate(divide='ignore', invalid='ignore'):
                var_asd = np.where(psd_i > 0, var_psd / (4.0 * psd_i), 0.0)
            variance[out_idx, :] = var_asd

        # Cache results for use by plot_psd()
        self._psd      = psd
        self._variance = variance
        self._freqs    = freqs
        self._channels = channels

        if return_freqs:
            return psd, variance, freqs
        return psd, variance

    def _estimators_from_moments(self, channel_pairs, methods):
        """
        Derive transfer function estimators from the shared statistical moment cache.

        Operates entirely on self._moments caache; _ensure_moments() must be called first.

        The estimators, for output channel o and input channel i, are:
          - rms-ratio        = sqrt(S_oo / S_ii)    (real)
          - cross-correlation = S_oi / S_ii         (complex)
          - phase-locked     = m_o / m_i            (complex; transducer sweep only)

        The 1-sigma uncertainty on each estimator's magnitude is propagated from
        the variances of the underlying moment means, computed via
        _variance_of_mean() from S and R (and m for phase-locked).

        Parameters
        ----------
        channel_pairs : list of [str, str]
            Each element is [channel_output, channel_input].
        methods : list of str
            Subset of VALID_TF_METHODS: 'rms-ratio', 'cross-correlation',
            'phase-locked'.

        Returns
        -------
        results : dict
            Keys are method names from ``methods``. Each value is a dict keyed by
            (channel_output, channel_input) tuples, one entry per channel pair.
            Each inner dict contains:
              'channel_output'    : str
              'channel_input'     : str
              'transfer_function' : np.ndarray; real for rms-ratio, complex
                                    for cross-correlation and phase-locked
              'transfer_sigma'    : np.ndarray; 1-sigma magnitude uncertainty
              'freqs'             : np.ndarray, shape (n_freqs,)
              'method'            : str
        """
        moments = self._moments
        chans   = moments['channels']
        freqs   = np.array(moments['freqs'])
        S       = moments['S']
        R       = moments['R']
        m       = moments['m']
        counts  = moments['counts']

        # phase-locked needs the mean phasors, which are not available (and not
        # meaningful) for random-phase continuous data.
        if ('phase-locked' in methods) and (m is None):
            raise ValueError(
                "The 'phase-locked' estimator requires the mean phasors, which "
                "are not available for the continuous data path."
            )

        results = {method: {} for method in methods}

        # Loop over each channel pair and derive the requested estimators.
        for ch_out, ch_in in channel_pairs:
            o = chans.index(ch_out)
            i = chans.index(ch_in)

            # Auto-spectra (PSDs) and the cross-spectrum for this pair.
            psd_out = S[o, o, :].real
            psd_in  = S[i, i, :].real
            cross   = S[o, i, :]            # complex cross-spectrum S_oi

            # Variances of the auto-spectra means: Var(S_ii) = (R_ii - S_ii^2)/(N-1)
            var_psd_out = self._variance_of_mean(R[o, o, :].real, psd_out ** 2, counts)
            var_psd_in  = self._variance_of_mean(R[i, i, :].real, psd_in ** 2, counts)

            if 'rms-ratio' in methods:
                # rms-ratio = sqrt(S_oo / S_ii). Relative uncertainty of a square
                # root of a ratio: sigma/T = 0.5 sqrt(Var(P_o)/P_o^2 + Var(P_i)/P_i^2).
                with np.errstate(divide='ignore', invalid='ignore'):
                    tf = np.sqrt(psd_out / psd_in)
                    rel = 0.5 * np.sqrt(
                        var_psd_out / (psd_out ** 2)
                        + var_psd_in / (psd_in ** 2)
                    )
                    sigma = tf * rel
                results['rms-ratio'][(ch_out, ch_in)] = {
                    'channel_output':    ch_out,
                    'channel_input':     ch_in,
                    'transfer_function': tf,
                    'transfer_sigma':    sigma,
                    'freqs':             freqs,
                    'method':            'rms-ratio',
                }

            if 'cross-correlation' in methods:
                # cross-correlation = S_oi / S_ii (complex H1 estimator).
                # Var(S_oi) = (⟨|a_o|^2 |a_i|^2⟩ - |S_oi|^2)/(N-1) = (R_oi - |S_oi|^2)/(N-1).
                abs_cross = np.abs(cross)
                var_cross = self._variance_of_mean(R[o, i, :].real, abs_cross ** 2, counts)
                with np.errstate(divide='ignore', invalid='ignore'):
                    tf = cross / psd_in
                    rel = np.sqrt(
                        var_cross / (abs_cross ** 2)
                        + var_psd_in / (psd_in ** 2)
                    )
                    sigma = np.abs(tf) * rel
                results['cross-correlation'][(ch_out, ch_in)] = {
                    'channel_output':    ch_out,
                    'channel_input':     ch_in,
                    'transfer_function': tf,
                    'transfer_sigma':    sigma,
                    'freqs':             freqs,
                    'method':            'cross-correlation',
                }

            if 'phase-locked' in methods:
                # phase-locked = ⟨a_o⟩ / ⟨a_i⟩ (complex). The variance of each mean
                # phasor is Var(⟨a⟩) = (⟨|a|^2⟩ - |⟨a⟩|^2)/(N-1) = (S_ii - |m_i|^2)/(N-1).
                m_out = m[o, :]
                m_in  = m[i, :]
                abs_m_out = np.abs(m_out)
                abs_m_in  = np.abs(m_in)
                var_m_out = self._variance_of_mean(psd_out, abs_m_out ** 2, counts)
                var_m_in  = self._variance_of_mean(psd_in, abs_m_in ** 2, counts)
                with np.errstate(divide='ignore', invalid='ignore'):
                    tf = m_out / m_in
                    rel = np.sqrt(
                        var_m_out / (abs_m_out ** 2)
                        + var_m_in / (abs_m_in ** 2)
                    )
                    sigma = np.abs(tf) * rel
                results['phase-locked'][(ch_out, ch_in)] = {
                    'channel_output':    ch_out,
                    'channel_input':     ch_in,
                    'transfer_function': tf,
                    'transfer_sigma':    sigma,
                    'freqs':             freqs,
                    'method':            'phase-locked',
                }

        return results

    def calc_transfer_function(self, channel_pairs=None, methods=None,
                               accel_gain=100.0, downsample_factor=1,
                               trace_length_samples=None, trace_length_msec=None,
                               verbose=True, force_overwrite=False):
        """
        Compute transfer function estimators for one or more channel pairs and
        cache them for the plotting.

          - 'rms-ratio':        sqrt(S_oo / S_ii)        (real)
          - 'cross-correlation': S_oi / S_ii             (complex)
          - 'phase-locked':     ⟨a_o⟩ / ⟨a_i⟩            (complex; transducer sweep only)

        Parameters
        ----------
        channel_pairs : list of [str, str], optional
            Each element is [channel_output, channel_input].
            E.g. [['Stage1', 'Ground'], ['Stage2', 'Ground']].
        methods : str or list of str, optional
            Which estimator(s) to compute. One or more of 'rms-ratio',
            'cross-correlation', 'phase-locked' (hyphens or underscores accepted),
            or 'all' to return every cached method.
        accel_gain : float, optional
            Accelerometer gain factor. Ignored for sweep data. Default: 100.0.
        downsample_factor : int, optional
            Process every Nth event. Ignored for sweep data. Default: 1.
        trace_length_samples : int, optional
            Overrides the native event length and rechunks the continuous data into
            traces of this many samples.
            Continuous data only. See _accumulate_moments_continuous() for the
            rechunking behavior. Default: None.
        trace_length_msec : float, optional
            Overrides the native event length and rechunks the continuous data into
            traces of this many milliseconds.
            Continuous data only. Default: None.
        verbose : bool, optional
            Continuous data only. Print progress information. Default: True.
        force_overwrite : bool, optional
            If True, ignore the shared moment cache and recompute it from scratch
            (re-reading the data) before deriving the estimators. Default: False.

        Returns
        -------
        list of dict  (when ``methods=None``)
            One dict per pair, each containing:
              'channel_output'    : str
              'channel_input'     : str
              'transfer_function' : np.ndarray, shape (n_freqs,); real amplitude
              'transfer_sigma'    : np.ndarray, shape (n_freqs,); 1-sigma uncertainty
              'freqs'             : np.ndarray, shape (n_freqs,)

        dict of {str: dict of {tuple: dict}}  (when ``methods`` is provided)
            Top-level keys are the requested method names
            ('rms-ratio', 'cross-correlation', 'phase-locked').
            Each value is a dict keyed by (channel_output, channel_input) tuples, 
            one entry per channel pair.
        """
        # Normalize a bare pair like ['Stage1', 'Ground'] to [['Stage1', 'Ground']]
        if channel_pairs and isinstance(channel_pairs[0], str):
            channel_pairs = [channel_pairs]

        if channel_pairs is None:
            raise ValueError(
                "channel_pairs is required to compute transfer functions."
            )

        # Accept estimator names with either hyphens or underscores
        methods_norm = self._normalize_method_names(methods)

        # Valid estimators depend on the data path (phase-locked is sweep-only).
        if self._data_source == 'continuous_data':
            valid_methods = self.CONTINUOUS_TF_METHODS
        else:
            valid_methods = self.VALID_TF_METHODS

        # Resolve which estimators to compute. methods=None keeps the legacy
        # rms-ratio-only flat-list return; 'all' computes every valid estimator.
        legacy_return = methods_norm is None
        if legacy_return:
            compute_methods = ['rms-ratio']
        elif methods_norm == 'all':
            compute_methods = list(valid_methods)
        elif isinstance(methods_norm, str):
            compute_methods = [methods_norm]
        else:
            compute_methods = list(methods_norm)

        # Validate requested estimators against the data path.
        for method_name in compute_methods:
            if method_name not in valid_methods:
                raise ValueError(
                    f"Method '{method_name}' is not available for this data path. "
                    f"Valid methods: {valid_methods}."
                )

        # The channels needed are exactly those referenced by the pairs.
        tf_channels = []
        for pair in channel_pairs:
            for ch in pair:
                if ch not in tf_channels:
                    tf_channels.append(ch)

        # Ensure the moment cache covers those channels (reused from a prior
        # calc_psd() / calc_transfer_function() when possible, built otherwise;
        # force_overwrite forces a fresh recomputation).
        self._ensure_moments(
            channels=tf_channels,
            accel_gain=accel_gain,
            downsample_factor=downsample_factor,
            trace_length_samples=trace_length_samples,
            trace_length_msec=trace_length_msec,
            verbose=verbose,
            force_overwrite=force_overwrite,
        )

        # Derive the estimators and their uncertainties from the moment cache.
        results = self._estimators_from_moments(
            channel_pairs=channel_pairs,
            methods=compute_methods,
        )

        # Legacy path (methods=None): cache the pair-keyed rms-ratio results and
        # return the flat list with the 'method' key stripped.
        if legacy_return:
            self._upsert_tf_cache('rms-ratio', results['rms-ratio'])
            flat_results = []
            for pair_dict in results['rms-ratio'].values():
                legacy_dict = {k: v for k, v in pair_dict.items() if k != 'method'}
                flat_results.append(legacy_dict)
            return flat_results

        # Cache each computed estimator. Keys are already in the canonical
        # hyphenated form produced by _normalize_method_names().
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
            The estimation method key (e.g. 'rms-ratio').
        new_results : dict of {tuple: dict}
            New TF result dicts keyed by (channel_output, channel_input) tuple.
        """
        if self._transfer_functions is None:
            self._transfer_functions = {}

        existing = self._transfer_functions.get(method_name, {})

        # Report any already-cached pairs that are being replaced (for example,
        # continuous data estimators overwritten by a later sweep computation)
        overwritten = [pair for pair in new_results if pair in existing]
        if overwritten:
            pairs_str = ", ".join(f"{ch_out}/{ch_in}" for ch_out, ch_in in overwritten)
            print(
                f"Overwrote cached '{method_name}' transfer function for "
                f"channel pair(s): {pairs_str}."
            )

        # Merge: keep existing pairs, add/replace with new ones.
        merged = dict(existing)
        merged.update(new_results)
        self._transfer_functions[method_name] = merged

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
        # No downsampling requested, or fewer bins than target; use the full array
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
        'rms-ratio':       ('-',  ' (RMS Ratio)'),
        'cross-correlation': ('--', ' (Cross Correlation)'),
        'phase-locked':     ('-.', ' (Phase Locked)'),
    }

    @staticmethod
    def _normalize_method_names(methods):
        """
        Normalize transfer function estimator names to the hyphenated form.

        Parameters
        ----------
        methods : str or list of str or None
            A single estimator name, a list of estimator names, or None.

        Returns
        -------
        str or list of str or None
            The input with underscores replaced by hyphens in every string.
        """
        if methods is None:
            return None
        if isinstance(methods, str):
            return methods.replace('_', '-')
        return [
            m.replace('_', '-') if isinstance(m, str) else m
            for m in methods
        ]

    def _resolve_tf_pairs_to_plot(self, channel_pairs, methods):
        """
        Resolve which transfer function dicts to plot based on methods and channel_pairs.

        Parameters
        ----------
        channel_pairs : list of [str, str] or None
            Requested pairs. If None, all available pairs are used.
        methods : str or list of str or None
            One estimator name, a list of estimator names to overlay, 'all' to
            overlay every cached estimator, or None for the default (rms-ratio).
            For processed data, valid names are 'rms-ratio', 'cross-correlation',
            and 'phase-locked'. For continuous data, valid names are 'rms-ratio' and
            'cross-correlation'.

        Returns
        -------
        plot_groups : list of (list_of_tf_dicts, linestyle, label_suffix)
            Each group is a set of TF dicts to plot with the given style.
        """
        # Accept estimator names with either hyphens or underscores
        methods = self._normalize_method_names(methods)

        groups = []

        if methods is not None:
            # Explicit method selection. Reads the shared per-method cache, which
            # is populated by calc_transfer_function() for both sweep and
            # continuous data. Never computes here; the cache must already be
            # populated by a prior calc_transfer_function() call.
            if self._transfer_functions is None:
                raise RuntimeError(
                    "No transfer functions cached. "
                    "Call calc_transfer_function() first."
                )

            # Determine which methods to overlay.
            # methods can be a single string or a list of method names.
            if isinstance(methods, list):
                # Explicit list of methods to overlay
                for m in methods:
                    if m not in self.VALID_TF_METHODS:
                        raise ValueError(
                            f"Unknown method '{m}'. "
                            f"Valid: {self.VALID_TF_METHODS}"
                        )
                method_keys = list(methods)
            elif methods == 'all':
                # Overlay all cached methods
                method_keys = [
                    m for m in self.VALID_TF_METHODS
                    if m in self._transfer_functions and self._transfer_functions[m]
                ]
            elif methods in self.VALID_TF_METHODS:
                method_keys = [methods]
            else:
                raise ValueError(
                    f"Unknown method '{methods}'. "
                    f"Valid: {self.VALID_TF_METHODS + ('all',)}"
                )

            # Build plot groups: one per method
            show_suffix = len(method_keys) > 1
            for m in method_keys:
                if m not in self._transfer_functions:
                    raise RuntimeError(
                        f"Method '{m}' not cached. "
                        f"Call calc_transfer_function(methods=['{m}']) first."
                    )
                tf_by_pair = self._transfer_functions[m]
                ls, suffix = self._METHOD_LINESTYLES[m]

                if channel_pairs is not None:
                    filtered = []
                    for p in channel_pairs:
                        pair_key = (p[0], p[1])
                        if pair_key not in tf_by_pair:
                            raise RuntimeError(
                                f"Transfer function for pair {list(pair_key)} is not "
                                f"cached for method '{m}'. Call "
                                f"calc_transfer_function(channel_pairs=[{list(pair_key)}], "
                                f"methods=['{m}']) first."
                            )
                        filtered.append(tf_by_pair[pair_key])
                else:
                    filtered = list(tf_by_pair.values())

                groups.append((filtered, ls, suffix if show_suffix else ''))

        else:
            # Default behavior (methods=None): use rms-ratio from cache. Never
            # computes here; the cache must already be populated by a prior
            # calc_transfer_function() call.
            if self._transfer_functions is None:
                raise RuntimeError(
                    "No transfer functions cached. "
                    "Call calc_transfer_function() first."
                )

            # Default method is rms-ratio; read its pair-keyed cache.
            tf_by_pair = self._transfer_functions.get('rms-ratio', {})

            if channel_pairs is not None:
                pairs_list = []
                for p in channel_pairs:
                    pair_key = (p[0], p[1])
                    if pair_key not in tf_by_pair:
                        raise RuntimeError(
                            f"Transfer function for pair {list(pair_key)} is not "
                            f"cached (method 'rms-ratio'). Call "
                            f"calc_transfer_function(channel_pairs=[{list(pair_key)}]) "
                            f"first."
                        )
                    pairs_list.append(tf_by_pair[pair_key])
            else:
                pairs_list = list(tf_by_pair.values())

            groups.append((pairs_list, '-', ''))

        return groups

    def plot_transfer_function(self, channel_pairs=None, figsize=(14, 6), methods='all',
                               show_err=True, log_downsample=False):
        """
        Plot the transfer function magnitude for one or more channel pairs.

        Uses internally cached transfer functions from a prior calc_transfer_function()
        call.

        Parameters
        ----------
        channel_pairs : list of [str, str] or [str, str], optional
            Each element is [channel_output, channel_input]. 
            If None, all currently cached transfer functions are plotted.
        figsize : tuple of (float, float), optional
            Figure size passed to plt.subplots. Default: (14, 6).
        methods : str or list of str, optional
            Which estimation method(s) to plot. A single string is treated as the
            sole method; a list overlays several. Default: 'all'.
              - 'all': overlay all cached methods with distinct linestyles (default)
              - 'rms-ratio', 'cross-correlation', 'phase-locked': plot one method
              - A list like ['rms-ratio', 'phase-locked']: overlay selected methods
              - None: plot only rms-ratio (processed data)
        show_err : bool or list of bool, optional
            Whether to draw the 1-sigma uncertainty band for each method group.
            If a single bool, applies to all method groups. If a list, must have
            one entry per method group (same length as the resolved method list).
            Default: True.
        log_downsample : bool or int, optional
            If True, logarithmically downsample the plotted line and uncertainty
            band to _LOG_DOWNSAMPLE_DEFAULT_POINTS log-spaced frequency bins
            before drawing. If an int, use that many points. 
            Useful when getting matplotlib "cell block" errors.
            Default: False.

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
            channel_pairs=channel_pairs, methods=methods,
        )

        # Normalize show_err to a list with one bool per plot group
        n_groups = len(plot_groups)
        if isinstance(show_err, bool):
            show_err_per_group = [show_err] * n_groups
        else:
            show_err_per_group = list(show_err)
            if len(show_err_per_group) != n_groups:
                raise ValueError(
                    f"show_err has {len(show_err_per_group)} entries "
                    f"but there are {n_groups} method group(s) to plot. "
                    "Pass a single bool or a list matching the number of methods."
                )

        fig, ax = plt.subplots(figsize=figsize)
        prop_cycle_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

        all_tf_values = []
        color_index = 0

        # Plot each group (one group per method when overlaying multiple methods)
        for group_idx, (pairs_to_plot, linestyle, label_suffix) in enumerate(plot_groups):
            draw_uncertainty = show_err_per_group[group_idx]

            for i, tf_dict in enumerate(pairs_to_plot):
                ch_out = tf_dict['channel_output']
                ch_in  = tf_dict['channel_input']
                freqs  = tf_dict['freqs']
                tf_raw = tf_dict['transfer_function']
                tf_sig = tf_dict['transfer_sigma']
                color  = prop_cycle_colors[(color_index + i) % len(prop_cycle_colors)]

                # For complex TFs (cross-correlation, phase-locked), plot magnitude
                tf_mag = np.abs(tf_raw) if np.iscomplexobj(tf_raw) else tf_raw
                all_tf_values.append(tf_mag)

                # Optionally select a log-spaced subset of frequency indices. This
                # keeps the plotted polygon/line size manageable for continuous data PSDs
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

                    # Mask non-finite bounds. fill_between treats NaN as a polygon
                    # break, which avoids drawing to infinite y on log scale.
                    bad = ~np.isfinite(lower) | ~np.isfinite(upper)
                    lower = np.where(bad, np.nan, lower)
                    upper = np.where(bad, np.nan, upper)

                    # Draw uncertainty band
                    ax.fill_between(
                        plot_freqs,
                        lower,
                        upper,
                        color=color,
                        alpha=0.2,
                        linewidth=1,
                    )

            # For multi-method overlay, keep same color mapping across methods
            if methods != 'all' and not isinstance(methods, list):
                color_index = color_index + len(pairs_to_plot)

        # Set y-limits across all plotted transfer functions
        all_tf_concat = np.concatenate(all_tf_values)
        finite_vals = all_tf_concat[np.isfinite(all_tf_concat) & (all_tf_concat > 0)]
        if finite_vals.size > 0:
            ax.set_ylim(finite_vals.min() * 0.8, finite_vals.max() * 1.2)

        ax.set_xlabel("Frequency (Hz)", fontsize=16, fontweight='bold')

        ax.set_ylabel("Attenuation", fontsize=16, fontweight='bold')

        ax.grid(True, which="both", ls=":")
        ax.tick_params(direction='in')
        ax.legend()
        ax.tick_params(axis='both', which='both', direction='in')
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Secondary right-hand axis in dB, with ticks synced to the left axis
        secax = ax.secondary_yaxis(
            location="right",
            functions=(lambda y: 20.0 * np.log10(y), lambda db: 10.0 ** (db / 20.0)),
        )
        secax.set_ylabel("Attenuation (dB)", fontsize=16, fontweight='bold')
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
                                     methods=None):
        """
        Plot the transfer function phase for one or more channel pairs.

        Only works with complex-valued transfer function methods
        ('cross-correlation' or 'phase-locked'). The 'rms-ratio' method
        discards phase information and cannot be plotted here.

        Parameters
        ----------
        channel_pairs : list of [str, str] or [str, str], optional
            Each element is [channel_output, channel_input]. A bare pair like
            ['Stage1', 'Ground'] is also accepted.
            If None, all currently cached pairs for the given methods are plotted.
        figsize : tuple of (float, float), optional
            Figure size passed to plt.subplots. Default: (14, 6).
        methods : str or list of str, optional
            Which estimation method(s) to plot. A single string is treated as the
            sole method; a list overlays several. Valid: 'cross-correlation',
            'phase-locked', or 'all' (overlay all cached methods). Default:
            'phase-locked' if cached, else 'cross-correlation'.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        # Normalize a bare pair like ['Stage1', 'Ground'] to [['Stage1', 'Ground']]
        if channel_pairs is not None and channel_pairs and isinstance(channel_pairs[0], str):
            channel_pairs = [channel_pairs]

        # Accept estimator names with either hyphens or underscores
        methods = self._normalize_method_names(methods)

        # Determine which complex methods to plot
        complex_methods = ('cross-correlation', 'phase-locked')

        if methods is None:
            # Auto-select: prefer phase-locked if cached, else cross-correlation
            if (self._transfer_functions is not None
                    and 'phase-locked' in self._transfer_functions):
                methods = 'phase-locked'
            elif (self._transfer_functions is not None
                    and 'cross-correlation' in self._transfer_functions):
                methods = 'cross-correlation'
            else:
                raise RuntimeError(
                    "No complex-valued transfer functions cached. "
                    "Call calc_transfer_function(methods=['phase-locked']) or "
                    "calc_transfer_function(methods=['cross-correlation']) first."
                )

        # Reject rms-ratio (no phase info) whether passed as string or in a list
        if methods == 'rms-ratio':
            raise ValueError(
                "The 'rms-ratio' method discards phase information. "
                "Use 'cross-correlation' or 'phase-locked' for phase plots."
            )
        if isinstance(methods, list) and 'rms-ratio' in methods:
            raise ValueError(
                "The 'rms-ratio' method discards phase information. "
                "Use 'cross-correlation' or 'phase-locked' for phase plots."
            )

        # Build list of methods to overlay.
        # methods can be a single string or a list of method names.
        if isinstance(methods, list):
            for m in methods:
                if m not in complex_methods:
                    raise ValueError(
                        f"Unknown or non-complex method '{m}'. "
                        f"Valid for phase plots: {complex_methods}"
                    )
            method_keys = list(methods)
        elif methods == 'all':
            method_keys = [
                m for m in complex_methods
                if (self._transfer_functions is not None
                    and m in self._transfer_functions
                    and self._transfer_functions[m])
            ]
            if not method_keys:
                raise RuntimeError(
                    "No complex-valued transfer functions cached. "
                    "Call calc_transfer_function(methods=[...]) first."
                )
        elif methods in complex_methods:
            method_keys = [methods]
        else:
            raise ValueError(
                f"Unknown method '{methods}'. "
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
                    f"Call calc_transfer_function(methods=['{m}']) first."
                )

            tf_by_pair = self._transfer_functions[m]
            ls, suffix = self._METHOD_LINESTYLES[m]

            # Filter to requested pairs if specified
            if channel_pairs is not None:
                pairs_to_plot = []
                for p in channel_pairs:
                    pair_key = (p[0], p[1])
                    if pair_key not in tf_by_pair:
                        raise RuntimeError(
                            f"Transfer function for pair {list(pair_key)} is not "
                            f"cached for method '{m}'. Call "
                            f"calc_transfer_function(channel_pairs=[{list(pair_key)}], "
                            f"methods=['{m}']) first."
                        )
                    pairs_to_plot.append(tf_by_pair[pair_key])
            else:
                pairs_to_plot = list(tf_by_pair.values())

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

        ax.set_xlabel("Frequency (Hz)", fontsize=16, fontweight='bold')
        ax.set_ylabel("Phase (degrees)", fontsize=16, fontweight='bold')
        ax.set_ylim(-180, 180)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.grid(True, which="both", ls=":")
        ax.legend()
        ax.tick_params(axis='both', which='both', direction='in')
        ax.tick_params(axis='both', which='major', labelsize=16)

        return fig, ax

    def plot_psd(
        self,
        channels=None,
        figsize=(14, 6),
        show_err=True,
        colors=None,
        show_spectral_noise=True,
        spectral_noise_color='tab:red',
        spectral_noise_dict=None,
    ):
        """
        Plot the PSD for one or more channels.
        
        Uses internally cached PSDs from a prior calc_psd() call when available.
        If any requested channel is not yet cached, calc_psd() is called automatically
        for the full requested channel list before plotting.

        Parameters
        ----------
        channels : list of str, optional
            Channel names to plot, in the order they should appear.
            If None, all currently cached channels are plotted.
        figsize : tuple of (float, float), optional
            Figure size passed to plt.subplots. Default: (14, 6).
        show_err : bool, optional
            If True, draw a ±1-sigma uncertainty band around each
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
            (See how _NOISE_FLOOR_FREQS_HZ and _NOISE_FLOOR_G_PER_SQRT_HZ)

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        # Reject a bare string early: iterating it would split into characters
        # and produce a confusing downstream error about invalid channels
        if isinstance(channels, str):
            raise TypeError(
                f"channels must be array-like (e.g. a list of str), not a single "
                f"string. Did you mean channels=['{channels}']?"
            )

        # Resolve which channels to plot
        if channels is None:
            if self._channels is None:
                raise RuntimeError(
                    "No PSDs cached. Call calc_psd() first or pass channels= explicitly."
                )
            channels = self._channels

        # Ensure PSDs are available for every requested channel
        missing = [ch for ch in channels if self._channels is None or ch not in self._channels]
        if missing:
            if self._data_source == 'continuous_data':
                raise ValueError(
                    f"Channels {missing} were not included in the last calc_psd() call. "
                    f"Call calc_psd(channels=...) including {missing} before plotting."
                )
            self.calc_psd(channels=channels)

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
            if show_err:
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
        ax.set_xlabel("Frequency (Hz)", fontsize=16, fontweight='bold')
        ax.set_ylabel(r"RMS Amplitude ($g/\sqrt{\mathrm{Hz}}$)", fontsize=16, fontweight='bold')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
        ax.set_ylim([min_amp*0.8, max_amp*1.2])
        ax.tick_params(axis='both', which='both', direction='in')
        ax.tick_params(axis='both', which='major', labelsize=16)

        return fig, ax
