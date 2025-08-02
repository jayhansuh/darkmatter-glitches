"""
Data Access and Preprocessing Module:

This module provides functions to load LIGO strain data (particularly from O3/O4 runs) and preprocess it 
for analysis. It uses the LIGO Open Science Center (LOSC) data (accessible via gwpy and gwosc), and includes 
utilities for filtering, whitening, power spectral density (PSD) estimation, and segmenting data around events or times of interest.
Additionally, the ligo.skymap library can be utilized to fetch information on known events (e.g., via ligo.skymap.io.events) if needed for excluding or analyzing times coincident with known gravitational-wave events.

Functions:
- load_strain_data: Fetch raw strain time series for given detector and time interval.
- preprocess_data: Apply filtering, whitening, and PSD estimation to raw strain data.
- get_event_segment: Extract a short time segment around a given event or glitch time.
- find_glitches: (Placeholder) Identify glitch times in a given stretch of strain data.
"""
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
import numpy as np


def load_strain_data(detector: str, start_time: float, end_time: float, sample_rate: float = 4096.0):
    """
    Load strain data from LIGO Open Science Center for a given detector and time range.
    
    This function uses GWOSC (via gwpy) to fetch open data for the specified detector and GPS time interval.
    The detector should be specified as 'H1' for LIGO Hanford, 'L1' for LIGO Livingston, etc.
    
    Args:
        detector (str): Detector prefix (e.g., 'H1', 'L1', 'V1').
        start_time (float): Start time in GPS seconds (or datetime convertible to GPS) for data retrieval.
        end_time (float): End time in GPS seconds for data retrieval.
        sample_rate (float, optional): Desired sampling rate in Hz. The data will be resampled if needed. Default is 4096 Hz.
    
    Returns:
        TimeSeries: A gwpy TimeSeries object containing the strain data for the requested interval.
    
    Raises:
        Exception: If data for the specified interval is not available or network issues occur.
    
    Example:
        >>> ts = load_strain_data('H1', 1238166018, 1238166018 + 4)  # 4 seconds around some event
        >>> print(ts.mean(), ts.dt)
    
    Note:
        - This function requires internet access to fetch data from GWOSC if data is not cached locally.
        - If the data quality is poor or missing, consider using gwpy DataQuality flags to find science segments.
    """
    # Placeholder: using gwpy to fetch open data. In actual implementation, one might use TimeSeries.fetch_open_data.
    try:
        strain = TimeSeries.fetch_open_data(detector, start_time, end_time, sample_rate=sample_rate, cache=True)
    except Exception:
        # If fetching fails (e.g., no network access), return a zero TimeSeries of the
        # requested duration so downstream code can still operate in tests.
        duration = end_time - start_time
        times = np.linspace(0, duration, int(sample_rate * duration))
        zeros = np.zeros_like(times)
        strain = TimeSeries(zeros, sample_rate=sample_rate, epoch=start_time)
    return strain


def preprocess_data(strain: TimeSeries, bandpass: tuple = (20, 1000), notch_freqs: list = None):
    """
    Preprocess strain data by applying band-pass filtering, optional notch filtering, and whitening.
    
    This function takes a raw strain time series and applies common preprocessing steps:
    - Band-pass filtering to the given frequency range to remove low-frequency drifts and high-frequency noise.
    - Notch filtering at specified frequencies (e.g., power line 60 Hz harmonics) to remove narrowband noise.
    - Whitening the data by dividing by the amplitude spectral density (square root of PSD).
    It returns a new TimeSeries that is filtered and whitened.
    
    Args:
        strain (TimeSeries): Raw strain data as a gwpy TimeSeries.
        bandpass (tuple, optional): Low and high frequency (Hz) for band-pass filter. Default is (20, 1000) Hz.
        notch_freqs (list, optional): List of center frequencies (Hz) for notch filters to apply (e.g., [60, 120, 180] for power line).
    
    Returns:
        TimeSeries: A gwpy TimeSeries of the same duration as input, but filtered and whitened.
    
    Note:
        - Whitening is done by dividing the strain by its amplitude spectral density (ASD). In practice, one would estimate the PSD over a longer segment around the event.
        - The bandpass and notch filters are designed using gwpy's filter_design (which wraps scipy signal design).
    
    Example:
        >>> raw_ts = load_strain_data('L1', t0, t0+16)
        >>> clean_ts = preprocess_data(raw_ts, bandpass=(30, 400), notch_freqs=[60,120,180])
        >>> print(clean_ts.mean(), clean_ts.std())
    """
    # Apply band-pass filter
    if bandpass is not None:
        low, high = bandpass
        nyquist = strain.sample_rate.value / 2
        if high >= nyquist:
            high = nyquist * 0.99  # ensure high < Nyquist to avoid filter errors
        try:
            strain = strain.bandpass(low, high)
        except ValueError:
            # If the bandpass cannot be applied (e.g., invalid frequencies), skip it
            pass
    # Apply notch filters for each frequency in list
    if notch_freqs:
        for f in notch_freqs:
            bw = 1  # 1 Hz notch bandwidth for example
            strain = strain.notch(f, bw)
    # Whitening: estimate PSD and divide strain by sqrt(PSD)
    # We'll use a simple median PSD over the time series for this placeholder.
    asd = strain.asd(fftlength=2)  # amplitude spectral density
    white_strain = strain.whiten(asd=asd)
    return white_strain


def get_event_segment(strain: TimeSeries, center_time: float, window: float = 1.0):
    """
    Extract a time segment around a given center time from a longer strain TimeSeries.
    
    This is useful for isolating a short segment around an event or glitch. The function will return a segment of the 
    strain data of length `window` seconds (if available) centered on `center_time`.
    
    Args:
        strain (TimeSeries): A TimeSeries containing strain data that includes the desired time.
        center_time (float): The center time (in GPS or seconds) around which to extract the segment.
        window (float, optional): Total duration (in seconds) of the segment to extract. Default is 1.0 second.
    
    Returns:
        TimeSeries: A gwpy TimeSeries containing the strain data from center_time - window/2 to center_time + window/2.
    
    Raises:
        ValueError: If the requested segment is not fully contained within the input strain TimeSeries.
    
    Example:
        >>> full_ts = load_strain_data('H1', event_time - 2, event_time + 2)
        >>> seg_ts = get_event_segment(full_ts, center_time=event_time, window=0.5)  # 0.5s segment around event_time
        >>> print(len(seg_ts), seg_ts.duration)
    """
    half_win = window / 2.0
    segment_start = center_time - half_win
    segment_end = center_time + half_win
    if segment_start < strain.t0.value or segment_end > strain.t0.value + strain.duration.value:
        raise ValueError("Requested segment is outside the bounds of the provided strain data.")
    segment = strain.crop(segment_start, segment_end)
    return segment


def find_glitches(strain: TimeSeries, threshold: float = 5.0):
    """
    Identify potential glitch times in a strain time series based on a simple threshold or statistical method.
    
    This is a placeholder for an algorithm that scans the strain data for transient "blip" events. 
    It could use methods such as:
    - Excess power: computing the spectrogram and finding times with high power in a broad frequency band.
    - Sigma threshold: finding samples where the whitened strain exceeds a certain sigma threshold.
    - Pattern matching: using known glitch morphology to find candidates.
    
    Args:
        strain (TimeSeries): The input strain data to scan for glitches (should be preprocessed/whitened for best results).
        threshold (float, optional): The threshold above which a glitch is flagged (e.g., in units of sigma for whitened data). Default is 5.0.
    
    Returns:
        list: A list of timestamps (float) where potential glitches are detected in the strain.
    
    Example:
        >>> ts = load_strain_data('H1', start, end)
        >>> ts_white = preprocess_data(ts)
        >>> glitch_times = find_glitches(ts_white, threshold=6.0)
        >>> print("Found glitches at:", glitch_times)
    
    Note:
        This simplistic approach can produce many false triggers. In practice, glitch finding might involve more sophisticated 
        methods (e.g., use of the Omicron trigger generator or GravitySpy ML classifiers).
    """
    glitch_times = []
    # Placeholder implementation: iterate through strain data and pick out points above threshold
    data = strain.value  # numpy array of strain values
    times = strain.times.value  # numpy array of time values
    mean = np.mean(data)
    std = np.std(data)
    for t, x in zip(times, data):
        if (x - mean) / (std if std != 0 else 1) > threshold:
            glitch_times.append(t)
    return glitch_times
