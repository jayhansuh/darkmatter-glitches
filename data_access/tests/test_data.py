"""
Test cases for Data Access and Preprocessing Module.

These tests validate that the data loading and preprocessing functions behave as expected with given parameters,
using short time intervals and synthetic or known data samples.
"""
import numpy as np
from gwpy.timeseries import TimeSeries
from data_access import data


def test_load_strain_data_basic():
    """Test loading a small segment of strain data (network connectivity permitting)."""
    detector = 'H1'
    # Choose a known GPS time where data is available (e.g., around GW150914 event or just any time in O3).
    start = 1126259462  # example GPS (GW150914 peak time, but just as a test if open data is available)
    end = 1126259462 + 1  # 1 second of data
    ts = data.load_strain_data(detector, start, end, sample_rate=256.0)
    # After loading, we expect a TimeSeries with appropriate sampling.
    assert ts.dt.value == 1/256.0 or ts.sample_rate.value == 256.0, "Data should be resampled to 256 Hz"
    assert ts.duration.value == 1.0, "Duration of loaded data should be 1 second"


def test_preprocess_data_whitening():
    """Test that preprocessing returns a whitened TimeSeries with roughly unit variance (for white noise)."""
    # Create a fake TimeSeries with Gaussian noise for testing
    times = np.linspace(0, 4, 4*256)  # 4 seconds at 256 Hz
    strain_vals = np.random.normal(0, 1, len(times))
    fake_ts = TimeSeries(strain_vals, sample_rate=256.0, epoch=0)
    white_ts = data.preprocess_data(fake_ts, bandpass=(50, 100), notch_freqs=None)
    # After whitening, if the noise was white, the std should be ~1 (within some tolerance due to finite sample)
    std = white_ts.std().value
    assert 0.5 < std < 2.0, "Whitened data should have variance of order 1"
    # Check that bandpass was applied: frequencies outside 50-100 should be attenuated (not easy to test without Fourier transform, skip detailed check)


def test_get_event_segment_bounds():
    """Test that get_event_segment properly extracts the correct time window and handles out-of-bounds."""
    # Use a dummy time series for testing
    t = np.linspace(0, 10, 1001)  # 10 seconds sampled at 100 Hz
    vals = np.sin(2*np.pi*1*t)  # some dummy data
    ts = TimeSeries(vals, sample_rate=100.0, epoch=0)
    # Extract 2-second segment around t=5s
    seg = data.get_event_segment(ts, center_time=5.0, window=2.0)
    assert np.isclose(seg.duration.value, 2.0, atol=0.01), "Segment duration should be 2 seconds"
    # Out-of-bounds request should raise error
    try:
        data.get_event_segment(ts, center_time=9.5, window=2.0)
        raised = False
    except ValueError:
        raised = True
    assert raised, "Requesting segment partially outside data should raise ValueError"


def test_find_glitches_threshold():
    """Test that find_glitches flags times where signal exceeds threshold."""
    # Create a simple TimeSeries with one obvious outlier glitch
    t = np.linspace(0, 1, 101)  # 1 second at 100 Hz
    data_vals = np.random.normal(0, 1, len(t))
    # Introduce a "glitch" spike 10-sigma at t=0.5
    data_vals[50] += 10 * np.std(data_vals)
    ts = TimeSeries(data_vals, sample_rate=100.0, epoch=0)
    glitch_times = data.find_glitches(ts, threshold=5.0)
    # Check that the time around 0.5s is flagged
    assert any(abs(gt - 0.5) < 0.01 for gt in glitch_times), "Glitch at 0.5s should be detected"
    # Check that no false glitch at places where data is normal
    # (We can allow possibly random noise to trigger, but probability is low for 5-sigma in 101 points)
