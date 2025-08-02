"""
Test cases for Waveform Modeling Module.

These tests use synthetic parameters to verify that waveform generation routines produce outputs of expected type and shape.
"""
import numpy as np
from waveform_modeling import waveforms


def test_simulate_quadrupole_waveform_basic():
    """Test that simulate_quadrupole_waveform returns time and strain arrays of equal length."""
    t, h = waveforms.simulate_quadrupole_waveform(particle_mass=1e-10, distance=1e20, decay_fraction=0.5, sampling_rate=1024.0)
    # The function should return numpy arrays for time and strain
    assert isinstance(t, np.ndarray), "Time array should be a numpy array"
    assert isinstance(h, np.ndarray), "Strain array should be a numpy array"
    assert t.shape == h.shape, "Time and strain arrays must have the same shape"
    # Check that the waveform is centered (peak roughly at mid-time for this placeholder model)
    peak_index = np.argmax(h)
    assert np.isclose(t[peak_index], 0.1/2, atol=0.01), "Peak of quadrupole waveform should be near the center of the time series"


def test_simulate_recoil_waveform_basic():
    """Test that simulate_recoil_waveform returns non-negative half-sine pulse and correct output types."""
    t, h = waveforms.simulate_recoil_waveform(particle_mass=1e-9, distance=1e19, kick_velocity=1e3, sampling_rate=1000.0)
    assert isinstance(t, np.ndarray) and isinstance(h, np.ndarray), "Outputs should be numpy arrays"
    assert t.shape == h.shape, "Time and strain outputs must have same shape"
    # All strain values should be >= 0 due to half-sine implementation
    assert np.all(h >= 0), "Recoil waveform should be non-negative (half-sine pulse)"


def test_simulate_gaussian_burst_waveform_basic():
    """Test that simulate_gaussian_burst_waveform returns a Gaussian-like burst with correct length and type."""
    t, h = waveforms.simulate_gaussian_burst_waveform(particle_mass=1e-8, distance=1e21, burst_width=0.005, sampling_rate=2048.0)
    assert isinstance(t, np.ndarray) and isinstance(h, np.ndarray), "Outputs should be numpy arrays"
    assert t.shape == h.shape, "Time and strain arrays must have the same shape"
    # The peak of the Gaussian should be near the center of the time series
    peak_idx = np.argmax(h)
    center_idx = len(t)//2
    assert abs(peak_idx - center_idx) < len(t)*0.1, "Peak of Gaussian burst should be near center of array"
