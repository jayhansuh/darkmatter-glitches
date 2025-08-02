"""Test cases for Waveform Modeling Module.

These tests use synthetic parameters to verify that waveform generation routines
produce outputs of expected type and shape.
"""

import numpy as np
import pytest

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


def test_simulate_quadrupole_waveform_zero_asymmetry():
    """Zero asymmetry should yield no quadrupole strain."""
    particle_mass = 1e12
    distance = 1e9
    decay_fraction = 0.0
    t, h = waveforms.simulate_quadrupole_waveform(
        particle_mass, distance, decay_fraction, sampling_rate=1000.0
    )
    assert isinstance(t, np.ndarray) and isinstance(h, np.ndarray)
    assert t.shape == h.shape and h.size > 0
    assert np.allclose(h, 0.0)


def test_simulate_quadrupole_waveform_amplitude_scaling():
    """Quadrupole amplitude scales with asymmetry and 1/distance."""
    particle_mass = 1e5
    dist1 = 1e6
    dist2 = 2e6
    frac1 = 0.5
    frac2 = 1.0
    t1, h1 = waveforms.simulate_quadrupole_waveform(
        particle_mass, dist1, frac1, sampling_rate=1000.0
    )
    _, h2 = waveforms.simulate_quadrupole_waveform(
        particle_mass, dist1, frac2, sampling_rate=1000.0
    )
    _, h3 = waveforms.simulate_quadrupole_waveform(
        particle_mass, dist2, frac1, sampling_rate=1000.0
    )
    assert h2.max() == pytest.approx(2 * h1.max(), rel=1e-2)
    assert h3.max() == pytest.approx(0.5 * h1.max(), rel=1e-2)


def test_simulate_quadrupole_waveform_distance_zero_handling():
    """Distance of zero should fall back to 1.0 without infinities."""
    particle_mass = 1e6
    t0, h0 = waveforms.simulate_quadrupole_waveform(
        particle_mass, 0.0, decay_fraction=1.0
    )
    t1, h1 = waveforms.simulate_quadrupole_waveform(
        particle_mass, 1.0, decay_fraction=1.0
    )
    assert h0.shape == h1.shape
    assert np.allclose(h0, h1)
    assert np.isfinite(h0).all()


def test_simulate_recoil_waveform_no_kick():
    """Zero kick velocity should produce zero recoil strain."""
    t, h = waveforms.simulate_recoil_waveform(
        particle_mass=1e10, distance=1e7, kick_velocity=0.0, sampling_rate=512.0
    )
    assert h.shape == t.shape and h.size > 0
    assert np.allclose(h, 0.0)


def test_simulate_recoil_waveform_positive_strain():
    """Recoil strain is non-negative and scales with kick velocity."""
    particle_mass = 5e5
    distance = 5e5
    v1 = 1e3
    v2 = 2e3
    _, h1 = waveforms.simulate_recoil_waveform(particle_mass, distance, v1)
    _, h2 = waveforms.simulate_recoil_waveform(particle_mass, distance, v2)
    assert np.all(h1 >= 0) and np.all(h2 >= 0)
    assert h2.max() > h1.max()
    assert h2.max() == pytest.approx(2 * h1.max(), rel=0.1)


def test_simulate_gaussian_burst_waveform_zero_mass():
    """Zero mass should yield ~zero Gaussian burst strain."""
    t, h = waveforms.simulate_gaussian_burst_waveform(0.0, 1e3, burst_width=0.005)
    assert t.shape == h.shape and h.size > 0
    assert np.allclose(h, 0.0, atol=1e-12)


def test_simulate_gaussian_burst_waveform_amplitude_distance_scaling():
    """Gaussian burst amplitude scales with mass and 1/distance."""
    m1 = 1e-9
    m2 = 2e-9
    d1 = 1e4
    d2 = 2e4
    t, h1 = waveforms.simulate_gaussian_burst_waveform(m1, d1)
    _, h2 = waveforms.simulate_gaussian_burst_waveform(m2, d1)
    _, h3 = waveforms.simulate_gaussian_burst_waveform(m1, d2)
    peak1 = np.max(np.abs(h1))
    peak2 = np.max(np.abs(h2))
    peak3 = np.max(np.abs(h3))
    assert peak2 == pytest.approx(2 * peak1, rel=1e-2)
    assert peak3 == pytest.approx(0.5 * peak1, rel=1e-2)
    duration = t[-1] - t[0]
    assert 0.09 < duration < 0.11
    assert np.all(np.diff(t) > 0)
