"""
Test cases for Bayesian Search and Inference Module.

These tests use simple synthetic scenarios to ensure the Bayesian inference scaffolds behave consistently,
especially for injection and recovery functions.
"""
import numpy as np
from inference import search
from waveform_modeling import waveforms


def test_inject_and_recover_signal():
    """Test injecting a synthetic signal into noise and attempting a recovery (posterior) to see if parameters are close."""
    # Create dummy noise data (whitened noise ~ N(0,1))
    data = np.random.normal(0, 1, 1000)
    # Define injection parameters for a Gaussian burst waveform
    injection_params = {"particle_mass": 1e-9, "distance": 5e20, "burst_width": 0.01}
    injected_data = search.inject_signal(data, injection_params, waveforms.simulate_gaussian_burst_waveform)
    # Recover the signal (placeholder posterior)
    post = search.recover_signal(injected_data, waveforms.simulate_gaussian_burst_waveform)
    # Ensure posterior result contains keys for all injected parameters
    for key in injection_params.keys():
        assert key in post, f"Posterior results should contain {key}"
    # Check that recovered median is within an order of magnitude of true value (loose check due to placeholder)
    for key, true_val in injection_params.items():
        recovered_val = post[key]["median"]
        if true_val != 0:
            assert 0.1 < recovered_val/true_val < 10, f"Recovered {key} median is not within a reasonable range of true value"


def test_compute_bayes_factor_no_signal():
    """Test that Bayes factor is ~1 when comparing noise vs noise (no signal present)."""
    data = np.zeros(100)  # no signal, all zeros
    BF = search.compute_bayes_factor(data, waveforms.simulate_gaussian_burst_waveform, priors=search.define_priors())
    assert np.isfinite(BF), "Bayes factor should be a finite number"
    # If there's no signal, BF should not strongly favor signal (expect ~1)
    assert BF < 10, "Bayes factor should not strongly favor signal when none is present"
