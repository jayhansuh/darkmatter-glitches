"""
Test cases for Visualization and Plotting Module.

These tests ensure that the plotting functions produce figures of the correct type and that basic plotting logic works.
They do not display or save the figures here, just verify that a Figure is returned and axes labels are set.
"""
import numpy as np
from visualization import plotting


def test_plot_waveform_output():
    """Test that plot_waveform returns a Figure and labels are set correctly."""
    t = np.linspace(0, 1, 100)
    h = np.sin(2*np.pi*10*t)
    fig = plotting.plot_waveform(t, h, title="Test Waveform")
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure), "plot_waveform should return a Figure"
    ax = fig.axes[0]
    assert ax.get_title() == "Test Waveform", "Plot title not set correctly"


def test_plot_noise_spectrum_output():
    """Test that plot_noise_spectrum returns a Figure and uses log-log scale."""
    freqs = np.logspace(1, 2, 50)
    psd = 1e-22 * np.ones_like(freqs)
    fig = plotting.plot_noise_spectrum(freqs, psd)
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure), "plot_noise_spectrum should return a Figure"
    ax = fig.axes[0]
    # Check that x-axis is log scale
    assert ax.get_xscale() == 'log' and ax.get_yscale() == 'log', "Frequency and PSD axes should be log scale"


def test_plot_posterior_output():
    """Test that plot_posterior returns a Figure and histogram is plotted."""
    samples = np.random.normal(0, 1, 1000)
    fig = plotting.plot_posterior(samples, "test_param")
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure), "plot_posterior should return a Figure"
    ax = fig.axes[0]
    # Check x-axis label is set to parameter name
    assert ax.get_xlabel() == "test_param", "X-axis label should be parameter name"
    # The histogram should have at least one bar (there should be some patches)
    assert len(ax.patches) > 0, "Histogram bars (patches) should be present in posterior plot"


def test_plot_signal_glitch_overlay_output():
    """Test that plot_signal_glitch_overlay returns a Figure and both signals are plotted."""
    t = np.linspace(0, 0.1, 100)
    sig = np.sin(2*np.pi*50*t)
    glitch = 0.5 * np.sin(2*np.pi*50*t + 0.1)  # slight phase offset or amplitude difference
    fig = plotting.plot_signal_glitch_overlay(t, sig, glitch)
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure), "plot_signal_glitch_overlay should return a Figure"
    ax = fig.axes[0]
    # There should be two lines in the legend (DM Signal and Observed Glitch)
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "DM Signal" in legend_texts and "Observed Glitch" in legend_texts, "Legend should contain both signal and glitch labels"
