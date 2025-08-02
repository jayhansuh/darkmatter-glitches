"""
Visualization and Plotting Module:

This module provides routines to generate publication-quality figures related to the analysis:
- Plotting example waveforms from dark matter decay models.
- Plotting detector noise spectra and time-series with glitches.
- Plotting posterior distributions from Bayesian inference (e.g., corner plots or marginals).
- Overlays of glitch vs. signal waveforms for comparison.

Functions:
- plot_waveform: Plot a time series of a waveform.
- plot_noise_spectrum: Plot a noise Power Spectral Density (PSD) or amplitude spectral density.
- plot_posterior: Plot posterior distribution for a given parameter (or corner plot for multiple).
- plot_signal_glitch_overlay: Overlay a signal waveform with a glitch waveform for comparison.
"""
import matplotlib.pyplot as plt


def plot_waveform(time, strain, title="Waveform", ax=None):
    """
    Plot a gravitational wave strain time series.
    
    Creates a time-domain plot of a waveform, showing strain vs time. If an Axes object is provided, it plots on that; 
    otherwise, it creates a new figure. The plot is labeled with axes titles and an optional title.
    
    Args:
        time (array-like): Array of time values (seconds).
        strain (array-like): Array of strain values (dimensionless) corresponding to the time array.
        title (str, optional): Title for the plot. Default is "Waveform".
        ax (matplotlib.axes.Axes, optional): An existing Axes to plot on. If None, a new figure and axes are created.
    
    Returns:
        matplotlib.figure.Figure: The Figure object containing the plot (useful for saving or further manipulation).
    
    Example:
        >>> t, h = waveforms.simulate_gaussian_burst_waveform(1e-9, 1e20)
        >>> fig = plot_waveform(t, h, title="Example DM decay waveform")
        >>> fig.savefig("waveform.png")
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(time, strain, label='Strain')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Strain")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig


def plot_noise_spectrum(freqs, psd, ax=None):
    """
    Plot a noise power spectral density (PSD) or amplitude spectral density (ASD).
    
    This function creates a log-log plot of detector noise spectrum. If ax is not provided, it creates a new figure. 
    It properly labels the axes (Frequency and PSD/ASD).
    
    Args:
        freqs (array-like): Array of frequency values (Hz).
        psd (array-like): Array of PSD or ASD values (e.g., strain^2/Hz for PSD, or strain/√Hz for ASD).
        ax (matplotlib.axes.Axes, optional): Axes to plot on, if provided.
    
    Returns:
        matplotlib.figure.Figure: The Figure object with the noise spectrum plot.
    
    Example:
        >>> freqs = np.logspace(1, 3, 500)
        >>> psd = 1e-22 * np.ones_like(freqs)  # dummy flat PSD
        >>> fig = plot_noise_spectrum(freqs, psd)
        >>> fig.show()
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.loglog(freqs, psd, color='orange')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [strain^2/Hz]" if psd is not None else "ASD [strain/√Hz]")
    ax.set_title("Detector Noise Spectrum")
    ax.grid(True, which="both")
    return fig


def plot_posterior(parameter_samples, parameter_name, ax=None):
    """
    Plot the posterior distribution for a given parameter.
    
    This function takes a set of samples (or a distribution summary) for one parameter from the Bayesian analysis and produces a histogram or density plot.
    
    Args:
        parameter_samples (array-like): Samples from the posterior distribution of the parameter.
        parameter_name (str): Name of the parameter (for labeling).
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
    
    Returns:
        matplotlib.figure.Figure: The Figure object containing the posterior distribution plot.
    
    Example:
        >>> samples = np.random.normal(0, 1, 10000)  # example posterior samples
        >>> fig = plot_posterior(samples, "particle_mass")
        >>> fig.savefig("particle_mass_posterior.png")
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.hist(parameter_samples, bins=50, density=True, alpha=0.7)
    ax.set_xlabel(f"{parameter_name}")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Posterior of {parameter_name}")
    ax.grid(True)
    return fig


def plot_signal_glitch_overlay(time, signal_strain, glitch_strain, ax=None):
    """
    Overlay a signal waveform and a glitch waveform for comparison.
    
    This plot helps visualize differences or similarities between a hypothesized dark matter signal and an actual glitch.
    Both waveforms (signal and glitch) should be aligned in time for a meaningful comparison.
    
    Args:
        time (array-like): Time array common to both waveforms (seconds).
        signal_strain (array-like): Strain values for the simulated signal.
        glitch_strain (array-like): Strain values for the glitch (should be same length as signal_strain).
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
    
    Returns:
        matplotlib.figure.Figure: The Figure containing the overlay plot.
    
    Example:
        >>> t = np.linspace(0, 0.1, 1000)
        >>> signal = np.sin(2*np.pi*100*t) * np.exp(-((t-0.05)/0.01)**2)  # a fake signal
        >>> glitch = signal * 0.8  # assume glitch is similar but smaller amplitude
        >>> fig = plot_signal_glitch_overlay(t, signal, glitch)
        >>> fig.show()
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(time, signal_strain, label='DM Signal', linestyle='-', linewidth=2)
    ax.plot(time, glitch_strain, label='Observed Glitch', linestyle='--', linewidth=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Strain")
    ax.set_title("Signal vs Glitch Comparison")
    ax.legend()
    ax.grid(True)
    return fig
