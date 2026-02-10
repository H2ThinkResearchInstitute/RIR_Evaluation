"""
calculate_ir
================

This module contains an improved implementation for computing a room impulse
response (RIR) from a recorded logarithmic sine sweep.  The method is
based on deconvolving the recorded sweep with a properly scaled time‑reversed
version of the original sweep (an *inverse filter*).  For a sweep whose
instantaneous frequency changes exponentially from ``f1`` to ``f2`` over a
duration ``T``, the amplitude of the inverse filter must be multiplied by
an exponential factor to compensate for the sweep's varying spectral
density【321666997635485†L214-L275】.  Without this compensation the
resulting impulse response exhibits a skewed frequency response
【321666997635485†L214-L275】.

The function provided here automatically estimates the start and end
frequencies of the sweep when they are not given, constructs the
inverse filter, performs the deconvolution and returns a cropped impulse
response with a configurable pre‑silence and duration.

The implementation accepts numpy arrays and supports optional plotting of
the impulse response and its envelope for inspection.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from typing import Optional, Tuple


def _estimate_sweep_frequencies(sweep: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    """Estimate the start and end frequency of an exponential sweep.

    The instantaneous frequency is obtained from the unwrapped phase of the
    analytic signal.  Outliers are rejected using percentiles to avoid
    including spurious values caused by noise.

    Args:
        sweep: 1‑D array containing the original sweep signal.
        sample_rate: Sampling rate in Hz.

    Returns:
        A tuple ``(f_start, f_end)`` containing the estimated starting and
        ending frequencies in Hz.
    """
    # Form the analytic signal and unwrap its phase
    analytic = signal.hilbert(sweep)
    phase = np.unwrap(np.angle(analytic))
    # Compute instantaneous frequency; the derivative of phase yields
    # omega(t) = dφ/dt.  Divide by 2π to get frequency in Hz.
    inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
    # Discard extreme values to reduce the influence of noise
    f_start = np.percentile(inst_freq, 1)
    f_end = np.percentile(inst_freq, 99)
    # Ensure positive frequencies
    f_start = max(f_start, 1.0)
    f_end = max(f_end, f_start + 1.0)
    return float(f_start), float(f_end)


def calculate_ir(
    original_sweep_path: str,
    recorded_sweep: np.ndarray,
    sample_rate: int,
    *,
    f_start: Optional[float] = None,
    f_end: Optional[float] = None,
    pre_silence: float = 0.2,
    ir_duration: float = 4.0,
    plot_results: bool = False,
    override_onset: Optional[int] = None,
    return_onset: bool = False,
) -> Tuple[np.ndarray, int] | Tuple[np.ndarray, int, int]:
    """Compute a room impulse response from a recorded sine sweep.

    The function loads the original sweep, constructs a time‑reversed
    inverse filter with exponential amplitude compensation as described by
    Jojeck【321666997635485†L214-L275】, performs an FFT‑based convolution
    with the recorded signal and extracts a window centred on the direct
    sound.  If the starting and ending frequencies of the sweep are not
    provided, they are estimated automatically.

    Args:
        original_sweep_path: Path to the WAV file containing the test sweep.
        recorded_sweep: Mono recording of the played sweep (one channel of
            a stereo recording can be passed).  The array will be
            converted to floating point internally.
        sample_rate: Sampling rate of the recording in Hz.
        f_start: Optional starting frequency of the sweep (in Hz).  If
            ``None`` the frequency is estimated.
        f_end: Optional ending frequency of the sweep (in Hz).  If
            ``None`` the frequency is estimated.
        pre_silence: Amount of pre‑silence before the direct sound in
            seconds when cropping the impulse response.  The returned
            RIR will be trimmed so that the largest peak (assumed to
            represent the direct sound) occurs exactly at this offset
            (e.g. ``pre_silence=0.2`` yields a peak at 0.2 s), which
            simplifies alignment in audio editors.
        ir_duration: Duration in seconds of the returned impulse response
            segment (including pre_silence).
        plot_results: If ``True``, display the impulse response and its
            envelope.

    Returns:
        A tuple ``(rir, sample_rate)`` where ``rir`` is the cropped impulse
        response and ``sample_rate`` is the sampling rate of the result.

        If ``return_onset`` is ``True``, a third element is appended
        containing the index (in samples) of the detected or overridden
        direct sound onset used for cropping.
    """
    # Load the original sweep from disk and verify sample rate
    sweep_rate, sweep = wavfile.read(original_sweep_path)
    if sweep_rate != sample_rate:
        raise ValueError(
            f"Abtastrate des Sweeps ({sweep_rate} Hz) stimmt nicht mit Aufnahme ({sample_rate} Hz) überein."
        )
    
    # Convert to float64 for numerical stability.  Handle stereo sweeps by
    # converting to mono (averaging channels)
    if sweep.ndim == 2:
        sweep = sweep.mean(axis=1)
    sweep = np.asarray(sweep, dtype=np.float64)

    # Convert recorded sweep to float
    recorded = np.asarray(recorded_sweep, dtype=np.float64)

    # Estimate sweep start/end frequencies if not provided
    if f_start is None or f_end is None:
        f_start_est, f_end_est = _estimate_sweep_frequencies(sweep, sample_rate)
        f_start = f_start if f_start is not None else f_start_est
        f_end = f_end if f_end is not None else f_end_est

    # Duration and exponential sweep rate
    T = len(sweep) / sample_rate
    R = np.log(f_end / f_start)

    # Construct time vector for inverse filter
    t = np.arange(len(sweep)) / sample_rate

    # Compute amplitude compensation factor.  According to Farina's
    # exposition【321666997635485†L214-L275】, the inverse filter is the
    # time‑reversed sweep divided by k(t) = e^{t·R/T}.
    k = np.exp(t * R / T)
    inv_filter = sweep[::-1] / k

    # Normalise inverse filter to unit peak to avoid overflow in convolution
    inv_filter /= np.max(np.abs(inv_filter))

    # Perform convolution.  Using FFT is much faster for long signals
    rir_full = signal.fftconvolve(recorded, inv_filter, mode="full")

    # --- Determine the onset of the direct sound ---------------------------
    if override_onset is not None:
        # Allow callers to override the onset detection and use a fixed index.
        direct_index = int(override_onset)
    else:
        # Traditionally the largest peak in the impulse response corresponds
        # to the direct sound.  However, early reflections can sometimes be
        # stronger than the true direct arrival, especially when using
        # directional microphones or in highly reflective rooms【369875728283459†L116-L121】.
        # To mitigate this, we locate the first significant rise in the
        # envelope of the impulse response rather than simply selecting the
        # global maximum.  The Hilbert transform yields an analytic signal
        # whose magnitude provides a smooth amplitude envelope【369875728283459†L41-L70】.
        env = np.abs(signal.hilbert(rir_full))
        max_env = np.max(env)
        # Define a threshold as a fraction of the maximum.  Values below this
        # are considered noise or leakage and ignored.  A 20 % threshold has
        # been found to work well for typical swept‑sine measurements; adjust
        # if necessary.
        threshold = 0.2 * max_env
        # Find the first index where the envelope exceeds the threshold.  If
        # none is found (which would be pathological), fall back to the
        # position of the global maximum.
        onset_candidates = np.where(env >= threshold)[0]
        if len(onset_candidates) > 0:
            direct_index = int(onset_candidates[0])
        else:
            direct_index = int(np.argmax(env))

    pre_samples = int(pre_silence * sample_rate)
    total_samples = int((pre_silence + ir_duration) * sample_rate)
    # Crop the response so that the direct sound occurs at ``pre_silence`` seconds.
    start_idx = max(0, direct_index - pre_samples)
    end_idx = start_idx + total_samples
    rir_cropped = rir_full[start_idx:end_idx]

    # Normalise cropped impulse response to unit peak
    if np.max(np.abs(rir_cropped)) > 0:
        rir_cropped = rir_cropped / np.max(np.abs(rir_cropped))

    # Optionally plot the results
    if plot_results:
        import matplotlib.pyplot as plt

        # Show time in seconds with the direct sound at ``pre_silence``.
        time_axis = np.arange(len(rir_cropped)) / sample_rate

        # Envelope using Hilbert transform
        analytic_ir = signal.hilbert(rir_cropped)
        amplitude_envelope = np.abs(analytic_ir)
        log_envelope = 20 * np.log10(amplitude_envelope + 1e-20)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(time_axis, rir_cropped)
        ax1.set_title("Impulsantwort (berechnet)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        ax2.plot(time_axis, log_envelope)
        ax2.set_title("Logarithmische Einhüllende (dB)")
        ax2.set_ylabel("Amplitude (dB)")
        ax2.set_xlabel("Zeit (s)")
        ax2.grid(True)
        plt.tight_layout()
        plt.show()
        
    # Decide whether to return the onset index
    if return_onset:
        return rir_cropped.astype(np.float32), sample_rate, direct_index
    else:
        return rir_cropped.astype(np.float32), sample_rate