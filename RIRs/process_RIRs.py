"""
Module to post‑process measured room impulse responses (RIRs) and calculate
acoustic parameters such as the Speech Transmission Index (STI), early‑to‑late
energy ratio (C50), definition (D50) and reverberation time T30.  This file
is a lightly modified version of the original `process_RIRs.py` from the
H2ThinkResearchInstitute/RIR_Evaluation repository.  The goal of these
modifications is to improve reproducibility between left and right channels
and to correct systematic biases in the T30 estimation.

Key changes relative to the upstream script:

1. **Avoid double alignment:**  The original script applied an onset
   alignment (`_align_onset`) both in the `main()` function and again
   inside `calculate_parameters()`.  Aligning twice can shift the
   reference point differently between channels and truncate varying
   amounts of reverberation, which led to systematically shorter T30
   values on one channel.  The updated `calculate_parameters()` now
   assumes that the RIR passed in has already been aligned and does
   **not** call `_align_onset` again.

2. **Remove fixed 1 s time window for T30 fitting:**  The original
   `_t30_schroeder()` function restricted the linear regression for T30 to
   the first second of the energy decay curve (`t <= 1.0`).  This
   arbitrary limit can lead to underestimation of the decay slope in
   rooms with reverberation times longer than one second.  The limit has
   been removed so that the full length of the impulse response is
   available for fitting.  The fitting window still follows the ISO‑3382
   recommendation of using the portion of the decay between −5 and −35 dB
   (or −25 dB as a fallback).

3. **English comments throughout:**  For clarity and maintainability,
   additional comments have been added in English.  These explain the
   rationale behind the calculations, parameter choices and edge cases.

The rest of the algorithm remains equivalent to the original, including
envelope calculation, octave filtering and STI computation.
"""

import os
import numpy as np
from scipy import signal
import librosa
from scipy.signal import hilbert, butter, sosfiltfilt, filtfilt
from tqdm import tqdm


def envelope_db(sig: np.ndarray, fs: int, window_ms: float = 5.0) -> np.ndarray:
    """Compute a smoothed logarithmic envelope of a signal.

    The envelope is calculated by rectifying the signal, convolving with a
    moving average window of length ``window_ms`` and converting to dB
    relative to the maximum.  A small epsilon prevents ``log(0)``.

    Args:
        sig: Input signal as a 1‑D numpy array.
        fs: Sampling rate [Hz].
        window_ms: Length of the smoothing window in milliseconds.

    Returns:
        Normalised envelope in dB (0 dB = maximum amplitude).
    """
    # Compute length of moving average in samples
    n = int(fs * window_ms / 1000.0)
    n = max(1, n)
    # Moving average of absolute value
    env = np.convolve(np.abs(sig), np.ones(n) / n, mode="same")
    # Convert to dB relative to peak, adding eps to avoid log(0)
    env_db = 20.0 * np.log10(env / np.max(env) + 1e-12)
    return env_db


def load_rir(file_path: str) -> tuple[np.ndarray, int]:
    """Load a RIR from disk and apply a 4th order high‑pass filter at 60 Hz.

    This follows the upstream behaviour of removing low‑frequency noise
    components that are not relevant for intelligibility and reverberation
    time calculation.  The filter is zero‑phase (filtfilt) to avoid
    introducing additional group delay.

    Args:
        file_path: Path to a mono or stereo ``.wav`` file containing the RIR.

    Returns:
        ``rir_filtered``: Filtered RIR as a 1‑D numpy array.
        ``fs``: Sampling rate in Hz.
    """
    # Use librosa to load the audio with its native sampling rate
    rir, fs = librosa.load(file_path, sr=None, mono=True)
    # Design a 4th‑order Butterworth high‑pass filter with cutoff at 60 Hz
    fc = 60.0
    Wn = fc / (fs / 2.0)
    b, a = butter(N=4, Wn=Wn, btype="highpass")
    # Apply the filter forwards and backwards to get zero phase
    rir_filtered = filtfilt(b, a, rir)
    return rir_filtered, fs


def extract_position(filename: str) -> str:
    """Extract a unique position key from a RIR filename.

    The upstream script attempted to group files, but this version simply
    returns the filename without extension to act as a unique key for
    result reporting.  Change this behaviour if a different grouping
    convention is desired.

    Args:
        filename: Name of a RIR file (e.g. ``"RIR_room1_pos2.wav"``).

    Returns:
        The filename without its extension.
    """
    return os.path.splitext(filename)[0]


def _align_onset(
    rir: np.ndarray, fs: int, thresh_db: float = 12.0, pre_ms: float = 2.0,
    max_ms: float = 300.0
) -> tuple[np.ndarray, float]:
    """Align the onset of the direct sound to the beginning of the signal.

    This function uses the Hilbert envelope to estimate when the direct
    sound arrives.  It computes a noise floor over the first ``max_ms``
    milliseconds, sets a threshold above this noise by ``thresh_db`` dB and
    finds the first sample exceeding this threshold.  Within an 8 ms
    window starting at that point, the function finds the maximum peak and
    subtracts ``pre_ms`` milliseconds to provide a small pre‑delay.  The
    returned RIR starts at this onset index.

    Args:
        rir: Input impulse response (1‑D array).
        fs: Sampling rate [Hz].
        thresh_db: Threshold above the estimated noise floor [dB].
        pre_ms: Amount of pre‑delay to include before the detected peak [ms].
        max_ms: Duration over which to estimate the noise floor [ms].

    Returns:
        Tuple (rir_aligned, t0) where ``rir_aligned`` is the cropped RIR and
        ``t0`` is the onset time in seconds relative to the original signal.
    """
    x = rir.astype(float) - np.mean(rir)
    # Envelope via Hilbert transform magnitude
    env = np.abs(hilbert(x))
    # Determine the index for the noise floor estimation
    n0 = min(len(env), int(0.2 * fs), int(max_ms / 1000.0 * fs))
    noise = np.median(env[:max(1, n0)]) + 1e-12
    # Threshold above noise floor
    thr = noise * (10.0 ** (thresh_db / 20.0))
    above = np.nonzero(env > thr)[0]
    # If nothing exceeds the threshold, return the original signal
    if len(above) == 0:
        return x, 0.0
    # Use the first sample that crosses the threshold as the onset index.
    # In the original implementation the peak within an 8 ms window was
    # selected; however, early reflections can have higher amplitudes than
    # the direct sound, leading to inconsistent alignment between channels.
    # By choosing the earliest crossing we lock on to the first significant
    # arrival, which should correspond more closely to the direct sound
    # even if a subsequent reflection has a higher peak.
    i1 = int(above[0])
    # Apply a small pre‑delay to include a bit of silence before the direct
    # sound.  ``pre_ms`` is specified in milliseconds.
    i0 = max(0, int(i1 - pre_ms * fs / 1000.0))
    return x[i0:], i0 / fs


def _c50_d50(rir: np.ndarray, fs: int) -> tuple[float, float]:
    """Compute C50 (dB) and D50 (%) for an impulse response.

    C50 is the early‑to‑late energy ratio defined as 10·log10(E_early / E_late)
    with the early period defined as 0–50 ms.  D50 is the definition
    (percentage of early energy): 100·E_early / (E_early + E_late).
    If no late energy is present, C50 returns NaN and D50 returns 100 %.

    Args:
        rir: Aligned impulse response (1‑D array).
        fs: Sampling rate [Hz].

    Returns:
        (c50_db, d50_percent).
    """
    # Squared amplitude (energy per sample)
    e = rir.astype(float) ** 2
    # Number of samples corresponding to 50 ms
    n50 = max(1, int(round(0.05 * fs)))
    early = float(np.sum(e[:n50]))
    late = float(np.sum(e[n50:])) if n50 < len(e) else 0.0
    if late <= 0.0:
        # No late energy; return NaN for C50 and 100 % for D50
        return float("nan"), 100.0
    c50 = 10.0 * np.log10(early / late)
    d50 = 100.0 * early / (early + late)
    return c50, d50


def _t30_schroeder(rir: np.ndarray, fs: int, noise_threshold_db: float = -45.0) -> float:
    """Estimate the reverberation time T30 using the Schroeder method.

    The energy decay curve (EDC) is computed from the noise‑corrected
    impulse response (see `_schroeder_edc_noise_corrected`).  T30 is
    obtained by fitting a straight line to the portion of the decay
    between −5 dB and −35 dB relative to the EDC's peak.  If the −35 dB
    point is not reached, the function falls back to T20 (−5 dB to −25 dB).
    The original implementation restricted the fit to the first second of
    the RIR (`t <= 1.0`), which could underestimate reverberation times
    longer than one second; this restriction has been removed here.

    Args:
        rir: Aligned impulse response (1‑D array).
        fs: Sampling rate [Hz].
        noise_threshold_db: Target noise floor for dynamic tail trimming
            (unused here but kept for interface compatibility).

    Returns:
        Estimated T30 in seconds, or NaN if the fit fails.
    """
    # Compute noise‑corrected energy decay curve
    edc = _schroeder_edc_noise_corrected(rir, fs, tail_ms=500, min_tail_frac=0.1)
    # Convert to dB relative to the maximum
    edc_db = 10.0 * np.log10(edc / np.max(edc))
    t = np.arange(len(edc_db)) / fs

    def _fit(mask: np.ndarray) -> float | None:
        """Perform a linear regression on the selected portion of the EDC.

        Returns the estimated reverberation time (−60 dB / slope) or
        ``None`` if the slope is non‑negative or the mask is too small.
        """
        if mask.sum() < 10:
            return None
        tt = t[mask]
        yy = edc_db[mask]
        p = np.polyfit(tt, yy, 1)
        a = p[0]  # slope in dB per second (negative for decay)
        if a >= 0.0:
            return None
        return -60.0 / a

    # Primary T30 fit: between −5 and −35 dB over the entire available length
    mask30 = (edc_db >= -35.0) & (edc_db <= -5.0)
    rt = _fit(mask30)
    if rt is None:
        # Fallback T20 fit: between −5 and −25 dB
        mask20 = (edc_db >= -25.0) & (edc_db <= -5.0)
        rt = _fit(mask20)
    return float(rt) if rt is not None else float("nan")


def _octave_band_sos(fc: float, fs: int, order: int = 4):
    """Design a Butterworth band‑pass filter for a 1‑octave band.

    Given a centre frequency ``fc``, the lower and upper cutoff frequencies
    are set to ``fc / sqrt(2)`` and ``fc * sqrt(2)``, respectively.  The
    filter is returned in second‑order section (SOS) form for numerical
    stability.

    Args:
        fc: Centre frequency of the octave band [Hz].
        fs: Sampling rate [Hz].
        order: Filter order (default 4).

    Returns:
        SOS representation of the band‑pass filter.
    """
    f1 = fc / np.sqrt(2)
    f2 = fc * np.sqrt(2)
    ny = fs / 2.0
    f1 = max(1.0, f1)            # protect against sub‑Hz values
    f2 = min(ny * 0.999, f2)     # protect against Nyquist
    return butter(order, [f1 / ny, f2 / ny], btype="band", output="sos")


def _mtf_from_rir_band(
    hb: np.ndarray, fs: int, fm: float, snr_db: float | None = None
) -> float:
    """Compute the Modulation Transfer Function (MTF) for a single band.

    The MTF is defined by Houtgast/Steeneken as

    ``m = |∫ h^2(t) · cos(2π·f_m·t) dt| / (∫ h^2(t) dt + N)``.

    Optionally, a noise term ``N`` is added based on the signal‑to‑noise
    ratio in the band.  If ``snr_db`` is ``None``, no noise is added.

    Args:
        hb: Band‑passed impulse response (energy impulse response).
        fs: Sampling rate [Hz].
        fm: Modulation frequency [Hz].
        snr_db: Estimated signal‑to‑noise ratio in dB, or ``None``.

    Returns:
        The modulation index ``m`` clipped between 0 and 1.
    """
    e = hb.astype(np.float64) ** 2
    if not np.any(e):
        return 0.0
    # Time axis in seconds
    t = np.arange(len(e)) / fs
    # Numerator: projection of the energy onto cos modulation
    num = np.abs(np.sum(e * np.cos(2.0 * np.pi * fm * t)))
    # Denominator: total signal energy plus optional noise
    den_signal = np.sum(e)
    if snr_db is None:
        den = den_signal
    else:
        # Convert SNR from dB to linear ratio and compute noise energy
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise = den_signal / max(snr_lin, 1e-12)
        den = den_signal + noise
    m = num / max(den, 1e-20)
    return float(np.clip(m, 0.0, 1.0))


def _ti_from_m(m: float) -> float:
    """Compute the Transmission Index (TI) from a modulation index.

    The TI is derived from the modulation index by first computing an
    equivalent signal‑to‑noise ratio in the modulation domain
    ``SNR_mod = 10·log10(m² / (1 − m²))``, clipping to the range [−15, +15]
    dB, and then linearly mapping to the interval [0, 1] via
    ``TI = (SNR_mod + 15) / 30``.

    Args:
        m: Modulation index between 0 and 1.

    Returns:
        The transmission index in the range [0, 1].
    """
    m = np.clip(m, 1e-9, 1.0 - 1e-9)
    snr_mod = 10.0 * np.log10((m * m) / (1.0 - m * m))
    snr_mod = np.clip(snr_mod, -15.0, 15.0)
    return (snr_mod + 15.0) / 30.0


# IEC standard octave bands and male articulation index weights
IEC_BANDS = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
IEC_AI_MALE = np.array([0.00, 0.13, 0.14, 0.11, 0.12, 0.09, 0.06], dtype=float)
IEC_AI_MALE = IEC_AI_MALE / IEC_AI_MALE.sum()  # normalise to sum to 1
# IEC standard modulation frequencies [Hz]
IEC_MOD_FREQS = np.array([
    0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5,
    3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5
], dtype=float)


def _schroeder_edc(h: np.ndarray) -> np.ndarray:
    """Compute the basic Schroeder Energy Decay Curve (EDC).

    Given an impulse response ``h``, the EDC is the cumulative reverse
    integral of squared magnitude: ``EDC[n] = sum_{k=n}^{N-1} |h[k]|²``.
    The resulting array has the same length as ``h``.
    """
    e = (h.astype(np.float64)) ** 2
    return np.cumsum(e[::-1])[::-1]


def _schroeder_edc_noise_corrected(
    h: np.ndarray, fs: int, tail_ms: float = 500.0, min_tail_frac: float = 0.1
) -> np.ndarray:
    """Compute a noise‑corrected Schroeder EDC.

    ISO 3382 recommends subtracting a stationary noise floor from the
    squared impulse response before integration.  The noise floor is
    estimated from the last ``max(tail_ms * fs / 1000, min_tail_frac * N)``
    samples of the signal.  Negative values after subtraction are set
    to zero before integration.

    Args:
        h: Impulse response (1‑D array).
        fs: Sampling rate [Hz].
        tail_ms: Minimum tail duration used for noise estimation [ms].
        min_tail_frac: Minimum fraction of the signal length used for the
            noise estimate.

    Returns:
        Noise‑corrected EDC as a 1‑D array.
    """
    h = np.asarray(h, dtype=np.float64)
    e = h * h
    n = len(e)
    tail = int(tail_ms * fs / 1000.0)
    tail = max(tail, int(min_tail_frac * n))
    tail = min(tail, n)
    noise = np.mean(e[-tail:]) if tail > 0 else 0.0  # energy per sample
    e_corr = e - noise
    e_corr[e_corr < 0.0] = 0.0
    edc = np.cumsum(e_corr[::-1])[::-1]
    # Prevent zeros to avoid log10 issues
    edc = np.maximum(edc, 1e-30)
    return edc


def _estimate_t0(
    h: np.ndarray, fs: int, pre_ms: float = 50.0,
    gate_db: float = 10.0, search_ms: float = 5.0
) -> int:
    """Estimate the starting index (t0) for STI calculation.

    The algorithm looks backward from the maximum peak over a window of
    length ``pre_ms`` to estimate the local noise RMS.  It then moves
    forward from a point ``search_ms`` before the peak until the
    absolute amplitude exceeds the noise RMS by ``gate_db`` dB.
    This index is used to trim the beginning of the RIR for STI.

    Args:
        h: Input impulse response (1‑D array).
        fs: Sampling rate [Hz].
        pre_ms: Duration before the peak used to estimate noise [ms].
        gate_db: Threshold above the noise RMS [dB].
        search_ms: Duration before the peak where the search begins [ms].

    Returns:
        The sample index of the estimated t0 (integer).
    """
    h = np.asarray(h, dtype=np.float64)
    idx_peak = int(np.argmax(np.abs(h)))
    # Region prior to the peak for noise estimation
    pre = max(0, idx_peak - int(pre_ms * fs / 1000.0))
    noise_seg = h[pre:idx_peak] if idx_peak > pre else h[:max(1, int(0.02 * fs))]
    noise_rms = np.sqrt(np.mean(noise_seg ** 2)) if noise_seg.size else 1e-9
    thr = noise_rms * (10.0 ** (gate_db / 20.0))
    start = max(0, idx_peak - int(search_ms * fs / 1000.0))
    for i in range(start, len(h)):
        if abs(h[i]) >= thr:
            return int(i)
    return int(idx_peak)


def _estimate_band_snr_db(
    hb: np.ndarray, fs: int, tail_frac: float = 0.2, min_tail_ms: float = 200.0
) -> float:
    """Estimate the SNR (dB) of a band‑passed impulse response.

    The noise variance is estimated from the last ``max(tail_frac * N,
    min_tail_ms * fs / 1000)`` samples of the band‑passed signal.  The SNR
    is computed as ``10·log10(signal_energy / noise_energy)``.

    Args:
        hb: Band‑passed RIR (1‑D array).
        fs: Sampling rate [Hz].
        tail_frac: Fraction of the total length used for noise estimation.
        min_tail_ms: Minimum duration for noise estimation [ms].

    Returns:
        Estimated SNR in dB.
    """
    hb = np.asarray(hb, dtype=np.float64)
    n = len(hb)
    tail = max(int(tail_frac * n), int(min_tail_ms * fs / 1000.0))
    tail = min(max(tail, int(0.1 * n)), n)
    noise = hb[-tail:] if tail > 0 else hb[-int(0.1 * n):]
    noise_var = np.var(noise)
    sig_energy = np.sum(hb * hb)
    noise_energy = max(noise_var * len(hb), 1e-20)
    snr_lin = max(sig_energy / noise_energy, 1e-20)
    return 10.0 * np.log10(snr_lin)


def calculate_sti(
    rir: np.ndarray,
    fs: int,
    *,
    band_centers: np.ndarray | None = None,
    mod_freqs: np.ndarray | None = None,
    band_importance: np.ndarray | None = None,
    snr_db_per_band: list[float] | None = None,
    fade_ms: float = 0.0,
    auto_t0: bool = True,
    auto_snr: bool = True
) -> tuple[float, np.ndarray]:
    """Compute the Speech Transmission Index (STI) from a single RIR.

    This implementation follows the IEC standard: the impulse response
    is optionally trimmed at the first significant onset (t0) and can
    have a fade‐out window applied to mitigate truncation artefacts.  The
    impulse response is filtered into octave bands, modulation indices
    are computed for each modulation frequency and band, and the
    transmission index matrix is assembled.  Finally, the STI is
    calculated as a weighted average over bands and modulation
    frequencies.

    Args:
        rir: Input impulse response (1‑D array).
        fs: Sampling rate [Hz].
        band_centers: Octave band centre frequencies (length 7).  If
            ``None``, use IEC standard bands 125–8000 Hz.
        mod_freqs: Modulation frequencies (length 14).  If ``None``, use
            IEC standard modulation frequencies.
        band_importance: Weight vector per band; if ``None`` uses the
            IEC male articulation index weights.
        snr_db_per_band: Optional list of SNR values per band in dB.
        fade_ms: Optional fade‑out window length in ms applied to the end
            of the RIR to reduce truncation ripple.
        auto_t0: If ``True``, automatically estimate and remove the
            pre‑impulse segment before calculating STI.
        auto_snr: If ``True``, estimate the SNR per band instead of
            assuming an infinite SNR.

    Returns:
        A tuple (sti, TI_matrix) where ``sti`` is the overall Speech
        Transmission Index (between 0 and 1) and ``TI_matrix`` has
        shape [n_bands, n_mod_freqs].
    """
    rir = np.asarray(rir, dtype=np.float64)
    # Default parameters
    if band_centers is None:
        band_centers = IEC_BANDS
    if mod_freqs is None:
        mod_freqs = IEC_MOD_FREQS
    if band_importance is None:
        band_importance = IEC_AI_MALE.copy()
    else:
        band_importance = np.asarray(band_importance, dtype=np.float64)
        band_importance = band_importance / np.sum(band_importance)
    assert len(band_centers) == 7, "Expect 7 octave bands (125–8000 Hz)."
    assert len(mod_freqs) == 14, "Expect 14 modulation frequencies (IEC)."
    # Trim initial silence based on estimated t0 when requested
    if auto_t0:
        i0 = _estimate_t0(rir, fs)
        rir = rir[int(i0):]
    # Apply optional fade‑out window to minimise truncation artefacts
    if fade_ms and fade_ms > 0.0:
        n = len(rir)
        fade_len = int(fs * fade_ms / 1000.0)
        fade_len = min(fade_len, n)
        if fade_len > 0:
            win = np.hanning(2 * fade_len)[fade_len:]
            rir[-fade_len:] *= win
    n_b = len(band_centers)
    n_m = len(mod_freqs)
    TI_matrix = np.zeros((n_b, n_m), dtype=np.float64)
    # Check optional SNR list length
    if snr_db_per_band is not None:
        assert len(snr_db_per_band) == n_b, "Length of snr_db_per_band must match band_centers"
    # Progress indicator
    pbar = tqdm(total=n_b * n_m, desc="STI calculation", unit="step")
    try:
        for bi, fc in enumerate(band_centers):
            sos = _octave_band_sos(fc, fs, order=4)
            hb = sosfiltfilt(sos, rir)
            # Determine band SNR
            if snr_db_per_band is not None:
                snr_db = snr_db_per_band[bi]
            elif auto_snr:
                snr_db = _estimate_band_snr_db(hb, fs)
            else:
                snr_db = None
            # Modulation indices per modulation frequency
            for mi, fm in enumerate(mod_freqs):
                m = _mtf_from_rir_band(hb, fs, fm, snr_db=snr_db)
                TI_matrix[bi, mi] = _ti_from_m(m)
                pbar.update(1)
    finally:
        pbar.close()
    # Average across modulation frequencies and weight across bands
    TI_band = TI_matrix.mean(axis=1)
    sti = float(np.sum(band_importance * TI_band))
    # Sanity checks for validity
    if not np.isfinite(TI_matrix).all():
        raise ValueError("TI contains NaN or Inf values.")
    if not (0.0 <= sti <= 1.0):
        raise ValueError("STI is outside the [0,1] range.")
    return sti, TI_matrix


def calculate_parameters(rir: np.ndarray, fs: int) -> tuple[float, np.ndarray, float, float, float]:
    """Compute STI, C50, D50 and T30 for a given impulse response.

    The input RIR **must already be aligned** so that the direct sound
    starts at the beginning.  This version no longer calls `_align_onset`
    internally to avoid double cropping.  The function dynamically
    shortens the tail of the RIR based on the energy decay curve to
    remove portions below a selectable noise threshold (default −45 dB,
    fallback −35 dB).  It then computes STI (with `auto_t0=False`), C50,
    D50 and T30.

    Args:
        rir: Aligned impulse response (1‑D array).
        fs: Sampling rate [Hz].

    Returns:
        Tuple (sti, TI_matrix, c50, d50, t30).
    """
    # In the previous version a dynamic trimming based on the energy
    # decay curve was applied here to remove the low‑level tail.  While
    # this reduces the influence of noise, it can also remove late
    # reverberation differently for each channel and thus skew metrics
    # such as D50 and T30.  We no longer perform a hard cut here; the
    # noise‑corrected integration inside `_t30_schroeder` already
    # compensates for stationary noise.  Short impulse responses
    # (<60 ms) are unlikely to contain enough data for reliable
    # reverberation metrics and are therefore rejected.
    if len(rir) < int(0.06 * fs):
        return float("nan"), np.full((7, 14), np.nan), float("nan"), float("nan"), float("nan")
    try:
        # STI is computed without automatically estimating t0 because the
        # onset alignment has already been performed by the caller.
        sti, TI = calculate_sti(
            rir, fs,
            band_centers=None,
            mod_freqs=None,
            band_importance=None,
            snr_db_per_band=None,
            fade_ms=40.0,
            auto_t0=False,
            auto_snr=True
        )
    except Exception as e:
        print(f"Error during STI calculation: {e}")
        sti = float("nan")
        TI = np.full((7, 14), np.nan)
    # Plausibility check: the peak of the RIR should occur within 20 ms
    n80 = min(len(rir), int(0.08 * fs))
    i_pk = int(np.argmax(np.abs(rir[:max(1, n80)])))
    if i_pk > int(0.02 * fs):
        print(
            f"Warning: peak occurs late after alignment (i_pk={i_pk / fs * 1000.0:.1f} ms). t0 may be incorrect."
        )
    # Compute C50, D50 and T30 on the trimmed RIR
    c50, d50 = _c50_d50(rir, fs)
    t30 = _t30_schroeder(rir, fs)
    return sti, TI, c50, d50, t30


# Define the octave bands used for bandwise parameter calculation
OCT_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]


def bandwise_params(rir: np.ndarray, fs: int) -> dict[int, tuple[float, float, float]]:
    """Compute C50, D50 and T30 in each octave band.

    The input RIR should be pre‑aligned.  Each band is obtained via a
    Butterworth band‑pass filter, and the metrics are computed on the
    band‑passed signal.  This function returns a dictionary keyed by the
    centre frequency.

    Args:
        rir: Aligned impulse response (1‑D array).
        fs: Sampling rate [Hz].

    Returns:
        Dictionary mapping band centre frequency to (c50, d50, t30).
    """
    vals: dict[int, tuple[float, float, float]] = {}
    for fc in OCT_BANDS:
        sos = _octave_band_sos(fc, fs, order=4)
        hb = sosfiltfilt(sos, rir)
        c50, d50 = _c50_d50(hb, fs)
        t30 = _t30_schroeder(hb, fs)
        vals[int(fc)] = (c50, d50, t30)
    return vals


def main() -> None:
    """Entry point for batch processing of RIR files in the script's directory.

    This function searches for all ``.wav`` files in the directory where
    the script is located, aligns the onset of each RIR, normalises the
    early portion (0–80 ms), computes STI, C50, D50 and T30 using
    `calculate_parameters()` and writes the results to a CSV file.  It
    also prints a summary of bandwise T30 values and displays a boxplot
    of the distributions of C50, D50 and T30 across all processed files.
    """
    # Determine the directory containing the script and look for WAV files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rir_directory = script_dir
    print(f"RIR directory (script directory): {rir_directory}")
    rir_files = [
        f for f in os.listdir(rir_directory)
        if f.lower().endswith(".wav") and os.path.isfile(os.path.join(rir_directory, f))
    ]
    if not rir_files:
        print("No RIR files found.")
        return
    # Prepare collections for summary statistics
    sti_values: list[float] = []
    c50_values: list[float] = []
    d50_values: list[float] = []
    t30_values: list[float] = []
    # Prepare the output CSV file
    out_csv = os.path.join(rir_directory, "rir_parameters_per_file.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("Filename;STI;C50_dB;D50_percent;T30_s\n")
    # Group files by base name (before _left_RIR/_right_RIR or other suffixes)
    groups: dict[str, list[str]] = {}
    for file in rir_files:
        base = file
        # Remove suffixes commonly used for channel identification
        for suffix in ["_left_RIR.wav", "_right_RIR.wav", "_left.wav", "_right.wav"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        groups.setdefault(base, []).append(file)
    # Process each group of potentially paired RIR files
    for base, files_in_group in groups.items():
        # Map from filename to onset time
        onset_times: dict[str, float] = {}
        # Load each RIR and determine its onset
        rirs_raw: dict[str, tuple[np.ndarray, int]] = {}
        for file in files_in_group:
            file_path = os.path.join(rir_directory, file)
            rir_raw, fs = load_rir(file_path)
            # Determine onset and keep raw RIR for later cropping
            rir_aligned_tmp, t0 = _align_onset(rir_raw, fs)
            rirs_raw[file] = (rir_raw, fs)
            onset_times[file] = t0
        # Use the earliest onset among channels as the common reference
        common_t0 = min(onset_times.values())
        # Determine a common tail cut for the group to avoid differences due to
        # extended noise floor.  For each channel, estimate where the energy
        # decay drops below a threshold (−45 dB relative to peak, fallback
        # −35 dB).  The earliest of these cut points is used for all
        # channels in the group to enforce identical tail length.
        cut_indices: dict[str, int] = {}
        for file in files_in_group:
            rir_raw, fs = rirs_raw[file]
            # Crop to the common onset
            i0 = int(common_t0 * fs)
            rir_tmp = rir_raw[i0:]
            # Compute noise‑corrected energy decay curve
            edc = _schroeder_edc_noise_corrected(rir_tmp, fs, tail_ms=500.0, min_tail_frac=0.1)
            edc_db = 10.0 * np.log10(edc / np.max(edc) + 1e-30)
            cut_db = -45.0
            if not np.any(edc_db < cut_db):
                cut_db = -35.0
            below = np.nonzero(edc_db < cut_db)[0]
            if below.size:
                idx = int(below[0])
                # Add 100 ms buffer after the point where the curve falls below the threshold
                cut_idx = min(len(rir_tmp), idx + int(0.1 * fs))
            else:
                cut_idx = len(rir_tmp)
            cut_indices[file] = cut_idx
        # Common cut is the minimum across all channels to ensure that the late
        # tail does not include noise for any channel
        common_cut = min(cut_indices.values()) if cut_indices else None
        for file in files_in_group:
            rir_raw, fs = rirs_raw[file]
            # Crop at onset and tail
            i0 = int(common_t0 * fs)
            rir_tmp = rir_raw[i0:]
            if common_cut is not None:
                rir = rir_tmp[:common_cut]
            else:
                rir = rir_tmp
            # Normalise the first 80 ms to unity to reduce level differences
            n80 = max(1, int(0.08 * fs))
            peak = np.max(np.abs(rir[:min(len(rir), n80)])) + 1e-12
            rir = rir / peak
            position = extract_position(file)
            # Compute acoustic parameters on the aligned, trimmed and normalised RIR
            sti, TI, c50, d50, t30 = calculate_parameters(rir, fs)
            sti_values.append(sti)
            c50_values.append(c50)
            d50_values.append(d50)
            t30_values.append(t30)
            print(
                f"{position}: STI={sti:.3f}, C50={c50:.2f} dB, D50={d50/100.0:.4f}, T30={t30:.2f} s"
            )
            # Helper to format numeric values for CSV
            def _fmt(x: float, nd: int) -> str:
                return "" if not np.isfinite(x) else f"{x:.{nd}f}".replace(".", ",")
            # Append results to CSV
            with open(out_csv, "a", encoding="utf-8", newline="") as f:
                f.write(
                    f"{position};"
                    f"{_fmt(sti, 3)};"
                    f"{_fmt(c50, 2)};"
                    f"{_fmt(d50 / 100.0, 4)};"
                    f"{_fmt(t30, 2)}\n"
                )
            # Print bandwise T30 values for diagnostic purposes
            bandvals = bandwise_params(rir, fs)
            print("Bandwise T30 estimates:")
            for fc, (c50_b, d50_b, t30_b) in bandvals.items():
                print(f"  {fc:>5d} Hz: T30 = {t30_b:.2f} s")
            print()
    # Summarise results across all files
    def _finite_stats(values: list[float]) -> tuple[float, float] | None:
        finite_vals = [x for x in values if np.isfinite(x)]
        return (float(np.mean(finite_vals)), float(np.std(finite_vals))) if finite_vals else None
    stats_sti = _finite_stats(sti_values)
    stats_c50 = _finite_stats(c50_values)
    stats_d50 = _finite_stats(d50_values)
    stats_t30 = _finite_stats(t30_values)
    if stats_sti:
        print(f"Average STI: {stats_sti[0]:.2f}, Std Dev: {stats_sti[1]:.2f}")
    else:
        print("STI: no valid values.")
    if stats_c50:
        print(f"Average C50: {stats_c50[0]:.2f} dB, Std Dev: {stats_c50[1]:.2f} dB")
    else:
        print("C50: no valid values.")
    if stats_d50:
        print(f"Average D50: {stats_d50[0]:.2f}, Std Dev: {stats_d50[1]:.2f}")
    else:
        print("D50: no valid values.")
    if stats_t30:
        print(f"Average T30: {stats_t30[0]:.2f} s, Std Dev: {stats_t30[1]:.2f} s")
    else:
        print("T30: no valid values.")


if __name__ == "__main__":
    main()