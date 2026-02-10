import os
import numpy as np
from scipy import signal
from scipy.signal import hilbert, butter, sosfiltfilt, filtfilt
from tqdm import tqdm

# The following functions implement corrected calculations for room acoustic
# parameters (STI, C50, D50 and T30) based on the guidance in the DIN and ISO
# standards for room acoustics. They avoid some pitfalls present in earlier
# versions of the script, such as double onset alignment and arbitrary time
# limits for the T30 fit. The implementation largely follows the improved
# version found online, but is written here from scratch for clarity.

def envelope_db(sig: np.ndarray, fs: int, window_ms: float = 5.0) -> np.ndarray:
    """Compute a smoothed log‑amplitude envelope of a signal.

    Args:
        sig: Input signal array.
        fs: Sampling frequency in Hz.
        window_ms: Smoothing window length in ms.

    Returns:
        Envelope in dB relative to its maximum (0 dB at peak).
    """
    n = max(1, int(fs * window_ms / 1000.0))
    env = np.convolve(np.abs(sig), np.ones(n) / n, mode="same")
    env_db = 20.0 * np.log10(env / np.max(env) + 1e-12)
    return env_db

def load_rir(file_path: str) -> tuple[np.ndarray, int]:
    """Load a mono or stereo RIR and apply a high‑pass filter at 60 Hz.

    Returns:
        Filtered RIR (mono) and sampling rate.
    """
    import soundfile as sf  # use soundfile for reading/writing
    rir, fs = sf.read(file_path)
    if rir.ndim > 1:
        rir = np.mean(rir, axis=1)
    # high‑pass filter
    fc = 60.0
    Wn = fc / (fs / 2.0)
    b, a = butter(4, Wn, btype="highpass")
    rir = filtfilt(b, a, rir.astype(float))
    return rir, fs

def extract_position(filename: str) -> str:
    return os.path.splitext(filename)[0]

def _align_onset(rir: np.ndarray, fs: int, thresh_db: float = 20.0, pre_ms: float = 2.0, max_ms: float = 300.0) -> tuple[np.ndarray, float]:
    """Align RIR so that the direct sound occurs at the start.

    Uses the Hilbert envelope to find the first sample exceeding a
    noise‑adaptive threshold, then applies a small pre‑delay.

    Returns:
        Aligned RIR and onset time in seconds.
    """
    x = rir.astype(float) - np.mean(rir)
    env = np.abs(hilbert(x))
    n0 = min(len(env), int(0.2 * fs), int(max_ms / 1000.0 * fs))
    noise = np.median(env[:max(1, n0)]) + 1e-12
    thr = noise * (10.0 ** (thresh_db / 20.0))
    above = np.nonzero(env > thr)[0]
    if len(above) == 0:
        return x, 0.0
    i1 = int(above[0])
    i0 = max(0, int(i1 - pre_ms * fs / 1000.0))
    return x[i0:], i0 / fs

def _c50_d50(rir: np.ndarray, fs: int) -> tuple[float, float]:
    """Compute C50 (dB) and D50 (percent) from an aligned RIR."""
    e = rir.astype(float) ** 2
    n50 = max(1, int(round(0.05 * fs)))
    early = float(np.sum(e[:n50]))
    late = float(np.sum(e[n50:]))
    if late <= 0.0:
        return float("nan"), 100.0
    c50 = 10.0 * np.log10(early / late)
    d50 = 100.0 * early / (early + late)
    return c50, d50

def _schroeder_edc(h: np.ndarray) -> np.ndarray:
    e = (h.astype(float)) ** 2
    return np.cumsum(e[::-1])[::-1]

def _schroeder_edc_noise_corrected(h: np.ndarray, fs: int, tail_ms: float = 500.0, min_tail_frac: float = 0.1) -> np.ndarray:
    h = np.asarray(h, float)
    e = h * h
    n = len(e)
    tail = int(max(tail_ms * fs / 1000.0, min_tail_frac * n))
    tail = min(tail, n)
    noise = np.mean(e[-tail:]) if tail > 0 else 0.0
    e_corr = e - noise
    e_corr[e_corr < 0.0] = 0.0
    edc = np.cumsum(e_corr[::-1])[::-1]
    return np.maximum(edc, 1e-30)

def _t30_schroeder(rir: np.ndarray, fs: int) -> float:
    """Estimate T30 using ISO‑3382 (−5 to −35 dB slope). Fallback to T20."""
    edc = _schroeder_edc_noise_corrected(rir, fs, tail_ms=500.0, min_tail_frac=0.1)
    edc_db = 10.0 * np.log10(edc / np.max(edc))
    t = np.arange(len(edc_db)) / fs
    def _fit(mask: np.ndarray) -> float | None:
        if mask.sum() < 10:
            return None
        tt = t[mask]
        yy = edc_db[mask]
        a, b = np.polyfit(tt, yy, 1)
        if a >= 0.0:
            return None
        return -60.0 / a
    mask30 = (edc_db <= -5.0) & (edc_db >= -35.0)
    rt = _fit(mask30)
    if rt is None:
        mask20 = (edc_db <= -5.0) & (edc_db >= -25.0)
        rt = _fit(mask20)
    return float(rt) if rt is not None else float("nan")

def _octave_band_sos(fc: float, fs: int, order: int = 4):
    f1 = fc / np.sqrt(2)
    f2 = fc * np.sqrt(2)
    ny = fs / 2.0
    f1 = max(1.0, f1)
    f2 = min(ny * 0.999, f2)
    return butter(order, [f1 / ny, f2 / ny], btype="band", output="sos")

def _estimate_t0(h: np.ndarray, fs: int, pre_ms: float = 50.0, gate_db: float = 10.0, search_ms: float = 5.0) -> int:
    h = np.asarray(h, float)
    idx_peak = int(np.argmax(np.abs(h)))
    pre = max(0, idx_peak - int(pre_ms * fs / 1000.0))
    noise_seg = h[pre:idx_peak] if idx_peak > pre else h[:max(1, int(0.02 * fs))]
    noise_rms = np.sqrt(np.mean(noise_seg ** 2)) if noise_seg.size else 1e-9
    thr = noise_rms * (10.0 ** (gate_db / 20.0))
    start = max(0, idx_peak - int(search_ms * fs / 1000.0))
    for i in range(start, len(h)):
        if abs(h[i]) >= thr:
            return int(i)
    return int(idx_peak)

def _estimate_band_snr_db(hb: np.ndarray, fs: int, tail_frac: float = 0.2, min_tail_ms: float = 200.0) -> float:
    hb = np.asarray(hb, float)
    n = len(hb)
    tail = max(int(tail_frac * n), int(min_tail_ms * fs / 1000.0))
    tail = min(max(tail, int(0.1 * n)), n)
    noise = hb[-tail:] if tail > 0 else hb[-int(0.1 * n):]
    noise_var = np.var(noise)
    sig_energy = np.sum(hb * hb)
    noise_energy = max(noise_var * len(hb), 1e-20)
    snr_lin = max(sig_energy / noise_energy, 1e-20)
    return 10.0 * np.log10(snr_lin)

def _mtf_from_rir_band(hb: np.ndarray, fs: int, fm: float, snr_db: float | None = None) -> float:
    e = hb.astype(float) ** 2
    if not np.any(e):
        return 0.0
    t = np.arange(len(e)) / fs
    num = np.abs(np.sum(e * np.cos(2.0 * np.pi * fm * t)))
    den_signal = np.sum(e)
    if snr_db is None:
        den = den_signal
    else:
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise = den_signal / max(snr_lin, 1e-12)
        den = den_signal + noise
    m = num / max(den, 1e-20)
    return float(np.clip(m, 0.0, 1.0))

def _ti_from_m(m: float) -> float:
    m = np.clip(m, 1e-9, 1.0 - 1e-9)
    snr_mod = 10.0 * np.log10((m * m) / (1.0 - m * m))
    snr_mod = np.clip(snr_mod, -15.0, 15.0)
    return (snr_mod + 15.0) / 30.0

# Standard centre frequencies and modulation frequencies (IEC)
IEC_BANDS = np.array([125, 250, 500, 1000, 2000, 4000, 8000], float)
IEC_AI_MALE = np.array([0.00, 0.13, 0.14, 0.11, 0.12, 0.09, 0.06], float)
IEC_AI_MALE = IEC_AI_MALE / IEC_AI_MALE.sum()
IEC_MOD_FREQS = np.array([0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5], float)

def calculate_sti(rir: np.ndarray, fs: int, *, band_centers: np.ndarray | None = None, mod_freqs: np.ndarray | None = None, band_importance: np.ndarray | None = None, snr_db_per_band: list[float] | None = None, fade_ms: float = 0.0, auto_t0: bool = True, auto_snr: bool = True) -> tuple[float, np.ndarray]:
    rir = np.asarray(rir, float)
    if band_centers is None:
        band_centers = IEC_BANDS
    if mod_freqs is None:
        mod_freqs = IEC_MOD_FREQS
    if band_importance is None:
        band_importance = IEC_AI_MALE.copy()
    else:
        band_importance = np.asarray(band_importance, float)
        band_importance = band_importance / np.sum(band_importance)
    assert len(band_centers) == 7
    assert len(mod_freqs) == 14
    if auto_t0:
        i0 = _estimate_t0(rir, fs)
        rir = rir[int(i0):]
    if fade_ms and fade_ms > 0.0:
        n = len(rir)
        fade_len = min(n, int(fs * fade_ms / 1000.0))
        if fade_len > 0:
            win = np.hanning(2 * fade_len)[fade_len:]
            rir[-fade_len:] *= win
    n_b = len(band_centers)
    n_m = len(mod_freqs)
    TI_matrix = np.zeros((n_b, n_m), float)
    for bi, fc in enumerate(band_centers):
        sos = _octave_band_sos(fc, fs, order=4)
        hb = sosfiltfilt(sos, rir)
        if snr_db_per_band is not None:
            snr_db = snr_db_per_band[bi]
        elif auto_snr:
            snr_db = _estimate_band_snr_db(hb, fs)
        else:
            snr_db = None
        for mi, fm in enumerate(mod_freqs):
            m = _mtf_from_rir_band(hb, fs, fm, snr_db)
            TI_matrix[bi, mi] = _ti_from_m(m)
    TI_band = TI_matrix.mean(axis=1)
    sti = float(np.sum(band_importance * TI_band))
    if not np.isfinite(TI_matrix).all():
        raise ValueError("TI contains NaN or Inf values.")
    if not (0.0 <= sti <= 1.0):
        raise ValueError("STI outside [0,1].")
    return sti, TI_matrix

def calculate_parameters(rir: np.ndarray, fs: int) -> tuple[float, np.ndarray, float, float, float]:
    # Only compute if RIR has at least 60 ms of data
    if len(rir) < int(0.06 * fs):
        return float("nan"), np.full((7, 14), np.nan), float("nan"), float("nan"), float("nan")
    try:
        sti, TI = calculate_sti(rir, fs, fade_ms=40.0, auto_t0=False, auto_snr=True)
    except Exception as e:
        print(f"STI calculation error: {e}")
        sti = float("nan")
        TI = np.full((7, 14), np.nan)
    # Check direct sound occurs within 20 ms
    n80 = min(len(rir), int(0.08 * fs))
    i_pk = int(np.argmax(np.abs(rir[:max(1, n80)])))
    if i_pk > int(0.02 * fs):
        print(f"Warning: direct peak later than 20 ms (peak at {i_pk / fs:.3f}s)")
    c50, d50 = _c50_d50(rir, fs)
    t30 = _t30_schroeder(rir, fs)
    return sti, TI, c50, d50, t30

def bandwise_params(rir: np.ndarray, fs: int):
    bands = {}
    for fc in IEC_BANDS:
        sos = _octave_band_sos(fc, fs, order=4)
        hb = sosfiltfilt(sos, rir)
        c50, d50 = _c50_d50(hb, fs)
        t30 = _t30_schroeder(hb, fs)
        bands[int(fc)] = (c50, d50, t30)
    return bands

# For demonstration/testing, define a simple main that generates a synthetic
# RIR with known decay time and computes the parameters. This is used when
# executing the script directly.
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000  # sampling rate
    # Generate synthetic RIR: impulse at t=0 and exponential decay with RT60=0.6s
    t = np.arange(int(2.0 * fs)) / fs
    rt60 = 0.6  # expected reverberation time (T60)
    # Exponential decay: level falls by 60 dB over rt60 seconds
    decay = 10 ** (-t * 60.0 / (20.0 * rt60))
    rir = np.zeros_like(t)
    rir[0] = 1.0  # direct sound
    rir += decay * 0.1  # add decaying tail with lower amplitude
    # Align RIR (no pre-alignment needed as direct at start)
    sti, TI, c50, d50, t30 = calculate_parameters(rir, fs)
    print(f"Synthetic RIR parameters:\nSTI = {sti:.3f}, C50 = {c50:.2f} dB, D50 = {d50:.2f} %, T30 = {t30:.2f} s")

