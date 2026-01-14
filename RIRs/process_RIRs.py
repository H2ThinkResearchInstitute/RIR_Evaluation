import os
import re
import numpy as np
from scipy import signal
import librosa
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt, filtfilt
from tqdm import tqdm  # Fortschrittsbalken-Bibliothek

# Pfad zu den RIR-Dateien: im selben Ordner wie dieses Skript
script_dir = os.path.dirname(os.path.abspath(__file__))
rir_directory = script_dir
print(f"RIR-Ordner (Skriptordner): {rir_directory}")

# Alle .wav-Dateien im Verzeichnis auflisten
rir_files = [
    f for f in os.listdir(rir_directory)
    if f.lower().endswith(".wav")
    and os.path.isfile(os.path.join(rir_directory, f))
]

def envelope_db(sig, fs, window_ms=5):
        n = int(fs * window_ms / 1000)
        env = np.convolve(np.abs(sig), np.ones(n)/n, mode='same')
        env_db = 20 * np.log10(env / np.max(env) + 1e-12)  # normiert, +eps um log(0) zu vermeiden
        return env_db

# Funktion zum Laden einer RIR-Datei
def load_rir(file_path):
    rir, fs = librosa.load(file_path, sr=None)

    # Hochpassfilter
    fc = 60
    Wn = fc / (fs / 2)
    b, a = butter(N=4, Wn=Wn, btype='highpass')
    rir_filtered = filtfilt(b, a, rir)

    """
    # Zeitachse
    t = np.arange(len(rir)) / fs

    env_orig = envelope_db(rir, fs)
    env_filt = envelope_db(rir_filtered, fs)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, env_orig, label="Original", alpha=0.8)
    plt.plot(t, env_filt, label="Gefiltert (50 Hz HP)", alpha=0.8)
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude [dB, normiert]")
    plt.title("Einhüllende der RIR (logarithmischer Maßstab)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """

    return rir_filtered, fs

def extract_position(filename):
    parts = filename.split('_')
    pos_rep = parts[2].split('-')
    # Neu: keine Gruppierung/Mittelung mehr -> eindeutiger Key pro Datei (ohne Endung)
    return os.path.splitext(filename)[0]


def _align_onset(rir, fs, thresh_db=12.0, pre_ms=2.0, max_ms=300.0):
    x = rir.astype(float) - np.mean(rir)
    env = np.abs(signal.hilbert(x))

    # Noise aus z.B. ersten 200 ms (oder bis max_ms), damit median stabiler ist
    n0 = min(len(env), int(0.2 * fs), int(max_ms/1000 * fs))

    noise = np.median(env[:max(1, n0)]) + 1e-12
    thr = noise * (10.0 ** (thresh_db / 20.0))
    above = np.nonzero(env > thr)[0]
    if len(above) == 0:
        return x, 0.0
    i1 = above[0]

    # kürzeres Suchfenster, damit frühe Reflexionen/Modulationen weniger „übernehmen“
    win = slice(i1, min(len(x), i1 + int(0.008*fs)))  # 8 ms Fenster
    
    i_peak = int(np.argmax(np.abs(x[win])) + i1)
    i0 = max(0, int(i_peak - pre_ms * fs / 1000.0))
    return x[i0:], i0 / fs

def _c50_d50(rir, fs):
    e = rir.astype(float)**2
    n50 = max(1, int(round(0.05 * fs)))
    early = e[:n50].sum()
    late = e[n50:].sum() if n50 < len(e) else 0.0
    if late <= 0:
        return np.nan, 100.0
    c50 = 10.0 * np.log10(early / late)
    d50 = 100.0 * early / (early + late)
    return c50, d50

def _t30_schroeder(rir, fs, noise_threshold_db=-45.0):
    # Noise-corrected EDC (wichtig bei langen RIRs mit konstantem Noise-Floor)
    edc = _schroeder_edc_noise_corrected(rir, fs, tail_ms=500, min_tail_frac=0.1)
    edc_db = 10.0 * np.log10(edc / edc.max())

    t = np.arange(len(edc_db)) / fs
    
    # Primär: T30 Fit (-5..-35). Falls -35 nicht erreicht wird: Fallback auf T20 (-5..-25).
    def _fit(mask):
        if mask.sum() < 10:
            return None
        tt = t[mask]
        yy = edc_db[mask]
        p = np.polyfit(tt, yy, 1)
        a = p[0]
        if a >= 0:
            return None
        return -60.0 / a

    mask30 = (edc_db >= -35.0) & (edc_db <= -5.0) & (t <= 1.0)
    rt = _fit(mask30)
    if rt is None:
        mask20 = (edc_db >= -25.0) & (edc_db <= -5.0) & (t <= 1.0)
        rt = _fit(mask20)
    if rt is None:
        return np.nan

    return rt

def _octave_band_sos(fc, fs, order=4):
    """Butterworth-Filter als Oktavband (125–8000 Hz)."""
    f1 = fc / np.sqrt(2)
    f2 = fc * np.sqrt(2)
    ny = fs / 2.0
    f1 = max(1.0, f1)              # untere Grenze schützen
    f2 = min(ny * 0.999, f2)       # obere Grenze schützen
    return butter(order, [f1/ny, f2/ny], btype="band", output="sos")

def _mtf_from_rir_band(hb, fs, fm, snr_db=None):
    """
    Modulationsübertragungsfunktion m(fm) nach Houtgast/Steeneken:
    m = |∫ h^2(t) cos(2π fm t) dt| / (∫ h^2(t) dt + N)
    Optional: N wird aus SNR(dB) geschätzt (N = Signalenergie / SNR_linear).
    """
    e = hb.astype(np.float64)**2
    if not np.any(e):
        return 0.0

    t = np.arange(len(e)) / fs
    num = np.abs(np.sum(e * np.cos(2*np.pi*fm*t)))
    den_signal = np.sum(e)

    if snr_db is None:
        den = den_signal
    else:
        snr_lin = 10**(snr_db/10.0)
        noise = den_signal / max(snr_lin, 1e-12)
        den = den_signal + noise

    m = num / max(den, 1e-20)
    return float(np.clip(m, 0.0, 1.0))

def _ti_from_m(m):
    """
    Transmission Index je Modulationsfrequenz:
    SNR_mod = 10*log10( m^2 / (1 - m^2) ), begrenzt auf [-15, +15] dB
    TI = (SNR_mod + 15) / 30
    """
    m = np.clip(m, 1e-9, 1-1e-9)
    snr_mod = 10.0 * np.log10((m*m) / (1.0 - m*m))
    snr_mod = np.clip(snr_mod, -15.0, 15.0)
    return (snr_mod + 15.0) / 30.0

IEC_BANDS = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
IEC_AI_MALE = np.array([0.00, 0.13, 0.14, 0.11, 0.12, 0.09, 0.06], dtype=float)
IEC_AI_MALE = IEC_AI_MALE / IEC_AI_MALE.sum()  # normiert auf 1
IEC_MOD_FREQS = np.array([0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15,
                          4.0, 5.0, 6.3, 8.0, 10.0, 12.5], dtype=float)
def _schroeder_edc(h):
    e = (h.astype(np.float64))**2
    return np.cumsum(e[::-1])[::-1]

def _schroeder_edc_noise_corrected(h, fs, tail_ms=500, min_tail_frac=0.1):
    """
    ISO-3382-Style: Schätze stationäres Rauschen aus dem Tail und ziehe es sampleweise ab,
    bevor integriert wird. Das stabilisiert T20/T30 bei langen, verrauschten Aufnahmen.
    """
    h = np.asarray(h, dtype=np.float64)
    e = h*h
    n = len(e)
    tail = int(tail_ms * fs / 1000.0)
    tail = max(tail, int(min_tail_frac * n))
    tail = min(tail, n)
    noise = np.mean(e[-tail:]) if tail > 0 else 0.0  # Energie pro Sample
    e_corr = e - noise
    e_corr[e_corr < 0] = 0.0
    edc = np.cumsum(e_corr[::-1])[::-1]
    edc = np.maximum(edc, 1e-30)
    return edc

def _estimate_t0(h, fs, pre_ms=50, gate_db=10, search_ms=5):
    """Startmarke: ab ~max-Peak rückwärts ein kurzes Fenster, Rauschen messen,
       t0 = erster Sample ≥ gate_db über Rausch-RMS im Umfeld des Peaks."""
    h = np.asarray(h, dtype=np.float64)
    idx_peak = int(np.argmax(np.abs(h)))
    pre = max(0, idx_peak - int(pre_ms * fs / 1000))
    noise_seg = h[pre:idx_peak] if idx_peak > pre else h[:max(1, int(0.02*fs))]
    noise_rms = np.sqrt(np.mean(noise_seg**2)) if noise_seg.size else 1e-9
    thr = noise_rms * (10**(gate_db/20.0))
    start = max(0, idx_peak - int(search_ms * fs / 1000))
    for i in range(start, len(h)):
        if abs(h[i]) >= thr:
            return i
    return idx_peak

def _estimate_band_snr_db(hb, fs, tail_frac=0.2, min_tail_ms=200):
    hb = np.asarray(hb, dtype=np.float64)
    n = len(hb)
    tail = max(int(tail_frac*n), int(min_tail_ms*fs/1000))
    tail = min(max(tail, int(0.1*n)), n)
    noise = hb[-tail:] if tail > 0 else hb[-int(0.1*n):]
    noise_var = np.var(noise)
    sig_energy = np.sum(hb*hb)
    noise_energy = max(noise_var * len(hb), 1e-20)
    snr_lin = max(sig_energy / noise_energy, 1e-20)
    return 10*np.log10(snr_lin)


def calculate_sti(rir, fs, *,
                  band_centers=None,
                  mod_freqs=None,
                  band_importance=None,
                  snr_db_per_band=None,
                  fade_ms=0.0,
                  auto_t0=True,
                  auto_snr=True):
    """
    Berechnet den STI aus einer Einzel-RIR.
    Parameter:
      - rir: 1D-Array der Impulsantwort
      - fs: Abtastrate [Hz]
      - band_centers: Oktavband-Mittenfrequenzen (7 Bänder, 125–8000 Hz)
      - mod_freqs: Modulationsfrequenzen nach IEC (14 Werte)
      - band_importance: Gewichte je Oktavband (Standard: gleichgewichtet)
      - snr_db_per_band: optionale Liste gleicher Länge wie band_centers (Rausch-SNR in dB)
      - fade_ms: optionales Hann-Fenster am Ende gegen Clipping/Trunkation
    Rückgabe:
      - sti: Skalar
      - TI_matrix: Array [n_bands, n_mods] der Transmission Indices
    """
    rir = np.asarray(rir, dtype=np.float64)
    # Normierte IEC-Defaults
    if band_centers is None:
        band_centers = IEC_BANDS
    if mod_freqs is None:
        mod_freqs = IEC_MOD_FREQS
    if band_importance is None:
        band_importance = IEC_AI_MALE.copy()
    else:
        band_importance = np.asarray(band_importance, dtype=np.float64)
        band_importance = band_importance / np.sum(band_importance)
    assert len(band_centers) == 7, "Erwarte genau 7 Oktavbänder (125–8000 Hz)."
    assert len(mod_freqs) == 14, "Erwarte 14 Modulationsfrequenzen (IEC)."
    
    # t0 setzen und Anfang vor t0 kappen
    if auto_t0:
        i0 = _estimate_t0(rir, fs)
        rir = rir[i0:]

    if fade_ms and fade_ms > 0:
        n = len(rir)
        fade_len = int(fs * fade_ms / 1000.0)
        fade_len = min(fade_len, n)
        if fade_len > 0:
            win = np.hanning(2*fade_len)[fade_len:]  # sanfter Ausklang
            rir[-fade_len:] *= win

    n_b = len(band_centers)
    n_m = len(mod_freqs)
    TI_matrix = np.zeros((n_b, n_m), dtype=np.float64)

    if snr_db_per_band is not None:
        assert len(snr_db_per_band) == n_b, "snr_db_per_band Länge passt nicht zu band_centers"

    pbar = tqdm(total=n_b * n_m, desc="STI-Berechnung", unit="step")
    try:
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
                m = _mtf_from_rir_band(hb, fs, fm, snr_db=snr_db)
                TI_matrix[bi, mi] = _ti_from_m(m)
                pbar.update(1)
    finally:
        pbar.close()

    # Mittelung: erst über Modulationsfrequenzen (gleichgewichtet),
    # dann über Bänder mit band_importance.
    TI_band = TI_matrix.mean(axis=1)
    sti = float(np.sum(band_importance * TI_band))
    # Sanity-Checks
    if not np.isfinite(TI_matrix).all():
        raise ValueError("TI enthält NaN/Inf.")
    if not (0.0 <= sti <= 1.0):
        raise ValueError("STI außerhalb [0,1].")
    return sti, TI_matrix


def calculate_parameters(rir, fs):
    rir, t0 = _align_onset(rir, fs)

    # dynamisches Kürzen (noise-corrected): erst -45 dB, sonst wenigstens -35 dB
    edc = _schroeder_edc_noise_corrected(rir, fs, tail_ms=500, min_tail_frac=0.1)
    edc_db = 10*np.log10(edc/np.max(edc) + 1e-30)
    cut_db = -45.0
    if not np.any(edc_db < cut_db):
        cut_db = -35.0
    below = np.nonzero(edc_db < cut_db)[0]
    if below.size:
        idx = int(below[0])
        rir = rir[:min(len(rir), idx + int(0.1*fs))]  # +100 ms Puffer

    if len(rir) < int(0.06 * fs):
        return np.nan, np.nan, np.nan, np.nan

    try:
        sti, TI = calculate_sti(
            rir, fs,
            band_centers=None,
            mod_freqs=None,
            band_importance=None,
            snr_db_per_band=None,
            fade_ms=40.0,      # längerer Fade gegen Trunkations-Welligkeit
            auto_t0=False,
            auto_snr=True
        )
    except Exception as e:
        print(f"Fehler bei der STI-Berechnung: {e}")
        sti = np.nan
        TI = np.full((7, 14), np.nan)

    # Plausibilitätscheck: Peak muss sehr früh liegen
    n80 = min(len(rir), int(0.08*fs))
    i_pk = int(np.argmax(np.abs(rir[:max(1, n80)])))
    if i_pk > int(0.02*fs):  # Peak erst nach 20 ms -> verdächtig
        print(f"Warnung: Peak sehr spät nach Alignment (i_pk={i_pk/fs*1000:.1f} ms). t0 evtl. falsch.")

    c50, d50 = _c50_d50(rir, fs)
    t30 = _t30_schroeder(rir, fs)
    return sti, TI, c50, d50, t30

OCT_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]

def bandwise_params(rir, fs):
    vals = {}
    for fc in OCT_BANDS:
        sos = _octave_band_sos(fc, fs, order=4)
        hb = sosfiltfilt(sos, rir)
        c50, d50 = _c50_d50(hb, fs)
        t30 = _t30_schroeder(hb, fs)
        vals[fc] = (c50, d50, t30)
    return vals

# Hauptskript
def main():
    if not rir_files:
        return

    sti_values = []
    c50_values = []
    d50_values = []
    t30_values = []

    # --- Neu: Ergebnisdatei (pro RIR/Position) ---
    out_csv = os.path.join(rir_directory, "rir_parameters_per_file.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("Dateiname;STI;C50_dB;D50_percent;T30_s\n")

    for file in rir_files:
        file_path = os.path.join(rir_directory, file)
        rir, fs = load_rir(file_path)
        rir, _ = _align_onset(rir, fs)

        # Frühbereich-Normierung (0–80 ms), wie zuvor
        n80 = max(1, int(0.08 * fs))
        peak = np.max(np.abs(rir[:min(len(rir), n80)])) + 1e-12
        rir = rir / peak

        position = os.path.splitext(file)[0]

        sti, TI, c50, d50, t30 = calculate_parameters(rir, fs)
        sti_values.append(sti)
        c50_values.append(c50)
        d50_values.append(d50)
        t30_values.append(t30)
        print(f"{position}: t0-Ausrichtung OK | STI={sti:.3f} | TI-Bandmittel={np.nanmean(TI, axis=1)}")

        def _fmt(x, nd):
            if not np.isfinite(x):
                return ""
            return f"{x:.{nd}f}".replace(".", ",")

        with open(out_csv, "a", encoding="utf-8", newline="") as f:
            f.write(
                f"{position};"
                f"{_fmt(sti,3)};"
                f"{_fmt(c50,2)};"
                f"{_fmt(d50/100.0,4)};"
                f"{_fmt(t30,2)}\n"
            )

        # --- Neu: pro RIR/Position in Textdatei schreiben ---
        def _fmt(x, nd=3):
            return "NaN" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"

        # --- Test: T30 bandweise prüfen ---
        bandvals = bandwise_params(rir, fs)
        print("\nBandweise T30-Auswertung für Raum 115 (EWA):")
        for fc, (c50_b, d50_b, t30_b) in bandvals.items():
            print(f"  {fc:>5.0f} Hz: T30 = {t30_b:.2f} s")
        print()

    finite_sti = [x for x in sti_values if np.isfinite(x)]
    finite_c50 = [x for x in c50_values if np.isfinite(x)]
    finite_d50 = [x for x in d50_values if np.isfinite(x)]
    finite_t30 = [x for x in t30_values if np.isfinite(x)]

    if finite_sti:
        print(f"Durchschnittlicher sti: {np.mean(finite_sti):.2f}")
        print(f"Standardabweichung sti: {np.std(finite_sti):.2f}")
    else:
        print("sti: keine gültigen Werte (alle NaN).")

    if finite_c50:
        print(f"Durchschnittlicher c50: {np.mean(finite_c50):.2f}")
        print(f"Standardabweichung c50: {np.std(finite_c50):.2f}")
    else:
        print("C50: keine gültigen Werte (alle NaN).")

    if finite_d50:
        print(f"Durchschnittlicher d50: {np.mean(finite_d50):.2f}")
        print(f"Standardabweichung d50: {np.std(finite_d50):.2f}")
    else:
        print("D50: keine gültigen Werte (alle NaN).")

    if finite_t30:
        print(f"Durchschnittlicher t30: {np.mean(finite_t30):.2f}")
        print(f"Standardabweichung t30: {np.std(finite_t30):.2f}")
    else:
        print("T30: keine gültigen Werte (alle NaN).")

    plt.figure(figsize=(10, 6))
    plt.boxplot([c50_values, d50_values, t30_values], labels=['C50', 'D50', 'T30'])
    plt.title('Verteilung der akustischen Parameter')
    plt.ylabel('Wert')
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()
