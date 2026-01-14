import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

def calculate_impulse_response(original_sweep_path, recorded_sweep, sample_rate, plot_results=False):
    # 1. Sweep-Signale laden
    sample_rate_orig, original_sweep = wavfile.read(original_sweep_path)

    # 2. Abtastraten prüfen
    if sample_rate_orig != sample_rate:
        raise ValueError("Die Abtastraten der beiden Signale müssen übereinstimmen.")

    # 3. Aufnahme zuschneiden
    len_orig = len(original_sweep)
    len_rec = len(recorded_sweep)

    start_index = (len_rec - len_orig) // 2
    recorded_sweep_cropped = recorded_sweep[start_index:start_index + len_orig]

    # 4. Deconvolution: Berechnung der Impulsantwort
    impulse_response = signal.fftconvolve(recorded_sweep_cropped, original_sweep[::-1], mode='full')

    # 5. Normalisierung
    impulse_response /= np.max(np.abs(impulse_response))

    # 6. Berechnung der Einhüllenden
    analytic_signal = signal.hilbert(impulse_response)
    amplitude_envelope = np.abs(analytic_signal)
    log_envelope = 20 * np.log10(amplitude_envelope + 1e-20)

    # 7. Finde das Maximum der Einhüllenden
    t_max_index = np.argmax(np.abs(impulse_response))

    # 8. Impulsantwort zuschneiden, sodass das Maximum bei 0 Sekunden liegt und 4 Sekunden lang ist
    start_sample = max(0, t_max_index - int(0.2 * sample_rate))
    cropped_impulse_response = impulse_response[start_sample:start_sample + int(4 * sample_rate)]
    cropped_envelope = log_envelope[start_sample:start_sample + int(4 * sample_rate)]

    if plot_results:
        # 9. Zeitachse anpassen
        cropped_time_axis = np.linspace(-0.2, len(cropped_impulse_response) / sample_rate - 0.2, len(cropped_impulse_response))

        # 10. Plot der Impulsantwort und Einhüllenden
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Oberer Subplot: Impulsantwort
        ax1.plot(cropped_time_axis, cropped_impulse_response)
        ax1.set_title("Impulsantwort")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        # Unterer Subplot: Logarithmische Einhüllende
        ax2.plot(cropped_time_axis, cropped_envelope)
        ax2.set_title("Logarithmische Einhüllende (dB)")
        ax2.set_ylabel("Amplitude (dB)")
        ax2.set_xlabel("Zeit (s)")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    return cropped_impulse_response, sample_rate
