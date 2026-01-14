import os
import numpy as np
from calculate_ir import calculate_impulse_response
from scipy.io import wavfile

def process_folder(folder_path, original_sweep_path, plot_results=False):
    # Liste aller WAV-Dateien im Ordner
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav') and f != os.path.basename(original_sweep_path)]

    for file in files:
        recorded_sweep_path = os.path.join(folder_path, file)
        sample_rate, recorded_sweep = wavfile.read(recorded_sweep_path)

        # Trennung der Stereokanäle
        recorded_sweep_left  = recorded_sweep[:, 0]
        recorded_sweep_right = recorded_sweep[:, 1]

        rir_out_dir = os.path.join(folder_path, "RIRs")
        os.makedirs(rir_out_dir, exist_ok=True)

        output_ir_path_left = os.path.join(rir_out_dir, f"{os.path.splitext(file)[0]}_left_RIR.wav")
        output_ir_path_right = os.path.join(rir_out_dir, f"{os.path.splitext(file)[0]}_right_RIR.wav")

        print(f"Verarbeite {file}...")

        # Berechnung der RIR für den linken Kanal
        cropped_impulse_response_left, _ = calculate_impulse_response(original_sweep_path, recorded_sweep_left, sample_rate, plot_results)
        wavfile.write(output_ir_path_left, sample_rate, cropped_impulse_response_left.astype(np.float32))

        # Berechnung der RIR für den rechten Kanal
        cropped_impulse_response_right, _ = calculate_impulse_response(original_sweep_path, recorded_sweep_right, sample_rate, plot_results)
        wavfile.write(output_ir_path_right, sample_rate, cropped_impulse_response_right.astype(np.float32))

# Pfad zum Ordner mit den Aufnahmen
folder_path = os.getcwd()
original_sweep_path = os.path.join(folder_path, "Log_Sweep_50_Hz-20_kHz.wav")

# Aufruf der Funktion, plot_results=False ist Standard
process_folder(folder_path, original_sweep_path, plot_results=False)
