"""Batch processing for room impulse responses.

This module provides a function to iterate over a folder of recorded
logarithmic sine sweep measurements and compute the corresponding room
impulse responses (RIRs) for each stereo channel.  It relies on the
``calculate_ir`` function from :mod:`calculate_ir`, which performs
deconvolution with a properly scaled inverse filter and crops the
resulting impulse response such that the direct sound appears after a
configurable pre‑silence.

The function writes the processed RIRs to a subfolder named ``RIRs``
within the input directory.  The output filenames follow the pattern
``<original_filename>_left_RIR.wav`` and
``<original_filename>_right_RIR.wav``.

Example
-------
To process all WAV files in the current working directory using the
reference sweep ``Log_Sweep_50_Hz-20_kHz.wav``::

    from process_folder import process_folder

    folder = "/path/to/measurements"
    sweep_file = os.path.join(folder, "Log_Sweep_50_Hz-20_kHz.wav")
    process_folder(folder, sweep_file, pre_silence=0.2, ir_duration=4.0)

This will generate a ``RIRs`` subfolder containing the left and right
channel impulse responses for each measurement.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy.io import wavfile

from calculate_ir import calculate_ir


def process_folder(
    folder_path: str,
    original_sweep_path: str,
    *,
    pre_silence: float = 0.2,
    ir_duration: float = 4.0,
    f_start: Optional[float] = None,
    f_end: Optional[float] = None,
    plot_results: bool = False,
) -> None:
    """Process all measurement files in a folder to compute RIRs.

    The function scans ``folder_path`` for all ``.wav`` files except the
    reference sweep itself, assumes each file contains a stereo recording
    of a swept sine measurement, separates the two channels and applies
    :func:`calculate_ir` to each.

    Args:
        folder_path: Path to the directory containing recorded sweep files.
        original_sweep_path: Path to the reference sweep used for
            deconvolution.  This file is excluded from processing.
        pre_silence: Amount of silence (in seconds) to insert before the
            direct sound in the cropped RIR.  See :func:`calculate_ir`.
        ir_duration: Total length (in seconds) of the returned impulse
            response segment.  Must be long enough to capture the
            reverberation tail.
        f_start: Optional starting frequency of the sweep.  If ``None``,
            the function will estimate it from the sweep signal.
        f_end: Optional ending frequency of the sweep.  If ``None``,
            the function will estimate it from the sweep signal.
        plot_results: If ``True``, show the RIR and envelope plots for
            each channel during processing.  Useful for debugging.

    Returns:
        None.  The RIRs are written to ``folder_path/RIRs``.
    """
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path!r} ist kein gültiges Verzeichnis.")
    
    # Exclude the reference sweep from processing
    reference_name = os.path.basename(original_sweep_path)

    # List all WAV files in the directory
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav') and f != reference_name]
    if not files:
        print("Keine Aufnahmedateien zum Verarbeiten gefunden.")
        return
    
    # Create output directory
    rir_out_dir = os.path.join(folder_path, "RIRs")
    os.makedirs(rir_out_dir, exist_ok=True)

    # Process each file
    for fname in files:
        file_path = os.path.join(folder_path, fname)
        print(f"Verarbeite {fname}...")

        # Read the stereo recording
        sample_rate, data = wavfile.read(file_path)

        # Ensure float array; handle mono by expanding dims
        if data.ndim == 1:
            # Duplicate mono channel to stereo for consistency
            data = np.column_stack((data, data))

        # Separate channels
        left_channel = data[:, 0]
        right_channel = data[:, 1]

        # Determine the direct‑sound onsets for each channel.  The
        # ``calculate_ir`` function returns the raw onset index when
        # ``return_onset`` is True.  We use this to align both channels to
        # the earliest direct arrival.
        _, _, onset_left = calculate_ir(
            original_sweep_path,
            left_channel,
            sample_rate,
            f_start=f_start,
            f_end=f_end,
            pre_silence=pre_silence,
            ir_duration=ir_duration,
            plot_results=False,
            return_onset=True,
        )
        _, _, onset_right = calculate_ir(
            original_sweep_path,
            right_channel,
            sample_rate,
            f_start=f_start,
            f_end=f_end,
            pre_silence=pre_silence,
            ir_duration=ir_duration,
            plot_results=False,
            return_onset=True,
        )

        # Choose the earliest of the two onsets as a common reference.
        common_onset = min(onset_left, onset_right)
        # Compute the RIRs using the common onset to ensure identical
        # cropping across both channels.  plot_results is propagated.
        rir_left, _ = calculate_ir(
            original_sweep_path,
            left_channel,
            sample_rate,
            f_start=f_start,
            f_end=f_end,
            pre_silence=pre_silence,
            ir_duration=ir_duration,
            plot_results=plot_results,
            override_onset=common_onset,
        )
        rir_right, _ = calculate_ir(
            original_sweep_path,
            right_channel,
            sample_rate,
            f_start=f_start,
            f_end=f_end,
            pre_silence=pre_silence,
            ir_duration=ir_duration,
            plot_results=plot_results,
            override_onset=common_onset,
        )
        
        # Write results to separate files
        base, _ = os.path.splitext(fname)
        left_out_path = os.path.join(rir_out_dir, f"{base}_left_RIR.wav")
        right_out_path = os.path.join(rir_out_dir, f"{base}_right_RIR.wav")
        wavfile.write(left_out_path, sample_rate, rir_left.astype(np.float32))
        wavfile.write(right_out_path, sample_rate, rir_right.astype(np.float32))
    print(f"Verarbeitung abgeschlossen. RIRs wurden im Verzeichnis {rir_out_dir!r} gespeichert.")


if __name__ == "__main__":
    # If executed directly, process the current working directory using
    # a sweep named ``Log_Sweep_50_Hz-20_kHz.wav``.  This behaviour
    # mirrors the original example script.
    cwd = os.getcwd()
    default_sweep = os.path.join(cwd, "Log_Sweep_50_Hz-20_kHz.wav")
    if not os.path.isfile(default_sweep):
        raise FileNotFoundError(
            "Der referenzsweep 'Log_Sweep_50_Hz-20_kHz.wav' wurde im aktuellen Verzeichnis nicht gefunden."
        )
    process_folder(cwd, default_sweep)