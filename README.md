# RIR Evaluation – Minimal Example for Room Acoustics Analysis
This repository provides a minimal, reproducible workflow for the evaluation of room impulse responses (RIRs) and the derivation of room acoustic parameters.
It serves as a compact reference implementation accompanying experimental research on classroom acoustics, sound field structure, and speech intelligibility.

## Overview
The repository demonstrates how to:
- Compute room impulse responses from exponential sine sweep measurements
- Derive standard room acoustic parameters:
-- Reverberation Time (T30)
-- Clarity (C50)
-- Definition (D50)
-- Speech Transmission Index (STI)
- Evaluate spatial coherence as a proxy for sound field diffuseness
- Process and analyze measurement data in a reproducible way
The implementation follows established standards such as ISO 3382-1 and IEC 60268-16, and reflects the methodology used in the associated publications.

## Repository Structure
RIR_Evaluation/

│

├── RIRs/                                   # Example room impulse responses

├── calculate_ir.py                         # Compute room impulse responses from sweep recordings

├── process_folder.py                       # Batch evaluation of RIR files / parameter extraction

├── Log_Sweep_50_Hz-20_kHz.wav              # Excitation sweep signal

├── Mo_R114_1-1.wav                         # Example measurement

├── Mo_R114_1-2.wav

├── Mo_R114_1-3.wav

├── Mo_R114_2-1.wav

├── Mo_R114_2-2.wav

├── Mo_R114_2-3.wav

├── 26-03-27 Zwischen Absorption und Streuung – Einfluss der Raumkanten auf die akustische.pdf # Conference Proceeding

├── LICENSE

└── README.md


## Purpose
The repository demonstrates how to:
- Compute room impulse responses using exponential sine sweep measurements
- Derive standard room acoustic parameters:
-- Reverberation Time (T30)
-- Clarity (C50)
-- Definition (D50)
-- Speech Transmission Index (STI)
- Perform batch processing of measurement datasets
- Provide a minimal reproducible pipeline for research and validation
The implementation reflects the exact workflow used in the associated studies.

## Getting Started
### Requirements
- Python ≥ 3.1
- numpy
- scipy
- matplotlib
- librosa (optional, depending on feature extraction)

### 1. Compute Room Impulse Responses
Use the provided sweep signal (Log_Sweep_50_Hz-20_kHz.wav) and recorded measurements:
  python calculate_ir.py

This step performs:
- Time alignment
- Frequency-domain deconvolution
- RIR normalization

### 2. Process a Dataset
To extract acoustic parameters for multiple files:
  python process_folder.py

This will:
- Load RIRs
- Compute acoustic parameters
- Aggregate results across measurements

## Methodology
### RIR Calculation
- Based on exponential sine sweep deconvolution
- Inverse filter derived from time-reversed excitation signal
- Frequency-domain processing

### Acoustic Parameters
- T30: Schroeder integration (−5 dB to −35 dB extrapolation)
- C50 / D50: Early-to-total energy ratio (50 ms boundary)
- STI: Modulation transfer function approach

### Measurement Setup (Example Data)
- Dodecahedral loudspeaker (speech source approximation)
- Multiple source–receiver configurations
- Repeated measurements for robustness
- Sampling rate: 48.0 kHz

## Scientific Context
This repository is directly linked to the following research:
- “Zwischen Absorption und Streuung – Einfluss der Raumkanten auf die akustische Qualität von Unterrichtsräumen” Proceedings of the DAGA 2026 in Dresden
- “Effect of Edge-Based AcousticModifications on Speech Intelligibility and SNR Thresholds in Classrooms: A Field Study” Planned publication in MDPI Acoustics

## Notes
- The repository is intentionally minimalistic
- Focus is on transparency and reproducibility, not completeness
- The code is designed for:
-- Research prototyping
-- Method validation
-- Educational purposes

## Citation
If you use this code, please cite “Effect of Edge-Based AcousticModifications on Speech Intelligibility and SNR Thresholds in Classrooms: A Field Study”
