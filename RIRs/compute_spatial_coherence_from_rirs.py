from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from tkinter import FALSE
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import csd, welch


# =========================
# User-adjustable parameters
# =========================
COH_FMIN = 2
50.0           # lower limit for broadband coherence summary
COH_FMAX = 4000.0          # upper limit for broadband coherence summary
LATE_START_MS = 80.0       # late field starts after direct sound / early reflections
LATE_END_MS = 1000.0        # end of late-field window
NFFT_COH = 8192            # FFT size for spectral estimates on the late tail
NORMALIZE_RIRS = True      # peak-normalize each RIR before coherence calculation
ALIGN_TO_DIRECT_PEAK = False
PAIR_REGEX = re.compile(r"^(?P<base>.+)_(?P<side>left|right)_RIR$", re.IGNORECASE)


@dataclass
class ResultRow:
    pair: str
    room: str
    config: str
    repetition: str
    left_file: str
    right_file: str
    coh_mean_250_4000: float
    coh_median_250_4000: float
    coh_min_250_4000: float
    coh_max_250_4000: float
    coh_250_500: float
    coh_500_1000: float
    coh_1000_2000: float
    coh_2000_4000: float


def read_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    data, fs = sf.read(path, always_2d=True)
    x = data[:, 0].astype(np.float64) if data.shape[1] == 1 else data.mean(axis=1).astype(np.float64)
    return x, fs


def normalize(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x))
    return x if peak == 0 else x / peak


def align_to_peak(x: np.ndarray) -> np.ndarray:
    peak_idx = int(np.argmax(np.abs(x)))
    return x[peak_idx:]


def pad_to_same_length(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = max(len(x), len(y))
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return x, y


def band_average(freqs: np.ndarray, values: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(mask):
        return np.nan
    return float(np.mean(values[mask]))


def compute_late_coherence(h1: np.ndarray, h2: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    i0 = int(round((LATE_START_MS / 1000.0) * fs))
    i1 = int(round((LATE_END_MS / 1000.0) * fs))
    i1 = min(i1, len(h1), len(h2))
    if i1 <= i0 + 32:
        raise ValueError(
            "Late window too short. Check LATE_START_MS/LATE_END_MS or the available RIR length."
        )

    x = h1[i0:i1]
    y = h2[i0:i1]

    nperseg = min(2048, len(x))
    if nperseg < 256:
        nperseg = len(x)

    freqs, pxy = csd(x, y, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, nfft=NFFT_COH)
    _, pxx = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, nfft=NFFT_COH)
    _, pyy = welch(y, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, nfft=NFFT_COH)

    coh = (np.abs(pxy) ** 2) / np.maximum(pxx * pyy, 1e-20)
    coh = np.clip(coh, 0.0, 1.0)
    return freqs, coh


def parse_name(base_name: str) -> Tuple[str, str, str]:
    """Expected pattern like Mo_R114_3-2 -> room=R114, config=3, repetition=2."""
    m = re.search(r"R(\d+)_([0-9]+)-([0-9]+)$", base_name)
    if m:
        room = f"R{m.group(1)}"
        config = m.group(2)
        repetition = m.group(3)
        return room, config, repetition
    return "", "", ""


def find_pairs(base_dir: Path) -> List[Tuple[str, Path, Path]]:
    grouped: Dict[str, Dict[str, Path]] = {}
    for path in sorted(base_dir.glob("*.wav")):
        m = PAIR_REGEX.match(path.stem)
        if not m:
            continue
        base = m.group("base")
        side = m.group("side").lower()
        grouped.setdefault(base, {})[side] = path

    pairs: List[Tuple[str, Path, Path]] = []
    for base, files in sorted(grouped.items()):
        if "left" in files and "right" in files:
            pairs.append((base, files["left"], files["right"]))
    return pairs


def process_pair(base_name: str, left_path: Path, right_path: Path) -> ResultRow:
    h1, fs1 = read_audio_mono(left_path)
    h2, fs2 = read_audio_mono(right_path)
    if fs1 != fs2:
        raise ValueError(f"Sampling-rate mismatch: {left_path.name}={fs1}, {right_path.name}={fs2}")

    if ALIGN_TO_DIRECT_PEAK:
        h1 = align_to_peak(h1)
        h2 = align_to_peak(h2)

    h1, h2 = pad_to_same_length(h1, h2)

    if NORMALIZE_RIRS:
        h1 = normalize(h1)
        h2 = normalize(h2)

    freqs, coh = compute_late_coherence(h1, h2, fs1)
    mask = (freqs >= COH_FMIN) & (freqs <= COH_FMAX)
    coh_band = coh[mask]
    if len(coh_band) == 0:
        raise ValueError(f"No coherence bins in {COH_FMIN}..{COH_FMAX} Hz")

    room, config, repetition = parse_name(base_name)
    return ResultRow(
        pair=base_name,
        room=room,
        config=config,
        repetition=repetition,
        left_file=left_path.name,
        right_file=right_path.name,
        coh_mean_250_4000=float(np.mean(coh_band)),
        coh_median_250_4000=float(np.median(coh_band)),
        coh_min_250_4000=float(np.min(coh_band)),
        coh_max_250_4000=float(np.max(coh_band)),
        coh_250_500=band_average(freqs, coh, 250.0, 500.0),
        coh_500_1000=band_average(freqs, coh, 500.0, 1000.0),
        coh_1000_2000=band_average(freqs, coh, 1000.0, 2000.0),
        coh_2000_4000=band_average(freqs, coh, 2000.0, 4000.0),
    )


def write_csv(rows: List[ResultRow], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair",
            "room",
            "config",
            "repetition",
            "left_file",
            "right_file",
            "coh_mean_250_4000",
            "coh_median_250_4000",
            "coh_min_250_4000",
            "coh_max_250_4000",
            "coh_250_500",
            "coh_500_1000",
            "coh_1000_2000",
            "coh_2000_4000",
        ])
        for r in rows:
            writer.writerow([
                r.pair,
                r.room,
                r.config,
                r.repetition,
                r.left_file,
                r.right_file,
                f"{r.coh_mean_250_4000:.6f}",
                f"{r.coh_median_250_4000:.6f}",
                f"{r.coh_min_250_4000:.6f}",
                f"{r.coh_max_250_4000:.6f}",
                f"{r.coh_250_500:.6f}".replace(".", ",") if np.isfinite(r.coh_250_500) else "",
                f"{r.coh_500_1000:.6f}".replace(".", ",") if np.isfinite(r.coh_500_1000) else "",
                f"{r.coh_1000_2000:.6f}".replace(".", ",") if np.isfinite(r.coh_1000_2000) else "",
                f"{r.coh_2000_4000:.6f}".replace(".", ",") if np.isfinite(r.coh_2000_4000) else "",
            ])


def write_grouped_csv(rows: List[ResultRow], out_path: Path) -> None:
    groups: Dict[Tuple[str, str], List[ResultRow]] = {}
    for r in rows:
        groups.setdefault((r.room, r.config), []).append(r)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "room",
            "config",
            "n_repetitions",
            "coh_mean_250_4000_median",
            "coh_mean_250_4000_iqr_q1",
            "coh_mean_250_4000_iqr_q3",
        ])
        for (room, config), vals in sorted(groups.items(), key=lambda x: (x[0][0], int(x[0][1] or 0))):
            xs = np.array([v.coh_mean_250_4000 for v in vals], dtype=float)
            writer.writerow([
                room,
                config,
                len(vals),
                f"{np.median(xs):.6f}".replace(".", ","),
                f"{np.quantile(xs, 0.25):.6f}".replace(".", ","),
                f"{np.quantile(xs, 0.75):.6f}".replace(".", ","),
            ])


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pairs = find_pairs(base_dir)
    if not pairs:
        raise FileNotFoundError(
            "No left/right RIR pairs found next to the script. Expected names like 'Mo_R114_1-1_left_RIR.wav'."
        )

    results: List[ResultRow] = []
    errors: List[Tuple[str, str]] = []

    for base_name, left_path, right_path in pairs:
        try:
            result = process_pair(base_name, left_path, right_path)
            results.append(result)
            print(f"{base_name}: coherence(250-4000 Hz, late tail) = {result.coh_mean_250_4000:.3f}")
        except Exception as exc:
            errors.append((base_name, str(exc)))
            print(f"ERROR in {base_name}: {exc}")

    results.sort(key=lambda r: (r.room, int(r.config or 0), int(r.repetition or 0), r.pair))

    write_csv(results, base_dir / "coherence_results_per_pair.csv")
    write_grouped_csv(results, base_dir / "coherence_results_grouped.csv")

    if errors:
        with (base_dir / "coherence_errors.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pair", "error"])
            writer.writerows(errors)

    print("\nWritten:")
    print(f"  {base_dir / 'coherence_results_per_pair.csv'}")
    print(f"  {base_dir / 'coherence_results_grouped.csv'}")
    if errors:
        print(f"  {base_dir / 'coherence_errors.csv'}")


if __name__ == "__main__":
    main()
