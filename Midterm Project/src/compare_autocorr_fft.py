from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    y = data.astype(np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    if np.issubdtype(data.dtype, np.integer):
        y = y / max(float(np.iinfo(data.dtype).max), 1.0)
    return y, sr


def best_frame(y: np.ndarray, sr: int, frame_ms: float = 25.0) -> np.ndarray:
    frame_len = int(sr * frame_ms / 1000.0)
    if len(y) <= frame_len:
        return y
    hop = max(1, frame_len // 2)
    n = 1 + (len(y) - frame_len) // hop
    energies = []
    starts = []
    for i in range(n):
        s = i * hop
        fr = y[s : s + frame_len]
        starts.append(s)
        energies.append(float(np.mean(fr**2)))
    best_idx = int(np.argmax(energies))
    s = starts[best_idx]
    return y[s : s + frame_len]


def autocorr_f0(frame: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 400.0) -> tuple[float, np.ndarray, np.ndarray]:
    x = frame - np.mean(frame)
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    lags = np.arange(len(corr))

    min_lag = int(sr / fmax)
    max_lag = min(int(sr / fmin), len(corr) - 1)
    region = corr[min_lag : max_lag + 1]
    lag = int(np.argmax(region)) + min_lag
    f0 = float(sr / lag) if lag > 0 else float("nan")
    return f0, lags, corr


def fft_f0(frame: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 400.0) -> tuple[float, np.ndarray, np.ndarray]:
    x = frame - np.mean(frame)
    w = np.hanning(len(x))
    spec = np.fft.rfft(x * w)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return float("nan"), freqs, mag
    peak_idx = int(np.argmax(mag[band]))
    f0 = float(freqs[band][peak_idx])
    return f0, freqs, mag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Autocorrelation vs FFT comparison graph.")
    p.add_argument("--audio-path", type=Path, default=None, help="Audio file path. If omitted, first row from master metadata is used.")
    p.add_argument("--master-metadata", type=Path, default=Path("data/master_metadata.xlsx"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.audio_path is None:
        meta = pd.read_excel(args.master_metadata)
        args.audio_path = Path(str(meta.iloc[0]["Audio_Path"]))

    y, sr = load_wav_mono(args.audio_path)
    frame = best_frame(y, sr, frame_ms=25.0)

    f0_ac, lags, corr = autocorr_f0(frame, sr)
    f0_fft, freqs, mag = fft_f0(frame, sr)

    lag_ms = (lags / sr) * 1000.0

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(lag_ms, corr, color="tab:blue")
    ax[0].set_title(f"Autocorrelation (F0~{f0_ac:.1f} Hz)")
    ax[0].set_xlabel("Lag (ms)")
    ax[0].set_ylabel("Correlation")
    ax[0].grid(alpha=0.3)

    ax[1].plot(freqs, mag, color="tab:orange")
    ax[1].set_xlim(0, 500)
    ax[1].set_title(f"FFT Magnitude (F0~{f0_fft:.1f} Hz)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude")
    ax[1].grid(alpha=0.3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.audio_path.stem
    fig_path = args.output_dir / f"autocorr_vs_fft_{stem}.png"
    txt_path = args.output_dir / f"autocorr_vs_fft_{stem}.txt"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)

    txt_path.write_text(
        f"Audio: {args.audio_path}\n"
        f"Sample rate: {sr}\n"
        f"Autocorrelation F0 (Hz): {f0_ac:.2f}\n"
        f"FFT peak F0 (Hz): {f0_fft:.2f}\n"
        f"Difference (Hz): {abs(f0_ac - f0_fft):.2f}\n"
    )
    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {txt_path}")


if __name__ == "__main__":
    main()
