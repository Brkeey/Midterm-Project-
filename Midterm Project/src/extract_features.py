from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
import librosa


def frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(y) < frame_length:
        return np.empty((0, frame_length), dtype=y.dtype)
    n_frames = 1 + (len(y) - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (y.strides[0] * hop_length, y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides).copy()


def load_wav_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    try:
        sr, data = wavfile.read(audio_path)
    except ValueError:
        # Some files may have a .wav extension but contain non-RIFF formats.
        # Skip them gracefully.
        return np.array([], dtype=np.float32), 0
    y = data.astype(np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    if np.issubdtype(data.dtype, np.integer):
        max_abs = float(np.iinfo(data.dtype).max)
        y = y / max(max_abs, 1.0)
    return y, sr


def short_term_energy(frames: np.ndarray) -> np.ndarray:
    return np.sum(frames**2, axis=1) / np.maximum(frames.shape[1], 1)


def zero_crossing_rate(frames: np.ndarray, sr: int) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zero_crossings = np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / 2.0
    seconds_per_frame = frames.shape[1] / sr
    return zero_crossings / np.maximum(seconds_per_frame, 1e-9)


def autocorr_pitch(frame: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 400.0) -> float | None:
    frame = frame - np.mean(frame)
    if np.allclose(frame, 0):
        return None

    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2 :]

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    max_lag = min(max_lag, len(corr) - 1)
    if max_lag <= min_lag:
        return None

    region = corr[min_lag : max_lag + 1]
    best_lag = int(np.argmax(region)) + min_lag
    if best_lag <= 0:
        return None
    return float(sr / best_lag)


def spectral_features_fallback(y: np.ndarray, sr: int) -> dict[str, float]:
    n_fft = 2048
    hop = 512
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    frames = []
    for start in range(0, len(y) - n_fft + 1, hop):
        frames.append(y[start : start + n_fft] * window)
    if not frames:
        frames = [y[:n_fft] * window]

    mags = np.array([np.abs(np.fft.rfft(fr)) + 1e-12 for fr in frames])
    power = mags**2
    mag_sum = np.sum(mags, axis=1)

    centroid = np.sum(mags * freqs[None, :], axis=1) / np.maximum(mag_sum, 1e-12)
    bandwidth = np.sqrt(
        np.sum(mags * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / np.maximum(mag_sum, 1e-12)
    )
    cumulative = np.cumsum(power, axis=1)
    threshold = 0.85 * cumulative[:, -1][:, None]
    rolloff_idx = np.argmax(cumulative >= threshold, axis=1)
    rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]
    flatness = np.exp(np.mean(np.log(mags), axis=1)) / np.maximum(np.mean(mags, axis=1), 1e-12)

    return {
        "Spectral_Centroid_Mean": float(np.mean(centroid)),
        "Spectral_Bandwidth_Mean": float(np.mean(bandwidth)),
        "Spectral_Rolloff_Mean": float(np.mean(rolloff)),
        "Spectral_Flatness_Mean": float(np.mean(flatness)),
    }


def extract_file_features(
    audio_path: Path, frame_ms: float = 25.0, hop_ratio: float = 0.5
) -> dict[str, float | str | int]:
    y, sr = load_wav_mono(audio_path)
    if len(y) == 0:
        return {
            "File_Name": audio_path.name,
            "Sample_Rate": sr,
            "Duration_s": 0.0,
            "Avg_F0_Hz": np.nan,
            "Avg_ZCR_per_s": np.nan,
            "Avg_Energy": np.nan,
            "Voiced_Frame_Ratio": 0.0,
            "Spectral_Centroid_Mean": np.nan,
            "Spectral_Bandwidth_Mean": np.nan,
            "Spectral_Rolloff_Mean": np.nan,
            "Spectral_Flatness_Mean": np.nan,
            "MFCC1_Mean": np.nan,
            "MFCC2_Mean": np.nan,
            "MFCC3_Mean": np.nan,
            "MFCC4_Mean": np.nan,
            "MFCC5_Mean": np.nan,
        }

    frame_length = int(sr * frame_ms / 1000.0)
    hop_length = max(1, int(frame_length * hop_ratio))
    if frame_length <= 2 or len(y) < frame_length:
        return {
            "File_Name": audio_path.name,
            "Sample_Rate": sr,
            "Duration_s": len(y) / sr,
            "Avg_F0_Hz": np.nan,
            "Avg_ZCR_per_s": np.nan,
            "Avg_Energy": np.nan,
            "Voiced_Frame_Ratio": 0.0,
            "Spectral_Centroid_Mean": np.nan,
            "Spectral_Bandwidth_Mean": np.nan,
            "Spectral_Rolloff_Mean": np.nan,
            "Spectral_Flatness_Mean": np.nan,
            "MFCC1_Mean": np.nan,
            "MFCC2_Mean": np.nan,
            "MFCC3_Mean": np.nan,
            "MFCC4_Mean": np.nan,
            "MFCC5_Mean": np.nan,
        }

    frames = frame_signal(y, frame_length=frame_length, hop_length=hop_length)
    energy = short_term_energy(frames)
    zcr = zero_crossing_rate(frames, sr=sr)

    # Voiced heuristic: above low energy floor and moderate ZCR.
    e_thr = np.percentile(energy, 35)
    z_low, z_high = np.percentile(zcr, 10), np.percentile(zcr, 90)
    voiced_mask = (energy >= e_thr) & (zcr >= z_low) & (zcr <= z_high)

    f0_values: list[float] = []
    for idx, is_voiced in enumerate(voiced_mask):
        if not is_voiced:
            continue
        f0 = autocorr_pitch(frames[idx], sr=sr)
        if f0 is not None:
            f0_values.append(f0)

    voiced_energy = energy[voiced_mask] if np.any(voiced_mask) else energy
    voiced_zcr = zcr[voiced_mask] if np.any(voiced_mask) else zcr

    sc = sb = sroll = sf = np.nan
    m1 = m2 = m3 = m4 = m5 = np.nan
    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        sc = float(np.mean(centroid)) if centroid.size else np.nan
        sb = float(np.mean(bandwidth)) if bandwidth.size else np.nan
        sroll = float(np.mean(rolloff)) if rolloff.size else np.nan
        sf = float(np.mean(flatness)) if flatness.size else np.nan
        m1 = float(np.mean(mfcc[0])) if mfcc.shape[0] > 0 else np.nan
        m2 = float(np.mean(mfcc[1])) if mfcc.shape[0] > 1 else np.nan
        m3 = float(np.mean(mfcc[2])) if mfcc.shape[0] > 2 else np.nan
        m4 = float(np.mean(mfcc[3])) if mfcc.shape[0] > 3 else np.nan
        m5 = float(np.mean(mfcc[4])) if mfcc.shape[0] > 4 else np.nan
    except Exception:
        fallback = spectral_features_fallback(y=y, sr=sr)
        sc = fallback["Spectral_Centroid_Mean"]
        sb = fallback["Spectral_Bandwidth_Mean"]
        sroll = fallback["Spectral_Rolloff_Mean"]
        sf = fallback["Spectral_Flatness_Mean"]

    return {
        "File_Name": audio_path.name,
        "Sample_Rate": sr,
        "Duration_s": len(y) / sr,
        "Avg_F0_Hz": float(np.mean(f0_values)) if f0_values else np.nan,
        "Avg_ZCR_per_s": float(np.mean(voiced_zcr)) if len(voiced_zcr) else np.nan,
        "Avg_Energy": float(np.mean(voiced_energy)) if len(voiced_energy) else np.nan,
        "Voiced_Frame_Ratio": float(np.mean(voiced_mask)),
        "Spectral_Centroid_Mean": sc,
        "Spectral_Bandwidth_Mean": sb,
        "Spectral_Rolloff_Mean": sroll,
        "Spectral_Flatness_Mean": sf,
        "MFCC1_Mean": m1,
        "MFCC2_Mean": m2,
        "MFCC3_Mean": m3,
        "MFCC4_Mean": m4,
        "MFCC5_Mean": m5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract F0/ZCR/Energy features from WAV files.")
    parser.add_argument(
        "--master-metadata",
        type=Path,
        default=Path("data/master_metadata.xlsx"),
        help="Path to merged metadata file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/features.csv"),
        help="Output CSV for extracted features",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=25.0,
        help="Frame size in milliseconds (20-30 recommended)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = pd.read_excel(args.master_metadata)
    if "Audio_Path" not in meta.columns:
        raise ValueError("master metadata must include 'Audio_Path' column")

    rows: list[dict[str, float | str | int]] = []
    for _, row in meta.iterrows():
        audio_exists = bool(row.get("Audio_Exists", True))
        audio_str = str(row.get("Audio_Path", "") or "").strip()
        if not audio_exists or not audio_str:
            continue
        audio_path = Path(audio_str)
        if not audio_path.exists():
            continue
        feat = extract_file_features(audio_path, frame_ms=args.frame_ms)
        feat["Gender"] = row.get("Gender", "")
        feat["Age"] = row.get("Age", np.nan)
        rows.append(feat)

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Feature file created: {args.output}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
