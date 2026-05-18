from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile

from frequency_features import extract_frequency_plane_features, _empty_frequency_features


def resolve_audio_path(audio_path: Path, project_root: Path | None = None) -> Path | None:
    """Try common locations when metadata paths are relative."""
    if audio_path.exists():
        return audio_path

    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root / audio_path)
        candidates.append(project_root / audio_path.name)
        if "Dataset" in audio_path.parts:
            idx = audio_path.parts.index("Dataset")
            candidates.append(project_root / "data" / Path(*audio_path.parts[idx:]))
        if audio_path.parts[:1] == ("Midterm Project",):
            candidates.append(project_root / Path(*audio_path.parts[1:]))

    here = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            here / audio_path,
            here / "data" / "Dataset" / audio_path.name,
            here.parent / audio_path,
        ]
    )

    for cand in candidates:
        if cand.exists():
            return cand
    return None


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
    except (ValueError, OSError):
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


def _empty_time_features(file_name: str, sr: int = 0) -> dict[str, float | str | int]:
    out: dict[str, float | str | int] = {
        "File_Name": file_name,
        "Sample_Rate": sr,
        "Duration_s": 0.0,
        "Avg_F0_Hz": np.nan,
        "Avg_ZCR_per_s": np.nan,
        "Avg_Energy": np.nan,
        "Voiced_Frame_Ratio": 0.0,
    }
    out.update(_empty_frequency_features())
    return out


def extract_file_features(
    audio_path: Path, frame_ms: float = 25.0, hop_ratio: float = 0.5
) -> dict[str, float | str | int]:
    y, sr = load_wav_mono(audio_path)
    if len(y) == 0:
        return _empty_time_features(audio_path.name, sr=sr)

    frame_length = int(sr * frame_ms / 1000.0)
    hop_length = max(1, int(frame_length * hop_ratio))
    if frame_length <= 2 or len(y) < frame_length:
        out = _empty_time_features(audio_path.name, sr=sr)
        out["Duration_s"] = len(y) / sr
        return out

    frames = frame_signal(y, frame_length=frame_length, hop_length=hop_length)
    energy = short_term_energy(frames)
    zcr = zero_crossing_rate(frames, sr=sr)

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

    freq_feats = extract_frequency_plane_features(y, sr)

    out: dict[str, float | str | int] = {
        "File_Name": audio_path.name,
        "Sample_Rate": sr,
        "Duration_s": len(y) / sr,
        "Avg_F0_Hz": float(np.mean(f0_values)) if f0_values else np.nan,
        "Avg_ZCR_per_s": float(np.mean(voiced_zcr)) if len(voiced_zcr) else np.nan,
        "Avg_Energy": float(np.mean(voiced_energy)) if len(voiced_energy) else np.nan,
        "Voiced_Frame_Ratio": float(np.mean(voiced_mask)),
    }
    out.update(freq_feats)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract time-domain and frequency-plane features from WAV files."
    )
    parser.add_argument(
        "--master-metadata",
        type=Path,
        default=Path("data/master_metadata.xlsx"),
        help="Path to merged metadata file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/features_phase2.csv"),
        help="Output CSV for extracted features",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root for resolving relative audio paths",
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
    skipped = 0
    for _, row in meta.iterrows():
        audio_exists = bool(row.get("Audio_Exists", True))
        audio_str = str(row.get("Audio_Path", "") or "").strip()
        if not audio_exists or not audio_str:
            continue
        audio_path = resolve_audio_path(Path(audio_str), project_root=args.project_root)
        if audio_path is None:
            skipped += 1
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
    if skipped:
        print(f"Skipped (audio not found): {skipped}")


if __name__ == "__main__":
    main()
