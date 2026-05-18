"""Frequency-domain (spectral plane) features without librosa."""

from __future__ import annotations

import numpy as np


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mel_min, mel_max = _hz_to_mel(np.array([0.0, sr / 2.0]))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.searchsorted(fft_freqs, hz_points)
    bins = np.clip(bins, 0, len(fft_freqs) - 1)

    fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left or right <= center:
            continue
        for j in range(left, center):
            fb[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fb[i, j] = (right - j) / max(right - center, 1)
    return fb


def stft_power(y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    frames: list[np.ndarray] = []
    for start in range(0, len(y) - n_fft + 1, hop):
        frame = y[start : start + n_fft] * window
        mag = np.abs(np.fft.rfft(frame)) + 1e-12
        frames.append(mag**2)
    if not frames:
        mag = np.abs(np.fft.rfft(y[:n_fft] * window)) + 1e-12
        frames = [mag**2]
    power = np.stack(frames, axis=0)
    return power, freqs


def _stats(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0 or np.all(np.isnan(x)):
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x))


def _dct_matrix(n: int, k: int) -> np.ndarray:
    n = max(n, 1)
    mat = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        mat[i] = np.cos(np.pi * i * (np.arange(n) + 0.5) / n)
    return mat


def mfcc_features(power: np.ndarray, sr: int, n_mfcc: int = 13, n_mels: int = 40) -> np.ndarray:
    n_fft = (power.shape[1] - 1) * 2
    fb = _mel_filterbank(n_mels=n_mels, n_fft=n_fft, sr=sr)
    mel = power @ fb.T
    log_mel = np.log(mel + 1e-10)
    dct = _dct_matrix(n_mels, n_mfcc)
    return log_mel @ dct.T


def delta_features(feats: np.ndarray, width: int = 9) -> np.ndarray:
    if feats.shape[0] < 3:
        return np.zeros_like(feats)
    half = width // 2
    out = np.zeros_like(feats)
    for t in range(feats.shape[0]):
        s = max(0, t - half)
        e = min(feats.shape[0], t + half + 1)
        n = e - s
        if n < 2:
            continue
        denom = 2.0 * sum((i - (n - 1) / 2.0) ** 2 for i in range(n))
        if denom <= 0:
            continue
        num = sum((i - (n - 1) / 2.0) * feats[s + i] for i in range(n))
        out[t] = num / denom
    return out


def chroma_from_power(power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    chroma = np.zeros((power.shape[0], 12), dtype=np.float64)
    for f_idx, f in enumerate(freqs):
        if f <= 0:
            continue
        midi = 69.0 + 12.0 * np.log2(f / 440.0)
        pitch_class = int(round(midi)) % 12
        chroma[:, pitch_class] += power[:, f_idx]
    row_sum = np.sum(chroma, axis=1, keepdims=True) + 1e-12
    return chroma / row_sum


def spectral_contrast(power: np.ndarray, n_bands: int = 6) -> np.ndarray:
    n_freq = power.shape[1]
    edges = np.linspace(0, n_freq, n_bands + 1, dtype=int)
    contrasts: list[float] = []
    for b in range(n_bands):
        band = power[:, edges[b] : edges[b + 1]]
        if band.size == 0:
            contrasts.append(0.0)
            continue
        peak = np.percentile(band, 85, axis=1)
        valley = np.percentile(band, 15, axis=1)
        contrasts.append(float(np.mean(np.log((peak + 1e-12) / (valley + 1e-12)))))
    return np.array(contrasts, dtype=np.float64)


def extract_frequency_plane_features(y: np.ndarray, sr: int) -> dict[str, float]:
    if len(y) == 0 or sr <= 0:
        return _empty_frequency_features()

    y = y.astype(np.float64)
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    power, freqs = stft_power(y, sr)
    mags = np.sqrt(power)

    mfcc = mfcc_features(power, sr, n_mfcc=13)
    mfcc_delta = delta_features(mfcc)
    mfcc_delta2 = delta_features(mfcc_delta)
    chroma = chroma_from_power(power, freqs)

    mag_sum = np.sum(mags, axis=1) + 1e-12
    centroid = np.sum(mags * freqs[None, :], axis=1) / mag_sum
    bandwidth = np.sqrt(
        np.sum(mags * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / mag_sum
    )
    cumulative = np.cumsum(power, axis=1)
    rolloff_thr = 0.85 * cumulative[:, -1][:, None]
    rolloff_idx = np.argmax(cumulative >= rolloff_thr, axis=1)
    rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]
    flatness = np.exp(np.mean(np.log(mags), axis=1)) / np.maximum(np.mean(mags, axis=1), 1e-12)

    flux = np.mean(np.sqrt(np.sum(np.diff(mags, axis=0) ** 2, axis=1))) if mags.shape[0] > 1 else 0.0

    p = mags / np.sum(mags, axis=1, keepdims=True)
    skew = np.sum(p * ((freqs[None, :] - centroid[:, None]) / (bandwidth[:, None] + 1e-12)) ** 3, axis=1)
    kurt = np.sum(p * ((freqs[None, :] - centroid[:, None]) / (bandwidth[:, None] + 1e-12)) ** 4, axis=1)

    low = np.sum(power[:, freqs <= 500], axis=1)
    mid = np.sum(power[:, (freqs > 500) & (freqs <= 2000)], axis=1)
    high = np.sum(power[:, freqs > 2000], axis=1)
    total_e = low + mid + high + 1e-12

    contrast = spectral_contrast(power)

    out: dict[str, float] = {
        "Spectral_Flux_Mean": float(flux),
        "Spectral_Skewness_Mean": float(np.mean(skew)),
        "Spectral_Kurtosis_Mean": float(np.mean(kurt)),
        "Mel_LowBand_Ratio": float(np.mean(low / total_e)),
        "Mel_MidBand_Ratio": float(np.mean(mid / total_e)),
        "Mel_HighBand_Ratio": float(np.mean(high / total_e)),
        "Chroma_Energy_Total": float(np.mean(np.sum(chroma, axis=1))),
    }

    for i, c in enumerate(contrast, start=1):
        out[f"Spectral_Contrast_B{i}"] = float(c)
    out["Spectral_Contrast_Mean"] = float(np.mean(contrast))

    for idx in range(13):
        m, s = _stats(mfcc[:, idx])
        out[f"MFCC{idx + 1}_Mean"] = m
        out[f"MFCC{idx + 1}_Std"] = s
        dm, _ = _stats(mfcc_delta[:, idx])
        out[f"MFCC{idx + 1}_Delta_Mean"] = dm

    for idx in range(12):
        m, s = _stats(chroma[:, idx])
        out[f"Chroma{idx + 1}_Mean"] = m
        out[f"Chroma{idx + 1}_Std"] = s

    for name, arr in [
        ("Spectral_Centroid", centroid),
        ("Spectral_Bandwidth", bandwidth),
        ("Spectral_Rolloff", rolloff),
        ("Spectral_Flatness", flatness),
    ]:
        m, s = _stats(arr)
        out[f"{name}_Mean"] = m
        out[f"{name}_Std"] = s

    return out


def _empty_frequency_features() -> dict[str, float]:
    out: dict[str, float] = {
        "Spectral_Flux_Mean": np.nan,
        "Spectral_Skewness_Mean": np.nan,
        "Spectral_Kurtosis_Mean": np.nan,
        "Mel_LowBand_Ratio": np.nan,
        "Mel_MidBand_Ratio": np.nan,
        "Mel_HighBand_Ratio": np.nan,
        "Chroma_Energy_Total": np.nan,
        "Spectral_Contrast_Mean": np.nan,
    }
    for i in range(1, 7):
        out[f"Spectral_Contrast_B{i}"] = np.nan
    for idx in range(13):
        out[f"MFCC{idx + 1}_Mean"] = np.nan
        out[f"MFCC{idx + 1}_Std"] = np.nan
        out[f"MFCC{idx + 1}_Delta_Mean"] = np.nan
    for idx in range(12):
        out[f"Chroma{idx + 1}_Mean"] = np.nan
        out[f"Chroma{idx + 1}_Std"] = np.nan
    for name in ("Spectral_Centroid", "Spectral_Bandwidth", "Spectral_Rolloff", "Spectral_Flatness"):
        out[f"{name}_Mean"] = np.nan
        out[f"{name}_Std"] = np.nan
    return out
