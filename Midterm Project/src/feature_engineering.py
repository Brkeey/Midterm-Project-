"""Derived features for Phase-2 emotion model (literature-inspired combinations)."""

from __future__ import annotations

import numpy as np
import pandas as pd


TIME_DOMAIN_COLS = [
    "Avg_F0_Hz",
    "Avg_ZCR_per_s",
    "Avg_Energy",
    "Voiced_Frame_Ratio",
]

FREQUENCY_PLANE_PREFIXES = (
    "MFCC",
    "Chroma",
    "Spectral_",
    "Mel_",
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Avg_F0_Hz" in out.columns and "Avg_Energy" in out.columns:
        out["Log_F0"] = np.log1p(out["Avg_F0_Hz"].clip(lower=1.0))
        out["Log_Energy"] = np.log1p(out["Avg_Energy"].clip(lower=0.0))
        out["F0_Energy_Product"] = out["Avg_F0_Hz"] * out["Avg_Energy"]
        out["F0_per_Centroid"] = out["Avg_F0_Hz"] / out["Spectral_Centroid_Mean"].replace(0, np.nan)

    if "Spectral_Centroid_Mean" in out.columns and "Spectral_Bandwidth_Mean" in out.columns:
        out["Centroid_Bandwidth_Ratio"] = (
            out["Spectral_Centroid_Mean"] / out["Spectral_Bandwidth_Mean"].replace(0, np.nan)
        )

    if "Spectral_Rolloff_Mean" in out.columns and "Spectral_Centroid_Mean" in out.columns:
        out["Rolloff_minus_Centroid"] = out["Spectral_Rolloff_Mean"] - out["Spectral_Centroid_Mean"]

    if "Spectral_Flatness_Mean" in out.columns and "Avg_Energy" in out.columns:
        out["Noisiness_Index"] = out["Spectral_Flatness_Mean"] * out["Avg_Energy"]

    if "Avg_ZCR_per_s" in out.columns and "Avg_Energy" in out.columns:
        out["Activity_Index"] = out["Avg_ZCR_per_s"] * out["Avg_Energy"]

    if "Mel_HighBand_Ratio" in out.columns and "Mel_LowBand_Ratio" in out.columns:
        out["High_Low_Mel_Ratio"] = out["Mel_HighBand_Ratio"] / out["Mel_LowBand_Ratio"].replace(0, np.nan)

    mfcc_means = [c for c in out.columns if c.startswith("MFCC") and c.endswith("_Mean") and "Delta" not in c]
    if len(mfcc_means) >= 3:
        out["MFCC_Mean_Norm"] = np.linalg.norm(out[mfcc_means].fillna(0.0), axis=1)

    chroma_means = [c for c in out.columns if c.startswith("Chroma") and c.endswith("_Mean")]
    if len(chroma_means) >= 2:
        chroma_mat = out[chroma_means].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            out["Chroma_Variance"] = np.nanvar(chroma_mat, axis=1, ddof=1)

    return out


def select_model_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "File_Name",
        "Gender",
        "Age",
        "Feeling",
        "Emotion",
        "Sample_Rate",
        "Duration_s",
    }
    numeric = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]
    return numeric
