from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
from classifier import predict_emotion, predict_gender_rule_based


ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data" / "master_metadata.xlsx"
FEATURES_PATH = ROOT / "outputs" / "features.csv"
SUMMARY_PATH = ROOT / "outputs" / "evaluation_summary.txt"
STATS_PATH = ROOT / "outputs" / "class_statistics.csv"
CM_PATH = ROOT / "outputs" / "confusion_matrix.csv"


@st.cache_data
def load_data():
    meta = pd.read_excel(META_PATH)
    feat = pd.read_csv(FEATURES_PATH)
    stats = pd.read_csv(STATS_PATH) if STATS_PATH.exists() else pd.DataFrame()
    cm = pd.read_csv(CM_PATH) if CM_PATH.exists() else pd.DataFrame()
    return meta, feat, stats, cm


def parse_accuracy() -> str:
    if not SUMMARY_PATH.exists():
        return "N/A"
    txt = SUMMARY_PATH.read_text()
    for line in txt.splitlines():
        if "Overall Accuracy" in line:
            return line.split(":")[-1].strip()
    return "N/A"


def main() -> None:
    st.set_page_config(page_title="Speech Gender Classifier", layout="wide")
    st.title("Speech Analysis - Rule-based Gender Classifier")

    meta, feat, stats, cm = load_data()
    accuracy = parse_accuracy()

    st.subheader("Dataset Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Accuracy (%)", accuracy)
    c2.metric("Total Matched Files", int(meta["Audio_Exists"].sum()))
    c3.metric("Feature Rows", len(feat))

    st.markdown("---")
    st.subheader("Single Audio Prediction")

    merged = feat.merge(meta[["File_Name", "Audio_Path", "Feeling"]], on="File_Name", how="left")
    options = merged["File_Name"].dropna().tolist()
    selected = st.selectbox("Select an audio file", options)

    row = merged[merged["File_Name"] == selected].iloc[0]
    f0 = float(row["Avg_F0_Hz"]) if pd.notna(row["Avg_F0_Hz"]) else float("nan")
    zcr = float(row["Avg_ZCR_per_s"]) if pd.notna(row["Avg_ZCR_per_s"]) else float("nan")
    energy = float(row["Avg_Energy"]) if pd.notna(row["Avg_Energy"]) else float("nan")
    voiced_ratio = (
        float(row["Voiced_Frame_Ratio"]) if pd.notna(row["Voiced_Frame_Ratio"]) else float("nan")
    )
    sc = float(row["Spectral_Centroid_Mean"]) if pd.notna(row.get("Spectral_Centroid_Mean")) else float("nan")
    sb = float(row["Spectral_Bandwidth_Mean"]) if pd.notna(row.get("Spectral_Bandwidth_Mean")) else float("nan")
    sr = float(row["Spectral_Rolloff_Mean"]) if pd.notna(row.get("Spectral_Rolloff_Mean")) else float("nan")
    sf = float(row["Spectral_Flatness_Mean"]) if pd.notna(row.get("Spectral_Flatness_Mean")) else float("nan")
    m1 = float(row["MFCC1_Mean"]) if pd.notna(row.get("MFCC1_Mean")) else float("nan")
    m2 = float(row["MFCC2_Mean"]) if pd.notna(row.get("MFCC2_Mean")) else float("nan")
    m3 = float(row["MFCC3_Mean"]) if pd.notna(row.get("MFCC3_Mean")) else float("nan")
    m4 = float(row["MFCC4_Mean"]) if pd.notna(row.get("MFCC4_Mean")) else float("nan")
    m5 = float(row["MFCC5_Mean"]) if pd.notna(row.get("MFCC5_Mean")) else float("nan")
    pred = predict_gender_rule_based(f0)
    pred_emotion = predict_emotion(
        f0,
        zcr,
        energy,
        voiced_ratio,
        sc,
        sb,
        sr,
        sf,
        m1,
        m2,
        m3,
        m4,
        m5,
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("Predicted Class", pred)
    c5.metric("Actual Class", str(row.get("Gender", "")))
    c6.metric("Average F0 (Hz)", f"{f0:.2f}" if pd.notna(f0) else "NaN")
    c7, c8 = st.columns(2)
    c7.metric("Predicted Emotion", pred_emotion)
    c8.metric("Actual Emotion", str(row.get("Feeling", "")))

    audio_path = str(row.get("Audio_Path", ""))
    if audio_path and Path(audio_path).exists():
        st.audio(audio_path)
    else:
        st.warning("Audio file not found for playback.")

    st.markdown("---")
    st.subheader("Class Statistics")
    if not stats.empty:
        st.dataframe(stats, use_container_width=True)

    st.subheader("Confusion Matrix")
    if not cm.empty:
        st.dataframe(cm, use_container_width=True)


if __name__ == "__main__":
    main()
