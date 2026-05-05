from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


FEATURE_COLS = [
    "Avg_F0_Hz",
    "Avg_ZCR_per_s",
    "Avg_Energy",
    "Voiced_Frame_Ratio",
    "Spectral_Centroid_Mean",
    "Spectral_Bandwidth_Mean",
    "Spectral_Rolloff_Mean",
    "Spectral_Flatness_Mean",
    "MFCC1_Mean",
    "MFCC2_Mean",
    "MFCC3_Mean",
    "MFCC4_Mean",
    "MFCC5_Mean",
]
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]


def normalize_text(value: object) -> str:
    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def map_emotion_token(token: object) -> str | None:
    s = normalize_text(token)
    mapping = {
        "angry": "Angry",
        "ofkeli": "Angry",
        "furious": "Angry",
        "happy": "Happy",
        "mutlu": "Happy",
        "neutral": "Neutral",
        "notr": "Neutral",
        "ntr": "Neutral",
        "sad": "Sad",
        "uzgun": "Sad",
        "mutsuz": "Sad",
        "surprised": "Surprised",
        "saskin": "Surprised",
        "shocked": "Surprised",
    }
    return mapping.get(s)


def normalize_emotion_label(value: object, file_name: object = "") -> str | None:
    direct = map_emotion_token(value)
    if direct is not None:
        return direct

    s = normalize_text(value)
    if ".wav" in s:
        parts = s.replace(".wav", "").split("_")
        if len(parts) >= 2:
            from_name = map_emotion_token(parts[-2])
            if from_name is not None:
                return from_name

    fn = normalize_text(file_name).replace(".wav", "")
    if fn:
        parts = [p for p in fn.split("_") if p]
        for p in reversed(parts[-5:]):
            from_name = map_emotion_token(p)
            if from_name is not None:
                return from_name
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train emotion classifier from extracted audio features.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/features.csv"),
        help="Path to extracted features CSV",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/master_metadata.xlsx"),
        help="Path to master metadata Excel (contains Feeling labels)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save trained model and metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split ratio",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    metadata = pd.read_excel(args.metadata)

    df = features.merge(metadata[["File_Name", "Feeling"]], on="File_Name", how="left")
    df["Emotion"] = df.apply(
        lambda row: normalize_emotion_label(row.get("Feeling", ""), row.get("File_Name", "")),
        axis=1,
    )
    df = df.dropna(subset=["Emotion"]).copy()
    df = df[df["Emotion"].isin(EMOTIONS)].copy()
    available_features = [c for c in FEATURE_COLS if c in df.columns and df[c].notna().any()]
    if len(available_features) < 4:
        raise ValueError("Not enough feature columns found for emotion training.")

    if len(df) < 50:
        raise ValueError("Not enough labeled rows to train emotion model.")

    x = df[available_features]
    y = df["Emotion"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    best_name = "gradient_boosting_audio_only"
    model: Pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=80,
                    learning_rate=0.02,
                    max_depth=2,
                    subsample=0.65,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, labels=EMOTIONS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=EMOTIONS)
    cm_df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "emotion_model.joblib"
    summary_path = args.output_dir / "emotion_model_summary.txt"
    cm_path = args.output_dir / "emotion_confusion_matrix.csv"

    model_features = available_features
    joblib.dump({"model": model, "feature_cols": model_features}, model_path)
    cm_df.to_csv(cm_path)
    summary_path.write_text(
        f"Emotion Model Accuracy (%): {acc * 100:.2f}\n"
        f"Best Model: {best_name}\n"
        f"Train Samples: {len(x_train)}\n"
        f"Test Samples: {len(x_test)}\n\n"
        f"Used Features: {', '.join(model_features)}\n\n"
        f"Labels: {', '.join(EMOTIONS)}\n\n"
        f"{report}\n"
    )

    print(f"Emotion Model Accuracy (%): {acc * 100:.2f}")
    print(f"Saved: {model_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()
