from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from feature_engineering import add_engineered_features, select_model_feature_columns


EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]

# Phase-1 baseline columns (time + basic spectral means).
PHASE1_FEATURE_COLS = [
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


def build_candidates(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "gradient_boosting_phase2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=random_state,
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                    ),
                ),
            ]
        ),
        "random_forest_phase2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting_audio_only": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=random_state,
                        n_estimators=80,
                        learning_rate=0.02,
                        max_depth=2,
                        subsample=0.65,
                    ),
                ),
            ]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train emotion classifier from extracted audio features.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/features_phase2.csv"),
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
    parser.add_argument(
        "--phase",
        type=int,
        default=2,
        choices=[1, 2],
        help="1 = Phase-1 baseline features only, 2 = frequency-plane + engineered features",
    )
    return parser.parse_args()


def prepare_labeled_frame(features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(metadata[["File_Name", "Feeling"]], on="File_Name", how="left")
    df["Emotion"] = df.apply(
        lambda row: normalize_emotion_label(row.get("Feeling", ""), row.get("File_Name", "")),
        axis=1,
    )
    df = df.dropna(subset=["Emotion"]).copy()
    return df[df["Emotion"].isin(EMOTIONS)].copy()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    metadata = pd.read_excel(args.metadata)
    df = prepare_labeled_frame(features, metadata)

    if args.phase == 2:
        df = add_engineered_features(df)
        feature_cols = select_model_feature_columns(df)
    else:
        feature_cols = [c for c in PHASE1_FEATURE_COLS if c in df.columns and df[c].notna().any()]

    if len(feature_cols) < 4:
        raise ValueError("Not enough feature columns found for emotion training.")
    if len(df) < 50:
        raise ValueError("Not enough labeled rows to train emotion model.")

    x = df[feature_cols]
    y = df["Emotion"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    candidates = build_candidates()
    if args.phase == 1:
        candidates = {"gradient_boosting_audio_only": candidates["gradient_boosting_audio_only"]}

    best_name = ""
    best_model: Pipeline | None = None
    best_cv = -1.0

    for name, model in candidates.items():
        scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        mean_cv = float(np.mean(scores))
        if mean_cv > best_cv:
            best_cv = mean_cv
            best_name = name
            best_model = model

    assert best_model is not None
    best_model.fit(x_train, y_train)

    y_pred = best_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, labels=EMOTIONS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=EMOTIONS)
    cm_df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "emotion_model.joblib"
    summary_path = args.output_dir / "emotion_model_summary.txt"
    cm_path = args.output_dir / "emotion_confusion_matrix.csv"

    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "phase": args.phase,
            "model_name": best_name,
        },
        model_path,
    )
    cm_df.to_csv(cm_path)
    summary_path.write_text(
        f"Phase: {args.phase}\n"
        f"Emotion Model Accuracy (%): {acc * 100:.2f}\n"
        f"CV Accuracy (%): {best_cv * 100:.2f}\n"
        f"Best Model: {best_name}\n"
        f"Train Samples: {len(x_train)}\n"
        f"Test Samples: {len(x_test)}\n"
        f"Feature Count: {len(feature_cols)}\n\n"
        f"Used Features: {', '.join(feature_cols)}\n\n"
        f"Labels: {', '.join(EMOTIONS)}\n\n"
        f"{report}\n"
    )

    print(f"Phase: {args.phase}")
    print(f"Emotion Model Accuracy (%): {acc * 100:.2f}")
    print(f"CV Accuracy (%): {best_cv * 100:.2f}")
    print(f"Best Model: {best_name}")
    print(f"Saved: {model_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()
