from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from classifier import predict_gender_rule_based


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule-based gender classifier.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/features.csv"),
        help="Path to extracted features CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for evaluation outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features)
    if "Gender" not in df.columns or "Avg_F0_Hz" not in df.columns:
        raise ValueError("features CSV must include 'Gender' and 'Avg_F0_Hz' columns")

    df["Predicted"] = df["Avg_F0_Hz"].apply(predict_gender_rule_based)
    eval_df = df[df["Predicted"] != "Unknown"].copy()

    accuracy = (eval_df["Gender"] == eval_df["Predicted"]).mean() * 100.0

    labels = ["Male", "Female", "Child"]
    cm = pd.crosstab(
        eval_df["Gender"],
        eval_df["Predicted"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)

    stats = (
        eval_df.groupby("Gender", as_index=False)
        .agg(
            Number_of_Samples=("File_Name", "count"),
            Average_F0_Hz=("Avg_F0_Hz", "mean"),
            Std_F0_Hz=("Avg_F0_Hz", "std"),
        )
        .rename(columns={"Gender": "Class"})
    )
    class_success = (
        eval_df.assign(Correct=eval_df["Gender"] == eval_df["Predicted"])
        .groupby("Gender", as_index=False)["Correct"]
        .mean()
    )
    class_success["Success_%"] = class_success["Correct"] * 100.0
    class_success = class_success.drop(columns=["Correct"]).rename(columns={"Gender": "Class"})
    stats = stats.merge(class_success, on="Class", how="left")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_out = args.output_dir / "predictions.csv"
    cm_out = args.output_dir / "confusion_matrix.csv"
    stats_out = args.output_dir / "class_statistics.csv"
    summary_out = args.output_dir / "evaluation_summary.txt"

    eval_df.to_csv(pred_out, index=False)
    cm.to_csv(cm_out)
    stats.to_csv(stats_out, index=False)
    summary_out.write_text(
        f"Overall Accuracy (%): {accuracy:.2f}\n"
        f"Total Evaluated Samples: {len(eval_df)}\n"
        f"Unknown Predictions: {int((df['Predicted'] == 'Unknown').sum())}\n"
    )

    print(f"Overall Accuracy (%): {accuracy:.2f}")
    print(f"Saved: {pred_out}")
    print(f"Saved: {cm_out}")
    print(f"Saved: {stats_out}")
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
