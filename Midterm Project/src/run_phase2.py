"""Run Phase-2 pipeline: extract frequency features -> train emotion model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase-2 feature extraction and model training.")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--features-out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root
    src = root / "src"
    metadata = args.metadata or (root / "data" / "master_metadata.xlsx")
    features_out = args.features_out or (root / "outputs" / "features_phase2.csv")

    py = sys.executable
    subprocess.run(
        [
            py,
            str(src / "extract_features.py"),
            "--master-metadata",
            str(metadata),
            "--output",
            str(features_out),
            "--project-root",
            str(root),
        ],
        check=True,
        cwd=root,
    )
    # Keep legacy path for apps/scripts expecting outputs/features.csv
    legacy_features = root / "outputs" / "features.csv"
    if features_out != legacy_features and features_out.exists():
        legacy_features.write_bytes(features_out.read_bytes())

    subprocess.run(
        [
            py,
            str(src / "train_emotion_model.py"),
            "--features",
            str(features_out),
            "--metadata",
            str(metadata),
            "--output-dir",
            str(root / "outputs"),
            "--phase",
            "2",
        ],
        check=True,
        cwd=root,
    )
    print("Phase-2 pipeline completed.")


if __name__ == "__main__":
    main()
