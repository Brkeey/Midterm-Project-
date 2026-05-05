from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

from classifier import predict_emotion, predict_gender_rule_based


ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data" / "master_metadata.xlsx"
FEATURES_PATH = ROOT / "outputs" / "features.csv"
SUMMARY_PATH = ROOT / "outputs" / "evaluation_summary.txt"
STATS_PATH = ROOT / "outputs" / "class_statistics.csv"


def parse_overall_accuracy() -> str:
    if not SUMMARY_PATH.exists():
        return "N/A"
    txt = SUMMARY_PATH.read_text()
    for line in txt.splitlines():
        if "Overall Accuracy" in line:
            return line.split(":")[-1].strip()
    return "N/A"


class TkClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Speech Analysis - Tkinter UI")
        self.root.geometry("980x640")

        self.meta = pd.read_excel(META_PATH)
        self.features = pd.read_csv(FEATURES_PATH)
        self.stats = pd.read_csv(STATS_PATH) if STATS_PATH.exists() else pd.DataFrame()

        self.data = self.features.merge(
            self.meta[["File_Name", "Audio_Path", "Feeling"]],
            on="File_Name",
            how="left",
        )

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        accuracy = parse_overall_accuracy()
        ttk.Label(top, text=f"Overall Accuracy (%): {accuracy}", font=("Arial", 12, "bold")).pack(
            side=tk.LEFT, padx=(0, 20)
        )
        ttk.Label(
            top,
            text=f"Total Matched Files: {int(self.meta['Audio_Exists'].sum())}",
            font=("Arial", 11),
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(top, text=f"Feature Rows: {len(self.features)}", font=("Arial", 11)).pack(
            side=tk.LEFT
        )

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Left panel: selection
        left = ttk.LabelFrame(main, text="Single Audio Prediction", padding=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        ttk.Label(left, text="Select audio file:").pack(anchor="w")
        self.file_var = tk.StringVar()
        file_names = sorted(self.data["File_Name"].dropna().tolist())
        self.combo = ttk.Combobox(left, textvariable=self.file_var, values=file_names, width=70)
        self.combo.pack(fill=tk.X, pady=(6, 8))
        if file_names:
            self.combo.set(file_names[0])

        ttk.Button(left, text="Predict", command=self.predict_selected).pack(anchor="w", pady=(2, 10))

        self.result_text = tk.Text(left, height=14, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(
            tk.END,
            "Prediction details will appear here.\n\n"
            "Note: This UI predicts class from Avg_F0_Hz using rule-based thresholds.",
        )
        self.result_text.config(state=tk.DISABLED)

        # Right panel: class stats
        right = ttk.LabelFrame(main, text="Class Statistics", padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cols = ("Class", "Samples", "Avg_F0_Hz", "Std_F0_Hz", "Success_%")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        for _, r in self.stats.iterrows():
            self.tree.insert(
                "",
                tk.END,
                values=(
                    r.get("Class", ""),
                    int(r.get("Number_of_Samples", 0)),
                    f"{float(r.get('Average_F0_Hz', float('nan'))):.2f}",
                    f"{float(r.get('Std_F0_Hz', float('nan'))):.2f}",
                    f"{float(r.get('Success_%', float('nan'))):.2f}",
                ),
            )

    def _write_result(self, text: str) -> None:
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

    def predict_selected(self) -> None:
        file_name = self.file_var.get().strip()
        if not file_name:
            messagebox.showwarning("Warning", "Please select a file.")
            return

        row_df = self.data[self.data["File_Name"] == file_name]
        if row_df.empty:
            messagebox.showerror("Error", "Selected file not found in feature table.")
            return

        row = row_df.iloc[0]
        f0 = row.get("Avg_F0_Hz", float("nan"))
        pred = predict_gender_rule_based(float(f0)) if pd.notna(f0) else "Unknown"
        zcr = row.get("Avg_ZCR_per_s", float("nan"))
        energy = row.get("Avg_Energy", float("nan"))
        voiced_ratio = row.get("Voiced_Frame_Ratio", float("nan"))
        sc = row.get("Spectral_Centroid_Mean", float("nan"))
        sb = row.get("Spectral_Bandwidth_Mean", float("nan"))
        sroll = row.get("Spectral_Rolloff_Mean", float("nan"))
        sf = row.get("Spectral_Flatness_Mean", float("nan"))
        m1 = row.get("MFCC1_Mean", float("nan"))
        m2 = row.get("MFCC2_Mean", float("nan"))
        m3 = row.get("MFCC3_Mean", float("nan"))
        m4 = row.get("MFCC4_Mean", float("nan"))
        m5 = row.get("MFCC5_Mean", float("nan"))
        pred_emotion = (
            predict_emotion(
                float(f0),
                float(zcr),
                float(energy),
                float(voiced_ratio),
                float(sc),
                float(sb),
                float(sroll),
                float(sf),
                float(m1),
                float(m2),
                float(m3),
                float(m4),
                float(m5),
            )
            if pd.notna(f0) and pd.notna(zcr) and pd.notna(energy) and pd.notna(voiced_ratio)
            else "Unknown"
        )

        details = (
            f"File: {file_name}\n"
            f"Audio Path: {row.get('Audio_Path', '')}\n"
            f"Actual Class: {row.get('Gender', '')}\n"
            f"Predicted Class: {pred}\n"
            f"Actual Emotion: {row.get('Feeling', '')}\n"
            f"Predicted Emotion: {pred_emotion}\n"
            f"Avg_F0_Hz: {row.get('Avg_F0_Hz', float('nan'))}\n"
            f"Avg_ZCR_per_s: {row.get('Avg_ZCR_per_s', float('nan'))}\n"
            f"Avg_Energy: {row.get('Avg_Energy', float('nan'))}\n"
            f"Voiced_Frame_Ratio: {row.get('Voiced_Frame_Ratio', float('nan'))}\n"
        )
        self._write_result(details)


def main() -> None:
    root = tk.Tk()
    TkClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
