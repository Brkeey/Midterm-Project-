# Final Project - Phase 2 Technical Report (Group 17)

## 1) Introduction
Phase 2 extends the Phase-1 baseline by adding **frequency-plane (frekans düzlemi)** descriptors on top of time-domain features (F0, ZCR, energy, voiced ratio). A new emotion classifier is trained with literature-inspired engineered features and model selection.

Target classes: `Neutral`, `Happy`, `Angry`, `Sad`, `Surprised`.

## 2) New Frequency-Plane Features
Implemented in `src/frequency_features.py` (NumPy/SciPy only, no librosa dependency):

| Category | Features |
|---|---|
| MFCC | 13 coefficients: mean, std, delta-mean |
| Chroma | 12 pitch-class means and stds |
| Spectral shape | centroid/bandwidth/rolloff/flatness mean + std |
| Spectral dynamics | flux, skewness, kurtosis |
| Sub-band energy | low / mid / high mel-band ratios |
| Contrast | 6 contrast bands + mean |
| Other | chroma energy total |

## 3) Engineered Features (Literature-Inspired)
Added in `src/feature_engineering.py`:

- `Log_F0`, `Log_Energy`
- `F0_Energy_Product`, `F0_per_Centroid`
- `Centroid_Bandwidth_Ratio`, `Rolloff_minus_Centroid`
- `Noisiness_Index`, `Activity_Index`
- `High_Low_Mel_Ratio`, `MFCC_Mean_Norm`, `Chroma_Variance`

## 4) Model and Training
- Pipeline: median imputation + standardization + classifier
- Candidate models compared with 5-fold CV:
  - `gradient_boosting_phase2`
  - `random_forest_phase2`
- Selected model: **Gradient Boosting (Phase 2)**
- Split: stratified 75% / 25%, `random_state=42`

## 5) Results (Phase 1 vs Phase 2)

| Metric | Phase 1 | Phase 2 |
|---|---:|---:|
| Test accuracy | 43.05% | **59.60%** |
| CV accuracy | - | 54.88% |
| Feature count | 8 | 100 |

Per-class F1 (test set):

| Class | Phase 1 F1 | Phase 2 F1 |
|---|---:|---:|
| Angry | 0.42 | **0.68** |
| Happy | 0.47 | **0.58** |
| Neutral | 0.56 | **0.62** |
| Sad | 0.00 | **0.47** |
| Surprised | 0.21 | **0.47** |

Main gains come from MFCC/chroma/spectral-contrast information and better separability for `Sad` and `Surprised`.

## 6) How to Reproduce
From project root:

```bash
python3 src/run_phase2.py
```

Or step-by-step:

```bash
python3 src/extract_features.py --output outputs/features_phase2.csv
python3 src/train_emotion_model.py --features outputs/features_phase2.csv --phase 2
```

## 7) Scoreboard Submission
- Group: **17**
- Method: **Gradient Boosting + Frequency-Plane Features**
- Accuracy: **59.60%**
- Features: time-domain + MFCC/chroma/spectral contrast + engineered descriptors

## 8) References (Literature)
- T. Giannakopoulos, "Study of spectral and prosodic features for speech emotion recognition," pattern analysis literature on MFCC/prosody.
- Schuller et al., INTERSPEECH emotion challenges: MFCC + delta + spectral descriptors.
- Eyben et al., openSMILE feature sets: spectral shape, energy, voicing-related cues.

## 9) Team Contributions
- Feature engineering and frequency-plane extraction
- Model selection and evaluation
- Report and scoreboard update
