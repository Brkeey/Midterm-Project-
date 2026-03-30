from __future__ import annotations

import numpy as np


def predict_gender_rule_based(avg_f0_hz: float) -> str:
    """
    Basic pitch-based rule:
    - Male   : F0 < 170 Hz
    - Female : 170 <= F0 < 255 Hz
    - Child  : F0 >= 255 Hz
    """
    if np.isnan(avg_f0_hz):
        return "Unknown"
    if avg_f0_hz < 170:
        return "Male"
    if avg_f0_hz < 255:
        return "Female"
    return "Child"
