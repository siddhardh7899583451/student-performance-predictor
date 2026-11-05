"""
Generate a realistic synthetic student performance dataset.
Run once: python generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
n = 400

study_hours = np.clip(np.random.normal(5.2, 2.4, n), 0.5, 14)
attendance = np.clip(np.random.normal(78, 14, n), 35, 100)
previous_score = np.clip(np.random.normal(62, 16, n), 20, 95)
sleep_hours = np.clip(np.random.normal(6.5, 1.4, n), 3.0, 10)
assignments_completed = np.clip(np.random.binomial(10, 0.70, n), 0, 10).astype(float)

# Tuned linear relationship so scores spread across a realistic range
true_score = (
    8.0
    + 3.6 * study_hours
    + 0.18 * attendance
    + 0.28 * previous_score
    + 1.4 * sleep_hours
    + 0.9 * assignments_completed
    + np.random.normal(0, 5.5, n)
)
exam_score = np.clip(true_score, 15, 100)

df = pd.DataFrame({
    "study_hours": np.round(study_hours, 1),
    "attendance_pct": np.round(attendance, 1),
    "previous_score": np.round(previous_score, 1),
    "sleep_hours": np.round(sleep_hours, 1),
    "assignments_completed": assignments_completed,
    "exam_score": np.round(exam_score, 1),
})

out = Path(__file__).parent / "data" / "student_performance.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows → {out}")
print(df[["exam_score"]].describe().round(2))
print("\nSample:")
print(df.head(6).to_string(index=False))
