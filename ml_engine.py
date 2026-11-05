"""
Core ML engine for the Student Performance Predictor.
Covers: train/test split, scaling, Linear/Ridge regression,
evaluation metrics, residual analysis, coefficient interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


FEATURE_COLS = [
    "study_hours",
    "attendance_pct",
    "previous_score",
    "sleep_hours",
    "assignments_completed",
]
TARGET_COL = "exam_score"

# Plain-English explanations for each coefficient (positive direction)
FEATURE_HELP = {
    "study_hours": "Extra study hour → expected change in exam score",
    "attendance_pct": "One percentage-point higher attendance → expected score change",
    "previous_score": "One point higher previous score → expected score change",
    "sleep_hours": "Extra hour of sleep → expected score change",
    "assignments_completed": "One extra assignment completed → expected score change",
}


@dataclass
class TrainResult:
    model_name: str
    r2_train: float
    r2_test: float
    mae_test: float
    rmse_test: float
    cv_r2_mean: float
    cv_r2_std: float
    coefficients: Dict[str, float]
    intercept: float
    feature_names: List[str]
    y_test: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray
    n_train: int
    n_test: int


class StudentPerformanceEngine:
    """Train, evaluate and explain Linear / Ridge models on student data."""

    def __init__(self, data_path: Optional[str | Path] = None):
        base = Path(__file__).parent
        self.data_path = Path(data_path) if data_path else base / "data" / "student_performance.csv"
        self.df = pd.read_csv(self.data_path)
        self.scaler: Optional[StandardScaler] = None
        self.poly: Optional[PolynomialFeatures] = None
        self.model = None
        self.feature_names: List[str] = FEATURE_COLS.copy()
        self.last_result: Optional[TrainResult] = None

    def load_summary(self) -> Dict[str, Any]:
        """Return dataset overview for the UI."""
        desc = self.df.describe().round(2)
        corr = self.df[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).round(3)
        return {
            "n_rows": len(self.df),
            "features": FEATURE_COLS,
            "target": TARGET_COL,
            "describe": desc.to_dict(),
            "corr_with_target": corr.to_dict(),
            "head": self.df.head(8).to_dict(orient="records"),
        }

    def train(
        self,
        algorithm: str = "linear",
        alpha: float = 1.0,
        test_size: float = 0.2,
        use_scaling: bool = True,
        poly_degree: int = 1,
        random_state: int = 42,
    ) -> TrainResult:
        """
        Train a model and return a rich TrainResult for metrics + explainability.
        """
        X = self.df[FEATURE_COLS].values
        y = self.df[TARGET_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Optional polynomial expansion (degree 1 = original features)
        if poly_degree > 1:
            self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X_train = self.poly.fit_transform(X_train)
            X_test = self.poly.transform(X_test)
            self.feature_names = list(self.poly.get_feature_names_out(FEATURE_COLS))
        else:
            self.poly = None
            self.feature_names = FEATURE_COLS.copy()

        # Scaling (fit only on train – avoids leakage)
        if use_scaling:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            self.scaler = None

        if algorithm.lower() == "ridge":
            self.model = Ridge(alpha=alpha)
            model_name = f"Ridge (α={alpha})"
        else:
            self.model = LinearRegression()
            model_name = "Linear Regression"

        self.model.fit(X_train, y_train)

        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        residuals = y_test - y_pred_test

        # 5-fold CV on the (scaled / poly) training matrix for stability estimate
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="r2")

        coef_dict = {
            name: float(c) for name, c in zip(self.feature_names, self.model.coef_)
        }

        result = TrainResult(
            model_name=model_name,
            r2_train=float(r2_score(y_train, y_pred_train)),
            r2_test=float(r2_score(y_test, y_pred_test)),
            mae_test=float(mean_absolute_error(y_test, y_pred_test)),
            rmse_test=float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            cv_r2_mean=float(cv_scores.mean()),
            cv_r2_std=float(cv_scores.std()),
            coefficients=coef_dict,
            intercept=float(self.model.intercept_),
            feature_names=self.feature_names,
            y_test=y_test,
            y_pred=y_pred_test,
            residuals=residuals,
            n_train=len(y_train),
            n_test=len(y_test),
        )
        self.last_result = result
        return result

    def predict_one(self, values: Dict[str, float]) -> Dict[str, Any]:
        """Predict exam score for a single student profile."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        row = np.array([[values[c] for c in FEATURE_COLS]], dtype=float)

        if self.poly is not None:
            row = self.poly.transform(row)
        if self.scaler is not None:
            row = self.scaler.transform(row)

        pred = float(self.model.predict(row)[0])
        pred = float(np.clip(pred, 0, 100))

        # Contribution breakdown (only meaningful for linear/ridge without high poly)
        contributions = {}
        if self.poly is None or (hasattr(self.poly, "degree") and self.poly.degree == 1):
            for name, coef in zip(self.feature_names, self.model.coef_):
                # If scaled, contribution is on standardized scale – still useful relative
                contributions[name] = float(coef * row[0][self.feature_names.index(name)])

        return {
            "predicted_score": round(pred, 1),
            "intercept": round(self.last_result.intercept, 2) if self.last_result else None,
            "contributions": contributions,
            "model": self.last_result.model_name if self.last_result else "unknown",
        }

    def coefficient_table(self) -> pd.DataFrame:
        """Return a tidy table of coefficients + plain-English help."""
        if not self.last_result:
            return pd.DataFrame()
        rows = []
        for name, coef in self.last_result.coefficients.items():
            base = name.split(" ")[0] if " " in name else name
            help_text = FEATURE_HELP.get(base, "Interaction / polynomial term")
            rows.append(
                {
                    "feature": name,
                    "coefficient": round(coef, 4),
                    "interpretation": help_text,
                }
            )
        return pd.DataFrame(rows).sort_values("coefficient", key=abs, ascending=False)
