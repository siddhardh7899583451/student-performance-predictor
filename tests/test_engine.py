"""Basic unit tests for the StudentPerformanceEngine."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_engine import FEATURE_COLS, StudentPerformanceEngine


@pytest.fixture
def engine():
    return StudentPerformanceEngine()


def test_data_loads(engine):
    assert len(engine.df) == 400
    for col in FEATURE_COLS + ["exam_score"]:
        assert col in engine.df.columns


def test_train_linear(engine):
    result = engine.train(algorithm="linear", use_scaling=True, poly_degree=1)
    assert 0.5 < result.r2_test < 1.0
    assert result.mae_test < 15
    assert result.n_train + result.n_test == 400
    assert len(result.coefficients) == len(FEATURE_COLS)


def test_train_ridge(engine):
    result = engine.train(algorithm="ridge", alpha=5.0)
    assert result.model_name.startswith("Ridge")
    assert result.r2_test > 0.4


def test_predict_after_train(engine):
    engine.train()
    out = engine.predict_one(
        {
            "study_hours": 5.0,
            "attendance_pct": 80.0,
            "previous_score": 65.0,
            "sleep_hours": 7.0,
            "assignments_completed": 7.0,
        }
    )
    assert 0 <= out["predicted_score"] <= 100


def test_poly_degree_expands_features(engine):
    result = engine.train(poly_degree=2, use_scaling=True)
    # degree 2 on 5 features → 5 + 5 + C(5,2) = 20 terms
    assert len(result.feature_names) > len(FEATURE_COLS)
