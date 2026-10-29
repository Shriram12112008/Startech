"""
tests/test_pipeline.py
----------------------
Comprehensive test suite for 01PS2 Financial Risk & Fraud Analytics.
Validates:
  - Synthetic dataset generation & Kaggle compatibility
  - Stratified train/test splitting & SMOTE rebalancing
  - Financial domain feature engineering & scaling
  - Multi-model training (Logistic Regression, Random Forest, Isolation Forest)
  - Calibrated Risk Scoring & 3-Tier Policy Action Matrix
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_dataset import generate_synthetic_dataset
from src.data_loader import FraudDataLoader
from src.evaluate import RiskDecisionEngine, compute_metrics
from src.feature_engineering import FinancialFeatureEngineer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_raw_df():
    """Generates a small deterministic sample dataframe for fast unit tests."""
    return generate_synthetic_dataset(n_samples=500, fraud_ratio=0.04, random_state=42)


def test_synthetic_data_generation(sample_raw_df):
    """Verifies synthetic dataset schema, column requirements, and fraud ratios."""
    df = sample_raw_df
    assert len(df) == 500
    assert "Class" in df.columns
    assert "Amount" in df.columns
    assert "Time" in df.columns
    assert "location_distance" in df.columns
    assert "device_risk_score" in df.columns
    assert "tx_velocity_1h" in df.columns
    assert "tx_velocity_24h" in df.columns

    # Verify PCA components exist
    for i in range(1, 29):
        assert f"V{i}" in df.columns

    # Check target distribution
    assert df["Class"].nunique() == 2
    assert df["Class"].sum() > 0
    assert df["Amount"].min() >= 0.50
    assert df["device_risk_score"].min() >= 0.0
    assert df["device_risk_score"].max() <= 100.0


def test_data_loader_stratified_split(sample_raw_df):
    """Verifies that train/test split is strictly stratified."""
    loader = FraudDataLoader(random_state=42)
    X_train, X_test, y_train, y_test = loader.split_data(sample_raw_df, test_size=0.2, stratify=True)

    assert len(X_train) == 400
    assert len(X_test) == 100
    assert y_train.sum() > 0
    assert y_test.sum() > 0

    # Verify fraud ratio is preserved across partitions
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.02


def test_smote_resampling(sample_raw_df):
    """Verifies SMOTE minority oversampling elevates fraud sample representation."""
    loader = FraudDataLoader(random_state=42)
    X_train, _, y_train, _ = loader.split_data(sample_raw_df, test_size=0.2, stratify=True)

    initial_fraud = int(y_train.sum())
    X_res, y_res = loader.apply_smote(X_train, y_train, sampling_strategy=0.25)

    assert len(X_res) > len(X_train)
    assert int(y_res.sum()) > initial_fraud
    assert int(y_res.sum()) == int(len(X_res[y_res == 0]) * 0.25)


def test_feature_engineering_transformer(sample_raw_df):
    """Verifies behavioral and temporal signals are properly engineered and scaled."""
    engineer = FinancialFeatureEngineer(with_scaling=True)
    X = sample_raw_df.drop(columns=["Class"])

    X_trans = engineer.fit_transform(X)
    feature_names = engineer.get_feature_names_out()

    assert "hour_of_day" in feature_names
    assert "is_night" in feature_names
    assert "amount_log" in feature_names
    assert "velocity_ratio" in feature_names
    assert "impossible_travel" in feature_names
    assert "high_risk_device" in feature_names

    # Check finite numbers, no NaN
    assert np.all(np.isfinite(X_trans))
    assert X_trans.shape[0] == len(sample_raw_df)
    assert X_trans.shape[1] == len(feature_names)


def test_model_training_and_prediction(sample_raw_df):
    """Verifies all 3 models train and predict correctly without errors."""
    loader = FraudDataLoader(random_state=42)
    X_train, X_test, y_train, y_test = loader.split_data(sample_raw_df, test_size=0.2)

    engineer = FinancialFeatureEngineer()
    X_train_scaled = engineer.fit_transform(X_train)
    X_test_scaled = engineer.transform(X_test)

    # 1. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    assert len(lr_probs) == len(X_test)
    assert (lr_probs >= 0.0).all() and (lr_probs <= 1.0).all()

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
    assert len(rf_probs) == len(X_test)
    assert (rf_probs >= 0.0).all() and (rf_probs <= 1.0).all()

    # 3. Isolation Forest
    iso = IsolationForest(n_estimators=20, contamination=0.04, random_state=42)
    iso.fit(X_train_scaled)
    iso_scores = iso.decision_function(X_test_scaled)
    assert len(iso_scores) == len(X_test)


def test_risk_decision_engine_bands():
    """
    Enforces the 3-Tier Policy Action Matrix specified in the problem statement:
      - Low Risk (0–30 Score): Approve Instantly
      - Medium Risk (31–70 Score): Step-Up Auth / OTP Review
      - High Risk (71–100 Score): Block & Alert User
    """
    engine = RiskDecisionEngine(low_threshold=30, high_threshold=70)

    # Test Low Risk
    for score in [0, 10, 25, 30]:
        res = engine.categorize_risk(score)
        assert res["risk_band"] == "Low Risk"
        assert res["action"] == "Approve Instantly"
        assert res["action_code"] == "APPROVE"

    # Test Medium Risk
    for score in [31, 50, 65, 70]:
        res = engine.categorize_risk(score)
        assert res["risk_band"] == "Medium Risk"
        assert res["action"] == "Step-Up Auth / OTP Review"
        assert res["action_code"] == "STEP_UP_AUTH"

    # Test High Risk
    for score in [71, 85, 95, 100]:
        res = engine.categorize_risk(score)
        assert res["risk_band"] == "High Risk"
        assert res["action"] == "Block & Alert User"
        assert res["action_code"] == "BLOCK"


def test_rule_engine_explanations():
    """Verifies rule triggers fire for high amount, velocity, impossible travel, and device score."""
    engine = RiskDecisionEngine()

    suspicious_tx = {
        "Amount": 4500.0,
        "Time": 7200,  # 02:00 AM UTC
        "location_distance": 950.0,
        "device_risk_score": 92.0,
        "tx_velocity_1h": 6,
        "tx_velocity_24h": 10,
    }

    triggered = engine.evaluate_rules(suspicious_tx)
    rules = [r["rule"] for r in triggered]

    assert "RULE_HIGH_AMOUNT" in rules
    assert "RULE_VELOCITY_SPIKE" in rules
    assert "RULE_IMPOSSIBLE_TRAVEL" in rules
    assert "RULE_SUSPICIOUS_DEVICE" in rules
    assert "RULE_NOCTURNAL_BURST" in rules


def test_end_to_end_single_transaction_evaluation():
    """Verifies comprehensive end-to-end evaluation payload structure."""
    engine = RiskDecisionEngine()
    tx = {
        "Amount": 12.50,
        "Time": 43200,  # 12:00 PM
        "location_distance": 5.0,
        "device_risk_score": 10.0,
        "tx_velocity_1h": 1,
        "tx_velocity_24h": 2,
    }
    decision = engine.evaluate_transaction(tx)

    assert "risk_score" in decision
    assert "risk_band" in decision
    assert "action" in decision
    assert "action_code" in decision
    assert "triggered_rules" in decision
    assert 0 <= decision["risk_score"] <= 100
    assert decision["risk_band"] == "Low Risk"
    assert decision["action"] == "Approve Instantly"


def test_compute_metrics_correctness():
    """Verifies confusion matrix and metric calculation utility."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.7, 0.9, 0.8])

    m = compute_metrics(y_true, y_pred, y_prob)

    assert m["true_positives"] == 2
    assert m["false_positives"] == 1
    assert m["false_negatives"] == 0
    assert m["true_negatives"] == 3
    assert m["recall"] == 1.0
    assert m["precision"] == 2 / 3
    assert m["f1"] == pytest.approx(0.80, abs=0.01)
