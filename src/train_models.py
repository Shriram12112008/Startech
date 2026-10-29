"""
src/train_models.py
-------------------
Training and orchestration pipeline for 01PS2 Financial Risk & Fraud Analytics.
Trains 3 distinct architectures:
  1. Logistic Regression (Interpretable Linear Baseline with Balanced Weights)
  2. Random Forest (High Accuracy Non-Linear Ensemble Classifier)
  3. Isolation Forest (Unsupervised Anomaly Detection)
Computes Precision, Recall, and F1 Score for all models and serializes
the best production artifact to models/fraud_model.pkl.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Path adjustment for relative execution
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import FraudDataLoader
from src.evaluate import RiskDecisionEngine, evaluate_all_models
from src.feature_engineering import FinancialFeatureEngineer


def train_models_pipeline(
    data_path: Optional[str] = None,
    use_smote: bool = True,
    n_samples: int = 20000,
    fraud_ratio: float = 0.01,
    random_state: int = 42,
    output_model_path: str = "models/fraud_model.pkl",
) -> Dict[str, Any]:
    """
    Executes end-to-end data ingestion, feature engineering, model training,
    evaluation, and artifact serialization.
    """
    print("\n" + "=" * 80)
    print(" 01PS2 – FINANCIAL RISK & FRAUD ANALYTICS: MODEL TRAINING PIPELINE")
    print("=" * 80)

    # 1. Ingest Data
    loader = FraudDataLoader(data_path=data_path, random_state=random_state)
    df = loader.load_data(n_samples=n_samples, fraud_ratio=fraud_ratio)

    # 2. Stratified Train/Test Split
    X_train_raw, X_test_raw, y_train, y_test = loader.split_data(df, test_size=0.2, stratify=True)

    # 3. Feature Engineering & Scaling
    print("[Pipeline] Fitting FinancialFeatureEngineer (domain signals & robust scaling)...")
    feature_engineer = FinancialFeatureEngineer(with_scaling=True)
    X_train_scaled = feature_engineer.fit_transform(X_train_raw)
    X_test_scaled = feature_engineer.transform(X_test_raw)
    feature_names = feature_engineer.get_feature_names_out()
    print(f"[Pipeline] Total features engineered: {len(feature_names)}")

    # 4. Imbalance Handling (SMOTE & Class Weights)
    class_weights = loader.get_class_weights(y_train)

    if use_smote:
        print("[Pipeline] Applying SMOTE to synthesize minority fraud samples for supervised training...")
        # Convert back to DataFrame for imblearn compatibility
        X_train_df = feature_engineer.transform_df(X_train_raw)
        X_resampled_df, y_resampled = loader.apply_smote(X_train_df, y_train, sampling_strategy=0.20)
        X_train_fit = X_resampled_df.values
        y_train_fit = y_resampled.values
    else:
        X_train_fit = X_train_scaled
        y_train_fit = y_train.values

    # 5. Model 1: Logistic Regression (Baseline)
    print("\n[Model 1/3] Training Logistic Regression (Baseline with Balanced Weights)...")
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1500,
        random_state=random_state,
        solver="lbfgs",
    )
    lr_model.fit(X_train_fit, y_train_fit)
    print("[Model 1/3] Logistic Regression training complete.")

    # 6. Model 2: Random Forest (High Accuracy Classifier)
    print("\n[Model 2/3] Training Random Forest (Ensemble Classifier, Balanced Subsample)...")
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train_fit, y_train_fit)
    print("[Model 2/3] Random Forest training complete.")

    # 7. Model 3: Isolation Forest (Unsupervised Anomaly Detection)
    print("\n[Model 3/3] Training Isolation Forest (Unsupervised Anomaly Detector)...")
    # Isolation Forest is trained on legitimate transactions or entire training set
    iso_model = IsolationForest(
        n_estimators=120,
        contamination=float(fraud_ratio * 1.5),  # Approximate contamination rate
        random_state=random_state,
        n_jobs=-1,
    )
    iso_model.fit(X_train_scaled)
    print("[Model 3/3] Isolation Forest training complete.")

    # 8. Evaluation & Benchmarking
    models_dict = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "Isolation Forest": iso_model,
    }

    report_path = Path(output_model_path).parent / "evaluation_report.json"
    benchmarks = evaluate_all_models(
        models=models_dict,
        X_test_scaled=X_test_scaled,
        y_test=y_test.values,
        output_path=str(report_path),
    )

    # 9. Select Best Production Model
    # Compare F1 score among supervised models
    best_name = "Random Forest"
    best_model = rf_model
    if benchmarks["Logistic Regression"]["f1"] > benchmarks["Random Forest"]["f1"]:
        best_name = "Logistic Regression"
        best_model = lr_model

    print(f"[Pipeline] Best Model Selected for Production: {best_name} (F1 Score: {benchmarks[best_name]['f1']:.4f})")

    # 10. Instantiate Risk & Decision Policy Engine
    decision_engine = RiskDecisionEngine(
        model=best_model,
        feature_engineer=feature_engineer,
        low_threshold=30,
        high_threshold=70,
    )

    # 11. Serialize Artifact Bundle
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "all_models": models_dict,
        "feature_engineer": feature_engineer,
        "feature_names": feature_names,
        "benchmarks": benchmarks,
        "decision_engine": decision_engine,
        "thresholds": {"low": 30, "high": 70},
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(df),
            "fraud_ratio": fraud_ratio,
            "use_smote": use_smote,
            "random_state": random_state,
        },
    }

    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"[Pipeline] Production Model Artifact saved to: {out_path.resolve()}")
    print("=" * 80 + "\n")

    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Financial Risk & Fraud ML Models")
    parser.add_argument("--data-path", type=str, default=None, help="Path to input dataset CSV")
    parser.add_argument("--output", type=str, default="models/fraud_model.pkl", help="Output path for model artifact")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--n-samples", type=int, default=20000, help="Number of records to synthesize if missing")
    parser.add_argument("--fraud-ratio", type=float, default=0.01, help="Synthetic fraud ratio")
    args = parser.parse_args()

    train_models_pipeline(
        data_path=args.data_path,
        use_smote=not args.no_smote,
        n_samples=args.n_samples,
        fraud_ratio=args.fraud_ratio,
        output_model_path=args.output,
    )
