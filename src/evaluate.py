"""
src/evaluate.py
---------------
Model evaluation, benchmark computation, and Decision & Risk Scoring Engine
for 01PS2 Financial Risk & Fraud Analytics.
Categorizes predictions into actionable operational risk bands:
  - Low Risk (0–30 Score): Approve Instantly
  - Medium Risk (31–70 Score): Step-Up Auth / OTP Review
  - High Risk (71–100 Score): Block & Alert User
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class RiskDecisionEngine:
    """
    Financial Risk Scoring & Action Policy Engine.
    Maps probabilistic & anomaly outputs into calibrated 0-100 Risk Scores,
    enforcing automated risk tiers, step-up authentication triggers, and explainability rules.
    """

    # Defined Risk Bands per Hackathon Specs
    BAND_LOW_MAX = 30
    BAND_MED_MAX = 70

    def __init__(
        self,
        model=None,
        feature_engineer=None,
        low_threshold: int = 30,
        high_threshold: int = 70,
    ):
        self.model = model
        self.feature_engineer = feature_engineer
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def calculate_risk_score(
        self,
        prob_fraud: float,
        raw_tx: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Calibrates raw model probability into a normalized 0-100 risk score
        with financial domain heuristics.
        """
        # Base score from probabilistic model output (0 to 100)
        base_score = float(prob_fraud) * 100.0

        modifier = 0.0
        if raw_tx:
            # Rule modifier: Extreme velocity burst (>= 5 txs in 1 hour)
            v1h = raw_tx.get("tx_velocity_1h", 1)
            if v1h >= 5:
                modifier += 15.0

            # Rule modifier: Impossible travel or high distance (> 500 km)
            dist = raw_tx.get("location_distance", 0)
            if dist > 800:
                modifier += 15.0
            elif dist > 350:
                modifier += 8.0

            # Rule modifier: High device risk score (> 80)
            dev_score = raw_tx.get("device_risk_score", 0)
            if dev_score >= 85:
                modifier += 12.0

            # Rule modifier: High transaction amount (> $2,000)
            amt = raw_tx.get("Amount", 0)
            if amt > 2500:
                modifier += 10.0

        final_score = int(np.clip(np.round(base_score * 0.75 + modifier), 0, 100))
        return final_score

    def categorize_risk(self, risk_score: int) -> Dict[str, str]:
        """
        Categorizes risk score into actionable operational decision bands:
          - Low Risk (0–30 Score): Approve Instantly
          - Medium Risk (31–70 Score): Step-Up Auth / OTP Review
          - High Risk (71–100 Score): Block & Alert User
        """
        if risk_score <= self.low_threshold:
            return {
                "risk_band": "Low Risk",
                "action": "Approve Instantly",
                "action_code": "APPROVE",
                "color": "#10B981",  # Emerald Green
                "badge_class": "badge-low",
                "recommendation": "Transaction within normal behavioral parameters. Automated clearing approved.",
            }
        elif risk_score <= self.high_threshold:
            return {
                "risk_band": "Medium Risk",
                "action": "Step-Up Auth / OTP Review",
                "action_code": "STEP_UP_AUTH",
                "color": "#F59E0B",  # Amber / Warning Yellow
                "badge_class": "badge-medium",
                "recommendation": "Suspicious pattern detected. Require SMS/App OTP verification or manual agent review.",
            }
        else:
            return {
                "risk_band": "High Risk",
                "action": "Block & Alert User",
                "action_code": "BLOCK",
                "color": "#EF4444",  # Crimson Red
                "badge_class": "badge-high",
                "recommendation": "Severe risk anomaly detected. Transaction blocked immediately, account frozen, alert dispatched.",
            }

    def evaluate_rules(self, raw_tx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluates transparent heuristic fraud risk rules for explainability audit trails.
        """
        triggered = []

        amt = float(raw_tx.get("Amount", 0))
        if amt > 2000.0:
            triggered.append({
                "rule": "RULE_HIGH_AMOUNT",
                "severity": "HIGH",
                "description": f"Transaction amount (${amt:,.2f}) exceeds $2,000 baseline threshold.",
            })

        v1h = int(raw_tx.get("tx_velocity_1h", 1))
        if v1h >= 4:
            triggered.append({
                "rule": "RULE_VELOCITY_SPIKE",
                "severity": "CRITICAL" if v1h >= 6 else "HIGH",
                "description": f"Rapid velocity burst detected: {v1h} transactions attempted within 1 hour.",
            })

        dist = float(raw_tx.get("location_distance", 0))
        if dist > 500.0:
            triggered.append({
                "rule": "RULE_IMPOSSIBLE_TRAVEL",
                "severity": "HIGH",
                "description": f"Card used {dist:,.1f} km from primary registered location (potential impossible travel).",
            })

        dev = float(raw_tx.get("device_risk_score", 0))
        if dev >= 75.0:
            triggered.append({
                "rule": "RULE_SUSPICIOUS_DEVICE",
                "severity": "HIGH",
                "description": f"Device reputation score ({dev:.1f}/100) indicates proxy/VPN or emulator tampering.",
            })

        time_val = float(raw_tx.get("Time", 0))
        hour = ((time_val // 3600) % 24)
        if (hour >= 23 or hour <= 5) and (amt > 300 or v1h >= 2):
            triggered.append({
                "rule": "RULE_NOCTURNAL_BURST",
                "severity": "MEDIUM",
                "description": f"Off-peak transaction ({int(hour):02d}:00 UTC) with elevated monetary or velocity exposure.",
            })

        return triggered

    def evaluate_transaction(
        self,
        raw_tx: Dict[str, Any],
        model_override=None,
    ) -> Dict[str, Any]:
        """
        Full end-to-end evaluation of a single transaction.
        Returns comprehensive decision payload, calibrated risk score,
        action directives, and triggered rule explainability.
        """
        model = model_override or self.model

        # Calculate model probability
        if model is not None and self.feature_engineer is not None:
            X_vec = self.feature_engineer.transform_single(raw_tx)

            if hasattr(model, "predict_proba"):
                prob_fraud = float(model.predict_proba(X_vec)[0, 1])
            elif hasattr(model, "decision_function"):
                # For Isolation Forest or SVM
                score = float(model.decision_function(X_vec)[0])
                # Map negative anomaly score to probability (lower score = higher anomaly)
                prob_fraud = float(1.0 / (1.0 + np.exp(score * 3.5)))
            else:
                prob_fraud = float(model.predict(X_vec)[0])
        else:
            # Fallback heuristic calculation if model not passed
            dev = float(raw_tx.get("device_risk_score", 0)) / 100.0
            v1h = min(1.0, float(raw_tx.get("tx_velocity_1h", 1)) / 5.0)
            dist = min(1.0, float(raw_tx.get("location_distance", 0)) / 1000.0)
            prob_fraud = (dev * 0.45) + (v1h * 0.35) + (dist * 0.20)

        risk_score = self.calculate_risk_score(prob_fraud, raw_tx=raw_tx)
        decision_meta = self.categorize_risk(risk_score)
        triggered_rules = self.evaluate_rules(raw_tx)

        return {
            "risk_score": risk_score,
            "probability_fraud": float(np.round(prob_fraud, 4)),
            "risk_band": decision_meta["risk_band"],
            "action": decision_meta["action"],
            "action_code": decision_meta["action_code"],
            "color": decision_meta["color"],
            "recommendation": decision_meta["recommendation"],
            "triggered_rules": triggered_rules,
            "rule_count": len(triggered_rules),
            "transaction_data": raw_tx,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes financial classification metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": int(len(y_true)),
        "fraud_count": int(np.sum(y_true)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.5
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = float(np.mean(y_true))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = float(np.mean(y_true))

    return metrics


def evaluate_all_models(
    models: Dict[str, Any],
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    output_path: Optional[str] = "models/evaluation_report.json",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates all trained models (Logistic Regression, Random Forest, Isolation Forest),
    computes benchmark metrics, and formats an executive summary.
    """
    benchmarks = {}

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_scaled)[:, 1]
            preds = (probs >= 0.5).astype(int)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_scaled)
            probs = 1.0 / (1.0 + np.exp(scores * 3.5))
            raw_preds = model.predict(X_test_scaled)
            # IsolationForest predicts -1 for anomalies, 1 for normal
            preds = (raw_preds == -1).astype(int)
        else:
            preds = model.predict(X_test_scaled)
            probs = preds.astype(float)

        benchmarks[name] = compute_metrics(y_test, preds, probs)

    # Print comparative ASCII table
    print("\n" + "=" * 80)
    print(" 01PS2 – FINANCIAL RISK & FRAUD MODEL BENCHMARK RESULTS")
    print("=" * 80)
    header = f"{'Model':<24} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'ROC-AUC':<10} | {'PR-AUC':<10}"
    print(header)
    print("-" * 80)
    for model_name, m in benchmarks.items():
        row = f"{model_name:<24} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['roc_auc']:<10.4f} | {m['pr_auc']:<10.4f}"
        print(row)
    print("=" * 80 + "\n")

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(benchmarks, f, indent=2)
        print(f"[Evaluation] Report successfully saved to: {out_file.resolve()}")

    return benchmarks
