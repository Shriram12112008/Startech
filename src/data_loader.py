"""
src/data_loader.py
------------------
Data ingestion, validation, stratified splitting, and imbalance mitigation
(SMOTE and class weights) for 01PS2 Financial Risk & Fraud Analytics.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Import synthetic preparation fallback if file does not exist
try:
    from data.prepare_dataset import prepare_dataset, generate_synthetic_dataset
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.prepare_dataset import prepare_dataset, generate_synthetic_dataset


class FraudDataLoader:
    """
    Robust data loader and pipeline preprocessor for fraud detection.
    Handles Kaggle formats, missing datasets, severe class imbalances,
    and stratified sampling.
    """

    EXPECTED_FEATURES = [
        "Time", "Amount", "location_distance", "device_risk_score",
        "tx_velocity_1h", "tx_velocity_24h"
    ]

    def __init__(self, data_path: Optional[str] = None, random_state: int = 42):
        self.random_state = random_state
        self.data_path = self._resolve_data_path(data_path)

    def _resolve_data_path(self, path: Optional[str]) -> Path:
        """Finds or infers default dataset location."""
        if path and os.path.exists(path):
            return Path(path)

        # Check candidate locations relative to current script or project root
        candidates = [
            Path("data/creditcard_fraud.csv"),
            Path("data/creditcard.csv"),
            Path(__file__).resolve().parents[1] / "data" / "creditcard_fraud.csv",
            Path(__file__).resolve().parents[1] / "data" / "creditcard.csv",
        ]
        for c in candidates:
            if c.exists():
                return c

        # Default target path if not yet generated
        return Path(__file__).resolve().parents[1] / "data" / "creditcard_fraud.csv"

    def load_data(
        self,
        filepath: Optional[str] = None,
        force_synthetic: bool = False,
        n_samples: int = 20000,
        fraud_ratio: float = 0.01,
    ) -> pd.DataFrame:
        """
        Loads the dataset from disk or automatically generates high-fidelity
        synthetic counterpart if missing.
        """
        target = Path(filepath) if filepath else self.data_path

        if force_synthetic or not target.exists():
            print(f"[DataLoader] Dataset not found at '{target}'. Auto-generating synthetic benchmark...")
            df = prepare_dataset(
                output_path=str(target),
                n_samples=n_samples,
                fraud_ratio=fraud_ratio,
                random_state=self.random_state,
            )
        else:
            print(f"[DataLoader] Ingesting transaction dataset from: {target.resolve()}")
            df = pd.read_csv(target)

        # Basic sanity check
        if "Class" not in df.columns:
            raise ValueError("Target column 'Class' missing from loaded dataset.")

        fraud_count = df["Class"].sum()
        total_count = len(df)
        print(f"[DataLoader] Loaded {total_count:,} records | Legitimate: {total_count - fraud_count:,} | Fraud: {fraud_count:,} ({fraud_count/total_count*100:.3f}%)")
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Performs stratified train/test split to preserve fraud proportion.
        """
        X = df.drop(columns=["Class"])
        y = df["Class"]

        stratify_labels = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_labels,
        )

        print(f"[DataLoader] Train split: {len(X_train):,} samples (Fraud: {y_train.sum():,}) | "
              f"Test split: {len(X_test):,} samples (Fraud: {y_test.sum():,})")
        return X_train, X_test, y_train, y_test

    def get_class_weights(self, y_train: Union[pd.Series, np.ndarray]) -> Dict[int, float]:
        """
        Computes inverse frequency class weights to penalize fraud false negatives.
        """
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weight_dict = {cls: float(w) for cls, w in zip(classes, weights)}
        print(f"[DataLoader] Calculated balanced class weights: {weight_dict}")
        return weight_dict

    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sampling_strategy: float = 0.15,
        k_neighbors: int = 5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies Synthetic Minority Over-sampling Technique (SMOTE) to rebalance training data.
        Features are synthetically interpolated along the feature space between fraud neighbors.
        Gracefully falls back if imblearn is unavailable.
        """
        try:
            from imblearn.over_sampling import SMOTE
            minority_count = int(y_train.sum())
            k = min(k_neighbors, max(1, minority_count - 1))
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k,
                random_state=self.random_state,
            )
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(
                f"[DataLoader] SMOTE applied: minority boosted from {minority_count:,} to "
                f"{int(y_resampled.sum()):,} samples (Total train: {len(X_resampled):,})"
            )
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name="Class")
        except Exception as e:
            print(f"[DataLoader] SMOTE warning ({e}). Utilizing native class weight penalization.")
            return X_train, y_train
