"""
src/feature_engineering.py
--------------------------
Financial domain feature engineering, scaling, and transformation pipeline.
Engineers behavioral velocity ratios, temporal nocturnal indicators,
impossible travel flags, and heavy-tailed log-amount stabilizers.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant feature engineering transformer designed for
    financial fraud detection & risk analytics.
    """

    NUMERIC_COLS_TO_SCALE = [
        "Amount",
        "location_distance",
        "device_risk_score",
        "tx_velocity_1h",
        "tx_velocity_24h",
        "hour_of_day",
        "amount_log",
        "velocity_ratio",
    ]

    def __init__(self, with_scaling: bool = True):
        self.with_scaling = with_scaling
        self.scaler = RobustScaler()
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _extract_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives domain-specific behavioral and risk signals from raw transaction data.
        """
        data = df.copy()

        # 1. Temporal signals
        if "Time" in data.columns:
            data["hour_of_day"] = ((data["Time"] // 3600) % 24).astype(float)
        elif "hour_of_day" not in data.columns:
            data["hour_of_day"] = 12.0

        # Late-night transaction indicator (11 PM - 5 AM)
        data["is_night"] = (
            (data["hour_of_day"] >= 23.0) | (data["hour_of_day"] <= 5.0)
        ).astype(float)

        # 2. Monetary log transform to stabilize heavy tails
        if "Amount" in data.columns:
            data["amount_log"] = np.log1p(np.maximum(0.0, data["Amount"].values))
        else:
            data["amount_log"] = 0.0

        # 3. Velocity acceleration ratio (burst velocity vs 24h baseline)
        v1h = data["tx_velocity_1h"] if "tx_velocity_1h" in data.columns else 1.0
        v24h = data["tx_velocity_24h"] if "tx_velocity_24h" in data.columns else 1.0
        data["velocity_ratio"] = v1h / ((v24h / 24.0) + 1e-3)

        # 4. Behavioral anomaly flags
        loc_dist = data["location_distance"] if "location_distance" in data.columns else 0.0
        data["impossible_travel"] = ((loc_dist > 500.0) & (v1h > 1.0)).astype(float)

        dev_score = data["device_risk_score"] if "device_risk_score" in data.columns else 0.0
        data["high_risk_device"] = (dev_score >= 75.0).astype(float)

        # 5. Drop raw monotonically increasing Time column if present
        if "Time" in data.columns:
            data = data.drop(columns=["Time"])

        return data

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the scaler on the engineered feature representations.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_feat = self._extract_domain_features(X)

        # Determine which numeric columns exist to scale
        cols_to_scale = [c for c in self.NUMERIC_COLS_TO_SCALE if c in df_feat.columns]

        if self.with_scaling and cols_to_scale:
            self.scaler.fit(df_feat[cols_to_scale])

        self.feature_names_ = list(df_feat.columns)
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transforms raw transaction data into scaled feature matrix.
        """
        if not self.is_fitted_:
            raise RuntimeError("FinancialFeatureEngineer must be fitted before transforming data.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_feat = self._extract_domain_features(X)

        # Align columns to fitted feature set
        for col in self.feature_names_:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        df_feat = df_feat[self.feature_names_]

        if self.with_scaling:
            cols_to_scale = [c for c in self.NUMERIC_COLS_TO_SCALE if c in df_feat.columns]
            if cols_to_scale:
                df_feat[cols_to_scale] = self.scaler.transform(df_feat[cols_to_scale])

        return df_feat.values

    def transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method returning DataFrame with labeled column names.
        """
        arr = self.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_)

    def transform_single(self, transaction_dict: dict) -> np.ndarray:
        """
        Transforms a single transaction record for low-latency live inference.
        """
        df = pd.DataFrame([transaction_dict])
        return self.transform(df)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_
