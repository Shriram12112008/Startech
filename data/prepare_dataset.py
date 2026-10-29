"""
data/prepare_dataset.py
-----------------------
Dataset generation, enrichment, and preparation module for 01PS2 Financial Risk & Fraud Analytics.
Supports standard Kaggle Credit Card Fraud dataset format (Time, V1-V28, Amount, Class)
and synthesizes high-fidelity financial transaction features:
  - Transaction Amount ($)
  - Time (elapsed seconds)
  - Location Distance (km from habitual base)
  - Device Risk Score (0-100 device profiling metric)
  - Velocity Metrics (tx_velocity_1h, tx_velocity_24h)
  - Severe Fraud Class Imbalance (~0.5% - 1.5% fraud rate)
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    n_samples: int = 20000,
    fraud_ratio: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a high-fidelity synthetic credit card transaction dataset
    mirroring the Kaggle Credit Card Fraud distribution and financial risk features.

    Parameters:
        n_samples (int): Total number of transaction records to generate.
        fraud_ratio (float): Fraction of fraudulent transactions (e.g., 0.01 = 1%).
        random_state (int): Seed for deterministic reproduction.

    Returns:
        pd.DataFrame: High-fidelity fraud analytics dataframe.
    """
    rng = np.random.default_rng(random_state)

    n_fraud = int(np.round(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # 1. Transaction Time (simulating 48 hours = 172,800 seconds)
    time_legit = np.sort(rng.uniform(0, 172800, size=n_legit))
    # Fraud transactions often cluster in off-peak nighttime hours (e.g., 11 PM - 5 AM)
    night_offsets = rng.choice([0, 86400], size=n_fraud)
    time_fraud_night = rng.uniform(0, 21600, size=n_fraud) + night_offsets
    time_fraud_day = rng.uniform(0, 172800, size=n_fraud)
    time_fraud = np.where(rng.random(size=n_fraud) < 0.65, time_fraud_night, time_fraud_day)

    # 2. Transaction Amount ($)
    # Legit: log-normal distribution with median ~$35, typical retail basket
    amount_legit = rng.lognormal(mean=3.5, sigma=1.1, size=n_legit)
    amount_legit = np.clip(amount_legit, 0.50, 5000.0)

    # Fraud: Bimodal - micro-authorization card testing ($0.80 - $4.00) or high-ticket siphon ($800 - $6500)
    is_card_testing = rng.random(size=n_fraud) < 0.25
    amount_fraud_micro = rng.uniform(0.75, 4.50, size=n_fraud)
    amount_fraud_macro = rng.lognormal(mean=7.2, sigma=0.9, size=n_fraud)
    amount_fraud = np.where(is_card_testing, amount_fraud_micro, amount_fraud_macro)
    amount_fraud = np.clip(amount_fraud, 0.75, 12000.0)

    # 3. Location Distance (km from habitual base / primary cardholder residence)
    # Legit: majority local purchases (Gamma distribution, 95% < 35 km)
    distance_legit = rng.gamma(shape=1.8, scale=7.5, size=n_legit)
    # Fraud: cross-border, impossible travel velocity, or remote data centers
    distance_fraud = rng.gamma(shape=3.5, scale=180.0, size=n_fraud) + rng.uniform(50.0, 800.0, size=n_fraud)

    # 4. Device Risk / Profiling Score (0 to 100)
    # Legit: trusted devices, low risk reputation (mean ~18, std ~10)
    device_risk_legit = rng.normal(loc=18.0, scale=9.5, size=n_legit)
    # Fraud: VPN, Tor, rooted/jailbroken devices, browser spoofing (mean ~82, std ~12)
    device_risk_fraud = rng.normal(loc=82.0, scale=11.0, size=n_fraud)
    device_risk_legit = np.clip(device_risk_legit, 0.0, 100.0)
    device_risk_fraud = np.clip(device_risk_fraud, 0.0, 100.0)

    # 5. Velocity Metrics
    # tx_velocity_1h: Transactions in last 1 hour
    # Legit: mostly 1 transaction (Poisson lambda=1.15)
    vel_1h_legit = rng.poisson(lam=1.15, size=n_legit)
    # Fraud: rapid automated carding bursts (Poisson lambda=4.90)
    vel_1h_fraud = rng.poisson(lam=4.90, size=n_fraud) + 1

    # tx_velocity_24h: Transactions in last 24 hours
    vel_24h_legit = vel_1h_legit + rng.poisson(lam=2.8, size=n_legit)
    vel_24h_fraud = vel_1h_fraud + rng.poisson(lam=7.5, size=n_fraud) + 2

    # 6. Kaggle-compatible PCA Components V1 through V28
    # In Kaggle dataset, V4, V11, V12, V14, V17 are strong fraud discriminators
    pca_legit = rng.normal(0, 1, size=(n_legit, 28))
    pca_fraud = rng.normal(0, 1, size=(n_fraud, 28))

    # Inject realistic directional shifts into key PCA discriminators
    pca_fraud[:, 3] += rng.normal(3.8, 1.2, size=n_fraud)    # V4 (positive shift in fraud)
    pca_fraud[:, 10] += rng.normal(3.5, 1.0, size=n_fraud)   # V11 (positive shift in fraud)
    pca_fraud[:, 11] -= rng.normal(4.2, 1.3, size=n_fraud)   # V12 (negative shift in fraud)
    pca_fraud[:, 13] -= rng.normal(5.1, 1.4, size=n_fraud)   # V14 (strong negative shift)
    pca_fraud[:, 16] -= rng.normal(4.5, 1.2, size=n_fraud)   # V17 (negative shift)
    pca_fraud[:, 18] += rng.normal(1.8, 1.0, size=n_fraud)   # V19

    # Combine Legit and Fraud records
    legit_df = pd.DataFrame(pca_legit, columns=[f"V{i}" for i in range(1, 29)])
    legit_df["Time"] = time_legit
    legit_df["Amount"] = np.round(amount_legit, 2)
    legit_df["location_distance"] = np.round(distance_legit, 2)
    legit_df["device_risk_score"] = np.round(device_risk_legit, 1)
    legit_df["tx_velocity_1h"] = vel_1h_legit
    legit_df["tx_velocity_24h"] = vel_24h_legit
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(pca_fraud, columns=[f"V{i}" for i in range(1, 29)])
    fraud_df["Time"] = time_fraud
    fraud_df["Amount"] = np.round(amount_fraud, 2)
    fraud_df["location_distance"] = np.round(distance_fraud, 2)
    fraud_df["device_risk_score"] = np.round(device_risk_fraud, 1)
    fraud_df["tx_velocity_1h"] = vel_1h_fraud
    fraud_df["tx_velocity_24h"] = vel_24h_fraud
    fraud_df["Class"] = 1

    # Concatenate and shuffle
    df = pd.concat([legit_df, fraud_df], axis=0, ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df["Time"] = np.sort(df["Time"].values)  # Sort by arrival time

    return df


def enrich_kaggle_dataset(raw_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    If a real Kaggle creditcard.csv dataset is provided, enriches it with
    the missing location_distance, device_risk_score, and velocity metrics.
    """
    df = raw_df.copy()
    rng = np.random.default_rng(random_state)
    n = len(df)
    is_fraud = df["Class"].values == 1

    # Enrich location distance
    dist = np.where(
        is_fraud,
        rng.gamma(shape=3.5, scale=180.0, size=n) + 120.0,
        rng.gamma(shape=1.8, scale=7.5, size=n),
    )
    df["location_distance"] = np.round(dist, 2)

    # Enrich device risk score
    dev = np.where(
        is_fraud,
        rng.normal(loc=82.0, scale=11.0, size=n),
        rng.normal(loc=18.0, scale=9.5, size=n),
    )
    df["device_risk_score"] = np.round(np.clip(dev, 0.0, 100.0), 1)

    # Enrich velocity metrics
    v1h = np.where(
        is_fraud,
        rng.poisson(lam=4.9, size=n) + 1,
        rng.poisson(lam=1.15, size=n),
    )
    v24h = v1h + np.where(
        is_fraud,
        rng.poisson(lam=7.5, size=n) + 2,
        rng.poisson(lam=2.8, size=n),
    )
    df["tx_velocity_1h"] = v1h
    df["tx_velocity_24h"] = v24h

    return df


def prepare_dataset(
    input_path: str = None,
    output_path: str = "data/creditcard_fraud.csv",
    n_samples: int = 20000,
    fraud_ratio: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads raw Kaggle dataset if available, or generates a synthetic dataset.
    Saves clean dataset to output_path.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if input_path and os.path.exists(input_path):
        print(f"[Dataset Preparation] Loading raw Kaggle dataset from: {input_path}")
        raw_df = pd.read_csv(input_path)
        needed = {"location_distance", "device_risk_score", "tx_velocity_1h", "tx_velocity_24h"}
        if not needed.issubset(raw_df.columns):
            print("[Dataset Preparation] Enriching dataset with financial risk features...")
            df = enrich_kaggle_dataset(raw_df, random_state=random_state)
        else:
            df = raw_df
    else:
        print(f"[Dataset Preparation] Generating high-fidelity synthetic dataset ({n_samples} records, {fraud_ratio*100:.2f}% fraud rate)...")
        df = generate_synthetic_dataset(n_samples=n_samples, fraud_ratio=fraud_ratio, random_state=random_state)

    df.to_csv(output_file, index=False)
    print(f"[Dataset Preparation] Successfully prepared dataset: {output_file.resolve()}")
    print(f"[Dataset Preparation] Total records: {len(df):,}, Fraud records: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Fraud & Financial Risk Dataset")
    parser.add_argument("--input", type=str, default=None, help="Path to raw creditcard.csv (optional)")
    parser.add_argument("--output", type=str, default="data/creditcard_fraud.csv", help="Path for output dataset")
    parser.add_argument("--n-samples", type=int, default=20000, help="Number of records for synthetic data")
    parser.add_argument("--fraud-ratio", type=float, default=0.01, help="Fraud ratio for synthetic data")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    prepare_dataset(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        fraud_ratio=args.fraud_ratio,
        random_state=args.random_state,
    )
