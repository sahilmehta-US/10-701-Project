"""
causal_feature_select.py
------------------------
Read PCMCI results and prepare a clean dataset for LSTM training.

Usage:
    python causal_feature_select.py

Outputs (in results/pcmci_output/):
    - lstm_causal_input.csv   → ready-to-use dataset for LSTM
    - selected_features.json  → list of selected feature names
"""

import os
import json
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
CAUSAL_FEATURES_CSV  = "results/pcmci_output/causal_features.csv"
FORECAST_DATA_CSV    = "../data/results/gold_forecast_features_t5.csv"
OUTPUT_DIR           = "results/pcmci_output"
TARGET_COL           = "Gold Futures (COMEX) | target_log_return_t_plus_5"

P_VALUE_THRESHOLD    = 0.05   # only keep features below this p-value
USE_BEST_LAG_ONLY    = True   # True = method 1 (exact lag), False = method 2 (all lags)

SPLIT_DEFINITION = {
    "train_end":      "2018-12-31",
    "val_start":      "2019-01-01",
    "val_end":        "2021-12-31",
    "test_start":     "2022-01-01",
}
# ══════════════════════════════════════════════════════════════════════════


def load_causal_features() -> pd.DataFrame:
    """Load PCMCI output and filter by p-value threshold."""
    df = pd.read_csv(CAUSAL_FEATURES_CSV)
    df = df[df["p_value"] < P_VALUE_THRESHOLD].copy()
    df = df.sort_values("p_value").reset_index(drop=True)
    print(f"[INFO] {len(df)} significant causal links found (p < {P_VALUE_THRESHOLD})")
    return df


def select_columns(causal_df: pd.DataFrame, forecast_df: pd.DataFrame) -> list:
    """
    Match PCMCI features to column names in the forecast DataFrame.
    
    Method 1 (USE_BEST_LAG_ONLY=True):
        For each feature, pick the single lag with the lowest p-value.
        e.g. "CBOE Volatility Index | diff_1" at lag 1
            ->"CBOE Volatility Index | diff_1 | lag_1"

    Method 2 (USE_BEST_LAG_ONLY=False):
        Keep all available lags for every significant feature.
    """
    selected = []

    if USE_BEST_LAG_ONLY:
        # Pick best (lowest p-value) lag per feature
        best = causal_df.loc[causal_df.groupby("feature")["p_value"].idxmin()]
        for _, row in best.iterrows():
            col = f"{row['feature']} | lag_{int(row['lag'])}"
            if col in forecast_df.columns:
                selected.append(col)
                print(f"  [SELECT] {col}  (p={row['p_value']:.4e})")
            else:
                print(f"  [WARN] Column not found in forecast data: {col}")
    else:
        # Keep all lags for every significant feature
        unique_features = causal_df["feature"].unique()
        for feat in unique_features:
            feat_cols = [c for c in forecast_df.columns if c.startswith(feat + " | lag_")]
            if feat_cols:
                selected.extend(feat_cols)
                print(f"  [SELECT] {feat} — {len(feat_cols)} lag columns")
            else:
                print(f"  [WARN] No lag columns found for: {feat}")

    return selected


def split_dataset(df: pd.DataFrame):
    """Split into train / val / test using fixed chronological boundaries."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    train = df[df.index <= SPLIT_DEFINITION["train_end"]]
    val   = df[(df.index >= SPLIT_DEFINITION["val_start"]) &
               (df.index <= SPLIT_DEFINITION["val_end"])]
    test  = df[df.index >= SPLIT_DEFINITION["test_start"]]

    print(f"\n[SPLIT] Train : {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} rows)")
    print(f"[SPLIT] Val   : {val.index[0].date()} → {val.index[-1].date()}  ({len(val)} rows)")
    print(f"[SPLIT] Test  : {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} rows)")

    return train, val, test


def scale_splits(train, val, test, feature_cols):
    """
    Fit StandardScaler on train only, then apply to val and test.
    Returns scaled numpy arrays and the fitted scaler.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train = scaler.fit_transform(train[feature_cols].values)
    X_val   = scaler.transform(val[feature_cols].values)
    X_test  = scaler.transform(test[feature_cols].values)

    y_train = train[TARGET_COL].values
    y_val   = val[TARGET_COL].values
    y_test  = test[TARGET_COL].values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def save_outputs(selected_cols, train, val, test):
    """Save selected feature names and the full causal dataset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save feature list as JSON (easy for teammates to load)
    feature_info = {
        "method": "best_lag_only" if USE_BEST_LAG_ONLY else "all_lags",
        "p_value_threshold": P_VALUE_THRESHOLD,
        "target": TARGET_COL,
        "n_features": len(selected_cols),
        "features": selected_cols,
    }
    json_path = os.path.join(OUTPUT_DIR, "selected_features.json")
    with open(json_path, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"\n[SAVED] Feature list → {json_path}")

    # Save full causal dataset (unsplit, unscaled) for reference
    full = pd.concat([train, val, test])
    csv_path = os.path.join(OUTPUT_DIR, "lstm_causal_input.csv")
    full[selected_cols + [TARGET_COL]].to_csv(csv_path)
    print(f"[SAVED] Full causal dataset → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    # Step 1: Load PCMCI results
    print("=" * 60)
    print("Step 1: Loading PCMCI causal features")
    print("=" * 60)
    causal_df = load_causal_features()

    # Step 2: Load forecast feature dataset
    print("\n" + "=" * 60)
    print("Step 2: Loading forecast feature dataset")
    print("=" * 60)
    forecast_df = pd.read_csv(FORECAST_DATA_CSV)
    print(f"[INFO] Forecast data shape: {forecast_df.shape}")

    # Step 3: Select columns
    print("\n" + "=" * 60)
    print("Step 3: Selecting causal feature columns")
    print("=" * 60)
    selected_cols = select_columns(causal_df, forecast_df)
    print(f"\n[INFO] Total selected features: {len(selected_cols)}")

    if not selected_cols:
        raise ValueError("No matching columns found. Check feature names.")

    # Step 4: Split dataset
    print("\n" + "=" * 60)
    print("Step 4: Splitting dataset")
    print("=" * 60)
    train, val, test = split_dataset(forecast_df)

    # Step 5: Scale features
    print("\n" + "=" * 60)
    print("Step 5: Scaling features (fit on train only)")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = scale_splits(
        train, val, test, selected_cols
    )
    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_val shape:   {X_val.shape}")
    print(f"[INFO] X_test shape:  {X_test.shape}")

    # Step 6: Save outputs
    print("\n" + "=" * 60)
    print("Step 6: Saving outputs")
    print("=" * 60)
    save_outputs(selected_cols, train, val, test)

    print("\nDone!")

    # Return everything for direct use in LSTM training
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
        "feature_names": selected_cols,
    }


if __name__ == "__main__":
    main()