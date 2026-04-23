"""
granger_feature_select.py

Granger Causality feature selection for comparison with PCMCI.
Bivaraite causality discovery, not considering confounding variables in
existing features.

Runs pairwise Granger tests on the TRAINING split only, outputs a
feature list in the same JSON format as pcmci_discovery.py so that
train_all.py can swap between them seamlessly.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


#  CONFIG

DATA_CSV    = "data/results/gold_base_stationary_dropna.csv"
SPLIT_JSON  = "data/results/split_definition.json"
TARGET_COL  = "Gold Futures (COMEX) | log_return"
MAX_LAG     = 10
ALPHA       = 0.05
OUTPUT_DIR  = "granger_outputs"

PCMCI_JSON  = "PCMCI/results/pcmci_output/ontrain/selected_features.json"



def load_train_split() -> pd.DataFrame:
    """Load dataset filtered to training period only."""
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    with open(SPLIT_JSON) as f:
        splits = json.load(f)

    start = pd.Timestamp(splits["train"]["start_date"])
    end = pd.Timestamp(splits["train"]["end_date"])
    mask = (df["Date"] >= start) & (df["Date"] <= end)
    train_df = df.loc[mask].copy()

    # Keep only numeric columns (same as dataset.py convention)
    numeric_cols = [TARGET_COL] + [
        c for c in df.columns
        if c not in ("Date", TARGET_COL)
        and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    train_df = train_df[numeric_cols].dropna()

    print(f"Training split: {len(train_df)} rows, "
          f"{len(numeric_cols)} columns")
    print(f"Period: {start.date()} to {end.date()}")
    return train_df


def run_granger_tests(df: pd.DataFrame) -> dict:
    """
    Pairwise Granger causality: X_j -> target for all features.
    
    Side note: Granger is bivariate — it tests X_j vs target's own lags,
    but does not condition on other features. PCMCI's MCI test does
    condition on all causal parents, which is why comparing the two
    feature sets is the core experiment.
    """
    other_cols = [c for c in df.columns if c != TARGET_COL]
    results = []
    selected = []

    print(f"\n{'Feature':<55s} {'Lag':>3s} {'F':>8s} {'p-val':>10s} {'Sig':>4s}")
    print("-" * 85)

    for col in other_cols:
        test_data = df[[TARGET_COL, col]].dropna()
        if len(test_data) < MAX_LAG + 10:
            results.append({"feature": col, "significant": False, "error": "too few rows"})
            continue

        try:
            gc = grangercausalitytests(test_data, maxlag=MAX_LAG, verbose=False)

            best_lag, best_p, best_f = None, 1.0, 0.0
            all_lags = {}
            for lag in range(1, MAX_LAG + 1):
                f_stat = gc[lag][0]["ssr_ftest"][0]
                p_val = gc[lag][0]["ssr_ftest"][1]
                all_lags[lag] = {"f": round(float(f_stat), 4), "p": round(float(p_val), 6)}
                if p_val < best_p:
                    best_p, best_f, best_lag = p_val, f_stat, lag

            sig = bool(best_p < ALPHA)
            marker = "***" if sig else ""
            print(f"  {col:<55s} {best_lag:>3d} {best_f:>8.3f} {best_p:>10.6f} {marker}")

            results.append({
                "feature": col, "best_lag": best_lag,
                "f_stat": round(float(best_f), 4),
                "p_value": round(float(best_p), 6),
                "significant": sig, "all_lags": all_lags,
            })
            if sig:
                selected.append(col)

        except Exception as e:
            print(f"  {col:<55s} ERROR: {e}")
            results.append({"feature": col, "significant": False, "error": str(e)})

    print("-" * 85)
    print(f"\nSignificant: {len(selected)} / {len(other_cols)}")
    return {"full_results": results, "selected_features": selected}


def compare_with_pcmci(granger_feats: list) -> dict | None:
    """Compare Granger vs PCMCI feature sets and print analysis."""
    if not os.path.exists(PCMCI_JSON):
        print(f"\n[WARN] {PCMCI_JSON} not found — run pcmci_discovery.py first")
        return None

    with open(PCMCI_JSON) as f:
        pcmci_data = json.load(f)

    pcmci_set = set(pcmci_data.get("exogenous_features", []))
    granger_set = set(granger_feats)

    overlap = sorted(granger_set & pcmci_set)
    granger_only = sorted(granger_set - pcmci_set)
    pcmci_only = sorted(pcmci_set - granger_set)

    print("\n" + "=" * 70)
    print("GRANGER vs PCMCI COMPARISON")
    print("=" * 70)
    print(f"  Granger:  {len(granger_set)} features")
    print(f"  PCMCI:    {len(pcmci_set)} features")
    print(f"  Overlap:  {len(overlap)}")

    if overlap:
        print("\n  SHARED (robust causal drivers):")
        for f in overlap:
            print(f"    + {f}")
    if granger_only:
        print("\n  GRANGER-ONLY (possibly confounded — no conditioning):")
        for f in granger_only:
            print(f"    ? {f}")
    if pcmci_only:
        print("\n  PCMCI-ONLY (conditional effects):")
        for f in pcmci_only:
            print(f"    ! {f}")

    return {"overlap": overlap, "granger_only": granger_only, "pcmci_only": pcmci_only}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = load_train_split()
    gc = run_granger_tests(train_df)

    # Save in same format as PCMCI for easy swapping in train_all.py
    output = {
        "method": "granger_causality",
        "target": TARGET_COL,
        "max_lag": MAX_LAG,
        "alpha": ALPHA,
        "n_selected": len(gc["selected_features"]),
        "features": gc["selected_features"],
    }
    sel_path = os.path.join(OUTPUT_DIR, "granger_selected_features.json")
    with open(sel_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {sel_path}")

    full_path = os.path.join(OUTPUT_DIR, "granger_full_results.json")
    with open(full_path, "w") as f:
        json.dump(gc["full_results"], f, indent=2)
    print(f"[SAVED] {full_path}")

    comparison = compare_with_pcmci(gc["selected_features"])
    if comparison:
        comp_path = os.path.join(OUTPUT_DIR, "granger_vs_pcmci.json")
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"[SAVED] {comp_path}")


if __name__ == "__main__":
    main()