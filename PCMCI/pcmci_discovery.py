"""
pcmci_discovery.py
------------------
Run PCMCI causal discovery on macro-financial time series data
to identify causal drivers of gold price (GC=F).

Assumptions:
    1. Linear conditional independence (ParCorr) — only detects linear
    dependencies; nonlinear relationships in financial data are missed.
    2. Approximate stationarity — we assume the input data has been
    transformed (log returns, differencing) to be approximately
    stationary. Volatility clustering and structural breaks may
    still violate this.
    3. Causal sufficiency — no latent confounders. Variables not in the
    dataset (e.g. Fed policy expectations, geopolitical risk) could
    induce spurious edges.
    4. Lag truncation — MAX_LAG bounds the longest causal delay considered.
    Macro effects operating at longer horizons will be missed.
    5. Multiple testing — controlled via Benjamini-Hochberg FDR correction.

Requirements:
    pip install tigramite pandas numpy matplotlib

Usage:
    python pcmci_discovery.py
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import pickle

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

#  CONFIG
DATA_CSV     = "../data/results/gold_base_stationary_dropna.csv"
TARGET_COL   = "Gold Futures (COMEX) | log_return"
MAX_LAG      = 5 # performs better comparing longer lags, appeal to volatility of financial market
PC_ALPHA     = 0.05 # significance for the PC condition-selection step
MCI_ALPHA    = 0.05 # significance for the MCI test step
FDR_METHOD   = "fdr_bh" # Benjamini-Hochberg FDR correction
OUTPUT_DIR   = "results/pcmci_output/ontrain"
SPLIT_FILE = "../data/results/split_definition.json"

# NOTE on target alignment:
# PCMCI discovers the causal structure of the system: X_j(t-tau) -> Gold(t).
# The downstream LSTM may predict Gold(t+5), not Gold(t). This is acceptable
# because causal parents of gold returns are the same variables regardless
# of prediction horizon — only lag importance shifts. We select features
# across all lags 1..MAX_LAG to cover both short and medium-term effects.



def load_and_preprocess() -> tuple:
    """
    Load the dataset, drop NaNs, and return a (T x N) numpy array
    plus the list of variable names.

    The target column is moved to index 0 for convenience when
    extracting causal parents later.
    """
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)
    train_start = pd.Timestamp(splits["train"]["start_date"])
    train_end   = pd.Timestamp(splits["train"]["end_date"])

    before_len = len(df)
    df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
    print(f"Restricted to training split: "
        f"{train_start.date()} → {train_end.date()}")
    print(f"Rows: {before_len} → {len(df)}")
    df = df.set_index("Date")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Move target to index 0
    cols = [TARGET_COL] + [c for c in df.columns if c != TARGET_COL]
    df = df[cols]

    print(f"Dataset shape after preprocessing: {df.shape}")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    print(f"Variables ({len(cols)}):")
    for i, c in enumerate(cols):
        marker = " <- TARGET" if i == 0 else ""
        print(f"       [{i:2d}] {c}{marker}")
    print()

    return df.values.astype(float), cols


def run_pcmci(data: np.ndarray, var_names: list) -> dict:
    """
    Run PCMCI with ParCorr (linear partial correlation) and
    Benjamini-Hochberg FDR correction for multiple testing.

    Returns (results_dict, pcmci_object, tigramite_dataframe).
    """
    dataframe = pp.DataFrame(
        data,
        datatime=np.arange(len(data)),
        var_names=var_names,
    )

    cond_ind_test = ParCorr(significance="analytic")

    pcmci = PCMCI(dataframe=dataframe,
        cond_ind_test=cond_ind_test, verbosity=1,
    )

    results = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=PC_ALPHA,
        alpha_level=MCI_ALPHA,fdr_method=FDR_METHOD,
    )

    print("PCMCI finished\n")
    return results, pcmci, dataframe


def extract_causal_features(results: dict, var_names: list,target_idx: int = 0,
) -> tuple:
    """
    Extract features with a statistically significant causal link
    to the target variable (gold) at any lag

    Returns:
    causal_df : DataFrame with columns [feature, lag, val_mci, pval]
        Exogenous features (other variables -> gold). Sorted by pval.
    self_df : DataFrame with same columns
        Self-links (gold -> gold at various lags). Reported separately
        so downstream code can decide whether to include AR features.
    """
    p_matrix = results["p_matrix"]       # shape (N, N, tau_max+1)
    val_matrix = results["val_matrix"]

    exogenous_records = []
    self_records = []

    for j, feat in enumerate(var_names):
        for lag in range(1, MAX_LAG + 1):
            p = p_matrix[target_idx, j, lag]
            val = val_matrix[target_idx, j, lag]
            if p < MCI_ALPHA:
                record = {
                    "feature": feat,
                    "lag": lag,
                    "val_mci": val,
                    "pval": p,
                }
                if j == target_idx:
                    self_records.append(record)
                else:
                    exogenous_records.append(record)

    # pd.DataFrame([]) produces 0 columns, so .sort_values() would KeyError.
    # Guard with len check before sorting.
    causal_df = pd.DataFrame(exogenous_records)
    if len(causal_df) > 0:
        causal_df = causal_df.sort_values("pval").reset_index(drop=True)

    self_df = pd.DataFrame(self_records)
    if len(self_df) > 0:
        self_df = self_df.sort_values("pval").reset_index(drop=True)

    print(f"Exogenous causal links: {len(causal_df)}")
    print(f"Self-links (AR): {len(self_df)}")

    return causal_df, self_df


def save_results(results, causal_df, self_df, pcmci, dataframe):
    """
    Save:
        1. Raw PCMCI results (pickle)
        2. Causal features CSV (exogenous)
        3. Self-links CSV (autoregressive)
        4. selected_features.json — for downstream LSTM pipeline
        5. p-value heatmap
        6. Causal graph plot
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Raw PCMCI results
    pickle_path = os.path.join(OUTPUT_DIR, "pcmci_results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Raw results saved at {pickle_path}")

    # 2. Causal features CSV (exogenous only)
    csv_path = os.path.join(OUTPUT_DIR, "causal_features.csv")
    causal_df.to_csv(csv_path, index=False)
    print(f"Causal features saved at {csv_path}")

    # 3. Self-links CSV
    self_csv_path = os.path.join(OUTPUT_DIR, "self_links.csv")
    self_df.to_csv(self_csv_path, index=False)
    print(f"Self-links (AR) saved at {self_csv_path}")

    # 4. selected_features.json — the bridge to the LSTM pipeline
    #    Guard .unique() against empty DataFrames
    unique_features = (
        causal_df["feature"].unique().tolist() if len(causal_df) > 0 else []
    )
    feature_info = {
        "target": TARGET_COL,
        "pcmci_config": {
            "max_lag": MAX_LAG,
            "pc_alpha": PC_ALPHA,
            "mci_alpha": MCI_ALPHA,
            "fdr_method": FDR_METHOD,
            "independence_test": "ParCorr (linear)",
        },
        "n_exogenous_features": len(unique_features),
        "exogenous_features": unique_features,
        "has_significant_self_links": len(self_df) > 0,
        "all_causal_links": (
            causal_df.to_dict(orient="records") if len(causal_df) > 0 else []
        ),
        "self_links": (
            self_df.to_dict(orient="records") if len(self_df) > 0 else []
        ),
    }
    json_path = os.path.join(OUTPUT_DIR, "selected_features.json")
    with open(json_path, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"[SAVED] Feature selection saved at {json_path}")

    # Print summary table
    print("\n-- Exogenous causal features for gold --")
    if len(causal_df) > 0:
        print(causal_df.to_string(index=False))
    else:
        print("  (none found -- consider relaxing alpha or increasing MAX_LAG)")

    if len(self_df) > 0:
        print("\n-- Autoregressive (self) links --")
        print(self_df.to_string(index=False))

    # 5. p-value heatmap
    _plot_pvalue_heatmap(results, dataframe)

    # 6. Causal graph
    _plot_significant_links(results, dataframe)


def _plot_pvalue_heatmap(results, dataframe):
    """Plot the MCI p-value matrix for lag 1."""
    p_matrix = results["p_matrix"][:, :, 1]
    var_names = dataframe.var_names
    N = len(var_names)

    fig, ax = plt.subplots(figsize=(max(8, N), max(6, N - 2)))
    im = ax.imshow(p_matrix, vmin=0, vmax=0.1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(var_names, rotation=90, fontsize=8)
    ax.set_yticklabels(var_names, fontsize=8)
    ax.set_title("PCMCI p-value matrix (lag = 1)\ngreen = significant causal link")
    plt.colorbar(im, ax=ax, label="p-value")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pvalue_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] p-value heatmap   saved at {path}")


def _plot_significant_links(results, dataframe):
    """Use tigramite's built-in plotting for the causal graph."""
    try:
        from tigramite import plotting as tp

        tp.plot_graph(
            val_matrix=results["val_matrix"],
            graph=results["graph"],
            var_names=dataframe.var_names,
            link_colorbar_label="MCI val",
            node_colorbar_label="auto-MCI",
            figsize=(12, 8),
        )
        path = os.path.join(OUTPUT_DIR, "causal_graph.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Causal graph saved at {path}")
    except Exception as e:
        print(f"Could not plot causal graph: {e}")



#  MAIN
def main():
    # Load & preprocess
    data, var_names = load_and_preprocess()

    # Run PCMCI with FDR correction
    results, pcmci, dataframe = run_pcmci(data, var_names)

    # Extract significant causal features (exogenous + self-links)
    target_idx = var_names.index(TARGET_COL)
    causal_df, self_df = extract_causal_features(results, var_names, target_idx)

    save_results(results, causal_df, self_df, pcmci, dataframe)

    print("\nCheck the results/pcmci_output_ontrain/ folder.")
    return causal_df, self_df


if __name__ == "__main__":
    main()