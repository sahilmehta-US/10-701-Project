"""
pcmci_discovery.py
------------------
Run PCMCI causal discovery on macro-financial time series data
to identify causal drivers of gold price (GC=F).

Requirements:
    pip install tigramite pandas numpy matplotlib

Usage:
    python pcmci_discovery.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless backend — safe for all environments
import matplotlib.pyplot as plt
import os
import pickle

# ── tigramite ──────────────────────────────────────────────────────────────
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
cond_ind_test = GPDC()
# ══════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these before running
# ══════════════════════════════════════════════════════════════════════════
TARGET_COL   = "Gold Futures (COMEX) | log_return"
MAX_LAG      = 5
PC_ALPHA     = 0.05
MCI_ALPHA    = 0.05
OUTPUT_DIR   = "results/pcmci_output"
# ══════════════════════════════════════════════════════════════════════════


def load_and_preprocess() -> tuple:
    """
    Load the dataset, drop NaNs, and return a (T x N) numpy array
    plus the list of variable names.
    """
    # read from csv you get from running yfinance_data.py
    df = pd.read_csv("../data/results/gold_base_stationary_dropna.csv", parse_dates=["Date"])
    df = df.set_index("Date")

    # Move target column to index 0
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    cols = [TARGET_COL] + [c for c in df.columns if c != TARGET_COL]
    df   = df[cols]

    print(f"[INFO] Dataset shape after preprocessing: {df.shape}")
    print(f"[INFO] Date range: {df.index[0]} → {df.index[-1]}")
    print(f"[INFO] Variables ({len(cols)}): {cols}\n")

    return df.values.astype(float), cols
    
def run_pcmci(data: np.ndarray, var_names: list) -> dict:
    """
    Run PCMCI with ParCorr and return the results dict from tigramite.
    """
    # Wrap data in tigramite's DataFrame object
    dataframe = pp.DataFrame(
        data,
        datatime=np.arange(len(data)),
        var_names=var_names,
    )

    # Conditional independence test: linear partial correlation
    cond_ind_test = ParCorr(significance="analytic")

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=1,
    )

    print("[INFO] Running PCMCI ...")
    results = pcmci.run_pcmci(
        tau_max=MAX_LAG,
        pc_alpha=PC_ALPHA,
        alpha_level=MCI_ALPHA,
    )
    print("[INFO] PCMCI finished.\n")
    return results, pcmci, dataframe


def extract_causal_features(
    results: dict,
    var_names: list,
    target_idx: int = 0,
) -> pd.DataFrame:
    """
    Extract features that have a statistically significant causal link
    to the target variable (gold) at any lag.

    Returns a DataFrame with columns:
        feature | lag | val_mci | p_value
    sorted by p_value ascending.
    """
    p_matrix   = results["p_matrix"]    # shape (N, N, tau_max+1)
    val_matrix = results["val_matrix"]

    records = []
    for j, feat in enumerate(var_names):
        if j == target_idx:
            continue                    # skip self-links
        for lag in range(1, MAX_LAG + 1):
            p   = p_matrix[j, target_idx, lag]
            val = val_matrix[j, target_idx, lag]
            if p < MCI_ALPHA:
                records.append(
                    {"feature": feat, "lag": lag, "val_mci": val, "p_value": p}
                )

    df_causes = (
        pd.DataFrame(records)
        .sort_values("p_value")
        .reset_index(drop=True)
    )
    return df_causes


def save_results(results: dict, causal_df: pd.DataFrame, pcmci, dataframe):
    """Save raw results (pickle), causal feature CSV, and plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Raw PCMCI results (pickle) — useful for later analysis
    pickle_path = os.path.join(OUTPUT_DIR, "pcmci_results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[SAVED] Raw results  → {pickle_path}")

    # 2. Causal features CSV
    csv_path = os.path.join(OUTPUT_DIR, "causal_features.csv")
    causal_df.to_csv(csv_path, index=False)
    print(f"[SAVED] Causal features → {csv_path}")
    print("\nTop causal features for gold price:")
    print(causal_df.to_string(index=False))

    # 3. p-value matrix heatmap
    _plot_pvalue_heatmap(results, dataframe)

    # 4. Significant links graph
    _plot_significant_links(results, dataframe)


def _plot_pvalue_heatmap(results, dataframe):
    """Plot the MCI p-value matrix for lag 1."""
    p_matrix  = results["p_matrix"][:, :, 1]    # lag-1 slice
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
    print(f"[SAVED] p-value heatmap  → {path}")


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
        print(f"[SAVED] Causal graph     → {path}")
    except Exception as e:
        print(f"[WARN] Could not plot causal graph: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    # Step 1: Load & preprocess data
    data, var_names = load_and_preprocess()

    # Step 2: Run PCMCI
    results, pcmci, dataframe = run_pcmci(data, var_names)

    # Step 3: Extract significant causal features for gold
    target_idx = var_names.index(TARGET_COL)
    causal_df  = extract_causal_features(results, var_names, target_idx)

    # Step 4: Save everything
    save_results(results, causal_df, pcmci, dataframe)

    print("\nDone! Check the results/pcmci_output/ folder.")
    return causal_df


if __name__ == "__main__":
    main()