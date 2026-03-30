"""
train.py
--------
Train and evaluate LSTM models for gold price prediction.

Runs two experiments:
  1. LSTM-all:    trained on ALL features (baseline)
  2. LSTM-causal: trained on PCMCI-selected features only

Both use the same architecture, hyperparameters, and random seed
so the only difference is the input feature set.

Usage:
    python train.py
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

from dataset import make_dataloaders
from lstm import LSTM

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
CSV        = "../data/results/gold_base_stationary_dropna.csv"
SPLIT_JSON = "../data/results/split_definition.json"
TARGET_COL = "Gold Futures (COMEX) | log_return"
SEQ_LEN    = 20
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 5e-4
HIDDEN     = 64
NUM_LAYERS = 2
DROPOUT    = 0.2
LOSS_SCALE = 1e6   # multiply losses to make them readable

CAUSAL_FEATURES_JSON = "../PCMCI/results/pcmci_output/selected_features.json"
# ══════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Reset all random seeds for reproducibility between runs."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, optimizer, loss_fn, loader, training):
    """Run one epoch of training or evaluation."""
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def plot_losses(train_losses, val_losses, best_epoch, label, filename):
    """Plot training curves and save to file."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, color="steelblue", label="Train Loss")
    plt.plot(epochs, val_losses, color="tomato", label="Val Loss")
    plt.axvline(x=best_epoch, color="green", linestyle="--", label="Best Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train vs Val Loss — {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[SAVED] Loss plot -> {filename}")


def run(csv_file, split_json, seq_len, batch_size, target_col,
        epochs, lr, hidden, num_layers, dropout, loss_scale,
        feature_cols=None, label="LSTM"):
    """
    Train an LSTM, evaluate on test set, return results dict.

    Parameters
    ----------
    feature_cols : list[str] or None
        If None, use all columns. If a list, use only those columns.
    label : str
        Name for this experiment (used in prints and filenames).
    """
    # Reset seed so both experiments start from identical conditions
    set_seed(42)

    train_loader, val_loader, test_loader, n_features = make_dataloaders(
        csv_file, split_json, seq_len=seq_len, batch_size=batch_size,
        target_col=target_col, feature_cols=feature_cols,
    )

    print(f"[{label}] Features: {n_features}  |  "
          f"Train: {len(train_loader.dataset)}  "
          f"Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    model = LSTM(
        input_size=n_features,
        hidden_size=hidden,
        output_size=1,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, optimizer, loss_fn, train_loader, training=True)
        val_loss = run_epoch(model, optimizer, loss_fn, val_loader, training=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        gap = (val_loss - train_loss) * loss_scale
        marker = " <<" if gap > 0 else ""
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train={train_loss*loss_scale:7.2f}  "
              f"val={val_loss*loss_scale:7.2f}  "
              f"gap={gap:+7.2f}{marker}")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    # Plot training curves
    plot_losses(train_loss_list, val_loss_list, best_epoch,
                label=label,
                filename=f"losses_{label.lower().replace(' ', '_')}.png")

    # Evaluate on test set using best checkpoint
    model.load_state_dict(best_state)
    test_loss = run_epoch(model, optimizer, loss_fn, test_loader, training=False)

    print(f"[{label}] Best epoch: {best_epoch}")
    print(f"[{label}] Best val MSE:  {best_val_loss*loss_scale:.2f} (x{1/loss_scale})")
    print(f"[{label}] Test MSE:      {test_loss*loss_scale:.2f} (x{1/loss_scale})")

    return {
        "label": label,
        "n_features": n_features,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "test_mse": test_loss,
        "feature_cols": feature_cols,
    }


def load_causal_features():
    """
    Read PCMCI selected features from JSON.
    Returns the list of column names to use for LSTM-causal.

    Always includes the target column (gold log_return) so the LSTM
    can see gold's own history in its sliding window — even though
    PCMCI found no significant self-links, the autoregressive signal
    is captured implicitly through the sequence dimension.
    """
    with open(CAUSAL_FEATURES_JSON) as f:
        info = json.load(f)

    causal_cols = info["exogenous_features"]
    print(f"[INFO] PCMCI selected {len(causal_cols)} exogenous features:")
    for c in causal_cols:
        print(f"       - {c}")

    # Always include gold's own column for fair comparison
    # (LSTM-all sees it; LSTM-causal should too)
    if TARGET_COL not in causal_cols:
        causal_cols.append(TARGET_COL)
        print(f"       + {TARGET_COL} (added for AR signal)")

    return causal_cols


def main():
    # ── Experiment 1: LSTM-all (baseline) ────────────────────────────
    print("=" * 70)
    print("EXPERIMENT 1: LSTM-ALL (all features)")
    print("=" * 70)
    result_all = run(
        CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL,
        EPOCHS, LR, HIDDEN, NUM_LAYERS, DROPOUT, LOSS_SCALE,
        feature_cols=None,
        label="LSTM-all",
    )

    # ── Experiment 2: LSTM-causal (PCMCI features) ──────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: LSTM-CAUSAL (PCMCI-selected features)")
    print("=" * 70)
    causal_cols = load_causal_features()
    result_causal = run(
        CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL,
        EPOCHS, LR, HIDDEN, NUM_LAYERS, DROPOUT, LOSS_SCALE,
        feature_cols=causal_cols,
        label="LSTM-causal",
    )

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<16} {'Features':>8} {'Best Epoch':>10} "
          f"{'Val MSE':>12} {'Test MSE':>12}")
    print("-" * 60)
    for r in [result_all, result_causal]:
        print(f"{r['label']:<16} {r['n_features']:>8} {r['best_epoch']:>10} "
              f"{r['best_val_mse']*LOSS_SCALE:>12.2f} {r['test_mse']*LOSS_SCALE:>12.2f}")

    # Which model won?
    if result_causal["test_mse"] < result_all["test_mse"]:
        pct = (1 - result_causal["test_mse"] / result_all["test_mse"]) * 100
        print(f"\n>> LSTM-causal wins by {pct:.1f}% on test MSE")
    else:
        pct = (1 - result_all["test_mse"] / result_causal["test_mse"]) * 100
        print(f"\n>> LSTM-all wins by {pct:.1f}% on test MSE")


if __name__ == "__main__":
    main()