"""
train.py
--------
Train and evaluate LSTM models for gold price prediction.

Runs two experiments:
  1. LSTM-all:    trained on ALL features (baseline)
  2. LSTM-causal: trained on PCMCI-selected features only

Both use the same architecture, hyperparameters, and random seed
so the only difference is the input feature set.

Additional experiment:
  3. LSTM-all-reg: trained on ALL features with regularization on
     features not selected by PCMCI

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
REG_LAMBDA = 1e-4
L2_ALL_REGULARIZATION = "l2_noncausal"
L1_ALL_REGULARIZATION = "l1_noncausal"

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


def compute_regularization_penalty(model, reg_type, reg_col_idxs):
    """Regularise the first-layer input weights tied to selected features."""
    if reg_type is None or not reg_col_idxs:
        return None

    input_weights = model.lstm.weight_ih_l0[:, reg_col_idxs]
    if reg_type == "l1":
        return input_weights.abs().sum()
    if reg_type == "l2":
        return input_weights.pow(2).sum()
    raise ValueError(f"Unsupported regularization type: {reg_type}")


def run_epoch(model, optimizer, loss_fn, loader, training,
              reg_type=None, reg_col_idxs=None, reg_lambda=0.0):
    """Run one epoch of training or evaluation."""
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            base_loss = loss_fn(pred, y)
            loss = base_loss
            if training and reg_type is not None and reg_lambda > 0:
                penalty = compute_regularization_penalty(model, reg_type, reg_col_idxs)
                if penalty is not None:
                    loss = loss + reg_lambda * penalty
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
        feature_cols=None, label="LSTM",
        regularization=None, reg_lambda=REG_LAMBDA,
        causal_cols=None):
    """
    Train an LSTM, evaluate on test set, return results dict.

    Parameters
    ----------
    feature_cols : list[str] or None
        If None, use all columns. If a list, use only those columns.
    label : str
        Name for this experiment (used in prints and filenames).
    regularization : str or None
        One of None, "l1_noncausal", or "l2_noncausal".
    causal_cols : list[str] or None
        PCMCI-selected columns used to determine which all-feature
        inputs should be regularised.
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

    if regularization is None:
        reg_type = None
        reg_col_idxs = []
        reg_cols = []
    else:
        if regularization not in {"l1_noncausal", "l2_noncausal"}:
            raise ValueError(
                "regularization must be one of None, 'l1_noncausal', or 'l2_noncausal'"
            )
        if causal_cols is None:
            raise ValueError("causal_cols must be provided when regularization is enabled")
        reg_type = regularization.split("_", maxsplit=1)[0]
        causal_set = set(causal_cols)
        reg_cols = [col for col in train_loader.dataset.feature_cols if col not in causal_set]
        reg_col_idxs = [
            idx for idx, col in enumerate(train_loader.dataset.feature_cols)
            if col not in causal_set
        ]

    reg_desc = regularization or "none"
    print(f"[{label}] Regularization: {reg_desc}")
    if reg_type is not None:
        print(f"[{label}] Penalizing {len(reg_cols)} non-PCMCI features with lambda={reg_lambda}")

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
        train_loss = run_epoch(
            model, optimizer, loss_fn, train_loader, training=True,
            reg_type=reg_type, reg_col_idxs=reg_col_idxs, reg_lambda=reg_lambda,
        )
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
    input_weights = model.lstm.weight_ih_l0.detach().cpu().clone()

    print(f"[{label}] Best epoch:    {best_epoch}")
    print(f"[{label}] Best val MSE:  {best_val_loss*loss_scale:.2f} (x{1/loss_scale})")
    print(f"[{label}] Test MSE:      {test_loss*loss_scale:.2f} (x{1/loss_scale})")
    print(f"[{label}] Weights:       {tuple(input_weights.shape)}")

    return {
        "label": label,
        "n_features": n_features,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "test_mse": test_loss,
        "feature_cols": train_loader.dataset.feature_cols,
        "regularization": reg_desc,
        "reg_lambda": reg_lambda if reg_type is not None else None,
        "weights": input_weights,
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

    causal_cols = list(info["exogenous_features"])
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

    # ── Experiment 3: LSTM-all with regularization L2 ──────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: LSTM-ALL-REG-L2 (all features, regularize non-PCMCI)")
    print("=" * 70)
    print(f"[INFO] Regularization: {L2_ALL_REGULARIZATION}; Lambda: {REG_LAMBDA}")
    result_all_reg_l2 = run(
        CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL,
        EPOCHS, LR, HIDDEN, NUM_LAYERS, DROPOUT, LOSS_SCALE,
        feature_cols=None,
        label="LSTM-all-reg-l2",
        regularization=L2_ALL_REGULARIZATION,
        reg_lambda=REG_LAMBDA,
        causal_cols=causal_cols,
    )

    # ── Experiment 4: LSTM-all with regularization L1 ──────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: LSTM-ALL-REG-L1 (all features, regularize non-PCMCI)")
    print("=" * 70)
    print(f"[INFO] Regularization: {L1_ALL_REGULARIZATION}; Lambda: {REG_LAMBDA}")
    result_all_reg_l1 = run(
        CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL,
        EPOCHS, LR, HIDDEN, NUM_LAYERS, DROPOUT, LOSS_SCALE,
        feature_cols=None,
        label="LSTM-all-reg-l1",
        regularization=L1_ALL_REGULARIZATION,
        reg_lambda=REG_LAMBDA,
        causal_cols=causal_cols,
    )

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<16} {'Features':>8} {'Reg':>14} {'Lambda':>10} {'Best Epoch':>10} "
          f"{'Val MSE':>12} {'Test MSE':>12}")
    print("-" * 88)
    results = [result_all, result_causal, result_all_reg_l2, result_all_reg_l1]
    for r in results:
        lambda_str = f"{r['reg_lambda']:.0e}" if r['reg_lambda'] is not None else "-"
        print(f"{r['label']:<16} {r['n_features']:>8} {r['regularization']:>14} {lambda_str:>10} {r['best_epoch']:>10} "
              f"{r['best_val_mse']*LOSS_SCALE:>12.2f} {r['test_mse']*LOSS_SCALE:>12.2f}")

    # Which model won?
    best_result = min(results, key=lambda r: r["test_mse"])
    print(f"\n>> Best overall: {best_result['label']} "
          f"with test MSE {best_result['test_mse']*LOSS_SCALE:.2f}")

    print("\nPAIRWISE TEST COMPARISONS")
    print("-" * 76)
    comparisons = [
        (result_all, result_all_reg_l2),
        (result_all, result_all_reg_l1),
        (result_all_reg_l2, result_all_reg_l1),
        (result_all, result_causal),
        (result_all_reg_l2, result_causal),
        (result_all_reg_l1, result_causal),
    ]
    for result_a, result_b in comparisons:
        if np.isclose(result_a["test_mse"], result_b["test_mse"]):
            print(f">> {result_a['label']} ties {result_b['label']} on test MSE")
            continue

        winner, loser = (result_a, result_b)
        if result_b["test_mse"] < result_a["test_mse"]:
            winner, loser = result_b, result_a

        pct = (1 - winner["test_mse"] / loser["test_mse"]) * 100
        diff = (loser["test_mse"] - winner["test_mse"]) * LOSS_SCALE
        print(f">> {winner['label']} beats {loser['label']} by {pct:.1f}% "
              f"on test MSE ({diff:.2f} scaled MSE)")

    print("\nINPUT WEIGHT SUMMARY")
    print("-" * 76)
    for r in results:
        col_norms = r["weights"].norm(dim=0)
        causal_idxs = [i for i, col in enumerate(r["feature_cols"]) if col in causal_cols]
        other_idxs = [i for i, col in enumerate(r["feature_cols"]) if col not in causal_cols]

        all_avg = col_norms.mean().item()
        causal_avg = col_norms[causal_idxs].mean().item() if causal_idxs else 0.0
        other_avg = col_norms[other_idxs].mean().item() if other_idxs else 0.0

        print(f"{r['label']:<16} avg|w| all={all_avg:.4f}  "
              f"pcmci={causal_avg:.4f}  other={other_avg:.4f}")
        print(f"       - weight shape: {tuple(r['weights'].shape)}")
        print(f"       - feature split: pcmci={len(causal_idxs)}  other={len(other_idxs)}")


if __name__ == "__main__":
    main()