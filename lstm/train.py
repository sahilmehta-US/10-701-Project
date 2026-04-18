"""
train.py

Variants (each run as N seeds):
    1. LSTM-all: trained on ALL features (baseline)
    2. LSTM-causal: PCMCI-selected features only
    3. LSTM-granger: Granger-selected features
    4. LSTM-pca: PCA-reduced inputs (matched dim to causal)
    5. LSTM-reg-l1: all features + L1 penalty on non-causal input weights
    6. LSTM-reg-l2: all features + L2 penalty on non-causal input weights
    (decide not to include the below models for paper clarity)
    # 7. LSTM-directional: all features + MSE+directional loss
    # 8. LSTM-directional-causal: PCMCI features + MSE+directional loss

Usage:
    python train.py
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

from dataset import make_dataloaders
from lstm import LSTM

#  CONFIG
CSV        = "../data/results/gold_base_stationary_dropna.csv"
SPLIT_JSON = "../data/results/split_definition.json"
TARGET_COL = "Gold Futures (COMEX) | log_return"
SEQ_LEN    = 20
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 8e-4
HIDDEN     = 32
NUM_LAYERS = 2
DROPOUT    = 0.36
LOSS_SCALE = 1e6 # multiply losses for readable printout
REG_LAMBDA = 1e-4 # regularization strength for reg-l1 / reg-l2

L2_ALL_REGULARIZATION = "l2_noncausal"
L1_ALL_REGULARIZATION = "l1_noncausal"

# DEFAULT_LAMBDA_DIR = 0.3 # directional loss weight
EARLY_STOP_PATIENCE = 10 # epochs without val improvement before stopping
PCA_COMPONENTS = 4 # matched to |C| from PCMCI; adjust if PCMCI count changes

SEEDS = [0, 1, 2, 3, 4]   # multi seed experiment, can also change it to other seeds

CAUSAL_FEATURES_JSON = "../PCMCI/results/pcmci_output/ontrain/selected_features.json"
GRANGER_FEATURES_JSON = "../granger_outputs/granger_selected_features.json"

OUTPUT_DIR = "experiment_outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  REPRODUCIBILITY
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#  LOSS FUNCTIONS
# def directional_mse_loss(pred, target, lambda_dir=0.3):
#     """
#     MSE + soft directional penalty.
#     Uses sigmoid(-pred * target / eps) as a differentiable sign-mismatch indicator.
#     """
#     mse = ((pred - target) ** 2).mean()
#     # sigmoid(-pred*target / scale) is ~1 when signs disagree, ~0 when agree
#     eps = 1e-3
#     sign_mismatch = torch.sigmoid(-pred*target/eps)
#     return mse + lambda_dir * sign_mismatch.mean()


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

#  EPOCH LOOP
def run_epoch(model, optimizer, loss_fn, loader, training,
            reg_type=None, reg_col_idxs=None, reg_lambda=0.0):
    """Run one epoch of training or evaluation."""
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            if training and reg_type is not None:
                loss = loss + compute_regularization_penalty(model, reg_type, reg_col_idxs)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def compute_metrics(model, loader):
    """Compute MSE, MAE, RMSE, directional accuracy on a dataloader."""
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            preds_all.append(pred.cpu().numpy())
            targets_all.append(y.cpu().numpy())
    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)

    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))
    # Directional accuracy: sign agreement (treat zero as positive for stability)
    pred_sign = np.sign(preds)
    targ_sign = np.sign(targets)
    diracc = float(np.mean(pred_sign == targ_sign))

    return {"mse": mse, "mae": mae, "rmse": rmse, "directional_acc": diracc}

def plot_losses(train_losses, val_losses, best_epoch, label, filename):
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()

def run(csv_file, split_json, seq_len, batch_size, target_col,
        epochs, lr, hidden, num_layers, dropout, loss_scale,
        feature_cols=None, label="LSTM",
        regularization=None, reg_lambda=REG_LAMBDA, causal_cols=None,
        pca_components=None, seed=42,
        loss_type="mse", 
        # lambda_dir=DEFAULT_LAMBDA_DIR,
        early_stop=EARLY_STOP_PATIENCE, verbose=True):
    """
    Train an LSTM, evaluate on test set, return results dict with full metrics.

    feature_cols : list[str] or None
        If None, use all columns. If a list, use only those columns.
    regularization : None, "l1_noncausal", "l2_noncausal"
        Soft penalty on first-layer input weights for non-causal features.
    causal_cols : list[str]
        PCMCI-selected columns; required when regularization is set.
    pca_components : int or None
        If set, apply PCA to features with this many components.
    loss_type : "mse" or "directional"
        Training loss.
    lambda_dir : float
        Weight for directional penalty in directional loss.
    early_stop : int
        Patience for early stopping on val loss (epochs with no improvement).
    """
    set_seed(seed)

    train_loader, val_loader, test_loader, n_features = make_dataloaders(
        csv_file, split_json, seq_len=seq_len, batch_size=batch_size,
        target_col=target_col, feature_cols=feature_cols,
        pca_components=pca_components,
    )

    if verbose:
        print(f"[{label} seed={seed}] Features: {n_features}  |  "
                f"Train: {len(train_loader.dataset)}  "
                f"Val: {len(val_loader.dataset)}  "
                f"Test: {len(test_loader.dataset)}")

    # Regularization
    if regularization is None:
        reg_type = None
        reg_col_idxs = []
    else:
        if regularization not in {"l1_noncausal", "l2_noncausal"}:
            raise ValueError(f"Unknown regularization: {regularization}")
        if causal_cols is None:
            raise ValueError("causal_cols required when regularization enabled")
        reg_type = regularization.split("_", maxsplit=1)[0]
        causal_set = set(causal_cols)
        # The dataset exposes feature_cols after PCA / selection
        dataset_feature_cols = train_loader.dataset.feature_cols
        reg_col_idxs = [
            idx for idx, col in enumerate(dataset_feature_cols)
            if col not in causal_set
        ]
        if verbose:
            print(f"[{label}] Regularizing {len(reg_col_idxs)} "
                    f"non-PCMCI features, λ={reg_lambda}")

    # Model
    model = LSTM(input_size=n_features, hidden_size=hidden, output_size=1,
        num_layers=num_layers, dropout=dropout,).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # loss functions
    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    # elif loss_type == "directional":
    #     loss_fn = lambda pred, target: directional_mse_loss(pred, target, lambda_dir)
    #     if verbose:
    #         print(f"[{label}] Using directional MSE loss (λ_dir={lambda_dir})")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(model, optimizer, loss_fn, train_loader,
                            training=True,
                            reg_type=reg_type, reg_col_idxs=reg_col_idxs,
                            reg_lambda=reg_lambda)
        va_loss = run_epoch(model, optimizer, loss_fn, val_loader,
                            training=False)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs}  "
                    f"train={tr_loss*loss_scale:7.2f}  "
                    f"val={va_loss*loss_scale:7.2f}")

        if epochs_no_improve >= early_stop:
            if verbose:
                print(f"[early stop at epoch {epoch}, best was {best_epoch}]")
            break

    # Plot
    plot_losses(train_losses, val_losses, best_epoch,
                label=f"{label} seed={seed}",
                filename=f"losses_{label.lower().replace(' ', '_')}_seed{seed}.png")

    # Load best checkpoint, compute full metrics on val and test
    model.load_state_dict(best_state)
    model.to(device)
    val_metrics = compute_metrics(model, val_loader)
    test_metrics = compute_metrics(model, test_loader)

    if verbose:
        print(f"[{label} seed={seed}] Best epoch: {best_epoch}")
        print(f"  val  MSE={val_metrics['mse']*loss_scale:.2f}  "
            f"DirAcc={val_metrics['directional_acc']:.4f}")
        print(f"  test MSE={test_metrics['mse']*loss_scale:.2f}  "
            f"DirAcc={test_metrics['directional_acc']:.4f}")

    return {
        "label": label,
        "seed": seed,
        "n_features": n_features,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


#  FEATURES
def load_causal_features():
    """Load PCMCI-selected features. Always include TARGET_COL for AR signal."""
    with open(CAUSAL_FEATURES_JSON) as f:
        info = json.load(f)
    causal_cols = list(info["exogenous_features"])
    print(f"[INFO] PCMCI selected {len(causal_cols)} exogenous features:")
    for c in causal_cols:
        print(f"       - {c}")
    if TARGET_COL not in causal_cols:
        causal_cols.append(TARGET_COL)
        print(f"       + {TARGET_COL} (added for AR signal)")
    return causal_cols


def load_granger_features():
    """Load Granger-selected features. Filter to columns present in main CSV."""
    with open(GRANGER_FEATURES_JSON) as f:
        info = json.load(f)
    # Support either JSON schema
    granger_cols = info.get("features") or info.get("selected_features") or []
    granger_cols = list(granger_cols)

    # Filter to columns that exist in the LSTM dataset
    available = pd.read_csv(CSV, nrows=0).columns.tolist()
    granger_cols = [c for c in granger_cols if c in available]

    if TARGET_COL not in granger_cols:
        granger_cols.append(TARGET_COL)
    print(f"Granger selected {len(granger_cols)} features "
        f"(filtered to match LSTM dataset).")
    return granger_cols

#  MAIN: multi-seed, all variants
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load feature sets once
    causal_cols = load_causal_features()
    granger_cols = load_granger_features()

    all_results = {
        "LSTM-all": [],
        "LSTM-causal": [],
        "LSTM-granger": [],
        "LSTM-pca": [],
        "LSTM-reg-l1": [],
        "LSTM-reg-l2": [],
        # "LSTM-directional": [],
        # "LSTM-directional-causal": [],
    }

    shared_input = (CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL,
                EPOCHS, LR, HIDDEN, NUM_LAYERS, DROPOUT, LOSS_SCALE)

    for seed in SEEDS:
        print(f"\n{'#'*10}\n" + f"SEED = {seed}\n" + f"{'#'*10}")

        all_results["LSTM-all"].append(run(*shared_input, feature_cols=None, label="LSTM-all", seed=seed))
        all_results["LSTM-causal"].append(run(*shared_input, feature_cols=causal_cols, label="LSTM-causal", seed=seed))
        all_results["LSTM-granger"].append(run(*shared_input, feature_cols=granger_cols, label="LSTM-granger", seed=seed))
        all_results["LSTM-pca"].append(run(*shared_input, feature_cols=None, pca_components=PCA_COMPONENTS,
            label="LSTM-pca", seed=seed))
        all_results["LSTM-reg-l1"].append(run(*shared_input, feature_cols=None, label="LSTM-reg-l1",
            regularization="l1_noncausal", reg_lambda=REG_LAMBDA,
            causal_cols=causal_cols, seed=seed))
        all_results["LSTM-reg-l2"].append(run(*shared_input, feature_cols=None, label="LSTM-reg-l2",
            regularization="l2_noncausal", reg_lambda=REG_LAMBDA,
            causal_cols=causal_cols, seed=seed))

        # all_results["LSTM-directional"].append(run(
        #     *shared_input, feature_cols=None, label="LSTM-directional",
        #     loss_type="directional", lambda_dir=DEFAULT_LAMBDA_DIR, seed=seed))

        # all_results["LSTM-directional-causal"].append(run(
        #     *shared_input, feature_cols=causal_cols, label="LSTM-directional-causal",
        #     loss_type="directional", lambda_dir=DEFAULT_LAMBDA_DIR, seed=seed))

    #  summary
    print(f"Overall summary for ({len(SEEDS)} seeds)")
    header = (f"{'Model':<26} {'Features':>8}  "
                f"{'Test MSE':>12}  {'Test MAE':>14}  {'Test RMSE':>18}  "
                f"{'Test DirAcc':>15}  {'Val DirAcc':>15}")
    print(header)
    print("-" * len(header))

    summary_rows = []
    for name, results in all_results.items():
        if len(results) == 0:
            continue
        mses = np.array([r["test_metrics"]["mse"] * LOSS_SCALE for r in results])
        maes = np.array([r["test_metrics"]["mae"] for r in results])
        rmses = np.array([r["test_metrics"]["rmse"] for r in results])
        diraccs = np.array([r["test_metrics"]["directional_acc"] for r in results])
        val_diraccs = np.array([r["val_metrics"]["directional_acc"] for r in results])
        n_feat = results[0]["n_features"]

        row = {
            "model": name,
            "n_features": n_feat,
            "test_mse_mean": mses.mean(),   "test_mse_std": mses.std(ddof=1),
            "test_mae_mean": maes.mean(),   "test_mae_std": maes.std(ddof=1),
            "test_rmse_mean": rmses.mean(), "test_rmse_std": rmses.std(ddof=1),
            "test_diracc_mean": diraccs.mean(), "test_diracc_std": diraccs.std(ddof=1),
            "val_diracc_mean": val_diraccs.mean(), "val_diracc_std": val_diraccs.std(ddof=1),
        }
        summary_rows.append(row)

        print(f"{name:<26} {n_feat:>8}  "
                f"{mses.mean():>7.2f}±{mses.std(ddof=1):>4.2f}  "
                f"{maes.mean():>8.5f}±{maes.std(ddof=1):>7.5f}  "
                f"{rmses.mean():>8.5f}±{rmses.std(ddof=1):>7.5f}  "
                f"{diraccs.mean():>6.3f}±{diraccs.std(ddof=1):>6.3f}  "
                f"{val_diraccs.mean():>6.3f}±{val_diraccs.std(ddof=1):>6.3f}")

    # summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "multi_seed_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n{summary_csv}")

    # raw per-seed results record
    raw_json = os.path.join(OUTPUT_DIR, "multi_seed_raw_results.json")
    # strip out non-serializable torch tensors (already stripped — just dicts here)
    with open(raw_json, "w") as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2)
    print(f"{raw_json}")

    # Best by test MSE
    best_mse = min(summary_rows, key=lambda r: r["test_mse_mean"])
    best_dir = max(summary_rows, key=lambda r: r["test_diracc_mean"])
    print(f"\n>> Best model by mean test MSE: {best_mse['model']}")
    print(f">> Best model by mean test DirAcc: {best_dir['model']}")


if __name__ == "__main__":
    main()