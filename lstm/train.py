"""
train.py: Train and evaluate LSTM models for gold price prediction.

Six architectures are trained, each over several random seeds:
    1. LSTM-all: trained on ALL features (baseline)
    2. LSTM-causal: trained on PCMCI-selected features only
    3. LSTM-granger: trained on Granger-selected features
    4. LSTM-pca: trained on PCA-reduced inputs (dim matched to number of PCMCI-selected features)
    5. LSTM-reg-l1: ALL features, L1 penalty on non-PCMCI input weights
    6. LSTM-reg-l2: ALL features, L2 penalty on non-PCMCI input weights

All six share architecture and hyperparameters for controlled experiment;
the only differences are the input feature set and the regularizer.
Each is trained for every seed in SEEDS(loop through every seed);
metrics are reported per-seed & aggregated (mean +/- std) so model
comparisons can be judged against noise from different seeds.

Two summary styles can be produced at the end of training (controlled
by SUMMARY_STYLE): a pairwise / weight-norm view and a metrics view
(MSE/MAE/RMSE/DirAcc with CSV + JSON dumps). See summarize_pairwise
and summarize_metrics below.

usage: python train.py
"""

import itertools
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

import paths
from dataset import make_dataloaders
from lstm import LSTM


# CONFIG
CSV        = "../data/results/gold_base_stationary_dropna.csv"
SPLIT_JSON = "../data/results/split_definition.json"
TARGET_COL = "Gold Futures (COMEX) | log_return"
SEQ_LEN    = 20
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 6e-4 # tuned
HIDDEN     = 16   # tuned
NUM_LAYERS = 2    # tuned
DROPOUT    = 0.47 # tuned
USE_ATTENTION_GATE = False
ATTENTION_HIDDEN = None   # defaults to HIDDEN when None
ATTENTION_DROPOUT = 0.1
LOSS_SCALE = 1e6   # multiply losses to make them readable
REG_LAMBDA = 1e-4
L2_ALL_REGULARIZATION = "l2_noncausal"
L1_ALL_REGULARIZATION = "l1_noncausal"

PCA_COMPONENTS = 4   # matched to |C| from PCMCI; adjust if PCMCI count changes

# Early stopping: when enabled, stop training a run if val loss has not
# improved for EARLY_STOP_PATIENCE consecutive epochs.
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 10

# Verbose chatter inside run() (per-epoch lines, feature counts, etc.).
# End-of-experiment summary tables in main() always print regardless.
VERBOSE = True

# Which end-of-experiment summary to produce.
#   "pairwise" : per-arch aggregation + pairwise comparisons + weight summary
#   "metrics"  : MSE/MAE/RMSE/DirAcc table + CSV + raw JSON dumps
#   "both"     : print / save both
SUMMARY_STYLE = "both"

CAUSAL_FEATURES_JSON = "../PCMCI/results/pcmci_output/ontrain/selected_features.json"
GRANGER_FEATURES_JSON = "../granger_outputs/granger_selected_features.json"
CHECKPOINT_DIR = paths.CHECKPOINT_DIR
OUTPUT_DIR = "experiment_outputs"   # metrics-style CSV / JSON destination

# Multi-seed training: every experiment is trained once per seed so we
# can report mean +/- std and assess whether model-vs-model gaps exceed
# seed-to-seed variance.
SEEDS = [42, 123, 456, 789, 1024, 1234, 5678, 9101, 10086, 10701]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Reset all random seeds for reproducibility between runs"""
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


def compute_metrics(model, loader):
    """Compute MSE, MAE, RMSE, directional accuracy on a dataloader.

    Used by the "metrics" summary style; run() calls this at the end
    of training so both summary styles can share the same return dict.
    """
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


def plot_losses(train_losses, val_losses, best_epoch, label, filename,
                verbose=True):
    """Plot training curves and save to file.

    filename is used as-is (callers pass paths.losses_png(...)).
    """
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
    if verbose:
        print(f"*Saved* Loss plot -> {filename}")


def plot_aggregate_losses(results, arch_label):
    """Mean +/- std of train / val loss curves across seeds for one arch.

    Per-seed curves remain available in each seed subfolder; this view
    is the one that goes in the report, where seed noise should be a
    band rather than N overlapping lines.
    """
    if not results:
        return None
    # Safeguard: if seeds disagree on #epochs (e.g. early stop), truncate.
    min_epochs = min(len(r["train_losses"]) for r in results)
    # Scale raw MSE by LOSS_SCALE (1e6) so the y-axis matches the report
    # tables and the ~1e-4 validation curve stops being squashed against
    # the x-axis. Log-scaled y further prevents the large first-epoch
    # training spike from dominating the visible range.
    train_curves = np.stack([r["train_losses"][:min_epochs] for r in results]) * LOSS_SCALE
    val_curves = np.stack([r["val_losses"][:min_epochs] for r in results]) * LOSS_SCALE
    epochs = np.arange(1, min_epochs + 1)

    # With only one seed, .std(ddof=1) is undefined — fall back to zero.
    if len(results) > 1:
        tr_mean, tr_std = train_curves.mean(0), train_curves.std(0, ddof=1)
        va_mean, va_std = val_curves.mean(0), val_curves.std(0, ddof=1)
    else:
        tr_mean, tr_std = train_curves[0], np.zeros_like(train_curves[0])
        va_mean, va_std = val_curves[0], np.zeros_like(val_curves[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    # Show individual seeds faintly so outliers are still visible.
    for curves in train_curves:
        ax.plot(epochs, curves, color="steelblue", alpha=0.15, linewidth=0.8)
    for curves in val_curves:
        ax.plot(epochs, curves, color="tomato", alpha=0.15, linewidth=0.8)
    ax.plot(epochs, tr_mean, color="steelblue", linewidth=1.6,
            label=f"Train mean (N={len(results)})")
    # Log-y makes additive bands slightly odd (negative lower bound),
    # so clip to a small positive floor before shading.
    floor = max(1e-2, float(min(va_mean.min(), tr_mean.min()) * 0.5))
    ax.fill_between(epochs, np.maximum(tr_mean - tr_std, floor), tr_mean + tr_std,
                    color="steelblue", alpha=0.25, label="Train +/-1σ")
    ax.plot(epochs, va_mean, color="tomato", linewidth=1.6,
            label=f"Val mean (N={len(results)})")
    ax.fill_between(epochs, np.maximum(va_mean - va_std, floor), va_mean + va_std,
                    color="tomato", alpha=0.25, label="Val +/-1σ")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"MSE (x {LOSS_SCALE:.0e})")
    ax.set_title(f"Train vs Val Loss (aggregated over seeds) — {arch_label}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(paths.aggregate_dir(arch_label),
                            paths.LOSSES_FILENAME)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{arch_label}] Aggregate loss plot -> {out_path}")
    return out_path


def run(csv_file, split_json, seq_len, batch_size, target_col,
        epochs, lr, hidden, num_layers, dropout, use_attn_gate,
        attn_hidden, attn_dropout, loss_scale, feature_cols=None,
        label="LSTM", regularization=None, reg_lambda=REG_LAMBDA,
        causal_cols=None, pca_components=None, seed=42,
        use_early_stop=USE_EARLY_STOP,
        early_stop_patience=EARLY_STOP_PATIENCE,
        verbose=VERBOSE):
    """
    Train an LSTM, evaluate on test set, return results dict.

    Parameters:
    feature_cols : list[str] or None
        If None, use all columns. If a list, use only those columns.
    label : str
        Name for this experiment (used in prints and filenames).
        An attention-gate suffix and a _seed<N> suffix are appended
        automatically — see base_label / arch_label / label in the
        returned dict for the three nesting levels.
    regularization : str or None
        One of None, "l1_noncausal", or "l2_noncausal".
    causal_cols : list[str] or None
        PCMCI-selected columns used to determine which all-feature
        inputs should be regularised.
    pca_components : int or None
        If set, apply PCA to features with this many components.
    seed : int
        RNG seed. Also included in the on-disk label for checkpoint /
        plot filenames so different seeds never overwrite each other.
    use_early_stop : bool
        If True, stop training when val loss has not improved for
        early_stop_patience consecutive epochs.
    early_stop_patience : int
        Epoch patience for early stopping (only used when
        use_early_stop is True).
    verbose : bool
        If True, print per-epoch progress and per-run summary lines.
        End-of-experiment summary tables in main() print regardless.
    """
    # Layered label: base (experiment)  ->  arch (+attn flag)  ->  label (+seed)
    base_label = label
    arch_label = f"{base_label}-attn" if use_attn_gate else base_label
    label = f"{arch_label}_seed{seed}"

    set_seed(seed)

    train_loader, val_loader, test_loader, n_features = make_dataloaders(
        csv_file, split_json, seq_len=seq_len, batch_size=batch_size,
        target_col=target_col, feature_cols=feature_cols,
        pca_components=pca_components,
    )

    if verbose:
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
            raise ValueError("regularization must be one of None, 'l1_noncausal', or 'l2_noncausal'")
        if causal_cols is None:
            raise ValueError("causal_cols must be provided when regularization is enabled")
        reg_type = regularization.split("_", maxsplit=1)[0]
        causal_set = set(causal_cols)
        reg_cols = [col for col in train_loader.dataset.feature_cols if col not in causal_set]
        reg_col_idxs = [idx for idx, col in enumerate(train_loader.dataset.feature_cols)
            if col not in causal_set]

    reg_desc = regularization or "none"
    if verbose:
        print(f"[{label}] Regularization: {reg_desc}")
        if reg_type is not None:
            print(f"[{label}] Penalizing {len(reg_cols)} non-PCMCI features with lambda={reg_lambda}")

    model = LSTM(
        input_size=n_features,
        hidden_size=hidden,
        output_size=1,
        num_layers=num_layers,
        dropout=dropout,
        use_attn_gate=use_attn_gate,
        attn_hidden_size=attn_hidden,
        attn_dropout=attn_dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, optimizer, loss_fn, train_loader, training=True,
            reg_type=reg_type, reg_col_idxs=reg_col_idxs, reg_lambda=reg_lambda,
        )
        val_loss = run_epoch(model, optimizer, loss_fn, val_loader, training=False)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Epoch 1, every 10th, and the final epoch (if not early-stopped).
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"  Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss*loss_scale:7.2f}  "
                f"val={val_loss*loss_scale:7.2f}")

        if use_early_stop and epochs_no_improve >= early_stop_patience:
            if verbose:
                print(f"[{label}] Early stop at epoch {epoch} "
                    f"(best was {best_epoch}, patience={early_stop_patience})")
            break

    # Plot training curves into the per-run folder. The path already
    # encodes arch + seed, so the filename itself is just "losses.png".
    losses_png = paths.losses_png(arch_label, seed)
    os.makedirs(os.path.dirname(losses_png), exist_ok=True)
    plot_losses(train_loss_list, val_loss_list, best_epoch,
                label=label, filename=losses_png, verbose=verbose)

    # Load best checkpoint, then compute both the HEAD-style test_loss
    # (via run_epoch on the test loader) and the metrics-style dict
    # (MSE/MAE/RMSE/DirAcc). Keeping both lets either summary method
    # work without re-training.
    model.load_state_dict(best_state)
    test_loss = run_epoch(model, optimizer, loss_fn, test_loader, training=False)
    val_metrics = compute_metrics(model, val_loader)
    test_metrics = compute_metrics(model, test_loader)
    input_weights = model.lstm.weight_ih_l0.detach().cpu().clone()

    if verbose:
        print(f"[{label}] Best epoch:    {best_epoch}")
        print(f"[{label}] Best val MSE:  {best_val_loss*loss_scale:.2f} (x{1/loss_scale})")
        print(f"[{label}] Test MSE:      {test_loss*loss_scale:.2f} (x{1/loss_scale})")
        print(f"[{label}] Test DirAcc:   {test_metrics['directional_acc']:.4f}")
        print(f"[{label}] Weights:       {tuple(input_weights.shape)}")

    # Persist best checkpoint so downstream scripts (predict.py,
    # recover_prices.py) can run without retraining.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = paths.checkpoint_path(label, root=CHECKPOINT_DIR)
    torch.save(
        {
            "label": label,
            "state_dict": best_state,
            "config": {
                "input_size": n_features,
                "hidden_size": hidden,
                "num_layers": num_layers,
                "dropout": dropout,
                "use_attn_gate": use_attn_gate,
                "attn_hidden_size": attn_hidden,
                "attn_dropout": attn_dropout,
                "seq_len": seq_len,
                "target_col": target_col,
            },
            "feature_cols": train_loader.dataset.feature_cols,
            "csv_file": csv_file,
            "split_json": split_json,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_loss,
            "test_mse": test_loss,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "regularization": reg_desc,
            "reg_lambda": reg_lambda if reg_type is not None else None,
            "pca_components": pca_components,
        },
        ckpt_path,
    )
    if verbose:
        print(f"[{label}] Checkpoint:    {ckpt_path}")

    return {
        "label": label,
        "arch_label": arch_label,
        "base_label": base_label,
        "seed": seed,
        "n_features": n_features,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "test_mse": test_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_cols": train_loader.dataset.feature_cols,
        "regularization": reg_desc,
        "reg_lambda": reg_lambda if reg_type is not None else None,
        "pca_components": pca_components,
        "weights": input_weights,
        "uses_attention_gate": model.attn_gate is not None,
        "checkpoint_path": ckpt_path,
        "train_losses": np.asarray(train_loss_list, dtype=np.float64),
        "val_losses": np.asarray(val_loss_list, dtype=np.float64),
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
    print(f"PCMCI selected {len(causal_cols)} exogenous features:")
    for c in causal_cols:
        print(f"       - {c}")

    # Always include gold's own column for fair comparison
    # (LSTM-all sees it; LSTM-causal should too)
    if TARGET_COL not in causal_cols:
        causal_cols.append(TARGET_COL)
        print(f"       + {TARGET_COL} (added for AR signal)")

    return causal_cols


def load_granger_features():
    """Load Granger-selected features. Filter to columns present in main CSV.

    Mirrors load_causal_features: the Granger sweep produces its own
    feature list which may include columns the LSTM dataset doesn't
    carry, so we intersect against the CSV header before use and always
    append TARGET_COL for the AR signal.
    """
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
    print(f"Granger selected {len(granger_cols)} features " f"(filtered to match LSTM dataset).")
    return granger_cols


def build_experiments(causal_cols, granger_cols):
    """Return the list of experiment configurations to train over seeds.

    Each element is a kwargs dict passed (together with the shared
    hyperparameters) to run(). Keeping configurations declarative lets
    the seed loop and the aggregation code stay generic.
    """
    return [
        {
            "label": "LSTM-all",
            "feature_cols": None,
            "regularization": None,
            "causal_cols": None,
            "pca_components": None,
        },
        {
            "label": "LSTM-causal",
            "feature_cols": causal_cols,
            "regularization": None,
            "causal_cols": None,
            "pca_components": None,
        },
        {
            "label": "LSTM-granger",
            "feature_cols": granger_cols,
            "regularization": None,
            "causal_cols": None,
            "pca_components": None,
        },
        {
            "label": "LSTM-pca",
            "feature_cols": None,
            "regularization": None,
            "causal_cols": None,
            "pca_components": PCA_COMPONENTS,
        },
        {
            "label": "LSTM-reg-l1",
            "feature_cols": None,
            "regularization": L1_ALL_REGULARIZATION,
            "causal_cols": causal_cols,
            "pca_components": None,
        },
        {
            "label": "LSTM-reg-l2",
            "feature_cols": None,
            "regularization": L2_ALL_REGULARIZATION,
            "causal_cols": causal_cols,
            "pca_components": None,
        },
    ]


def aggregate_by_arch(results):
    """Group per-seed results by arch_label and compute mean / std."""
    grouped = {}
    for r in results:
        grouped.setdefault(r["arch_label"], []).append(r)

    agg = {}
    for arch, rs in grouped.items():
        test_mses = np.array([r["test_mse"] for r in rs])
        val_mses = np.array([r["best_val_mse"] for r in rs])
        epochs = np.array([r["best_epoch"] for r in rs])
        agg[arch] = {
            "results": rs,
            "n_seeds": len(rs),
            "test_mean": float(test_mses.mean()),
            "test_std": float(test_mses.std(ddof=1)) if len(test_mses) > 1 else 0.0,
            "val_mean": float(val_mses.mean()),
            "val_std": float(val_mses.std(ddof=1)) if len(val_mses) > 1 else 0.0,
            "epoch_mean": float(epochs.mean()),
            "epoch_std": float(epochs.std(ddof=1)) if len(epochs) > 1 else 0.0,
            "regularization": rs[0]["regularization"],
            "reg_lambda": rs[0]["reg_lambda"],
            "n_features": rs[0]["n_features"],
        }
    return agg


def summarize_pairwise(all_results, causal_cols):
    """HEAD-style summary.

    Prints per-seed + aggregated tables, pairwise architecture comparisons
    with a pooled-std noise band, and an input-weight norm summary split
    by PCMCI / non-PCMCI features.
    """
    # Per-seed table
    print("\n" + "=" * 96)
    print("PER-SEED RESULTS")
    print("=" * 96)
    print(f"{'Architecture':<22} {'seed':>6} {'n_feat':>7} "
        f"{'reg':>14} {'best_epoch':>11} {'Val MSE':>12} {'Test MSE':>12}")
    print("-" * 96)
    for r in all_results:
        print(f"{r['arch_label']:<22} {r['seed']:>6d} {r['n_features']:>7d} "
            f"{r['regularization']:>14} {r['best_epoch']:>11d} "
            f"{r['best_val_mse']*LOSS_SCALE:>12.2f} "
            f"{r['test_mse']*LOSS_SCALE:>12.2f}")

    # Aggregated table
    agg = aggregate_by_arch(all_results)
    print("\n" + "=" * 102)
    print(f"AGGREGATED OVER SEEDS  (N = {len(SEEDS)} seeds per architecture;  "
        f"mean +/- std, scaled by {LOSS_SCALE:.0e})")
    print("=" * 102)
    print(f"{'Architecture':<22} {'n_feat':>7} {'reg':>14} "
        f"{'best_epoch':>14} {'Val MSE':>20} {'Test MSE':>20}")
    print("-" * 102)
    sorted_archs = sorted(agg.items(), key=lambda kv: kv[1]["test_mean"])
    for arch, s in sorted_archs:
        val = f"{s['val_mean']*LOSS_SCALE:7.2f} +/- {s['val_std']*LOSS_SCALE:5.2f}"
        tst = f"{s['test_mean']*LOSS_SCALE:7.2f} +/- {s['test_std']*LOSS_SCALE:5.2f}"
        ep = f"{s['epoch_mean']:5.1f} +/- {s['epoch_std']:4.1f}"
        print(f"{arch:<22} {s['n_features']:>7d} {s['regularization']:>14} "
            f"{ep:>14} {val:>20} {tst:>20}")

    best_arch, best_stats = sorted_archs[0]
    print(f"\n>> Best mean test MSE: {best_arch}  "
        f"({best_stats['test_mean']*LOSS_SCALE:.2f} +/- "
        f"{best_stats['test_std']*LOSS_SCALE:.2f})")

    # ── Pairwise comparisons on aggregated means ──────────────────
    print("\nPAIRWISE AGGREGATED COMPARISONS  (effect size vs seed noise)")
    print("-" * 96)
    for a, b in itertools.combinations(agg.keys(), 2):
        sa, sb = agg[a], agg[b]
        # Put the winner (lower mean test) on the left of the comparison.
        if sb["test_mean"] < sa["test_mean"]:
            a, b = b, a
            sa, sb = sb, sa
        mean_gap = (sb["test_mean"] - sa["test_mean"]) * LOSS_SCALE
        pct = (1 - sa["test_mean"] / sb["test_mean"]) * 100
        # Pooled std as a rough noise band for the gap; with N seeds
        # per side, (std_a^2 + std_b^2)^0.5 is a cheap stand-in for SE.
        noise = np.sqrt(sa["test_std"] ** 2 + sb["test_std"] ** 2) * LOSS_SCALE
        marker = "  (within seed noise)" if mean_gap < noise else ""
        print(f">> {a} beats {b} by {pct:5.2f}%  "
            f"(Δ={mean_gap:6.2f},  noise≈{noise:5.2f}){marker}")

    # Weight summary (averaged over seeds)
    print("\nINPUT WEIGHT SUMMARY  (averaged over seeds)")
    print("-" * 96)
    for arch, s in sorted_archs:
        # Average per-feature column norms across the seeds for this arch.
        stacked = torch.stack([r["weights"].norm(dim=0) for r in s["results"]])
        col_norms_mean = stacked.mean(dim=0)
        col_norms_std = stacked.std(dim=0, unbiased=False)

        feat_cols = s["results"][0]["feature_cols"]
        causal_idxs = [i for i, c in enumerate(feat_cols) if c in causal_cols]
        other_idxs = [i for i, c in enumerate(feat_cols) if c not in causal_cols]

        all_avg = col_norms_mean.mean().item()
        causal_avg = (col_norms_mean[causal_idxs].mean().item() if causal_idxs else 0.0)
        other_avg = (col_norms_mean[other_idxs].mean().item() if other_idxs else 0.0)

        print(f"{arch:<22} avg|w| all={all_avg:.4f}  "
            f"pcmci={causal_avg:.4f}  other={other_avg:.4f}")
        print(f"       - weight shape: {tuple(s['results'][0]['weights'].shape)}  "
            f"(per-col std across seeds: "
            f"mean={col_norms_std.mean().item():.4f})")
        print(f"       - feature split: pcmci={len(causal_idxs)}  " f"other={len(other_idxs)}")

    return agg


def summarize_metrics(all_results, output_dir=OUTPUT_DIR):
    """Metrics-style summary (colleague's version).

    Groups results by arch_label, prints MSE/MAE/RMSE/DirAcc with
    mean +/- std, writes multi_seed_summary.csv, and dumps a raw
    per-seed JSON (minus non-serializable tensors).
    """
    os.makedirs(output_dir, exist_ok=True)

    grouped = {}
    for r in all_results:
        grouped.setdefault(r["arch_label"], []).append(r)

    print("\n" + "=" * 112)
    print(f"METRICS SUMMARY  (N = {len(SEEDS)} seeds per architecture; "
        f"Test MSE scaled by {LOSS_SCALE:.0e})")
    print("=" * 112)
    header = (f"{'Model':<22} {'Features':>8}  "
            f"{'Test MSE':>14}  {'Test MAE':>18}  {'Test RMSE':>18}  "
            f"{'Test DirAcc':>15}  {'Val DirAcc':>15}")
    print(header)
    print("-" * len(header))

    summary_rows = []
    for name, results in grouped.items():
        if not results:
            continue
        mses = np.array([r["test_metrics"]["mse"] * LOSS_SCALE for r in results])
        maes = np.array([r["test_metrics"]["mae"] for r in results])
        rmses = np.array([r["test_metrics"]["rmse"] for r in results])
        diraccs = np.array([r["test_metrics"]["directional_acc"] for r in results])
        val_diraccs = np.array([r["val_metrics"]["directional_acc"] for r in results])
        n_feat = results[0]["n_features"]

        std = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
        row = {
            "model": name,
            "n_features": n_feat,
            "test_mse_mean": float(mses.mean()),
            "test_mse_std": float(std(mses)),
            "test_mae_mean": float(maes.mean()),
            "test_mae_std": float(std(maes)),
            "test_rmse_mean": float(rmses.mean()),
            "test_rmse_std": float(std(rmses)),
            "test_diracc_mean": float(diraccs.mean()),
            "test_diracc_std": float(std(diraccs)),
            "val_diracc_mean": float(val_diraccs.mean()),
            "val_diracc_std": float(std(val_diraccs)),
        }
        summary_rows.append(row)

        print(f"{name:<22} {n_feat:>8}  "
            f"{mses.mean():>7.2f}+/-{std(mses):>5.2f}  "
            f"{maes.mean():>8.5f}+/-{std(maes):>8.5f}  "
            f"{rmses.mean():>8.5f}+/-{std(rmses):>8.5f}  "
            f"{diraccs.mean():>6.3f}+/-{std(diraccs):>6.3f}  "
            f"{val_diraccs.mean():>6.3f}+/-{std(val_diraccs):>6.3f}")

    if not summary_rows:
        return None

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "multi_seed_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n*Saved* {summary_csv}")

    # Raw per-seed JSON dump (strip non-serializable fields)
    raw_json = os.path.join(output_dir, "multi_seed_raw_results.json")
    serializable = {}
    for arch, results in grouped.items():
        serializable[arch] = [
            {
                "label": r["label"],
                "seed": r["seed"],
                "n_features": r["n_features"],
                "best_epoch": r["best_epoch"],
                "val_metrics": r["val_metrics"],
                "test_metrics": r["test_metrics"],
                "regularization": r["regularization"],
                "pca_components": r["pca_components"],
            }
            for r in results
        ]
    with open(raw_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"*Saved* {raw_json}")

    # Best by test MSE / DirAcc
    best_mse = min(summary_rows, key=lambda r: r["test_mse_mean"])
    best_dir = max(summary_rows, key=lambda r: r["test_diracc_mean"])
    print(f"\n>> Best model by mean test MSE:    {best_mse['model']}")
    print(f">> Best model by mean test DirAcc:  {best_dir['model']}")

    return summary_rows


def main():
    causal_cols = load_causal_features()
    granger_cols = load_granger_features()
    experiments = build_experiments(causal_cols, granger_cols)

    # Training sweep
    all_results = []
    for exp_idx, exp in enumerate(experiments, start=1):
        for seed_idx, seed in enumerate(SEEDS, start=1):
            print("\n" + "=" * 78)
            print(
                f"EXPERIMENT {exp_idx}/{len(experiments)} "
                f"({exp['label']})  —  "
                f"seed {seed_idx}/{len(SEEDS)} = {seed}"
            )
            print("=" * 78)
            if exp["regularization"]:
                print(f"Regularization: {exp['regularization']}; "
                    f"Lambda: {REG_LAMBDA}")
            if exp["pca_components"]:
                print(f"PCA components: {exp['pca_components']}")

            r = run(
                CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL,
                EPOCHS, LR, HIDDEN, NUM_LAYERS, DROPOUT, USE_ATTENTION_GATE,
                ATTENTION_HIDDEN, ATTENTION_DROPOUT, LOSS_SCALE,
                feature_cols=exp["feature_cols"],
                label=exp["label"],
                regularization=exp["regularization"],
                reg_lambda=REG_LAMBDA,
                causal_cols=exp["causal_cols"],
                pca_components=exp["pca_components"],
                seed=seed,
                use_early_stop=USE_EARLY_STOP,
                early_stop_patience=EARLY_STOP_PATIENCE,
                verbose=VERBOSE,
            )
            all_results.append(r)

    # Aggregate loss plots (mean +/- std across seeds, per arch)
    # The per-seed losses.png files in each run folder remain the
    # ground truth; these aggregate views go in the report.
    grouped = {}
    for r in all_results:
        grouped.setdefault(r["arch_label"], []).append(r)
    for arch, rs in grouped.items():
        plot_aggregate_losses(rs, arch)

    # End-of-experiment summary(ies)
    if SUMMARY_STYLE not in {"pairwise", "metrics", "both"}:
        raise ValueError(
            f"SUMMARY_STYLE must be 'pairwise', 'metrics', or 'both'; "
            f"got {SUMMARY_STYLE!r}"
        )
    if SUMMARY_STYLE in {"pairwise", "both"}:
        summarize_pairwise(all_results, causal_cols)
    if SUMMARY_STYLE in {"metrics", "both"}:
        summarize_metrics(all_results, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
