"""
train.py
--------
Train and evaluate LSTM models for gold price prediction.

Four architectures are trained, each over several random seeds:
  1. LSTM-all:        trained on ALL features
  2. LSTM-causal:     trained on PCMCI-selected features only
  3. LSTM-all-reg-l2: ALL features, L2 penalty on non-PCMCI input weights
  4. LSTM-all-reg-l1: ALL features, L1 penalty on non-PCMCI input weights

All four share architecture and hyperparameters; the only differences
are the input feature set and the regularizer. Each is trained for
every seed in `SEEDS`; metrics are reported per-seed AND aggregated
(mean ± std) so model-vs-model comparisons can be judged against
seed-to-seed noise.

Usage:
    python train.py
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

import paths
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
USE_ATTENTION_GATE = False
ATTENTION_HIDDEN = None   # defaults to HIDDEN when None
ATTENTION_DROPOUT = 0.1
LOSS_SCALE = 1e6   # multiply losses to make them readable
REG_LAMBDA = 1e-4
L2_ALL_REGULARIZATION = "l2_noncausal"
L1_ALL_REGULARIZATION = "l1_noncausal"

CAUSAL_FEATURES_JSON = "../PCMCI/results/pcmci_output/selected_features.json"
CHECKPOINT_DIR = paths.CHECKPOINT_DIR

# Multi-seed training: every experiment is trained once per seed so we
# can report mean ± std and assess whether model-vs-model gaps exceed
# seed-to-seed variance.
SEEDS = [42, 123, 456, 789, 1024]
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


def plot_aggregate_losses(results, arch_label):
    """Mean ± std of train / val loss curves across seeds for one arch.

    Per-seed curves remain available in each seed subfolder; this view
    is the one that goes in the report, where seed noise should be a
    band rather than N overlapping lines.
    """
    if not results:
        return None
    # Safeguard: if seeds disagree on #epochs (e.g. early stop), truncate.
    min_epochs = min(len(r["train_losses"]) for r in results)
    train_curves = np.stack([r["train_losses"][:min_epochs] for r in results])
    val_curves = np.stack([r["val_losses"][:min_epochs] for r in results])
    epochs = np.arange(1, min_epochs + 1)

    # With only one seed, `.std(ddof=1)` is undefined — fall back to zero.
    if len(results) > 1:
        tr_mean, tr_std = train_curves.mean(0), train_curves.std(0, ddof=1)
        va_mean, va_std = val_curves.mean(0), val_curves.std(0, ddof=1)
    else:
        tr_mean, tr_std = train_curves[0], np.zeros_like(train_curves[0])
        va_mean, va_std = val_curves[0], np.zeros_like(val_curves[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    # Show individual seeds faintly so outliers are still visible.
    for r in results:
        ax.plot(epochs, r["train_losses"][:min_epochs],
                color="steelblue", alpha=0.15, linewidth=0.8)
        ax.plot(epochs, r["val_losses"][:min_epochs],
                color="tomato", alpha=0.15, linewidth=0.8)
    ax.plot(epochs, tr_mean, color="steelblue", linewidth=1.6,
            label=f"Train mean (N={len(results)})")
    ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std,
                    color="steelblue", alpha=0.25, label="Train ±1σ")
    ax.plot(epochs, va_mean, color="tomato", linewidth=1.6,
            label=f"Val mean (N={len(results)})")
    ax.fill_between(epochs, va_mean - va_std, va_mean + va_std,
                    color="tomato", alpha=0.25, label="Val ±1σ")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Train vs Val Loss (aggregated over seeds) — {arch_label}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
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
        causal_cols=None, seed=42):
    """
    Train an LSTM, evaluate on test set, return results dict.

    Parameters
    ----------
    feature_cols : list[str] or None
        If None, use all columns. If a list, use only those columns.
    label : str
        Name for this experiment (used in prints and filenames).
        An attention-gate suffix and a `_seed<N>` suffix are appended
        automatically — see `base_label` / `arch_label` / `label` in the
        returned dict for the three nesting levels.
    regularization : str or None
        One of None, "l1_noncausal", or "l2_noncausal".
    causal_cols : list[str] or None
        PCMCI-selected columns used to determine which all-feature
        inputs should be regularised.
    seed : int
        RNG seed. Also included in the on-disk label for checkpoint /
        plot filenames so different seeds never overwrite each other.
    """
    # Layered label: base (experiment)  →  arch (+attn flag)  →  label (+seed)
    base_label = label
    arch_label = f"{base_label}-attn" if use_attn_gate else base_label
    label = f"{arch_label}_seed{seed}"

    set_seed(seed)

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
        use_attn_gate=use_attn_gate,
        attn_hidden_size=attn_hidden,
        attn_dropout=attn_dropout,
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

    # Plot training curves into the per-run folder. The path already
    # encodes arch + seed, so the filename itself is just "losses.png".
    losses_png = paths.losses_png(arch_label, seed)
    os.makedirs(os.path.dirname(losses_png), exist_ok=True)
    plot_losses(train_loss_list, val_loss_list, best_epoch,
                label=label, filename=losses_png)

    # Evaluate on test set using best checkpoint
    model.load_state_dict(best_state)
    test_loss = run_epoch(model, optimizer, loss_fn, test_loader, training=False)
    input_weights = model.lstm.weight_ih_l0.detach().cpu().clone()

    print(f"[{label}] Best epoch:    {best_epoch}")
    print(f"[{label}] Best val MSE:  {best_val_loss*loss_scale:.2f} (x{1/loss_scale})")
    print(f"[{label}] Test MSE:      {test_loss*loss_scale:.2f} (x{1/loss_scale})")
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
            "regularization": reg_desc,
            "reg_lambda": reg_lambda if reg_type is not None else None,
        },
        ckpt_path,
    )
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
        "feature_cols": train_loader.dataset.feature_cols,
        "regularization": reg_desc,
        "reg_lambda": reg_lambda if reg_type is not None else None,
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
    print(f"[INFO] PCMCI selected {len(causal_cols)} exogenous features:")
    for c in causal_cols:
        print(f"       - {c}")

    # Always include gold's own column for fair comparison
    # (LSTM-all sees it; LSTM-causal should too)
    if TARGET_COL not in causal_cols:
        causal_cols.append(TARGET_COL)
        print(f"       + {TARGET_COL} (added for AR signal)")

    return causal_cols


def build_experiments(causal_cols):
    """Return the list of experiment configurations to train over seeds.

    Each element is a kwargs dict passed (together with the shared
    hyperparameters) to `run()`. Keeping configurations declarative lets
    the seed loop and the aggregation code stay generic.
    """
    return [
        {
            "label": "LSTM-all",
            "feature_cols": None,
            "regularization": None,
            "causal_cols": None,
        },
        {
            "label": "LSTM-causal",
            "feature_cols": causal_cols,
            "regularization": None,
            "causal_cols": None,
        },
        {
            "label": "LSTM-all-reg-l2",
            "feature_cols": None,
            "regularization": L2_ALL_REGULARIZATION,
            "causal_cols": causal_cols,
        },
        {
            "label": "LSTM-all-reg-l1",
            "feature_cols": None,
            "regularization": L1_ALL_REGULARIZATION,
            "causal_cols": causal_cols,
        },
    ]


def aggregate_by_arch(results):
    """Group per-seed results by `arch_label` and compute mean / std."""
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


def main():
    causal_cols = load_causal_features()
    experiments = build_experiments(causal_cols)

    # ── Training sweep ─────────────────────────────────────────────
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
                print(f"[INFO] Regularization: {exp['regularization']}; "
                      f"Lambda: {REG_LAMBDA}")

            r = run(
                CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL,
                EPOCHS, LR, HIDDEN, NUM_LAYERS, DROPOUT, USE_ATTENTION_GATE,
                ATTENTION_HIDDEN, ATTENTION_DROPOUT, LOSS_SCALE,
                feature_cols=exp["feature_cols"],
                label=exp["label"],
                regularization=exp["regularization"],
                reg_lambda=REG_LAMBDA,
                causal_cols=exp["causal_cols"],
                seed=seed,
            )
            all_results.append(r)

    # ── Aggregate loss plots (mean ± std across seeds, per arch) ──
    # The per-seed losses.png files in each run folder remain the
    # ground truth; these aggregate views go in the report.
    grouped = {}
    for r in all_results:
        grouped.setdefault(r["arch_label"], []).append(r)
    for arch, rs in grouped.items():
        plot_aggregate_losses(rs, arch)

    # ── Per-seed table ─────────────────────────────────────────────
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

    # ── Aggregated table ──────────────────────────────────────────
    agg = aggregate_by_arch(all_results)
    print("\n" + "=" * 102)
    print(f"AGGREGATED OVER SEEDS  (N = {len(SEEDS)} seeds per architecture;  "
          f"mean ± std, scaled by {LOSS_SCALE:.0e})")
    print("=" * 102)
    print(f"{'Architecture':<22} {'n_feat':>7} {'reg':>14} "
          f"{'best_epoch':>14} {'Val MSE':>20} {'Test MSE':>20}")
    print("-" * 102)
    # Sort architectures by mean test MSE so the best is on top.
    sorted_archs = sorted(agg.items(), key=lambda kv: kv[1]["test_mean"])
    for arch, s in sorted_archs:
        val = f"{s['val_mean']*LOSS_SCALE:7.2f} ± {s['val_std']*LOSS_SCALE:5.2f}"
        tst = f"{s['test_mean']*LOSS_SCALE:7.2f} ± {s['test_std']*LOSS_SCALE:5.2f}"
        ep = f"{s['epoch_mean']:5.1f} ± {s['epoch_std']:4.1f}"
        print(f"{arch:<22} {s['n_features']:>7d} {s['regularization']:>14} "
              f"{ep:>14} {val:>20} {tst:>20}")

    best_arch, best_stats = sorted_archs[0]
    print(f"\n>> Best mean test MSE: {best_arch}  "
          f"({best_stats['test_mean']*LOSS_SCALE:.2f} ± "
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
        # Pooled std as a rough noise band for the gap; with N=5 seeds
        # per side, (std_a^2 + std_b^2)^0.5 is a cheap stand-in for SE.
        noise = np.sqrt(sa["test_std"] ** 2 + sb["test_std"] ** 2) * LOSS_SCALE
        marker = "  (within seed noise)" if mean_gap < noise else ""
        print(f">> {a} beats {b} by {pct:5.2f}%  "
              f"(Δ={mean_gap:6.2f},  noise≈{noise:5.2f}){marker}")

    # ── Weight summary (averaged over seeds) ──────────────────────
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
        causal_avg = (col_norms_mean[causal_idxs].mean().item()
                      if causal_idxs else 0.0)
        other_avg = (col_norms_mean[other_idxs].mean().item()
                     if other_idxs else 0.0)

        print(f"{arch:<22} avg|w| all={all_avg:.4f}  "
              f"pcmci={causal_avg:.4f}  other={other_avg:.4f}")
        print(f"       - weight shape: {tuple(s['results'][0]['weights'].shape)}  "
              f"(per-col std across seeds: "
              f"mean={col_norms_std.mean().item():.4f})")
        print(f"       - feature split: pcmci={len(causal_idxs)}  "
              f"other={len(other_idxs)}")


if __name__ == "__main__":
    main()