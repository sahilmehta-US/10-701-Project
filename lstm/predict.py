"""
predict.py
----------
Load a trained LSTM checkpoint and dump predicted vs actual gold
log-returns on the test split.

Output:
    lstm/predictions/<label>_returns.csv
    columns: Date, y_true, y_pred

Usage:
    python predict.py                  # runs every checkpoint in lstm/checkpoints/
    python predict.py --label LSTM-all # runs one checkpoint
    python predict.py --checkpoint path/to/file.pt
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import make_splits
from lstm import LSTM

CHECKPOINT_DIR = "checkpoints"
PREDICTIONS_DIR = "predictions"
BATCH_SIZE = 256


def build_model(config):
    """Reconstruct an LSTM with the same architecture used during training."""
    return LSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=1,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_attn_gate=config["use_attn_gate"],
        attn_hidden_size=config["attn_hidden_size"],
        attn_dropout=config["attn_dropout"],
    )


def predict_from_checkpoint(ckpt_path, out_dir=PREDICTIONS_DIR, device=None):
    """Run inference on the test split and write a (Date, y_true, y_pred) CSV."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    label = ckpt["label"]
    config = ckpt["config"]
    feature_cols = ckpt["feature_cols"]

    print(f"[{label}] Loaded {ckpt_path}")
    print(f"[{label}] Best epoch={ckpt['best_epoch']}  "
          f"test_mse={ckpt['test_mse']:.3e}  "
          f"reg={ckpt['regularization']}")

    # Rebuild the exact test split the model was evaluated on. The scaler
    # is refit on the training slice, matching what happened in train.py.
    _, _, test_ds = make_splits(
        ckpt["csv_file"],
        ckpt["split_json"],
        seq_len=config["seq_len"],
        target_col=config["target_col"],
        feature_cols=feature_cols,
    )

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            pred = model(X.to(device)).detach().cpu().numpy()
            preds.append(np.atleast_1d(pred))
    y_pred = np.concatenate(preds)
    y_true = test_ds.y
    dates = pd.to_datetime(test_ds.dates)

    assert len(y_pred) == len(y_true) == len(dates), (
        f"length mismatch: pred={len(y_pred)} true={len(y_true)} dates={len(dates)}"
    )

    df = pd.DataFrame({
        "Date": dates,
        "y_true": y_true.astype(np.float64),
        "y_pred": y_pred.astype(np.float64),
    })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}_returns.csv")
    df.to_csv(out_path, index=False)

    mse = float(((df["y_pred"] - df["y_true"]) ** 2).mean())
    print(f"[{label}] Wrote {len(df)} predictions -> {out_path}")
    print(f"[{label}] Recomputed test MSE = {mse:.3e}  "
          f"(checkpoint reported {ckpt['test_mse']:.3e})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint. Overrides --label.")
    parser.add_argument("--label", type=str, default=None,
                        help="Checkpoint label (e.g. LSTM-all). Loads "
                             "checkpoints/<label>.pt.")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--out-dir", type=str, default=PREDICTIONS_DIR)
    args = parser.parse_args()

    if args.checkpoint is not None:
        paths = [args.checkpoint]
    elif args.label is not None:
        paths = [os.path.join(args.checkpoint_dir, f"{args.label}.pt")]
    else:
        paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
        if not paths:
            raise SystemExit(
                f"No checkpoints found in {args.checkpoint_dir}/. "
                f"Run train.py first."
            )

    for p in paths:
        predict_from_checkpoint(p, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
