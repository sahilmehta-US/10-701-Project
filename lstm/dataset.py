import json
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


TARGET_COL = "Gold Futures (COMEX) | log_return"

class GoldBaseStationaryDataset(Dataset):
    """
    Sliding-window dataset for LSTM gold-price-return prediction.

    Each sample is a pair (X, y) where:
      X  — float32 tensor of shape (seq_len, n_features)
           containing the standardised feature values for the
           preceding `seq_len` trading days.
      y  — float32 scalar: the gold log-return on the *next* day.

    Parameters
    ----------
    csv_file  : path to gold_base_stationary_dropna.csv
    seq_len   : look-back window length (default 20 trading days)
    target_col: column name to predict (defaults to gold log-return)
    feature_cols : explicit list of columns to use as features;
                   defaults to every numeric column including the target
    scaler    : fitted StandardScaler; if None a new one is fitted on
                this split's feature data (use the train-split scaler
                when constructing val/test splits)
    """

    def __init__(self, csv_file, seq_len=20, target_col=TARGET_COL, feature_cols=None, scaler=None):
        df = pd.read_csv(csv_file, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != "Date"]

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len

        features = df[feature_cols].values.astype(np.float32)
        targets = df[target_col].values.astype(np.float32)

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(features)
        self.scaler = scaler

        features_scaled = scaler.transform(features).astype(np.float32)

        # Build (X, y) pairs: X = window [t, t+seq_len), y = target at t+seq_len
        self.X = []
        self.y = []
        for i in range(len(features_scaled) - seq_len):
            self.X.append(features_scaled[i : i + seq_len])
            self.y.append(targets[i + seq_len])

        self.X = np.array(self.X, dtype=np.float32)  # (N, seq_len, n_features)
        self.y = np.array(self.y, dtype=np.float32)  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

    @property
    def n_features(self):
        return self.X.shape[2]


# ---------------------------------------------------------------------------
# Factory helper – produces train / val / test splits using date boundaries
# defined in a split_definition.json file (or explicit date strings).
# The StandardScaler is fitted only on the training rows to prevent leakage.
# ---------------------------------------------------------------------------

def make_splits(csv_file, split_json, seq_len=20, target_col=TARGET_COL, feature_cols=None):
    """
    Split the CSV into train / val / test datasets using the date boundaries
    defined in a split_definition.json file.

    The val and test windows each extend back by `seq_len` rows into the
    preceding split so the first LSTM window has a full history, matching
    the same convention used during training.
    """
    with open(split_json) as f:
        spec = json.load(f)

    df = pd.read_csv(csv_file, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    def _rows(start, end):
        mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
        return df.loc[mask].reset_index(drop=True)

    train_df = _rows(spec["train"]["start_date"], spec["train"]["end_date"])
    val_df   = _rows(spec["validation"]["start_date"], spec["validation"]["end_date"])
    test_df  = _rows(spec["test"]["start_date"], spec["test"]["end_date"])

    # Prepend the last `seq_len` rows of the preceding split so the first
    # window of each subsequent split has a complete history.
    raw_val_df = val_df
    val_df  = pd.concat([train_df.iloc[-seq_len:], raw_val_df], ignore_index=True)
    test_df = pd.concat([raw_val_df.iloc[-seq_len:], test_df], ignore_index=True)

    def _write_tmp(dataframe):
        path = csv_file + f".tmp_{id(dataframe)}.csv"
        dataframe.to_csv(path, index=False)
        return path

    train_path = _write_tmp(train_df)
    val_path   = _write_tmp(val_df)
    test_path  = _write_tmp(test_df)

    try:
        train_ds = GoldBaseStationaryDataset(train_path, seq_len=seq_len, target_col=target_col, feature_cols=feature_cols)
        val_ds   = GoldBaseStationaryDataset(val_path,   seq_len=seq_len, target_col=target_col, feature_cols=train_ds.feature_cols, scaler=train_ds.scaler)
        test_ds  = GoldBaseStationaryDataset(test_path,  seq_len=seq_len, target_col=target_col, feature_cols=train_ds.feature_cols, scaler=train_ds.scaler)
    finally:
        for p in (train_path, val_path, test_path):
            os.remove(p)

    return train_ds, val_ds, test_ds


def make_dataloaders(csv_file, split_json, seq_len=20, batch_size=64,
                     target_col=TARGET_COL, feature_cols=None,
                     pca_components=None):
    """
    Convenience wrapper around make_splits that returns DataLoaders directly.

    Training loader has shuffle=False to preserve temporal order.

    Parameters
    ----------
    csv_file     : path to the stationary CSV
    split_json   : path to split_definition.json
    seq_len      : LSTM look-back window length (default 20)
    batch_size   : mini-batch size (default 64)
    target_col   : column to predict
    feature_cols : columns to use as features; defaults to all numeric columns
    pca_components : int or None
        If set, fit a PCA on the TRAINING features only (after StandardScaler),
        and transform val/test with the same fitted PCA. This is the
        leakage-safe protocol: train sees train statistics only.

    Returns
    -------
    train_loader, val_loader, test_loader, n_features
    """
    from torch.utils.data import DataLoader

    train_ds, val_ds, test_ds = make_splits(
        csv_file, split_json, seq_len=seq_len,
        target_col=target_col, feature_cols=feature_cols,
    )

    # ── PCA: fit on train only, transform all splits ────────────────
    if pca_components is not None:
        pca = PCA(n_components=pca_components)

        # Flatten (N, seq_len, n_features) -> (N*seq_len, n_features) for PCA
        n, s, f = train_ds.X.shape
        flat_train = train_ds.X.reshape(-1, f)
        pca.fit(flat_train)

        train_ds.X = pca.transform(flat_train).reshape(n, s, -1).astype(np.float32)

        for ds in [val_ds, test_ds]:
            n2, s2, f2 = ds.X.shape
            ds.X = pca.transform(ds.X.reshape(-1, f2)).reshape(n2, s2, -1).astype(np.float32)

        # Overwrite feature_cols so downstream code doesn't confuse PCA
        # components with original named features.
        pc_names = [f"PC{i+1}" for i in range(pca_components)]
        for ds in [train_ds, val_ds, test_ds]:
            ds.feature_cols = pc_names

        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"[PCA] {f} features -> {pca_components} components "
              f"({explained:.1f}% variance explained)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    n_features = train_ds.X.shape[2]  # correct whether or not PCA was applied
    return train_loader, val_loader, test_loader, n_features