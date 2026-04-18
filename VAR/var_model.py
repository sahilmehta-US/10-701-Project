import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import VAR
from dataclasses import dataclass

# Configuration
DATA_ROOT = "../data/pipeline_steps"
DATASET_NAME = "step_2_merged_features_dropna.csv"
SPLIT_FILE = "../results/split_definition.json"
TARGET_COL = "Gold Futures (COMEX) | log_return"
MAXLAGS = None
IC = "aic"
TREND = "c"
HORIZONS = [1,5]
OUTPUT_DIR = "var_outputs"
USE_COLS = [
    "Gold Futures (COMEX) | log_return",
    "U.S. Dollar Index | log_return",
    "USD / JPY Exchange Rate | log_return",
    "S&P 500 | log_return",
    "CBOE Volatility Index | diff_1",
    "ICE BofA MOVE Index | diff_1",
    "WTI Crude Oil Futures | log_return",
    "Silver Futures | log_return",
    "Copper Futures | log_return",
    "US 10Y Treasury Yield | diff_1",
    "US 3M Treasury Yield | diff_1",
]
# Data Structures
@dataclass
class SplitRange:
    start_date: pd.Timestamp
    end_date: pd.Timestamp

@dataclass
class DataWrapper:
    frame: pd.DataFrame
    columns: list[str]
    target_col: str
    date_col: str

# Construct Data
def load_json(path):
    with path.open("r") as f:
        return json.load(f)
    
def load_data(path, target):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    cols = []
    for col in df.columns:
        if col == "Date":
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted.astype(float)
            cols.append(col)
    df = df[["Date"] + cols].dropna(how="any").reset_index(drop=True)
    if target not in cols:
        raise ValueError(f"Target column '{target}' not found in data columns: {cols}")
    return DataWrapper(frame=df, columns = cols, target_col = target, date_col = "Date")

def make_split_range(split_range, split):
    split_block = split_range[split]
    return SplitRange(
        start_date = pd.Timestamp(split_block["start_date"]),
        end_date=pd.Timestamp(split_block["end_date"])
    )
    
def filter_by_date(df, date_col, split):
    mask = (df[date_col] >= split.start_date) & (df[date_col] <= split.end_date)
    return df.loc[mask].copy()

def write_summary(path, fitted_var, config):
    with path.open("w") as f:
        f.write("VAR baseline summary\n")
        f.write("=" * 100 + "\n\n")
        f.write(json.dumps(config, indent = 2, default=str))
        f.write('\n\n')
        f.write(str(fitted_var.summary()))        

# Build VAR Model
def choose_maxlags(n_train_rows, n_series, maxlags):
    if maxlags is not None:
        if maxlags < 1:
            raise ValueError("maxlags must be a positive integer")
        return maxlags
    heuristic = min(20, max(2, n_train_rows // (10 * max(n_series, 1))))
    return int(heuristic)

def choose_lags(train, maxlags, ic, trend):
    model = VAR(train)
    order = model.select_order(maxlags = maxlags, trend = trend)
    selected = getattr(order, ic)
    if selected is None or (isinstance(selected, float) and np.isnan(selected)):
        raise ValueError(f"Information criterion {ic} did not return a valid lag order.")
    return max(1, min(int(selected), maxlags))

def fit_var(train, lags, trend):
    model = VAR(train)
    fitted = model.fit(lags, trend = trend)
    if fitted.k_ar != lags:
        raise ValueError(f"Fitted model has {fitted.k_ar} lags, expected {lags}.")
    return fitted

def forecast_var(fitted, full_data, date_series, start_ind, horizon, target_col):
    k_ar = fitted.k_ar
    target_loc = full_data.columns.get_loc(target_col)
    rows = []
    for origin_ind in range(start_ind, len(full_data) - horizon):
        history = full_data.iloc[:origin_ind + 1]
        if len(history) < k_ar:
            continue
        init_vals = history.iloc[-k_ar:].to_numpy()
        forecast = fitted.forecast(y=init_vals, steps = horizon)
        pred = float(forecast[horizon-1, target_loc])
        actual = float(full_data.iloc[origin_ind + horizon, target_loc])
        rows.append({
            "origin_date": pd.Timestamp(date_series.iloc[origin_ind]),
            "target_date": pd.Timestamp(date_series.iloc[origin_ind + horizon]),
            "horizon": horizon,
            "actual": actual,
            "pred": pred,
            "error": actual - pred,
        })
    return pd.DataFrame(rows)

def add_labels(preds, split_ranges):
    preds = preds.copy()
    preds["split"] = "outside"
    for split, split_range in split_ranges.items():
        mask = (preds["target_date"] >= split_range.start_date) & (preds["target_date"] <= split_range.end_date)
        preds.loc[mask, "split"] = split
    return preds

def compute_metrics(preds):
    metrics = []
    for (split, horizon), group in preds.groupby(["split", "horizon"], dropna = False):
        if group.empty or split == "outside":
            continue
        actual = group["actual"].to_numpy()
        pred = group["pred"].to_numpy()
        mae = float(mean_absolute_error(actual, pred))
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        denom = np.where(np.abs(actual) < 1e-12, np.nan, np.abs(actual))
        mape = float(np.nanmean(np.abs(actual - pred) / denom)) if np.isfinite(denom).any() else float("nan")
        directional_acc = float(np.mean(np.sign(actual) == np.sign(pred)))
        metrics.append({
            "split": split,
            "horizon": int(horizon),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_acc": directional_acc,
        })
    return pd.DataFrame(metrics)

def main():
    data_root = Path(DATA_ROOT)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents = True, exist_ok = True)
    csv_path = data_root / DATASET_NAME
    split_path = data_root / SPLIT_FILE
    bundle = load_data(csv_path, TARGET_COL)
    bundle.frame = bundle.frame[["Date"] + USE_COLS]
    bundle.columns = USE_COLS
    splits = load_json(split_path)
    
    train_split = make_split_range(splits, "train")
    val_split = make_split_range(splits, "validation")
    test_split = make_split_range(splits, "test")
    split_ranges = {"train": train_split, "validation": val_split, "test": test_split}
    
    full_data = bundle.frame.copy()
    train_df = filter_by_date(full_data, bundle.date_col, train_split)
    val_df = filter_by_date(full_data, bundle.date_col, val_split)
    test_df = filter_by_date(full_data, bundle.date_col, test_split)
    
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the splits is empty after filtering by date.")
    
    train_endog = train_df[bundle.columns].copy()
    full_endog = full_data[bundle.columns].copy()
    maxlags = choose_maxlags(len(train_endog), train_endog.shape[1], MAXLAGS)
    lags = choose_lags(train_endog, maxlags, IC, TREND)
    fitted = fit_var(train_endog, lags, TREND)
    
    fit_forecast_orig = len(train_df) - 1
    all_preds = []
    for horizon in sorted(set(HORIZONS)):
        preds = forecast_var(fitted, full_endog, full_data[bundle.date_col], fit_forecast_orig, horizon, bundle.target_col)
        preds = add_labels(preds, split_ranges)
        preds.to_csv(output_dir / f"var_predictions_h{horizon}.csv", index = False)
        all_preds.append(preds)
    
    combined_preds = pd.concat(all_preds, axis = 0, ignore_index=True)
    metrics = compute_metrics(combined_preds)
    metrics.to_csv(output_dir / "var_metrics.csv", index = False)
    
    config = {
        "data_root" : DATA_ROOT,
        "dataset_name" : DATASET_NAME,
        "split_file" : SPLIT_FILE,
        "date_column": bundle.date_col,
        "target_column": bundle.target_col,
        "n_endogenous_series": len(bundle.columns),
        "endogenous_columns": bundle.columns,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "lag_order_selection_ic": IC,
        "selected_lag_order": lags,
        "maxlags_searched": maxlags,
        "trend": TREND,
        "horizons": sorted(set(HORIZONS)),
        "nobs_used_for_forecast": fitted.nobs,
    }
    with (output_dir / "var_config.json").open("w") as f:
        json.dump(config, f, indent = 4)
    write_summary(output_dir / 'var_summary.txt', fitted, config)
    print(json.dumps(config, indent = 4))
    print(metrics.to_string(index=False))
    
    
    

if __name__ == "__main__":
    main()
    