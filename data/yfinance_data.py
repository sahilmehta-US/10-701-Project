import json
import os
import shutil
import numpy as np
import pandas as pd
import yfinance as yf

TICKER_TO_NAME = {
    "GC=F": "Gold Futures (COMEX)",
    "DX-Y.NYB": "U.S. Dollar Index",
    "EURUSD=X": "Euro / USD Exchange Rate",
    "JPY=X": "USD / JPY Exchange Rate",
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^VIX": "CBOE Volatility Index",
    "^MOVE": "ICE BofA MOVE Index",
    "CL=F": "WTI Crude Oil Futures",
    "BZ=F": "Brent Crude Oil Futures",
    "SI=F": "Silver Futures",
    "HG=F": "Copper Futures",
    "GLD": "SPDR Gold Shares ETF",
    "^TNX": "US 10Y Treasury Yield",
    "^IRX": "US 3M Treasury Yield",
    "^TYX": "US 30Y Treasury Yield",
}

DOWNLOAD_TICKERS = [
    "GC=F", "DX-Y.NYB", "EURUSD=X", "JPY=X",
    "^GSPC", "^IXIC", "^VIX", "^MOVE",
    "CL=F", "BZ=F", "SI=F", "HG=F", "GLD",
    "^TNX", "^IRX", "^TYX",
]

NAME_TO_TICKER = {name: ticker for ticker, name in TICKER_TO_NAME.items()}

FEATURE_DECISIONS = {
    "Gold Futures (COMEX)": {
        "keep": True,
        "role": "target",
        "decision_reason": "Keep as the target variable for the first causal-discovery dataset.",
    },
    "U.S. Dollar Index": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep as the core broad dollar factor in the smaller causal set.",
    },
    "USD / JPY Exchange Rate": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep one FX series and prefer USD/JPY over EUR/USD to reduce redundancy with the dollar index.",
    },
    "S&P 500": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep one broad equity benchmark and prefer S&P 500 over NASDAQ Composite for a cleaner market factor.",
    },
    "CBOE Volatility Index": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep as the main equity risk and sentiment proxy.",
    },
    "ICE BofA MOVE Index": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep as the rates-volatility channel even though it has more missingness than the other retained series.",
    },
    "WTI Crude Oil Futures": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep the oil factor and prefer WTI over Brent because it starts earlier and avoids duplicate crude exposure.",
    },
    "Silver Futures": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep as a nearby precious-metals signal that is related to, but not a near-duplicate of, gold futures.",
    },
    "Copper Futures": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep as an industrial commodity indicator for growth-sensitive demand.",
    },
    "Brent Crude Oil Futures": {
        "keep": False,
        "role": "drop_or_defer",
        "decision_reason": "Drop because it starts late, adds missingness, and overlaps heavily with WTI crude oil.",
    },
    "SPDR Gold Shares ETF": {
        "keep": False,
        "role": "drop_or_defer",
        "decision_reason": "Drop because it is too close to gold futures and likely adds near-duplicate information.",
    },
    "Euro / USD Exchange Rate": {
        "keep": False,
        "role": "drop_or_defer",
        "decision_reason": "Drop to avoid carrying both DXY and a highly overlapping EUR/USD currency channel.",
    },
    "NASDAQ Composite": {
        "keep": False,
        "role": "drop_or_defer",
        "decision_reason": "Drop to avoid redundancy with S&P 500 and keep the causal feature set smaller.",
    },
    "US 10Y Treasury Yield": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep as the primary real interest rate proxy, a direct theoretical driver of gold prices.",
    },
    "US 3M Treasury Yield": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep to construct yield spread (10Y - 3M) as a recession indicator.",
    },
    "US 30Y Treasury Yield": {
        "keep": False,
        "role": "drop_or_defer",
        "decision_reason": "Drop to avoid redundancy with 10Y yield; 10Y is the standard gold benchmark.",
    },
}

PRICE_LIKE_SERIES = {
    "Gold Futures (COMEX)",
    "U.S. Dollar Index",
    "Euro / USD Exchange Rate",
    "USD / JPY Exchange Rate",
    "S&P 500",
    "NASDAQ Composite",
    "WTI Crude Oil Futures",
    "Brent Crude Oil Futures",
    "Silver Futures",
    "Copper Futures",
    "SPDR Gold Shares ETF",
}

VOLATILITY_SERIES = {
    "CBOE Volatility Index",
    "ICE BofA MOVE Index",
    "US 10Y Treasury Yield",
    "US 3M Treasury Yield",
    "US 30Y Treasury Yield",
}

PREDICTOR_LAGS = (1, 5, 10, 20)

STEP_OUTPUT_DIR = "pipeline_steps"
RESULTS_DIR = "results"

SPLIT_DEFINITION = {
    "split_type": "chronological",
    "shuffle": False,
    "applies_to": [
        "gold_base_stationary.csv",
        "gold_base_stationary_dropna.csv",
        "gold_forecast_features_t1.csv",
        "gold_forecast_features_t5.csv",
    ],
    "train": {
        "start_date": "2006-01-03",
        "end_date": "2018-12-31",
    },
    "validation": {
        "start_date": "2019-01-01",
        "end_date": "2021-12-31",
    },
    "test": {
        "start_date": "2022-01-01",
        "end_date": "2024-12-31",
    },
    "notes": "Use fixed chronological splits for all time-series experiments. Do not randomly split observations.",
}

NO_SCALING_POLICY = {
    "scaling_applied_in_preprocessing_handoff": False,
    "applies_to": [
        "gold_base_stationary.csv",
        "gold_base_stationary_dropna.csv",
        "gold_forecast_features_t1.csv",
        "gold_forecast_features_t5.csv",
    ],
    "reason": [
        "Scaling must be fit on the training split only.",
        "Global standardization before the split would leak information from validation and test periods.",
        "The preprocessing handoff should be cleaned, aligned, transformed, and lagged if needed, but not globally standardized.",
    ],
    "instruction_for_downstream_use": "Fit any scaler on the training split only, then apply that fitted scaler to validation and test splits.",
}

FINAL_HANDOFF_FILE_ALIASES = {
    "step_8_gold_base_stationary.csv": "gold_base_stationary.csv",
    "step_8_gold_base_stationary_dropna.csv": "gold_base_stationary_dropna.csv",
    "step_9_gold_forecast_features_t1.csv": "gold_forecast_features_t1.csv",
    "step_9_gold_forecast_features_t5.csv": "gold_forecast_features_t5.csv",
}

PAIRED_ALTERNATIVES = [
    {
        "pair_name": "Dollar proxy pair",
        "option_a": "U.S. Dollar Index",
        "option_b": "Euro / USD Exchange Rate",
        "chosen_option": "U.S. Dollar Index",
        "paired_alternative_reason": "Both capture closely related dollar strength information, so keeping both may muddy causal interpretation.",
    },
    {
        "pair_name": "Equity index pair",
        "option_a": "S&P 500",
        "option_b": "NASDAQ Composite",
        "chosen_option": "S&P 500",
        "paired_alternative_reason": "Both are broad U.S. equity market proxies, so keeping both may add redundant market exposure.",
    },
    {
        "pair_name": "Crude oil pair",
        "option_a": "WTI Crude Oil Futures",
        "option_b": "Brent Crude Oil Futures",
        "chosen_option": "WTI Crude Oil Futures",
        "paired_alternative_reason": "Both track crude oil conditions, and Brent also starts later and adds missingness.",
    },
    {
        "pair_name": "Gold exposure pair",
        "option_a": "Gold Futures (COMEX)",
        "option_b": "SPDR Gold Shares ETF",
        "chosen_option": "Gold Futures (COMEX)",
        "paired_alternative_reason": "GLD is a near-duplicate gold exposure, so keeping both can muddy interpretation of gold-specific effects.",
    },
]

def save_data(df, filename):
    df.to_csv(filename, index=True)

def download_yfinance_data(start="2006-01-01", end="2025-01-01", interval="1d", rename_columns=True):
    df = yf.download(
        tickers=DOWNLOAD_TICKERS,
        start=start,
        end=end,      # end is exclusive
        interval=interval,
        auto_adjust=True,
        group_by="column",
        threads=False,
        progress=False,
        keepna=False,
        repair=False,
        multi_level_index=True
    )
    if rename_columns and isinstance(df.columns, pd.MultiIndex):
        unique_tickers = df.columns.get_level_values(1).unique()
        new_names = [TICKER_TO_NAME.get(t, t) for t in unique_tickers]
        df.columns = df.columns.set_levels(new_names, level=1)
    return df

def build_raw_download_metadata(start, end, interval):
    return {
        "start_date": start,
        "end_date": end,
        "interval": interval,
        "tickers": DOWNLOAD_TICKERS,
        "auto_adjust": True,
        "end_is_exclusive": True,
        "notes": "Raw close-only file is saved exactly as downloaded before any filling, transformation, lagging, or target construction.",
    }

def filter_by_price_types(df, price_types, flatten=False):
    mask = df.columns.get_level_values(0).isin(price_types)
    result = df.loc[:, mask].copy()
    if flatten and len(price_types) == 1:
        result.columns = result.columns.get_level_values(1)
    return result

def print_statistics(df, price_types, columns):
    """Print range, mean, and other useful statistics for each column."""
    mask = (
        df.columns.get_level_values(0).isin(price_types)
        & df.columns.get_level_values(1).isin(columns)
    )
    filtered = df.loc[:, mask]
    print(filtered.describe().to_string())
    print()

def get_missing_runs(series):
    runs = []
    missing_mask = series.isna().tolist()
    for start_idx, is_missing in enumerate(missing_mask):
        if not is_missing:
            continue
        if start_idx > 0 and missing_mask[start_idx - 1]:
            continue
        end_idx = start_idx
        while end_idx + 1 < len(missing_mask) and missing_mask[end_idx + 1]:
            end_idx += 1
        runs.append({
            "start": series.index[start_idx],
            "end": series.index[end_idx],
            "length": end_idx - start_idx + 1,
        })
    return runs

def is_leading_run(series, run):
    return run["start"] == series.index[0]

def classify_missingness(missing_count, max_gap, leading_gap):
    if missing_count == 0:
        return "No missing values"
    if leading_gap >= 20:
        return "Something more serious"
    if max_gap <= 2:
        return "Holiday-like only"
    return "Something more serious"

def summarize_missingness(series, peer_missing_median):
    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()
    missing_count = int(series.isna().sum())
    runs = get_missing_runs(series)
    max_gap = max((run["length"] for run in runs), default=0)
    leading_gap = runs[0]["length"] if runs and runs[0]["start"] == series.index[0] else 0
    classification = classify_missingness(missing_count, max_gap, leading_gap)

    if classification == "Holiday-like only":
        note = "Missing values are limited to short 1-2 day market-calendar gaps."
    elif leading_gap >= 20:
        note = f"Long leading gap before the series begins ({leading_gap} rows)."
        internal_runs = [run for run in runs if run["start"] != series.index[0]]
        if internal_runs:
            longest_internal_run = max(internal_runs, key=lambda run: run["length"])
            note += (
                " Also has an internal gap of "
                f"{longest_internal_run['length']} rows "
                f"({longest_internal_run['start'].strftime('%Y-%m-%d')} to "
                f"{longest_internal_run['end'].strftime('%Y-%m-%d')})."
            )
    elif missing_count > peer_missing_median:
        longest_run = max(runs, key=lambda run: run["length"])
        note = (
            f"More missing values than the typical series ({missing_count} vs median "
            f"{int(peer_missing_median)}) and a longest gap of {longest_run['length']} rows "
            f"({longest_run['start'].strftime('%Y-%m-%d')} to "
            f"{longest_run['end'].strftime('%Y-%m-%d')})."
        )
    else:
        long_runs = [run for run in runs if run["length"] >= 3]
        if long_runs:
            run_descriptions = ", ".join(
                f"{run['length']} rows ({run['start'].strftime('%Y-%m-%d')} to {run['end'].strftime('%Y-%m-%d')})"
                for run in long_runs
            )
            note = f"Contains multi-day historical gaps beyond holiday-like closures: {run_descriptions}."
        else:
            note = "Missing values are mild overall."

    return {
        "first_valid_date": first_valid.strftime("%Y-%m-%d") if pd.notna(first_valid) else None,
        "last_valid_date": last_valid.strftime("%Y-%m-%d") if pd.notna(last_valid) else None,
        "missing_values": missing_count,
        "max_consecutive_missing_gap": int(max_gap),
        "missing_gap_assessment": classification,
        "audit_note": note,
    }

def build_data_audit_report(df):
    missing_counts = df.isna().sum()
    peer_missing_median = float(missing_counts.median())
    report = pd.DataFrame.from_dict(
        {
            column: summarize_missingness(df[column], peer_missing_median)
            for column in df.columns
        },
        orient="index",
    )
    report.index.name = "series"
    return report

def build_feature_selection_report(audit_report):
    report = pd.DataFrame.from_dict(FEATURE_DECISIONS, orient="index")
    report.index.name = "series"
    report["missing_gap_assessment"] = audit_report["missing_gap_assessment"]
    report["audit_note"] = audit_report["audit_note"]
    return report

def select_feature_set(df, selection_report):
    kept_columns = selection_report.index[selection_report["keep"]].tolist()
    return df.loc[:, kept_columns].copy()

def forward_fill_short_gaps(series, max_gap=2):
    filled = series.copy()
    for run in get_missing_runs(series):
        if is_leading_run(series, run) or run["length"] > max_gap:
            continue
        previous_idx = series.index.get_loc(run["start"]) - 1
        previous_value = series.iloc[previous_idx]
        if pd.isna(previous_value):
            continue
        fill_mask = (filled.index >= run["start"]) & (filled.index <= run["end"])
        filled.loc[fill_mask] = previous_value
    return filled

def clean_feature_set(df, max_forward_fill_gap=2):
    cleaned_df = df.copy()
    for column in cleaned_df.columns:
        cleaned_df[column] = forward_fill_short_gaps(
            cleaned_df[column],
            max_gap=max_forward_fill_gap,
        )
    return cleaned_df

def build_cleaning_report(df, max_forward_fill_gap=2):
    rows = {}
    for column in df.columns:
        series = df[column]
        runs = get_missing_runs(series)
        short_fillable_runs = [
            run for run in runs
            if (not is_leading_run(series, run)) and run["length"] <= max_forward_fill_gap
        ]
        protected_runs = [
            run for run in runs
            if is_leading_run(series, run) or run["length"] > max_forward_fill_gap
        ]

        if column == "ICE BofA MOVE Index":
            reason = (
                "Allow only short forward fill for 1-2 day gaps. Longer MOVE gaps require manual review "
                "because they may not be simple calendar closures, so they are left missing."
            )
        elif protected_runs:
            reason = (
                "Apply only short forward fill for 1-2 day holiday-like gaps. Leave leading or longer gaps "
                "missing to avoid leaking future information into the past."
            )
        else:
            reason = (
                "Only short 1-2 day market-calendar gaps are forward-filled. No backward fill is used."
            )

        rows[column] = {
            "fill_method": f"forward_fill_only_up_to_{max_forward_fill_gap}_rows",
            "short_gaps_filled": len(short_fillable_runs),
            "remaining_unfilled_gaps": len(protected_runs),
            "cleaning_reason": reason,
        }

    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "series"
    return report

def compute_log_return(series):
    previous = series.shift(1)
    valid_mask = (series > 0) & (previous > 0)
    log_return = pd.Series(np.nan, index=series.index, dtype="float64")
    log_return.loc[valid_mask] = np.log(series.loc[valid_mask] / previous.loc[valid_mask])
    return log_return

def compute_first_difference(series):
    return series.diff()

def compute_future_log_return(series, horizon):
    future = series.shift(-horizon)
    valid_mask = (series > 0) & (future > 0)
    future_log_return = pd.Series(np.nan, index=series.index, dtype="float64")
    future_log_return.loc[valid_mask] = np.log(future.loc[valid_mask] / series.loc[valid_mask])
    return future_log_return

def compute_momentum(series, window):
    previous = series.shift(window)
    valid_mask = (series > 0) & (previous > 0)
    momentum = pd.Series(np.nan, index=series.index, dtype="float64")
    momentum.loc[valid_mask] = (series.loc[valid_mask] / previous.loc[valid_mask]) - 1.0
    return momentum

def compute_moving_average_ratio(series, window):
    moving_average = series.rolling(window=window, min_periods=window).mean()
    valid_mask = (series > 0) & (moving_average > 0)
    ma_ratio = pd.Series(np.nan, index=series.index, dtype="float64")
    ma_ratio.loc[valid_mask] = (series.loc[valid_mask] / moving_average.loc[valid_mask]) - 1.0
    return ma_ratio

def compute_rolling_volatility(series, window):
    return compute_log_return(series).rolling(window=window, min_periods=window).std()

def build_stationary_base_dataset(close_df):
    stationary_data = {}

    for column in close_df.columns:
        series = close_df[column]
        if column in PRICE_LIKE_SERIES:
            stationary_data[f"{column} | log_return"] = compute_log_return(series)
        elif column in VOLATILITY_SERIES:
            stationary_data[f"{column} | diff_1"] = compute_first_difference(series)
            stationary_data[f"{column} | log_return"] = compute_log_return(series)

    stationary_df = pd.DataFrame(stationary_data, index=close_df.index)
    return stationary_df

def build_stationary_transform_report(close_df):
    rows = {}
    for column in close_df.columns:
        if column in PRICE_LIKE_SERIES:
            rows[column] = {
                "stationary_columns": f"{column} | log_return",
                "transform_reason": (
                    "Treat as a price-like series and use log returns to create a more stationary base feature."
                ),
            }
        elif column in VOLATILITY_SERIES:
            rows[column] = {
                "stationary_columns": (
                    f"{column} | diff_1; {column} | log_return"
                ),
                "transform_reason": (
                    "Keep both first difference and log return in exploratory processing so the better-behaved "
                    "version can be chosen later."
                ),
            }

    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "series"
    return report

def build_target_dataset(close_df):
    gold_series = close_df["Gold Futures (COMEX)"]
    target_df = pd.DataFrame(
        {
            "Gold Futures (COMEX) | target_log_return_t_plus_1": compute_future_log_return(
                gold_series,
                horizon=1,
            ),
            "Gold Futures (COMEX) | target_log_return_t_plus_5": compute_future_log_return(
                gold_series,
                horizon=5,
            ),
        },
        index=close_df.index,
    )
    return target_df

def build_target_definition_report():
    report = pd.DataFrame(
        [
            {
                "target_column": "Gold Futures (COMEX) | target_log_return_t_plus_1",
                "source_series": "Gold Futures (COMEX)",
                "horizon_days": 1,
                "target_definition": "Log return from day t close to day t+1 close.",
                "alignment": "Predictors are observed at day t and the target is realized after day t.",
            },
            {
                "target_column": "Gold Futures (COMEX) | target_log_return_t_plus_5",
                "source_series": "Gold Futures (COMEX)",
                "horizon_days": 5,
                "target_definition": "Log return from day t close to day t+5 close.",
                "alignment": "Predictors are observed at day t and the target is realized after day t.",
            },
        ]
    )
    return report

def build_rolling_feature_dataset(close_df):
    rolling_data = {}
    for column in close_df.columns:
        rolling_data[f"{column} | momentum_5d | lag_1"] = compute_momentum(close_df[column], window=5).shift(1)
        rolling_data[f"{column} | momentum_10d | lag_1"] = compute_momentum(close_df[column], window=10).shift(1)
        rolling_data[f"{column} | rolling_volatility_20d | lag_1"] = compute_rolling_volatility(
            close_df[column],
            window=20,
        ).shift(1)
        rolling_data[f"{column} | moving_average_ratio_20d | lag_1"] = compute_moving_average_ratio(
            close_df[column],
            window=20,
        ).shift(1)
    rolling_df = pd.DataFrame(rolling_data, index=close_df.index)
    return rolling_df

def build_rolling_feature_report(close_df):
    rows = {}
    for column in close_df.columns:
        rows[column] = {
            "rolling_columns": (
                f"{column} | momentum_5d | lag_1; "
                f"{column} | momentum_10d | lag_1; "
                f"{column} | rolling_volatility_20d | lag_1; "
                f"{column} | moving_average_ratio_20d | lag_1"
            ),
            "rolling_reason": (
                "Add a small disciplined set of rolling forecasting features and lag them by 1 day to avoid leakage."
            ),
        }
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "source_series"
    return report

def build_lagged_predictor_dataset(stationary_df, lags=PREDICTOR_LAGS):
    lagged_data = {}
    for column in stationary_df.columns:
        for lag in lags:
            lagged_data[f"{column} | lag_{lag}"] = stationary_df[column].shift(lag)
    lagged_df = pd.DataFrame(lagged_data, index=stationary_df.index)
    return lagged_df

def build_lagged_predictor_report(stationary_df, lags=PREDICTOR_LAGS):
    rows = {}
    for column in stationary_df.columns:
        lagged_columns = [f"{column} | lag_{lag}" for lag in lags]
        rows[column] = {
            "lagged_columns": "; ".join(lagged_columns),
            "lagging_reason": (
                "Lag predictors before handoff so each row uses only historical information relative to the future target."
            ),
        }
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "source_predictor"
    return report

def build_forecasting_handoff_dataset(lagged_predictor_df, rolling_feature_df, target_df, drop_missing=True):
    forecasting_df = lagged_predictor_df.copy()
    if rolling_feature_df is not None and not rolling_feature_df.empty:
        forecasting_df = forecasting_df.join(rolling_feature_df, how="inner")
    forecasting_df = forecasting_df.join(target_df, how="inner")
    if drop_missing:
        forecasting_df = forecasting_df.dropna(how="any")
    return forecasting_df

def build_base_stationary_output(stationary_df):
    return stationary_df.copy()

def build_base_stationary_dropna_output(base_stationary_df):
    return base_stationary_df.dropna(how="any").copy()

def build_single_target_forecast_output(forecasting_handoff_df, target_column):
    predictor_columns = [
        column for column in forecasting_handoff_df.columns
        if " | target_log_return_t_plus_" not in column
    ]
    return forecasting_handoff_df[predictor_columns + [target_column]].copy()

def build_redundancy_check_report():
    report = pd.DataFrame(PAIRED_ALTERNATIVES)
    report["keep_both_in_causal_version"] = False
    report["caution"] = (
        "Treat as paired alternatives. Keeping both in the causal version may muddy interpretation."
    )
    return report

def build_date_dictionary_row(output_file):
    return {
        "output_file": output_file,
        "column_name": "Date",
        "raw_source_ticker": "",
        "source_series": "Date",
        "human_readable_meaning": "Trading date for the observation row.",
        "transform_used": "None",
        "units_or_interpretation": "Calendar date in YYYY-MM-DD format.",
        "feature_role": "date",
    }

def describe_base_transform(transform_name):
    if transform_name == "log_return":
        return (
            "Daily log return",
            "Log return, approximately the daily percent change in decimal units.",
        )
    if transform_name == "diff_1":
        return (
            "First difference",
            "One-day change in index points.",
        )
    raise ValueError(f"Unsupported base transform: {transform_name}")

def describe_rolling_transform(transform_name):
    if transform_name == "momentum_5d":
        return (
            "5-day momentum",
            "Simple 5-trading-day return in decimal units.",
        )
    if transform_name == "momentum_10d":
        return (
            "10-day momentum",
            "Simple 10-trading-day return in decimal units.",
        )
    if transform_name == "rolling_volatility_20d":
        return (
            "20-day rolling volatility",
            "20-trading-day rolling standard deviation of daily log returns.",
        )
    if transform_name == "moving_average_ratio_20d":
        return (
            "20-day moving-average ratio",
            "Relative deviation from the 20-trading-day moving average, computed as price / moving average - 1.",
        )
    raise ValueError(f"Unsupported rolling transform: {transform_name}")

def build_feature_dictionary_row(output_file, column_name):
    if " | target_log_return_t_plus_" in column_name:
        source_series, target_transform = column_name.split(" | target_log_return_t_plus_")
        horizon = int(target_transform)
        return {
            "output_file": output_file,
            "column_name": column_name,
            "raw_source_ticker": NAME_TO_TICKER.get(source_series, ""),
            "source_series": source_series,
            "human_readable_meaning": f"{horizon}-day-ahead gold log return target measured from day t to day t+{horizon}.",
            "transform_used": f"Future log return over {horizon} trading day(s)",
            "units_or_interpretation": "Forward log return in decimal units; positive values mean gold rose over the forecast horizon.",
            "feature_role": "target",
        }

    parts = column_name.split(" | ")
    source_series = parts[0]
    raw_source_ticker = NAME_TO_TICKER.get(source_series, "")

    if len(parts) == 2:
        transform_name = parts[1]
        transform_used, units = describe_base_transform(transform_name)
        feature_role = "target_series" if source_series == "Gold Futures (COMEX)" else "raw predictor"
        return {
            "output_file": output_file,
            "column_name": column_name,
            "raw_source_ticker": raw_source_ticker,
            "source_series": source_series,
            "human_readable_meaning": f"{transform_used} of {source_series}.",
            "transform_used": transform_used,
            "units_or_interpretation": units,
            "feature_role": feature_role,
        }

    if len(parts) == 3 and parts[2].startswith("lag_"):
        transform_name = parts[1]
        lag_name = parts[2]
        lag_days = int(lag_name.split("_")[1])

        if transform_name in {"log_return", "diff_1"}:
            base_transform, units = describe_base_transform(transform_name)
        else:
            base_transform, units = describe_rolling_transform(transform_name)

        feature_role = "rolling feature" if transform_name.startswith(("momentum", "rolling_volatility", "moving_average_ratio")) else "lagged predictor"
        return {
            "output_file": output_file,
            "column_name": column_name,
            "raw_source_ticker": raw_source_ticker,
            "source_series": source_series,
            "human_readable_meaning": f"{lag_days}-day lag of the {base_transform.lower()} for {source_series}.",
            "transform_used": f"{base_transform}; then lagged by {lag_days} trading day(s)",
            "units_or_interpretation": units,
            "feature_role": feature_role,
        }

    raise ValueError(f"Unrecognized feature column format: {column_name}")

def build_feature_dictionary(base_stationary_df, base_stationary_dropna_df, forecast_feature_t1_df, forecast_feature_t5_df):
    rows = [build_date_dictionary_row("gold_base_stationary.csv")]
    rows.extend(
        build_feature_dictionary_row("gold_base_stationary.csv", column_name)
        for column_name in base_stationary_df.columns
    )
    rows.append(build_date_dictionary_row("gold_base_stationary_dropna.csv"))
    rows.extend(
        build_feature_dictionary_row("gold_base_stationary_dropna.csv", column_name)
        for column_name in base_stationary_dropna_df.columns
    )
    rows.append(build_date_dictionary_row("gold_forecast_features_t1.csv"))
    rows.extend(
        build_feature_dictionary_row("gold_forecast_features_t1.csv", column_name)
        for column_name in forecast_feature_t1_df.columns
    )
    rows.append(build_date_dictionary_row("gold_forecast_features_t5.csv"))
    rows.extend(
        build_feature_dictionary_row("gold_forecast_features_t5.csv", column_name)
        for column_name in forecast_feature_t5_df.columns
    )
    return pd.DataFrame(rows)

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

def ensure_step_output_dir():
    os.makedirs(STEP_OUTPUT_DIR, exist_ok=True)

def step_file(filename):
    ensure_step_output_dir()
    return os.path.join(STEP_OUTPUT_DIR, filename)

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def result_file(filename):
    ensure_results_dir()
    return os.path.join(RESULTS_DIR, filename)

def export_final_handoff_aliases(file_aliases=FINAL_HANDOFF_FILE_ALIASES):
    for source_filename, alias_filename in file_aliases.items():
        shutil.copyfile(step_file(source_filename), result_file(alias_filename))

def main():
    start_date = "2006-01-01"
    end_date = "2025-01-01"
    interval = "1d"

    # Step 1. Raw close prices
    # - Download the original yfinance dataset.
    # - Keep only Close prices.
    # - Save the untouched raw close-price table and source metadata.
    close_df = download_yfinance_data(start=start_date, end=end_date, interval=interval)
    close_df = filter_by_price_types(close_df, price_types=["Close"], flatten=True)
    save_data(close_df, step_file("step_1_raw_close_prices.csv"))
    save_json(
        build_raw_download_metadata(start=start_date, end=end_date, interval=interval),
        step_file("step_1_raw_close_prices_metadata.json"),
    )

    # Step 2. Data inspection report
    # - Audit each raw series before any cleaning.
    # - Record valid-date range, missing-value count, largest gap, and missingness severity.
    audit_report = build_data_audit_report(close_df)
    save_data(audit_report, step_file("step_2_data_audit_report.csv"))

    # Step 3. Feature selection report
    # - Decide which downloaded series to keep and which to drop or defer.
    # - Document the reason for each keep/drop decision.
    selection_report = build_feature_selection_report(audit_report)
    save_data(selection_report, step_file("step_3_feature_selection_report.csv"))
    close_df = select_feature_set(close_df, selection_report)

    # Step 4. Cleaned close-price dataset
    # - Apply conservative cleaning to the selected close-price series.
    # - Allow only short forward fill, never backward fill.
    # - Leave longer or leading gaps missing.
    cleaning_report = build_cleaning_report(close_df, max_forward_fill_gap=2)
    save_data(cleaning_report, step_file("step_4_cleaning_report.csv"))
    cleaned_close_df = clean_feature_set(close_df, max_forward_fill_gap=2)
    save_data(cleaned_close_df, step_file("step_4_cleaned_close_prices.csv"))

    # Step 5. Stationary base dataset
    # - Convert cleaned close-price series into stationary-style features.
    # - Use log returns for price-like series.
    # - Keep both first-difference and log-return variants for VIX and MOVE.
    stationary_report = build_stationary_transform_report(cleaned_close_df)
    save_data(stationary_report, step_file("step_5_stationary_transform_report.csv"))
    stationary_df = build_stationary_base_dataset(cleaned_close_df)
    save_data(stationary_df, step_file("step_5_stationary_base_data.csv"))

    # Step 6. Target definition dataset
    # - Create explicit gold forecasting targets.
    # - Include 1-day-ahead and 5-day-ahead gold log-return targets.
    # - Define the prediction target before building lagged features.
    target_report = build_target_definition_report()
    save_data(target_report, step_file("step_6_target_definition_report.csv"))
    target_df = build_target_dataset(cleaned_close_df)
    save_data(target_df, step_file("step_6_target_data.csv"))

    # Step 7. Lagged forecasting handoff dataset
    # - Lag all stationary predictors by 1, 5, 10, and 20 trading days.
    # - Join those lagged predictors with the future gold targets.
    # - Produce a leak-safe forecasting table without rolling features.
    lagged_predictor_report = build_lagged_predictor_report(stationary_df, lags=PREDICTOR_LAGS)
    save_data(lagged_predictor_report, step_file("step_7_lagged_predictor_report.csv"))
    lagged_predictor_df = build_lagged_predictor_dataset(stationary_df, lags=PREDICTOR_LAGS)
    save_data(lagged_predictor_df, step_file("step_7_lagged_predictor_data.csv"))
    forecasting_handoff_df = build_forecasting_handoff_dataset(
        lagged_predictor_df,
        None,
        target_df,
        drop_missing=True,
    )
    save_data(forecasting_handoff_df, step_file("step_7_forecasting_handoff_data.csv"))

    # Step 8. Final stationary output
    # - Save the main stationary handoff file for causal and exploratory work.
    # - Also save a dropna version for methods that require complete cases.
    base_stationary_df = build_base_stationary_output(stationary_df)
    save_data(base_stationary_df, step_file("step_8_gold_base_stationary.csv"))
    base_stationary_dropna_df = build_base_stationary_dropna_output(base_stationary_df)
    save_data(base_stationary_dropna_df, step_file("step_8_gold_base_stationary_dropna.csv"))

    # Step 9. Rolling forecasting features
    # - Add a small disciplined set of rolling forecasting features:
    #   5-day momentum, 10-day momentum, 20-day rolling volatility, and 20-day moving-average ratio.
    # - Lag those rolling features by 1 day before joining them with lagged predictors and targets.
    # - Save the final forecasting handoff files, including separate t+1 and t+5 target versions.
    rolling_feature_report = build_rolling_feature_report(cleaned_close_df)
    save_data(rolling_feature_report, step_file("step_9_rolling_feature_report.csv"))
    rolling_feature_df = build_rolling_feature_dataset(cleaned_close_df)
    save_data(rolling_feature_df, step_file("step_9_rolling_feature_data.csv"))
    forecasting_handoff_with_rolling_df = build_forecasting_handoff_dataset(
        lagged_predictor_df,
        rolling_feature_df,
        target_df,
        drop_missing=True,
    )
    save_data(forecasting_handoff_with_rolling_df, step_file("step_9_forecasting_handoff_with_rolling_data.csv"))
    save_data(forecasting_handoff_with_rolling_df, step_file("step_9_gold_forecast_features.csv"))
    forecast_feature_t1_df = build_single_target_forecast_output(
        forecasting_handoff_with_rolling_df,
        "Gold Futures (COMEX) | target_log_return_t_plus_1",
    )
    save_data(forecast_feature_t1_df, step_file("step_9_gold_forecast_features_t1.csv"))
    forecast_feature_t5_df = build_single_target_forecast_output(
        forecasting_handoff_with_rolling_df,
        "Gold Futures (COMEX) | target_log_return_t_plus_5",
    )
    save_data(forecast_feature_t5_df, step_file("step_9_gold_forecast_features_t5.csv"))

    # INFO: everything starting from here is for information only, there is no more data processing
    # Step 10. Chronological split definition
    # - Save the fixed train/validation/test boundaries.
    # - Prevent random splitting and keep downstream experiments aligned.
    save_json(SPLIT_DEFINITION, result_file("split_definition.json"))

    # Step 11. Redundancy check report
    # - Record paired alternatives such as DXY vs EUR/USD, S&P vs NASDAQ, WTI vs Brent, and Gold vs GLD.
    # - Warn that keeping both sides of a pair may muddy causal interpretation.
    redundancy_report = build_redundancy_check_report()
    save_data(redundancy_report, result_file("redundancy_check_report.csv"))

    # Step 12. Feature dictionary
    # - Document every final output column.
    # - Record the raw source ticker, meaning, transform, interpretation, and feature role.
    feature_dictionary = build_feature_dictionary(
        base_stationary_df,
        base_stationary_dropna_df,
        forecast_feature_t1_df,
        forecast_feature_t5_df,
    )
    feature_dictionary.to_csv(result_file("feature_dictionary.csv"), index=False)

    # Step 13. No-scaling policy
    # - State explicitly that scaling is not part of preprocessing handoff.
    # - Remind downstream users to fit scaling only on the training split.
    save_json(NO_SCALING_POLICY, result_file("no_scaling_policy.json"))

    # Step 14. Final handoff aliases
    # - Copy the true downstream data files from the step-artifact folder into the results folder using clean names.
    # - Keep the numbered files in the inspection folder as the provenance trail and the results folder as the package to send.
    export_final_handoff_aliases()

if __name__ == "__main__":
    main()