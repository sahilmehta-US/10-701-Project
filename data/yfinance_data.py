"""
Download and preprocess Yahoo Finance daily closes into a stationary feature panel.

Pipeline: raw close prices (step 1) → audit → selection → cleaning → stationary transforms
(steps 2–5 under pipeline_steps/) → gold base copies (step 6) → split metadata,
reports, feature dictionary, and results/ aliases (steps 7–11). For Yahoo plus FRED in one
panel, run merged_data.py after this script.
"""
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
        "decision_reason": "Primary gold exposure; downstream models typically use its daily log return as the prediction target.",
    },
    "U.S. Dollar Index": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Keep as the core broad dollar factor in the retained feature set.",
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
        "decision_reason": "Drop to avoid redundancy with S&P 500 and keep the feature set smaller.",
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

STEP_OUTPUT_DIR = "pipeline_steps"
RESULTS_DIR = "results"

SPLIT_DEFINITION = {
    "split_type": "chronological",
    "shuffle": False,
    "applies_to": [
        "gold_base_stationary.csv",
        "gold_base_stationary_dropna.csv",
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
    ],
    "reason": [
        "Scaling must be fit on the training split only.",
        "Global standardization before the split would leak information from validation and test periods.",
        "The preprocessing handoff should be cleaned and transformed, but not globally standardized.",
    ],
    "instruction_for_downstream_use": "Fit any scaler on the training split only, then apply that fitted scaler to validation and test splits.",
}

FINAL_HANDOFF_FILE_ALIASES = {
    "step_6_gold_base_stationary.csv": "gold_base_stationary.csv",
    "step_6_gold_base_stationary_dropna.csv": "gold_base_stationary_dropna.csv",
}

PAIRED_ALTERNATIVES = [
    {
        "pair_name": "Dollar proxy pair",
        "option_a": "U.S. Dollar Index",
        "option_b": "Euro / USD Exchange Rate",
        "chosen_option": "U.S. Dollar Index",
        "paired_alternative_reason": "Both capture closely related dollar strength information, so keeping both adds redundant FX exposure.",
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
        "paired_alternative_reason": "GLD is a near-duplicate gold exposure, so keeping both is usually unnecessary.",
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
        "notes": (
            "Raw close-only file is saved exactly as downloaded before cleaning, stationary transforms, "
            "or any later handoff steps in this script."
        ),
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
                    "Emit both first difference and log return for this volatility or yield-like series; "
                    "use whichever column you need for modeling or merge downstream."
                ),
            }

    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "series"
    return report

def build_base_stationary_output(stationary_df):
    return stationary_df.copy()

def build_base_stationary_dropna_output(base_stationary_df):
    return base_stationary_df.dropna(how="any").copy()

def build_redundancy_check_report():
    report = pd.DataFrame(PAIRED_ALTERNATIVES)
    report["keep_both_in_reduced_feature_set"] = False
    report["caution"] = (
        "Treat as paired alternatives. Keeping both may add redundant information in the feature set."
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

def build_feature_dictionary_row(output_file, column_name):
    parts = column_name.split(" | ")
    source_series = parts[0]
    raw_source_ticker = NAME_TO_TICKER.get(source_series, "")

    if len(parts) == 2:
        transform_name = parts[1]
        transform_used, units = describe_base_transform(transform_name)
        feature_role = "gold" if source_series == "Gold Futures (COMEX)" else "predictor"
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

    raise ValueError(f"Unrecognized feature column format: {column_name}")

def build_feature_dictionary(base_stationary_df, base_stationary_dropna_df):
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
    """Run the full Yahoo pipeline: steps 1–6 in pipeline_steps/, then results/ docs and handoff (7–11)."""
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

    # Step 6. Final stationary output
    # - Save the full stationary panel and a row-wise dropna copy for complete-case modeling.
    base_stationary_df = build_base_stationary_output(stationary_df)
    save_data(base_stationary_df, step_file("step_6_gold_base_stationary.csv"))
    base_stationary_dropna_df = build_base_stationary_dropna_output(base_stationary_df)
    save_data(base_stationary_dropna_df, step_file("step_6_gold_base_stationary_dropna.csv"))

    # INFO: from step 7 onward, outputs go under results/ (no new step_N_* files in pipeline_steps/).
    # Step 7. Chronological split definition
    # - Write results/split_definition.json (train/val/test date ranges for downstream experiments).
    save_json(SPLIT_DEFINITION, result_file("split_definition.json"))

    # Step 8. Redundancy check report
    # - Record paired alternatives such as DXY vs EUR/USD, S&P vs NASDAQ, WTI vs Brent, and Gold vs GLD.
    # - Note that keeping both sides of a pair usually adds redundant exposure.
    redundancy_report = build_redundancy_check_report()
    save_data(redundancy_report, result_file("redundancy_check_report.csv"))

    # Step 9. Feature dictionary
    # - Document each column in the stationary handoff (gold_base_stationary / dropna).
    # - Record source ticker, transform, interpretation, and feature role.
    feature_dictionary = build_feature_dictionary(
        base_stationary_df,
        base_stationary_dropna_df,
    )
    feature_dictionary.to_csv(result_file("feature_dictionary.csv"), index=False)

    # Step 10. No-scaling policy
    # - State explicitly that scaling is not part of preprocessing handoff.
    # - Remind downstream users to fit scaling only on the training split.
    save_json(NO_SCALING_POLICY, result_file("no_scaling_policy.json"))

    # Step 11. Final handoff aliases
    # - Copy step_6_gold_base_stationary*.csv from pipeline_steps/ into results/ as gold_base_stationary*.csv.
    # - Numbered step files stay in pipeline_steps/ as provenance; results/ holds the short names models consume.
    export_final_handoff_aliases()

if __name__ == "__main__":
    main()