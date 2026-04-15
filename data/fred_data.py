import json
import os
import shutil
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests


FRED_SERIES_TO_NAME: Dict[str, str] = {
    "DGS10": "10Y Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "DGS1": "1Y Treasury Yield",
    "DTB3": "3M Treasury Bill Yield",
    "DFF": "Effective Federal Funds Rate",
    "SOFR": "Secured Overnight Financing Rate",
    "T10YIE": "10Y Breakeven Inflation",
    "DFII10": "10Y Treasury Real Yield",
    "DEXUSEU": "USD / EUR Exchange Rate (FRED)",
    "DEXJPUS": "JPY / USD Exchange Rate (FRED)",
    "DEXCHUS": "CNY / USD Exchange Rate (FRED)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "BAMLC0A4CBBB": "ICE BofA US BBB OAS",
    "DCOILWTICO": "WTI Spot Price (FRED)",
}

DOWNLOAD_SERIES: List[str] = list(FRED_SERIES_TO_NAME.keys())
NAME_TO_SERIES_ID: Dict[str, str] = {name: sid for sid, name in FRED_SERIES_TO_NAME.items()}

SERIES_METADATA: Dict[str, Dict[str, str]] = {
    "10Y Treasury Yield": {
        "category": "rates",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "2Y Treasury Yield": {
        "category": "rates",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "1Y Treasury Yield": {
        "category": "rates",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "3M Treasury Bill Yield": {
        "category": "rates",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "Effective Federal Funds Rate": {
        "category": "policy_rates",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "Secured Overnight Financing Rate": {
        "category": "policy_rates",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "10Y Breakeven Inflation": {
        "category": "inflation_expectations",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "10Y Treasury Real Yield": {
        "category": "real_rates",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "USD / EUR Exchange Rate (FRED)": {
        "category": "fx",
        "units": "usd_per_eur",
        "frequency": "daily",
        "transform_hint": "level_and_log_return",
    },
    "JPY / USD Exchange Rate (FRED)": {
        "category": "fx",
        "units": "jpy_per_usd",
        "frequency": "daily",
        "transform_hint": "level_and_log_return",
    },
    "CNY / USD Exchange Rate (FRED)": {
        "category": "fx",
        "units": "cny_per_usd",
        "frequency": "daily",
        "transform_hint": "level_and_log_return",
    },
    "ICE BofA US High Yield OAS": {
        "category": "credit",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "ICE BofA US BBB OAS": {
        "category": "credit",
        "units": "percent",
        "frequency": "daily",
        "transform_hint": "level_and_diff",
    },
    "WTI Spot Price (FRED)": {
        "category": "commodities",
        "units": "usd_per_barrel",
        "frequency": "daily",
        "transform_hint": "level_and_log_return",
    },
}

FEATURE_DECISIONS = {
    "10Y Treasury Yield": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Core long-end rates series for gold, real-rates, and macro-financial channel analysis.",
    },
    "2Y Treasury Yield": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Shorter-horizon Treasury rate used directly and for term-spread construction.",
    },
    "1Y Treasury Yield": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Useful intermediate point on the curve for additional rates-shape information.",
    },
    "3M Treasury Bill Yield": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Short-end cash-like yield that supports curve and policy-spread features.",
    },
    "Effective Federal Funds Rate": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Canonical policy-rate proxy from FRED for monetary-policy conditions.",
    },
    "Secured Overnight Financing Rate": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Modern overnight funding benchmark that complements the Fed funds rate.",
    },
    "10Y Breakeven Inflation": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Market-implied inflation expectations are plausible drivers of gold demand and pricing.",
    },
    "10Y Treasury Real Yield": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Real yields are especially important in many economic narratives for gold prices.",
    },
    "USD / EUR Exchange Rate (FRED)": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Adds a liquid FX channel from FRED without depending on yfinance yet.",
    },
    "JPY / USD Exchange Rate (FRED)": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Provides an additional major-currency market channel relevant to risk and safe-haven flows.",
    },
    "CNY / USD Exchange Rate (FRED)": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Adds an Asia-linked FX channel that may matter for commodity and global demand narratives.",
    },
    "ICE BofA US High Yield OAS": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Captures risk-premium conditions in lower-quality credit markets.",
    },
    "ICE BofA US BBB OAS": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Captures broader investment-grade credit-spread stress.",
    },
    "WTI Spot Price (FRED)": {
        "keep": True,
        "role": "feature",
        "decision_reason": "Useful macro commodity control directly from FRED when yfinance oil is not yet merged.",
    },
}

STEP_OUTPUT_DIR = "pipeline_steps"
RESULTS_DIR = "results"

PREDICTOR_LAGS = (1, 5, 10, 20)

FINAL_HANDOFF_FILE_ALIASES = {
    "step_4_fred_cleaned_series.csv": "fred_base_daily.csv",
    "step_4_fred_cleaned_series_dropna.csv": "fred_base_daily_dropna.csv",
    "step_5_fred_derived_features.csv": "fred_derived_features.csv",
    "step_5_fred_derived_features_dropna.csv": "fred_derived_features_dropna.csv",
    "step_6_fred_lagged_features.csv": "fred_lagged_features.csv",
    "step_6_fred_lagged_features_dropna.csv": "fred_lagged_features_dropna.csv",
}


# -----------------------------
# Environment and IO helpers
# -----------------------------

def load_env_file(env_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ if absent."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_fred_api_key(env_path: str = ".env") -> str:
    load_env_file(env_path=env_path)
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY was not found. Put FRED_API_KEY=... in your .env file or export it in the environment."
        )
    return api_key


def ensure_step_output_dir() -> None:
    os.makedirs(STEP_OUTPUT_DIR, exist_ok=True)


def step_file(filename: str) -> str:
    ensure_step_output_dir()
    return os.path.join(STEP_OUTPUT_DIR, filename)


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def result_file(filename: str) -> str:
    ensure_results_dir()
    return os.path.join(RESULTS_DIR, filename)


def save_data(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, index=True)


def save_json(data: dict, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def export_final_handoff_aliases(file_aliases: Dict[str, str] = FINAL_HANDOFF_FILE_ALIASES) -> None:
    for source_filename, alias_filename in file_aliases.items():
        shutil.copyfile(step_file(source_filename), result_file(alias_filename))


# -----------------------------
# FRED download logic
# -----------------------------

def fetch_fred_series(
    series_id: str,
    api_key: str,
    start_date: str,
    end_date: str,
    max_retries: int = 5,
    backoff_seconds: float = 1.5,
) -> pd.Series:
    """Download one FRED series as a pandas Series indexed by date, with retry/backoff."""
    import time

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()

            if "observations" not in payload:
                raise RuntimeError(
                    f"Unexpected FRED response for {series_id}: missing 'observations'."
                )

            records = []
            for row in payload["observations"]:
                value = row.get("value", ".")
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    numeric_value = np.nan
                records.append((pd.Timestamp(row["date"]), numeric_value))

            if not records:
                return pd.Series(
                    dtype="float64",
                    name=FRED_SERIES_TO_NAME.get(series_id, series_id),
                )

            index = pd.DatetimeIndex([date for date, _ in records], name="Date")
            values = [value for _, value in records]
            return pd.Series(
                values,
                index=index,
                name=FRED_SERIES_TO_NAME.get(series_id, series_id),
                dtype="float64",
            )

        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(backoff_seconds * (2 ** attempt))

    raise RuntimeError(
        f"Failed to download FRED series {series_id} after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def download_fred_data(
    start: str = "2006-01-01",
    end: str = "2024-12-31",
    series_ids: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download the configured FRED panel into one wide dataframe."""
    if series_ids is None:
        series_ids = DOWNLOAD_SERIES
    if api_key is None:
        api_key = get_fred_api_key(".env")

    all_series = []
    for series_id in series_ids:
        series = fetch_fred_series(series_id=series_id, api_key=api_key, start_date=start, end_date=end)
        all_series.append(series)

    if not all_series:
        return pd.DataFrame()

    df = pd.concat(all_series, axis=1, sort = True)
    df.index.name = "Date"
    return df


def build_raw_download_metadata(start: str, end: str) -> dict:
    return {
        "source": "FRED API",
        "start_date": start,
        "end_date": end,
        "series_ids": DOWNLOAD_SERIES,
        "series_name_map": FRED_SERIES_TO_NAME,
        "notes": (
            "Raw FRED level series saved before any filling, spread construction, differencing, or log-return transforms. "
            "FRED series may have different publication calendars and missing-value patterns."
        ),
    }


# -----------------------------
# Audit / cleaning helpers
# -----------------------------

def get_missing_runs(series: pd.Series) -> List[dict]:
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


def is_leading_run(series: pd.Series, run: dict) -> bool:
    return bool(len(series.index) > 0 and run["start"] == series.index[0])


def classify_missingness(missing_count: int, max_gap: int, leading_gap: int) -> str:
    if missing_count == 0:
        return "No missing values"
    if leading_gap >= 20:
        return "Late-start or structurally missing"
    if max_gap <= 5:
        return "Short-calendar gaps only"
    return "Contains material gaps"


def summarize_missingness(series: pd.Series, peer_missing_median: float) -> dict:
    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()
    missing_count = int(series.isna().sum())
    runs = get_missing_runs(series)
    max_gap = max((run["length"] for run in runs), default=0)
    leading_gap = runs[0]["length"] if runs and runs[0]["start"] == series.index[0] else 0
    classification = classify_missingness(missing_count, max_gap, leading_gap)

    if classification == "No missing values":
        note = "No missing values over the requested date range."
    elif classification == "Short-calendar gaps only":
        note = "Missing values appear limited to short reporting or calendar gaps."
    elif leading_gap >= 20:
        note = f"Long leading gap before the series begins ({leading_gap} rows)."
    elif missing_count > peer_missing_median:
        longest_run = max(runs, key=lambda run: run["length"])
        note = (
            f"More missing values than the typical series ({missing_count} vs median {int(peer_missing_median)}); "
            f"longest gap is {longest_run['length']} rows from "
            f"{longest_run['start'].strftime('%Y-%m-%d')} to {longest_run['end'].strftime('%Y-%m-%d')}."
        )
    else:
        note = "Contains some gaps that may need cautious forward fill or later exclusion."

    return {
        "first_valid_date": first_valid.strftime("%Y-%m-%d") if pd.notna(first_valid) else None,
        "last_valid_date": last_valid.strftime("%Y-%m-%d") if pd.notna(last_valid) else None,
        "missing_values": missing_count,
        "max_consecutive_missing_gap": int(max_gap),
        "missing_gap_assessment": classification,
        "audit_note": note,
    }


def build_data_audit_report(df: pd.DataFrame) -> pd.DataFrame:
    missing_counts = df.isna().sum()
    peer_missing_median = float(missing_counts.median()) if len(missing_counts) else 0.0
    report = pd.DataFrame.from_dict(
        {
            column: summarize_missingness(df[column], peer_missing_median)
            for column in df.columns
        },
        orient="index",
    )
    report.index.name = "series"
    return report


def build_feature_selection_report(audit_report: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame.from_dict(FEATURE_DECISIONS, orient="index")
    report.index.name = "series"
    if not audit_report.empty:
        report["missing_gap_assessment"] = audit_report["missing_gap_assessment"]
        report["audit_note"] = audit_report["audit_note"]
    report["series_id"] = [NAME_TO_SERIES_ID.get(name, "") for name in report.index]
    report["category"] = [SERIES_METADATA.get(name, {}).get("category", "") for name in report.index]
    report["units"] = [SERIES_METADATA.get(name, {}).get("units", "") for name in report.index]
    return report


def select_feature_set(df: pd.DataFrame, selection_report: pd.DataFrame) -> pd.DataFrame:
    kept_columns = selection_report.index[selection_report["keep"]].tolist()
    return df.loc[:, kept_columns].copy()


def forward_fill_short_gaps(series: pd.Series, max_gap: int = 5) -> pd.Series:
    filled = series.copy()
    for run in get_missing_runs(series):
        if is_leading_run(series, run) or run["length"] > max_gap:
            continue
        previous_idx = series.index.get_loc(run["start"]) - 1
        if previous_idx < 0:
            continue
        previous_value = series.iloc[previous_idx]
        if pd.isna(previous_value):
            continue
        fill_mask = (filled.index >= run["start"]) & (filled.index <= run["end"])
        filled.loc[fill_mask] = previous_value
    return filled


def clean_feature_set(df: pd.DataFrame, max_forward_fill_gap: int = 5) -> pd.DataFrame:
    cleaned_df = df.copy()
    for column in cleaned_df.columns:
        cleaned_df[column] = forward_fill_short_gaps(cleaned_df[column], max_gap=max_forward_fill_gap)
    return cleaned_df


def build_cleaning_report(df: pd.DataFrame, max_forward_fill_gap: int = 5) -> pd.DataFrame:
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
        rows[column] = {
            "fill_method": f"forward_fill_only_up_to_{max_forward_fill_gap}_rows",
            "short_gaps_filled": len(short_fillable_runs),
            "remaining_unfilled_gaps": len(protected_runs),
            "cleaning_reason": (
                "Apply only short forward fill for reporting/calendar gaps. Leave leading or longer gaps missing "
                "to avoid introducing stronger assumptions."
            ),
        }
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "series"
    return report


# -----------------------------
# Feature construction
# -----------------------------

def compute_log_return(series: pd.Series) -> pd.Series:
    previous = series.shift(1)
    valid_mask = (series > 0) & (previous > 0)
    log_return = pd.Series(np.nan, index=series.index, dtype="float64")
    log_return.loc[valid_mask] = np.log(series.loc[valid_mask] / previous.loc[valid_mask])
    return log_return


def compute_first_difference(series: pd.Series) -> pd.Series:
    return series.diff()


def build_derived_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    derived = pd.DataFrame(index=df.index)

    # Keep a clean level copy for straightforward merging or inspection later.
    for column in df.columns:
        derived[f"{column} | level"] = df[column]

    # Curve and policy spreads.
    if {"10Y Treasury Yield", "2Y Treasury Yield"}.issubset(df.columns):
        derived["10Y-2Y Treasury Spread | level"] = df["10Y Treasury Yield"] - df["2Y Treasury Yield"]
    if {"10Y Treasury Yield", "3M Treasury Bill Yield"}.issubset(df.columns):
        derived["10Y-3M Treasury Spread | level"] = df["10Y Treasury Yield"] - df["3M Treasury Bill Yield"]
    if {"2Y Treasury Yield", "Effective Federal Funds Rate"}.issubset(df.columns):
        derived["2Y Minus Fed Funds Spread | level"] = df["2Y Treasury Yield"] - df["Effective Federal Funds Rate"]
    if {"10Y Treasury Yield", "10Y Treasury Real Yield"}.issubset(df.columns):
        derived["10Y Term Inflation Wedge | level"] = df["10Y Treasury Yield"] - df["10Y Treasury Real Yield"]

    # Stationary-style transforms.
    for column in df.columns:
        hint = SERIES_METADATA.get(column, {}).get("transform_hint", "level_and_diff")
        if hint == "level_and_log_return":
            derived[f"{column} | log_return"] = compute_log_return(df[column])
        else:
            derived[f"{column} | diff_1"] = compute_first_difference(df[column])

    # Diffs of spreads can be useful for forecasting and causal-discovery baselines.
    for spread_col in [
        "10Y-2Y Treasury Spread | level",
        "10Y-3M Treasury Spread | level",
        "2Y Minus Fed Funds Spread | level",
        "10Y Term Inflation Wedge | level",
    ]:
        if spread_col in derived.columns:
            derived[spread_col.replace(" | level", " | diff_1")] = compute_first_difference(derived[spread_col])

    return derived


def build_derived_feature_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for column in df.columns:
        hint = SERIES_METADATA.get(column, {}).get("transform_hint", "level_and_diff")
        if hint == "level_and_log_return":
            rows[column] = {
                "derived_columns": f"{column} | level; {column} | log_return",
                "derivation_reason": "Keep the level and add a daily log return because this series behaves like a price or FX level.",
            }
        else:
            rows[column] = {
                "derived_columns": f"{column} | level; {column} | diff_1",
                "derivation_reason": "Keep the level and add a first difference because this series is a yield, spread, rate, or credit condition indicator.",
            }

    spread_rows = {
        "10Y-2Y Treasury Spread": {
            "derived_columns": "10Y-2Y Treasury Spread | level; 10Y-2Y Treasury Spread | diff_1",
            "derivation_reason": "Classic term-spread proxy for curve slope and recession / macro regime information.",
        },
        "10Y-3M Treasury Spread": {
            "derived_columns": "10Y-3M Treasury Spread | level; 10Y-3M Treasury Spread | diff_1",
            "derivation_reason": "Alternative yield-curve slope using the 3M bill rate instead of the 2Y node.",
        },
        "2Y Minus Fed Funds Spread": {
            "derived_columns": "2Y Minus Fed Funds Spread | level; 2Y Minus Fed Funds Spread | diff_1",
            "derivation_reason": "Captures how the 2Y Treasury sits relative to the effective policy rate.",
        },
        "10Y Term Inflation Wedge": {
            "derived_columns": "10Y Term Inflation Wedge | level; 10Y Term Inflation Wedge | diff_1",
            "derivation_reason": "Approximate nominal-minus-real wedge related to expected inflation compensation.",
        },
    }

    rows.update(spread_rows)
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "source_or_construct"
    return report


def build_lagged_feature_dataset(derived_df: pd.DataFrame, lags: tuple = PREDICTOR_LAGS) -> pd.DataFrame:
    """Create a FRED-only lagged feature table from the derived feature panel."""
    lagged_data = {}
    for column in derived_df.columns:
        for lag in lags:
            lagged_data[f"{column} | lag_{lag}"] = derived_df[column].shift(lag)
    return pd.DataFrame(lagged_data, index=derived_df.index)


def build_lagged_feature_report(derived_df: pd.DataFrame, lags: tuple = PREDICTOR_LAGS) -> pd.DataFrame:
    rows = {}
    for column in derived_df.columns:
        lagged_columns = [f"{column} | lag_{lag}" for lag in lags]
        rows[column] = {
            "lagged_columns": "; ".join(lagged_columns),
            "lagging_reason": (
                "Create observation-date lags for standalone FRED-only forecasting experiments. "
                "These are useful for inspection and baseline modeling, while a later merger script can still "
                "recompute a unified lag policy across the combined Yahoo + FRED panel."
            ),
        }
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "derived_feature"
    return report


def build_feature_dictionary(base_daily_df: pd.DataFrame, derived_df: pd.DataFrame, lagged_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    rows.append({
        "output_file": "fred_base_daily.csv",
        "column_name": "Date",
        "source_series_id": "",
        "source_series": "Date",
        "human_readable_meaning": "Observation date for the daily FRED panel.",
        "transform_used": "None",
        "units_or_interpretation": "Calendar date in YYYY-MM-DD format.",
        "feature_role": "date",
    })

    for column in base_daily_df.columns:
        rows.append({
            "output_file": "fred_base_daily.csv",
            "column_name": column,
            "source_series_id": NAME_TO_SERIES_ID.get(column, ""),
            "source_series": column,
            "human_readable_meaning": f"Daily level for {column} downloaded from FRED.",
            "transform_used": "Level",
            "units_or_interpretation": SERIES_METADATA.get(column, {}).get("units", ""),
            "feature_role": "raw feature",
        })

    rows.append({
        "output_file": "fred_derived_features.csv",
        "column_name": "Date",
        "source_series_id": "",
        "source_series": "Date",
        "human_readable_meaning": "Observation date for the derived daily FRED feature panel.",
        "transform_used": "None",
        "units_or_interpretation": "Calendar date in YYYY-MM-DD format.",
        "feature_role": "date",
    })

    for column in derived_df.columns:
        base_name, transform_name = column.split(" | ", 1)
        source_series_id = NAME_TO_SERIES_ID.get(base_name, "")

        if transform_name == "level":
            transform_used = "Level"
            units = SERIES_METADATA.get(base_name, {}).get("units", "") if source_series_id else "derived spread level in percentage points"
        elif transform_name == "diff_1":
            transform_used = "First difference"
            units = "One-day change in the underlying level"
        elif transform_name == "log_return":
            transform_used = "Daily log return"
            units = "Daily log return in decimal units"
        else:
            transform_used = transform_name
            units = ""

        rows.append({
            "output_file": "fred_derived_features.csv",
            "column_name": column,
            "source_series_id": source_series_id,
            "source_series": base_name,
            "human_readable_meaning": f"{transform_used} for {base_name}.",
            "transform_used": transform_used,
            "units_or_interpretation": units,
            "feature_role": "derived feature",
        })

    if lagged_df is not None:
        rows.append({
            "output_file": "fred_lagged_features.csv",
            "column_name": "Date",
            "source_series_id": "",
            "source_series": "Date",
            "human_readable_meaning": "Observation date for the lagged daily FRED feature panel.",
            "transform_used": "None",
            "units_or_interpretation": "Calendar date in YYYY-MM-DD format.",
            "feature_role": "date",
        })

        for column in lagged_df.columns:
            parts = column.split(" | ")
            if len(parts) == 3 and parts[2].startswith("lag_"):
                base_name = parts[0]
                base_transform = parts[1]
                lag_name = parts[2]
                lag_days = int(lag_name.split("_")[1])
            else:
                base_name = column
                base_transform = "derived value"
                lag_days = None

            source_series_id = NAME_TO_SERIES_ID.get(base_name, "")
            if base_transform == "level":
                base_transform_label = "Level"
            elif base_transform == "diff_1":
                base_transform_label = "First difference"
            elif base_transform == "log_return":
                base_transform_label = "Daily log return"
            else:
                base_transform_label = base_transform

            lag_phrase = f"{lag_days}-day lag" if lag_days is not None else "Lagged value"
            rows.append({
                "output_file": "fred_lagged_features.csv",
                "column_name": column,
                "source_series_id": source_series_id,
                "source_series": base_name,
                "human_readable_meaning": f"{lag_phrase} of the {base_transform_label.lower()} for {base_name}.",
                "transform_used": f"{base_transform_label}; then lagged by {lag_days} trading day(s)" if lag_days is not None else base_transform_label,
                "units_or_interpretation": "Lagged version of the corresponding derived FRED feature.",
                "feature_role": "lagged predictor",
            })

    return pd.DataFrame(rows)


def build_no_scaling_policy() -> dict:
    return {
        "scaling_applied_in_preprocessing_handoff": False,
        "applies_to": [
            "fred_base_daily.csv",
            "fred_base_daily_dropna.csv",
            "fred_derived_features.csv",
            "fred_derived_features_dropna.csv",
            "fred_lagged_features.csv",
            "fred_lagged_features_dropna.csv",
        ],
        "reason": [
            "Scaling must be fit on the training split only after the time-series split is chosen.",
            "Global scaling before splitting would leak future information into earlier periods.",
            "This script is intended only to download, audit, clean, derive, and optionally lag features from FRED data.",
        ],
    }


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> None:
    start_date = "2006-01-01"
    end_date = "2024-12-31"

    # Step 1. Raw FRED levels.
    raw_df = download_fred_data(start=start_date, end=end_date)
    save_data(raw_df, step_file("step_1_raw_fred_series.csv"))
    save_json(build_raw_download_metadata(start=start_date, end=end_date), step_file("step_1_raw_fred_series_metadata.json"))

    # Step 2. Audit the raw panel.
    audit_report = build_data_audit_report(raw_df)
    save_data(audit_report, step_file("step_2_fred_data_audit_report.csv"))

    # Step 3. Selection report and retained raw panel.
    selection_report = build_feature_selection_report(audit_report)
    save_data(selection_report, step_file("step_3_fred_feature_selection_report.csv"))
    selected_df = select_feature_set(raw_df, selection_report)
    save_data(selected_df, step_file("step_3_fred_selected_raw_series.csv"))

    # Step 4. Conservative cleaning.
    cleaning_report = build_cleaning_report(selected_df, max_forward_fill_gap=5)
    save_data(cleaning_report, step_file("step_4_fred_cleaning_report.csv"))
    cleaned_df = clean_feature_set(selected_df, max_forward_fill_gap=5)
    save_data(cleaned_df, step_file("step_4_fred_cleaned_series.csv"))
    save_data(cleaned_df.dropna(how="any"), step_file("step_4_fred_cleaned_series_dropna.csv"))

    # Step 5. Derived rates / spreads / returns panel.
    derived_report = build_derived_feature_report(cleaned_df)
    save_data(derived_report, step_file("step_5_fred_derived_feature_report.csv"))
    derived_df = build_derived_feature_dataset(cleaned_df)
    save_data(derived_df, step_file("step_5_fred_derived_features.csv"))
    save_data(derived_df.dropna(how="any"), step_file("step_5_fred_derived_features_dropna.csv"))

    # Step 6. Optional standalone lagged FRED-only predictor panel.
    lagged_feature_report = build_lagged_feature_report(derived_df, lags=PREDICTOR_LAGS)
    save_data(lagged_feature_report, step_file("step_6_fred_lagged_feature_report.csv"))
    lagged_df = build_lagged_feature_dataset(derived_df, lags=PREDICTOR_LAGS)
    save_data(lagged_df, step_file("step_6_fred_lagged_features.csv"))
    save_data(lagged_df.dropna(how="any"), step_file("step_6_fred_lagged_features_dropna.csv"))

    # Step 7. Informational files for downstream use.
    feature_dictionary = build_feature_dictionary(cleaned_df, derived_df, lagged_df=lagged_df)
    feature_dictionary.to_csv(result_file("feature_dictionary.csv"), index=False)
    save_json(build_no_scaling_policy(), result_file("no_scaling_policy.json"))

    # Step 8. Clean alias copies for downstream usage.
    export_final_handoff_aliases()


if __name__ == "__main__":
    main()
