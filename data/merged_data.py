import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pandas as pd


PREDICTOR_LAGS: Tuple[int, ...] = (1, 5, 10, 20)

MERGE_STEP_OUTPUT_DIR = "pipeline_steps"
MERGE_RESULTS_DIR = "results"

SHARED_STEP_DIR = Path("pipeline_steps")
SHARED_RESULTS_DIR = Path("results")

YF_ARCHIVE_STEP_DIR = Path("yfinance_pipeline_steps")
YF_ARCHIVE_RESULTS_DIR = Path("yfinance_results")

FRED_ARCHIVE_STEP_DIR = Path("fred_pipeline_steps")
FRED_ARCHIVE_RESULTS_DIR = Path("fred_results")

YF_SHARED_BASE_STATIONARY_PATH = SHARED_STEP_DIR / "step_5_stationary_base_data.csv"
YF_SHARED_TARGET_PATH = SHARED_STEP_DIR / "step_6_target_data.csv"
FRED_SHARED_DERIVED_PATH = SHARED_STEP_DIR / "step_5_fred_derived_features.csv"

YF_BASE_STATIONARY_PATH = YF_ARCHIVE_STEP_DIR / "step_5_stationary_base_data.csv"
YF_TARGET_PATH = YF_ARCHIVE_STEP_DIR / "step_6_target_data.csv"
FRED_DERIVED_PATH = FRED_ARCHIVE_STEP_DIR / "step_5_fred_derived_features.csv"

FINAL_HANDOFF_FILE_ALIASES = {
    "step_2_merged_unlagged_features.csv": "gold_base_stationary.csv",
    "step_2_merged_unlagged_features_dropna.csv": "gold_base_stationary_dropna.csv",
    "step_3_merged_unified_lagged_features.csv": "lagged_features.csv",
    "step_3_merged_unified_lagged_features_dropna.csv": "lagged_features_dropna.csv",
    "step_4_gold_forecast_features.csv": "gold_forecast_features.csv",
    "step_4_gold_forecast_features_t1.csv": "gold_forecast_features_t1.csv",
    "step_4_gold_forecast_features_t5.csv": "gold_forecast_features_t5.csv",
}

DEFAULT_SPLIT_DEFINITION = {
    "split_type": "chronological",
    "shuffle": False,
    "applies_to": [
        "gold_base_stationary.csv",
        "gold_base_stationary_dropna.csv",
        "lagged_features.csv",
        "lagged_features_dropna.csv",
        "gold_forecast_features.csv",
        "gold_forecast_features_t1.csv",
        "gold_forecast_features_t5.csv",
    ],
    "train": {"start_date": "2006-01-03", "end_date": "2018-12-31"},
    "validation": {"start_date": "2019-01-01", "end_date": "2021-12-31"},
    "test": {"start_date": "2022-01-01", "end_date": "2024-12-31"},
    "notes": "Use fixed chronological splits for all merged time-series experiments. Do not randomly split observations.",
}

NO_SCALING_POLICY = {
    "scaling_applied_in_preprocessing_handoff": False,
    "applies_to": list(FINAL_HANDOFF_FILE_ALIASES.values()),
    "reason": [
        "Scaling must be fit on the training split only.",
        "Global standardization before the split would leak information from validation and test periods.",
        "This merged handoff should be cleaned, merged, and uniformly lagged, but not globally standardized.",
    ],
    "instruction_for_downstream_use": "Fit any scaler on the training split only, then apply that fitted scaler to validation and test splits.",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def step_file(filename: str) -> Path:
    ensure_dir(Path(MERGE_STEP_OUTPUT_DIR))
    return Path(MERGE_STEP_OUTPUT_DIR) / filename


def result_file(filename: str) -> Path:
    ensure_dir(Path(MERGE_RESULTS_DIR))
    return Path(MERGE_RESULTS_DIR) / filename


def save_data(df: pd.DataFrame, filename: Path) -> None:
    df.to_csv(filename, index=True)


def save_json(data: dict, filename: Path) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_upstream_pipeline_if_needed(script_path: Path, sentinel_paths: Sequence[Path], module_name: str) -> None:
    if all(path.exists() for path in sentinel_paths):
        return
    module = load_module_from_path(module_name, script_path)
    if not hasattr(module, "main"):
        raise RuntimeError(f"{script_path} does not define a main() function.")
    module.main()


def read_indexed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "Date"
    return df


def assert_unique_columns(df: pd.DataFrame, label: str) -> None:
    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{label} contains duplicate columns: {duplicates}")


def maybe_get_split_definition(yfinance_script_path: Path) -> dict:
    try:
        module = load_module_from_path("yfinance_data_for_split", yfinance_script_path)
        split_definition = getattr(module, "SPLIT_DEFINITION", None)
        if isinstance(split_definition, dict):
            merged = dict(split_definition)
            merged["applies_to"] = list(FINAL_HANDOFF_FILE_ALIASES.values())
            merged["notes"] = (
                "Use fixed chronological splits for all merged time-series experiments "
                "after Yahoo and FRED feature alignment. Do not randomly split observations."
            )
            return merged
    except Exception:
        pass
    return DEFAULT_SPLIT_DEFINITION


def build_merge_report(yf_df: pd.DataFrame, fred_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "input_panel": "yfinance_stationary_panel",
            "rows": int(len(yf_df)),
            "columns": int(yf_df.shape[1]),
            "start_date": yf_df.index.min().strftime("%Y-%m-%d") if not yf_df.empty else None,
            "end_date": yf_df.index.max().strftime("%Y-%m-%d") if not yf_df.empty else None,
            "notes": "Yahoo Finance stationary-style feature panel before merged lagging.",
        },
        {
            "input_panel": "fred_derived_panel",
            "rows": int(len(fred_df)),
            "columns": int(fred_df.shape[1]),
            "start_date": fred_df.index.min().strftime("%Y-%m-%d") if not fred_df.empty else None,
            "end_date": fred_df.index.max().strftime("%Y-%m-%d") if not fred_df.empty else None,
            "notes": "FRED derived macro / rates feature panel before merged lagging.",
        },
        {
            "input_panel": "merged_unlagged_panel",
            "rows": int(len(merged_df)),
            "columns": int(merged_df.shape[1]),
            "start_date": merged_df.index.min().strftime("%Y-%m-%d") if not merged_df.empty else None,
            "end_date": merged_df.index.max().strftime("%Y-%m-%d") if not merged_df.empty else None,
            "notes": "Merged panel aligned to Yahoo Finance trading dates using a left join from the Yahoo panel.",
        },
    ]
    return pd.DataFrame(rows)


def merge_feature_panels(yf_stationary_df: pd.DataFrame, fred_derived_df: pd.DataFrame) -> pd.DataFrame:
    assert_unique_columns(yf_stationary_df, "Yahoo feature panel")
    assert_unique_columns(fred_derived_df, "FRED feature panel")

    overlapping = sorted(set(yf_stationary_df.columns).intersection(fred_derived_df.columns))
    if overlapping:
        fred_derived_df = fred_derived_df.rename(columns={column: f"FRED::{column}" for column in overlapping})

    merged_df = yf_stationary_df.join(fred_derived_df, how="left")
    merged_df = merged_df.sort_index()
    merged_df.index.name = "Date"
    return merged_df


def build_lagged_feature_dataset(feature_df: pd.DataFrame, lags: Iterable[int] = PREDICTOR_LAGS) -> pd.DataFrame:
    lagged_data = {}
    for column in feature_df.columns:
        for lag in lags:
            lagged_data[f"{column} | lag_{lag}"] = feature_df[column].shift(lag)
    lagged_df = pd.DataFrame(lagged_data, index=feature_df.index)
    lagged_df.index.name = "Date"
    return lagged_df


def build_lagged_feature_report(feature_df: pd.DataFrame, lags: Iterable[int] = PREDICTOR_LAGS) -> pd.DataFrame:
    rows = {}
    for column in feature_df.columns:
        lagged_columns = [f"{column} | lag_{lag}" for lag in lags]
        rows[column] = {
            "lagged_columns": "; ".join(lagged_columns),
            "lagging_reason": (
                "Apply one uniform lagging policy after Yahoo and FRED features are merged "
                "so all predictors are aligned consistently."
            ),
        }
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "source_predictor"
    return report


def build_forecast_handoff_dataset(
    lagged_df: pd.DataFrame,
    target_df: pd.DataFrame,
    drop_missing: bool = True,
) -> pd.DataFrame:
    handoff_df = lagged_df.join(target_df, how="inner")
    handoff_df = handoff_df.sort_index()
    if drop_missing:
        handoff_df = handoff_df.dropna(how="any")
    handoff_df.index.name = "Date"
    return handoff_df


def build_single_target_output(forecast_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    predictor_columns = [
        column
        for column in forecast_df.columns
        if not column.startswith("Gold Futures (COMEX) | target_log_return_t_plus_")
    ]
    return forecast_df[predictor_columns + [target_column]].copy()


def build_feature_dictionary(
    merged_unlagged_df: pd.DataFrame,
    lagged_df: pd.DataFrame,
    forecast_t1_df: pd.DataFrame,
    forecast_t5_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add_date_row(output_file: str, meaning: str) -> None:
        rows.append(
            {
                "output_file": output_file,
                "column_name": "Date",
                "source_panel": "merged",
                "source_series": "Date",
                "human_readable_meaning": meaning,
                "transform_used": "None",
                "units_or_interpretation": "Calendar date in YYYY-MM-DD format.",
                "feature_role": "date",
            }
        )

    add_date_row(
        "gold_base_stationary.csv",
        "Observation date for the merged Yahoo Finance and FRED feature panel before unified lagging.",
    )
    for column in merged_unlagged_df.columns:
        rows.append(
            {
                "output_file": "gold_base_stationary.csv",
                "column_name": column,
                "source_panel": infer_source_panel(column),
                "source_series": infer_base_series_name(column),
                "human_readable_meaning": f"Merged unlagged feature for {infer_base_series_name(column)}.",
                "transform_used": infer_transform_description(column),
                "units_or_interpretation": infer_units(column),
                "feature_role": "merged feature",
            }
        )

    add_date_row(
        "lagged_features.csv",
        "Observation date for the uniformly lagged merged predictor panel.",
    )
    for column in lagged_df.columns:
        rows.append(
            {
                "output_file": "lagged_features.csv",
                "column_name": column,
                "source_panel": infer_source_panel(column),
                "source_series": infer_base_series_name(column),
                "human_readable_meaning": f"Uniformly lagged merged predictor for {infer_base_series_name(column)}.",
                "transform_used": infer_transform_description(column),
                "units_or_interpretation": infer_units(column),
                "feature_role": "lagged predictor",
            }
        )

    add_date_row(
        "gold_forecast_features_t1.csv",
        "Observation date for the merged lagged predictor panel with the 1-day-ahead gold target.",
    )
    for column in forecast_t1_df.columns:
        rows.append(
            {
                "output_file": "gold_forecast_features_t1.csv",
                "column_name": column,
                "source_panel": infer_source_panel(column),
                "source_series": infer_base_series_name(column),
                "human_readable_meaning": infer_target_or_feature_meaning(column),
                "transform_used": infer_transform_description(column),
                "units_or_interpretation": infer_units(column),
                "feature_role": infer_feature_role(column),
            }
        )

    add_date_row(
        "gold_forecast_features_t5.csv",
        "Observation date for the merged lagged predictor panel with the 5-day-ahead gold target.",
    )
    for column in forecast_t5_df.columns:
        rows.append(
            {
                "output_file": "gold_forecast_features_t5.csv",
                "column_name": column,
                "source_panel": infer_source_panel(column),
                "source_series": infer_base_series_name(column),
                "human_readable_meaning": infer_target_or_feature_meaning(column),
                "transform_used": infer_transform_description(column),
                "units_or_interpretation": infer_units(column),
                "feature_role": infer_feature_role(column),
            }
        )

    return pd.DataFrame(rows)


def infer_source_panel(column_name: str) -> str:
    if column_name.startswith("Gold Futures (COMEX) | target_log_return_t_plus_"):
        return "yfinance_target"
    if column_name.startswith("FRED::"):
        return "fred"

    base = infer_base_series_name(column_name)
    fred_base_names = {
        "10Y Treasury Yield",
        "2Y Treasury Yield",
        "1Y Treasury Yield",
        "3M Treasury Bill Yield",
        "Effective Federal Funds Rate",
        "Secured Overnight Financing Rate",
        "10Y Breakeven Inflation",
        "10Y Treasury Real Yield",
        "USD / EUR Exchange Rate (FRED)",
        "JPY / USD Exchange Rate (FRED)",
        "CNY / USD Exchange Rate (FRED)",
        "WTI Spot Price (FRED)",
        "ICE BofA US High Yield OAS",
        "ICE BofA US BBB OAS",
        "10Y-2Y Treasury Spread",
        "10Y-3M Treasury Spread",
        "2Y Minus Fed Funds Spread",
        "10Y Term Inflation Wedge",
    }
    if base in fred_base_names:
        return "fred"
    return "yfinance"


def infer_base_series_name(column_name: str) -> str:
    if column_name.startswith("FRED::"):
        column_name = column_name.removeprefix("FRED::")
    if " | lag_" in column_name:
        column_name = column_name.rsplit(" | lag_", 1)[0]
    if " | target_log_return_t_plus_" in column_name:
        return column_name.split(" | target_log_return_t_plus_", 1)[0]
    return column_name.split(" | ", 1)[0]


def infer_transform_description(column_name: str) -> str:
    if " | target_log_return_t_plus_" in column_name:
        horizon = column_name.rsplit("_", 1)[1]
        return f"Future log return over {horizon} trading day(s)"

    working = column_name.removeprefix("FRED::")
    lag_text = ""
    if " | lag_" in working:
        working, lag_part = working.rsplit(" | lag_", 1)
        lag_text = f"; then lagged by {lag_part} trading day(s)"

    if " | " not in working:
        return f"Level{lag_text}"

    transform = working.split(" | ", 1)[1]
    mapping = {
        "log_return": "Daily log return",
        "diff_1": "First difference",
        "level": "Level",
        "momentum_5": "5-day momentum",
        "momentum_10": "10-day momentum",
        "rolling_volatility_20": "20-day rolling volatility",
        "moving_average_ratio_20": "20-day moving-average ratio",
    }
    return f"{mapping.get(transform, transform)}{lag_text}"


def infer_units(column_name: str) -> str:
    if " | target_log_return_t_plus_" in column_name:
        return "Forward log return in decimal units"
    if "log_return" in column_name or "momentum_" in column_name or "moving_average_ratio_" in column_name:
        return "Decimal return-style quantity"
    if "rolling_volatility_" in column_name:
        return "Rolling standard deviation in decimal units"
    if "diff_1" in column_name:
        return "One-day change in the underlying level"
    if "level" in column_name or "Treasury" in column_name or "Federal Funds" in column_name or "OAS" in column_name:
        return "Level of the underlying series"
    return "See source series definition"


def infer_target_or_feature_meaning(column_name: str) -> str:
    if " | target_log_return_t_plus_" in column_name:
        horizon = column_name.rsplit("_", 1)[1]
        return f"{horizon}-day-ahead gold log return target measured from day t to day t+{horizon}."
    if " | lag_" in column_name:
        base = infer_base_series_name(column_name)
        lag = column_name.rsplit(" | lag_", 1)[1]
        return f"{lag}-day lag of the merged feature for {base}."
    return f"Merged feature for {infer_base_series_name(column_name)}."


def infer_feature_role(column_name: str) -> str:
    if " | target_log_return_t_plus_" in column_name:
        return "target"
    if " | lag_" in column_name:
        return "lagged predictor"
    return "merged feature"


def export_final_handoff_aliases() -> None:
    for source_filename, alias_filename in FINAL_HANDOFF_FILE_ALIASES.items():
        shutil.copyfile(step_file(source_filename), result_file(alias_filename))


def clear_directory_contents(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def copy_directory_contents(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    if not src_dir.exists():
        return
    for item in src_dir.iterdir():
        destination = dst_dir / item.name
        if item.is_file():
            shutil.copy2(item, destination)
        elif item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)


def archive_shared_outputs(step_archive_dir: Path, results_archive_dir: Path) -> None:
    clear_directory_contents(step_archive_dir)
    clear_directory_contents(results_archive_dir)
    copy_directory_contents(SHARED_STEP_DIR, step_archive_dir)
    copy_directory_contents(SHARED_RESULTS_DIR, results_archive_dir)


def run_and_archive_yfinance(yfinance_script: Path) -> None:
    clear_directory_contents(SHARED_STEP_DIR)
    clear_directory_contents(SHARED_RESULTS_DIR)
    run_upstream_pipeline_if_needed(
        yfinance_script,
        sentinel_paths=[YF_SHARED_BASE_STATIONARY_PATH, YF_SHARED_TARGET_PATH],
        module_name="yfinance_data_for_merge",
    )
    archive_shared_outputs(YF_ARCHIVE_STEP_DIR, YF_ARCHIVE_RESULTS_DIR)
    clear_directory_contents(SHARED_STEP_DIR)
    clear_directory_contents(SHARED_RESULTS_DIR)


def run_and_archive_fred(fred_script: Path) -> None:
    clear_directory_contents(SHARED_STEP_DIR)
    clear_directory_contents(SHARED_RESULTS_DIR)
    run_upstream_pipeline_if_needed(
        fred_script,
        sentinel_paths=[FRED_SHARED_DERIVED_PATH],
        module_name="fred_data_for_merge",
    )
    archive_shared_outputs(FRED_ARCHIVE_STEP_DIR, FRED_ARCHIVE_RESULTS_DIR)
    clear_directory_contents(SHARED_STEP_DIR)
    clear_directory_contents(SHARED_RESULTS_DIR)


def ensure_archived_upstream_outputs(yfinance_script: Path, fred_script: Path, rerun_upstream: bool) -> None:
    yf_missing = not (YF_BASE_STATIONARY_PATH.exists() and YF_TARGET_PATH.exists())
    fred_missing = not FRED_DERIVED_PATH.exists()

    if rerun_upstream or yf_missing:
        run_and_archive_yfinance(yfinance_script)

    if rerun_upstream or fred_missing:
        run_and_archive_fred(fred_script)


def main(run_upstream_first: bool = True) -> None:
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    yfinance_script = script_dir / "yfinance_data.py"
    fred_script = script_dir / "fred_data.py"

    ensure_dir(YF_ARCHIVE_STEP_DIR)
    ensure_dir(YF_ARCHIVE_RESULTS_DIR)
    ensure_dir(FRED_ARCHIVE_STEP_DIR)
    ensure_dir(FRED_ARCHIVE_RESULTS_DIR)

    ensure_archived_upstream_outputs(
        yfinance_script=yfinance_script,
        fred_script=fred_script,
        rerun_upstream=run_upstream_first,
    )

    clear_directory_contents(SHARED_STEP_DIR)
    clear_directory_contents(SHARED_RESULTS_DIR)

    yf_stationary_df = read_indexed_csv(YF_BASE_STATIONARY_PATH)
    fred_derived_df = read_indexed_csv(FRED_DERIVED_PATH)
    target_df = read_indexed_csv(YF_TARGET_PATH)

    merged_unlagged_df = merge_feature_panels(yf_stationary_df, fred_derived_df)
    merged_unlagged_df = merged_unlagged_df.drop(columns=[
        "Secured Overnight Financing Rate | level",
        "Secured Overnight Financing Rate | diff_1",
    ], errors="ignore")
    # Drop SOFR (starts 2018-04, truncates training data from 4900 to 1750 rows)
    # Fed Funds Rate serves as close substitute
    merged_unlagged_dropna_df = merged_unlagged_df.dropna(how="any")

    save_json(
        {
            "yfinance_stationary_input": str(YF_BASE_STATIONARY_PATH),
            "yfinance_target_input": str(YF_TARGET_PATH),
            "fred_derived_input": str(FRED_DERIVED_PATH),
            "yfinance_archive_step_dir": str(YF_ARCHIVE_STEP_DIR),
            "yfinance_archive_results_dir": str(YF_ARCHIVE_RESULTS_DIR),
            "fred_archive_step_dir": str(FRED_ARCHIVE_STEP_DIR),
            "fred_archive_results_dir": str(FRED_ARCHIVE_RESULTS_DIR),
            "lag_policy": list(PREDICTOR_LAGS),
            "merge_alignment": (
                "Left join on Yahoo Finance trading dates, then apply uniform lagging "
                "to the merged predictor panel."
            ),
            "upstream_output_behavior": (
                "Yahoo and FRED each run into the shared pipeline_steps/results folders, "
                "are archived into source-specific folders, and then merged outputs are "
                "written back into the shared pipeline_steps/results folders."
            ),
        },
        step_file("step_1_merge_inputs_metadata.json"),
    )

    merge_report = build_merge_report(yf_stationary_df, fred_derived_df, merged_unlagged_df)
    save_data(merge_report, step_file("step_2_merge_report.csv"))
    save_data(merged_unlagged_df, step_file("step_2_merged_unlagged_features.csv"))
    save_data(merged_unlagged_dropna_df, step_file("step_2_merged_unlagged_features_dropna.csv"))

    lagged_feature_report = build_lagged_feature_report(merged_unlagged_df, lags=PREDICTOR_LAGS)
    lagged_feature_df = build_lagged_feature_dataset(merged_unlagged_df, lags=PREDICTOR_LAGS)
    lagged_feature_dropna_df = lagged_feature_df.dropna(how="any")
    save_data(lagged_feature_report, step_file("step_3_merged_unified_lagged_feature_report.csv"))
    save_data(lagged_feature_df, step_file("step_3_merged_unified_lagged_features.csv"))
    save_data(lagged_feature_dropna_df, step_file("step_3_merged_unified_lagged_features_dropna.csv"))

    forecast_df = build_forecast_handoff_dataset(lagged_feature_df, target_df, drop_missing=True)
    forecast_t1_df = build_single_target_output(
        forecast_df,
        "Gold Futures (COMEX) | target_log_return_t_plus_1",
    )
    forecast_t5_df = build_single_target_output(
        forecast_df,
        "Gold Futures (COMEX) | target_log_return_t_plus_5",
    )
    save_data(forecast_df, step_file("step_4_gold_forecast_features.csv"))
    save_data(forecast_t1_df, step_file("step_4_gold_forecast_features_t1.csv"))
    save_data(forecast_t5_df, step_file("step_4_gold_forecast_features_t5.csv"))

    feature_dictionary = build_feature_dictionary(
        merged_unlagged_df,
        lagged_feature_df,
        forecast_t1_df,
        forecast_t5_df,
    )
    feature_dictionary.to_csv(result_file("feature_dictionary.csv"), index=False)
    save_json(maybe_get_split_definition(yfinance_script), result_file("split_definition.json"))
    save_json(NO_SCALING_POLICY, result_file("no_scaling_policy.json"))
    export_final_handoff_aliases()


if __name__ == "__main__":
    main()
