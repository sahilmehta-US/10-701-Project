"""
Merge Yahoo Finance stationary features with FRED derived features on Yahoo trading dates.

Reads yfinance_pipeline_steps/step_5_stationary_base_data.csv and
fred_pipeline_steps/step_5_fred_derived_features.csv, left-joins FRED on the Yahoo index,
writes pipeline_steps/step_1_merge_inputs_metadata.json and step_2_merged_* CSVs,
then copies handoff aliases and split metadata into results/.
"""
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Sequence

import pandas as pd


MERGE_STEP_OUTPUT_DIR = "pipeline_steps"
MERGE_RESULTS_DIR = "results"

SHARED_STEP_DIR = Path("pipeline_steps")
SHARED_RESULTS_DIR = Path("results")

YF_ARCHIVE_STEP_DIR = Path("yfinance_pipeline_steps")
YF_ARCHIVE_RESULTS_DIR = Path("yfinance_results")

FRED_ARCHIVE_STEP_DIR = Path("fred_pipeline_steps")
FRED_ARCHIVE_RESULTS_DIR = Path("fred_results")

YF_SHARED_BASE_STATIONARY_PATH = SHARED_STEP_DIR / "step_5_stationary_base_data.csv"
FRED_SHARED_DERIVED_PATH = SHARED_STEP_DIR / "step_5_fred_derived_features.csv"

YF_BASE_STATIONARY_PATH = YF_ARCHIVE_STEP_DIR / "step_5_stationary_base_data.csv"
FRED_DERIVED_PATH = FRED_ARCHIVE_STEP_DIR / "step_5_fred_derived_features.csv"

FINAL_HANDOFF_FILE_ALIASES = {
    "step_2_merged_features.csv": "gold_base_stationary.csv",
    "step_2_merged_features_dropna.csv": "gold_base_stationary_dropna.csv",
}

DEFAULT_SPLIT_DEFINITION = {
    "split_type": "chronological",
    "shuffle": False,
    "applies_to": [
        "gold_base_stationary.csv",
        "gold_base_stationary_dropna.csv",
    ],
    "train": {"start_date": "2006-01-03", "end_date": "2018-12-31"},
    "validation": {"start_date": "2019-01-01", "end_date": "2021-12-31"},
    "test": {"start_date": "2022-01-01", "end_date": "2024-12-31"},
    "notes": "Use fixed chronological splits for all merged time-series experiments. Do not randomly split observations.",
}

# After the merge, row-wise ``dropna(how="any")`` requires every column to be
# non-null. A single late-start or patchy FRED series otherwise deletes almost
# all pre-2020 Yahoo rows and empty splits (e.g. LSTM train 2006–2018).
MERGED_COLUMN_MAX_MISSING_FRACTION = 0.05

NO_SCALING_POLICY = {
    "scaling_applied_in_preprocessing_handoff": False,
    "applies_to": list(FINAL_HANDOFF_FILE_ALIASES.values()),
    "reason": [
        "Scaling must be fit on the training split only.",
        "Global standardization before the split would leak information from validation and test periods.",
        "This merged handoff should be cleaned and merged, but not globally standardized.",
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
            "notes": "Yahoo Finance stationary-style feature panel before merge.",
        },
        {
            "input_panel": "fred_derived_panel",
            "rows": int(len(fred_df)),
            "columns": int(fred_df.shape[1]),
            "start_date": fred_df.index.min().strftime("%Y-%m-%d") if not fred_df.empty else None,
            "end_date": fred_df.index.max().strftime("%Y-%m-%d") if not fred_df.empty else None,
            "notes": "FRED derived macro / rates feature panel before merge.",
        },
        {
            "input_panel": "merged_panel",
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


def build_feature_dictionary(merged_df: pd.DataFrame) -> pd.DataFrame:
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
        "Observation date for the merged Yahoo Finance and FRED feature panel.",
    )
    for column in merged_df.columns:
        rows.append(
            {
                "output_file": "gold_base_stationary.csv",
                "column_name": column,
                "source_panel": infer_source_panel(column),
                "source_series": infer_base_series_name(column),
                "human_readable_meaning": f"Merged feature for {infer_base_series_name(column)}.",
                "transform_used": infer_transform_description(column),
                "units_or_interpretation": infer_units(column),
                "feature_role": "merged feature",
            }
        )

    return pd.DataFrame(rows)


def infer_source_panel(column_name: str) -> str:
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
    return column_name.split(" | ", 1)[0]


def infer_transform_description(column_name: str) -> str:
    working = column_name.removeprefix("FRED::")

    if " | " not in working:
        return "Level"

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
    return mapping.get(transform, transform)


def infer_units(column_name: str) -> str:
    if "log_return" in column_name or "momentum_" in column_name or "moving_average_ratio_" in column_name:
        return "Decimal return-style quantity"
    if "rolling_volatility_" in column_name:
        return "Rolling standard deviation in decimal units"
    if "diff_1" in column_name:
        return "One-day change in the underlying level"
    if "level" in column_name or "Treasury" in column_name or "Federal Funds" in column_name or "OAS" in column_name:
        return "Level of the underlying series"
    return "See source series definition"


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
        sentinel_paths=[YF_SHARED_BASE_STATIONARY_PATH],
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
    yf_missing = not YF_BASE_STATIONARY_PATH.exists()
    fred_missing = not FRED_DERIVED_PATH.exists()

    if rerun_upstream or yf_missing:
        run_and_archive_yfinance(yfinance_script)

    if rerun_upstream or fred_missing:
        run_and_archive_fred(fred_script)


def main(run_upstream_first: bool = True) -> None:
    """Run upstream Yahoo/FRED pipelines if needed, merge panels, write step_1–2 artifacts and results handoff."""
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

    merged_df = merge_feature_panels(yf_stationary_df, fred_derived_df)
    merged_df = merged_df.drop(columns=[
        "Secured Overnight Financing Rate | level",
        "Secured Overnight Financing Rate | diff_1",
    ], errors="ignore")
    # Drop SOFR (starts 2018-04, truncates training data from 4900 to 1750 rows)
    # Fed Funds Rate serves as close substitute

    missing_frac = merged_df.isna().mean()
    sparse_mask = missing_frac > MERGED_COLUMN_MAX_MISSING_FRACTION
    dropped_sparse_columns = sorted(sparse_mask[sparse_mask].index.astype(str).tolist())
    merged_df = merged_df.drop(columns=dropped_sparse_columns, errors="ignore")

    merged_dropna_df = merged_df.dropna(how="any")

    save_json(
        {
            "yfinance_stationary_input": str(YF_BASE_STATIONARY_PATH),
            "fred_derived_input": str(FRED_DERIVED_PATH),
            "yfinance_archive_step_dir": str(YF_ARCHIVE_STEP_DIR),
            "yfinance_archive_results_dir": str(YF_ARCHIVE_RESULTS_DIR),
            "fred_archive_step_dir": str(FRED_ARCHIVE_STEP_DIR),
            "fred_archive_results_dir": str(FRED_ARCHIVE_RESULTS_DIR),
            "merge_alignment": (
                "Left join on Yahoo Finance trading dates for the merged predictor panel."
            ),
            "upstream_output_behavior": (
                "Yahoo and FRED each run into the shared pipeline_steps/results folders, "
                "are archived into source-specific folders, and then merged outputs are "
                "written back into the shared pipeline_steps/results folders."
            ),
            "merged_column_max_missing_fraction": MERGED_COLUMN_MAX_MISSING_FRACTION,
            "dropped_sparse_columns": dropped_sparse_columns,
            "dropped_sparse_columns_note": (
                "Removed before row-wise complete-case filtering so one patchy FRED column "
                "does not erase most of the Yahoo calendar."
            ),
        },
        step_file("step_1_merge_inputs_metadata.json"),
    )

    merge_report = build_merge_report(yf_stationary_df, fred_derived_df, merged_df)
    save_data(merge_report, step_file("step_2_merge_report.csv"))
    save_data(merged_df, step_file("step_2_merged_features.csv"))
    save_data(merged_dropna_df, step_file("step_2_merged_features_dropna.csv"))

    feature_dictionary = build_feature_dictionary(merged_df)
    feature_dictionary.to_csv(result_file("feature_dictionary.csv"), index=False)
    save_json(maybe_get_split_definition(yfinance_script), result_file("split_definition.json"))
    save_json(NO_SCALING_POLICY, result_file("no_scaling_policy.json"))
    export_final_handoff_aliases()


if __name__ == "__main__":
    main()
